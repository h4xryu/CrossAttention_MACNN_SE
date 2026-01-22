"""
ECG Dataset Classes

다양한 ECG 데이터 포맷을 지원하는 Dataset 클래스들입니다.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("ecg_standard")
class ECGStandardDataset(Dataset):
    """
    Standard ECG Dataset (opt1 style)

    기존 실험용 ECG Dataset입니다.
    - 입력: ECG (1, 1, L) + RR features (7,)
    - 2D Conv 모델용 형식
    - RR features는 별도로 전달

    Args:
        data: (N, L) ECG segments
        labels: (N,) class labels
        rr_features: (N, 7) RR features
        patient_ids: (N,) patient IDs
        sample_ids: (N,) sample indices
    """

    def __init__(self, data, labels, rr_features, patient_ids, sample_ids):
        # (N, L) -> (N, 1, 1, L) for 2D Conv model
        self.data = torch.FloatTensor(data).unsqueeze(1).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        self.rr_features = torch.FloatTensor(rr_features)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.data[idx],           # (1, 1, L)
            self.labels[idx],         # scalar
            self.rr_features[idx],    # (7,)
            self.patient_ids[idx],    # scalar
            idx                       # sample index
        )


@DATASET_REGISTRY.register("daeac")
class DAEACDataset(Dataset):
    """
    DAEAC Paper Style Dataset (opt2 style)

    DAEAC 논문 재현용 Dataset입니다.
    - 3채널 입력: [ECG, pre_rr_ratio, near_pre_rr_ratio]
    - RR features를 ECG 길이로 repeat
    - 입력: (1, 3, L) tensor + (2,) RR features

    Args:
        data: (N, L) ECG segments
        labels: (N,) class labels
        rr_features: (N, 2) RR features [pre_rr_ratio, near_pre_rr_ratio]
        patient_ids: (N,) patient IDs
        sample_ids: (N,) sample indices
    """

    def __init__(self, data, labels, rr_features, patient_ids, sample_ids):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.rr_features = np.asarray(rr_features, dtype=np.float32)
        self.patient_ids = np.asarray(patient_ids, dtype=np.int64)
        self.sample_ids = np.asarray(sample_ids)
        self.seq_len = data.shape[1]

        # Validate rr_features shape
        self._validate_rr_features()

    def _validate_rr_features(self):
        """RR features shape 검증"""
        if self.rr_features.ndim != 2:
            raise ValueError(
                f"Expected rr_features to be 2D (n_samples, 2), "
                f"got {self.rr_features.ndim}D with shape {self.rr_features.shape}"
            )
        if self.rr_features.shape[1] != 2:
            raise ValueError(
                f"Expected rr_features to have 2 columns, "
                f"got shape {self.rr_features.shape}"
            )
        if len(self.rr_features) != len(self.labels):
            raise ValueError(
                f"rr_features length ({len(self.rr_features)}) != "
                f"labels length ({len(self.labels)})"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # ECG segment
        ecg = self.data[idx]  # (L,)

        # RR features
        rr = self.rr_features[idx]  # (2,)

        # RR features를 ECG 길이로 repeat
        pre_rr_ratio = np.full(self.seq_len, rr[0], dtype=np.float32)
        near_pre_rr_ratio = np.full(self.seq_len, rr[1], dtype=np.float32)

        # Stack to (3, L) → (1, 3, L)
        x = np.stack([ecg, pre_rr_ratio, near_pre_rr_ratio], axis=0)
        x = x[np.newaxis, :, :]  # (1, 3, L)

        return (
            torch.from_numpy(x),                          # (1, 3, L)
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.from_numpy(rr),                         # (2,)
            torch.tensor(self.patient_ids[idx], dtype=torch.long),
            torch.tensor(self.sample_ids[idx], dtype=torch.long),
        )


@DATASET_REGISTRY.register("ecg_multichannel")
class ECGMultiChannelDataset(Dataset):
    """
    Multi-channel ECG Dataset

    다중 리드 ECG 데이터를 위한 Dataset입니다.

    Args:
        data: (N, C, L) ECG segments with C channels
        labels: (N,) class labels
        rr_features: (N, D) RR features
        patient_ids: (N,) patient IDs
        sample_ids: (N,) sample indices
    """

    def __init__(self, data, labels, rr_features, patient_ids, sample_ids):
        self.data = torch.FloatTensor(data)
        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(1)  # (N, L) -> (N, 1, L)

        # Add height dimension for 2D conv: (N, C, L) -> (N, 1, C, L)
        self.data = self.data.unsqueeze(1)

        self.labels = torch.LongTensor(labels)
        self.rr_features = torch.FloatTensor(rr_features)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.data[idx],           # (1, C, L)
            self.labels[idx],
            self.rr_features[idx],
            self.patient_ids[idx],
            idx
        )


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader from dataset.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def get_dataloaders(
    train_data, train_labels, train_rr, train_pids, train_sids,
    valid_data, valid_labels, valid_rr, valid_pids, valid_sids,
    test_data, test_labels, test_rr, test_pids, test_sids,
    dataset_name: str = "daeac",
    batch_size: int = 256,
    num_workers: int = 4
):
    """
    Create train/valid/test dataloaders.

    Args:
        train_*, valid_*, test_*: Data for each split
        dataset_name: Name of dataset class in registry
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    DatasetClass = DATASET_REGISTRY.get(dataset_name)

    train_ds = DatasetClass(train_data, train_labels, train_rr, train_pids, train_sids)
    valid_ds = DatasetClass(valid_data, valid_labels, valid_rr, valid_pids, valid_sids)
    test_ds = DatasetClass(test_data, test_labels, test_rr, test_pids, test_sids)

    train_loader = create_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = create_dataloader(valid_ds, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = create_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
