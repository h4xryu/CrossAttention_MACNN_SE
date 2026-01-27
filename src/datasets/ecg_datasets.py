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
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        rr_features = np.asarray(rr_features, dtype=np.float32)
        patient_ids = np.asarray(patient_ids, dtype=np.int64)
        sample_ids = np.asarray(sample_ids, dtype=np.int64)

        seq_len = data.shape[1]
        n_samples = len(data)

        # RR features가 7개 컬럼인 경우 처음 2개만 선택 (캐시 호환성)
        if rr_features.ndim == 2 and rr_features.shape[1] == 7:
            print(f"  Warning: RR features has 7 columns, selecting first 2 columns for DAEAC format")
            rr_features = rr_features[:, :2]

        # Validate rr_features shape
        self._validate_rr_features(rr_features, labels)

        # 사전 계산: RR features를 ECG 길이로 repeat하여 3채널 데이터 생성
        # (N, L), (N, L), (N, L) -> (N, 3, L) -> (N, 1, 3, L)
        pre_rr_expanded = np.broadcast_to(
            rr_features[:, 0:1], (n_samples, seq_len)
        ).astype(np.float32)
        near_pre_rr_expanded = np.broadcast_to(
            rr_features[:, 1:2], (n_samples, seq_len)
        ).astype(np.float32)

        # Stack and add channel dimension: (N, 1, 3, L)
        x_data = np.stack([data, pre_rr_expanded, near_pre_rr_expanded], axis=1)
        x_data = x_data[:, np.newaxis, :, :]

        # 미리 torch tensor로 변환 (한 번만)
        self.x_tensor = torch.from_numpy(x_data)
        self.labels_tensor = torch.from_numpy(labels)
        self.rr_tensor = torch.from_numpy(rr_features)
        self.patient_ids_tensor = torch.from_numpy(patient_ids)
        self.sample_ids_tensor = torch.from_numpy(sample_ids)

        # 기존 인터페이스 호환성을 위한 alias
        self.labels = self.labels_tensor

    def _validate_rr_features(self, rr_features, labels):
        """RR features shape 검증"""
        if rr_features.ndim != 2:
            raise ValueError(
                f"Expected rr_features to be 2D (n_samples, 2), "
                f"got {rr_features.ndim}D with shape {rr_features.shape}"
            )
        if rr_features.shape[1] != 2:
            raise ValueError(
                f"Expected rr_features to have 2 columns, "
                f"got shape {rr_features.shape}"
            )
        if len(rr_features) != len(labels):
            raise ValueError(
                f"rr_features length ({len(rr_features)}) != "
                f"labels length ({len(labels)})"
            )

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        # 단순 인덱싱만 - 연산 없음
        return (
            self.x_tensor[idx],           # (1, 3, L)
            self.labels_tensor[idx],      # scalar
            self.rr_tensor[idx],          # (2,)
            self.patient_ids_tensor[idx], # scalar
            self.sample_ids_tensor[idx],  # scalar
        )


@DATASET_REGISTRY.register("opt3")
class Opt3Dataset(Dataset):
    """
    Opt3 Dataset: Early fusion (3D input) + Late fusion (7D RR features)
    
    opt2처럼 3D 입력 (ECG + 2 RR ratios)을 사용하되,
    추가로 7차원 RR features를 late fusion으로 결합합니다.
    
    Args:
        data: (N, L) ECG segments
        labels: (N,) class labels
        rr_features_2d: (N, 2) RR features for early fusion [pre_rr_ratio, near_pre_rr_ratio]
        rr_features_7d: (N, 7) RR features for late fusion
        patient_ids: (N,) patient IDs
        sample_ids: (N,) sample indices
    """
    
    def __init__(self, data, labels, rr_features_2d, rr_features_7d, patient_ids, sample_ids):
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        rr_features_2d = np.asarray(rr_features_2d, dtype=np.float32)
        rr_features_7d = np.asarray(rr_features_7d, dtype=np.float32)
        patient_ids = np.asarray(patient_ids, dtype=np.int64)
        sample_ids = np.asarray(sample_ids, dtype=np.int64)

        seq_len = data.shape[1]
        n_samples = len(data)

        # RR features_2d가 7개 컬럼인 경우 처음 2개만 선택 (캐시 호환성)
        if rr_features_2d.ndim == 2 and rr_features_2d.shape[1] == 7:
            print(f"  Warning: rr_features_2d has 7 columns, selecting first 2 columns for Opt3 format")
            rr_features_2d = rr_features_2d[:, :2]

        # Validate shapes
        if rr_features_2d.shape[1] != 2:
            raise ValueError(f"Expected rr_features_2d to have 2 columns, got {rr_features_2d.shape}")
        if rr_features_7d.shape[1] != 7:
            raise ValueError(f"Expected rr_features_7d to have 7 columns, got {rr_features_7d.shape}")
        if len(rr_features_2d) != len(labels) or len(rr_features_7d) != len(labels):
            raise ValueError("RR features length must match labels length")

        # 사전 계산: RR features를 ECG 길이로 repeat하여 3채널 데이터 생성
        pre_rr_expanded = np.broadcast_to(
            rr_features_2d[:, 0:1], (n_samples, seq_len)
        ).astype(np.float32)
        near_pre_rr_expanded = np.broadcast_to(
            rr_features_2d[:, 1:2], (n_samples, seq_len)
        ).astype(np.float32)

        # Stack and add channel dimension: (N, 1, 3, L)
        x_data = np.stack([data, pre_rr_expanded, near_pre_rr_expanded], axis=1)
        x_data = x_data[:, np.newaxis, :, :]

        # 미리 torch tensor로 변환 (한 번만)
        self.x_tensor = torch.from_numpy(x_data)
        self.labels_tensor = torch.from_numpy(labels)
        self.rr_7d_tensor = torch.from_numpy(rr_features_7d)
        self.patient_ids_tensor = torch.from_numpy(patient_ids)
        self.sample_ids_tensor = torch.from_numpy(sample_ids)

        # 기존 인터페이스 호환성을 위한 alias
        self.labels = self.labels_tensor

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        # 단순 인덱싱만 - 연산 없음
        return (
            self.x_tensor[idx],           # (1, 3, L) - early fusion input
            self.labels_tensor[idx],      # scalar
            self.rr_7d_tensor[idx],       # (7,) - late fusion RR features
            self.patient_ids_tensor[idx], # scalar
            self.sample_ids_tensor[idx],  # scalar
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
