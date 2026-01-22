"""
ECG Dataset 클래스들

확장 가능한 구조:
- ECGDataset: 기존 실험용 (opt1) - 1D/2D Conv 입력
- DAEACDataset: DAEAC 논문용 (opt2) - 3채널 입력 (ECG + 2 RR ratios)
- 향후 새로운 Dataset 클래스 추가 가능 (opt3+)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE, RR_FEATURE_OPTION


# =============================================================================
# opt1: 기존 실험용 Dataset (Cross-Attention 연구)
# =============================================================================

class ECGDataset(Dataset):
    """
    기존 실험용 ECG Dataset (opt1)

    입력 형태: ECG (1, 1, L) + RR features (7,)
    - 2D Conv 모델용 형식
    - RR features는 별도로 전달

    Args:
        ecg_data: (N, L) ECG segments
        rr_features: (N, 7) RR features
        labels: (N,) class labels
        patient_ids: (N,) patient IDs
        sample_ids: (N,) sample indices
    """

    def __init__(self, ecg_data, rr_features, labels, patient_ids, sample_ids):
        # (N, L) -> (N, 1, 1, L) for 2D Conv model
        self.ecg_data = torch.FloatTensor(ecg_data).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, L)
        self.rr_features = torch.FloatTensor(rr_features)
        self.labels = torch.LongTensor(labels)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.ecg_data[idx],       # (1, 1, L)
            self.rr_features[idx],    # (7,)
            self.labels[idx],
            self.patient_ids[idx],
            idx
        )


# =============================================================================
# opt2: DAEAC 논문용 Dataset
# =============================================================================

class DAEACDataset(Dataset):
    """
    DAEAC 논문용 ECG Dataset (opt2)

    Reference:
        "Inter-patient ECG arrhythmia heartbeat classification based on
        unsupervised domain adaptation" (Wang et al., 2021)

    입력 형태: 3채널 (1, 3, L)
    - Channel 0: ECG segment (128 samples)
    - Channel 1: pre_rr_ratio (repeated to 128)
    - Channel 2: near_pre_rr_ratio (repeated to 128)

    현재 MACNN_SE 모델은 2D Conv를 사용하므로 출력 형태:
        (B, 1, 3, L) - lead=1, 3채널을 height로 사용

    Args:
        data: (N, L) ECG segments
        rr_features: (N, 2) [pre_rr_ratio, near_pre_rr_ratio]
        labels: (N,) class labels
        patient_ids: (N,) patient IDs
        sample_ids: (N,) sample indices
    """

    def __init__(self, data, rr_features, labels, patient_ids, sample_ids):
        self.data = data
        self.rr_features = rr_features
        self.labels = labels
        self.patient_ids = patient_ids
        self.sample_ids = sample_ids
        self.seq_len = data.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # ECG segment
        ecg = self.data[idx]  # (L,)

        # RR features를 ECG와 같은 길이로 repeat
        rr = self.rr_features[idx]  # (2,)
        pre_rr_ratio = np.full(self.seq_len, rr[0], dtype=np.float32)
        near_pre_rr_ratio = np.full(self.seq_len, rr[1], dtype=np.float32)

        # 3채널 입력 생성: (3, L) -> 2D Conv 형식: (1, 3, L)
        x = np.stack([ecg, pre_rr_ratio, near_pre_rr_ratio], axis=0)  # (3, L)
        x = x[np.newaxis, :, :]  # (1, 3, L) for 2D Conv

        return (
            torch.from_numpy(x).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.from_numpy(rr).float(),
            torch.tensor(self.patient_ids[idx], dtype=torch.long),
            torch.tensor(self.sample_ids[idx], dtype=torch.long),
        )


# =============================================================================
# Dataset Factory
# =============================================================================

def get_dataset_class(rr_option: str = None):
    """
    RR 옵션에 맞는 Dataset 클래스 반환

    Args:
        rr_option: "opt1", "opt2", ... (None이면 config에서 읽음)

    Returns:
        Dataset class
    """
    if rr_option is None:
        rr_option = RR_FEATURE_OPTION

    dataset_map = {
        "opt1": ECGDataset,
        "opt2": DAEACDataset,
        # "opt3": NewResearchDataset,  # 향후 확장
    }

    if rr_option not in dataset_map:
        raise ValueError(f"Unknown RR option: '{rr_option}'. Available: {list(dataset_map.keys())}")

    return dataset_map[rr_option]


def create_dataset(data, rr_features, labels, patient_ids, sample_ids, rr_option: str = None):
    """
    RR 옵션에 맞는 Dataset 인스턴스 생성

    Args:
        data: ECG segments
        rr_features: RR features
        labels: class labels
        patient_ids: patient IDs
        sample_ids: sample indices
        rr_option: "opt1", "opt2", ... (None이면 config에서 읽음)

    Returns:
        Dataset instance
    """
    DatasetClass = get_dataset_class(rr_option)
    return DatasetClass(data, rr_features, labels, patient_ids, sample_ids)


# =============================================================================
# DataLoader Helper Functions
# =============================================================================

def get_dataloaders(
    train_data, train_rr, train_labels, train_patient_ids, train_sample_ids,
    test_data, test_rr, test_labels, test_patient_ids, test_sample_ids,
    batch_size: int = BATCH_SIZE,
    rr_option: str = None,
    num_workers: int = 4
) -> tuple:
    """
    Train/Test DataLoader 생성

    Args:
        train_*: 학습 데이터
        test_*: 테스트 데이터
        batch_size: 배치 크기
        rr_option: RR feature 옵션
        num_workers: DataLoader worker 수

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = create_dataset(
        train_data, train_rr, train_labels, train_patient_ids, train_sample_ids, rr_option
    )
    test_dataset = create_dataset(
        test_data, test_rr, test_labels, test_patient_ids, test_sample_ids, rr_option
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def get_dataloaders_with_valid(
    train_data, train_rr, train_labels, train_patient_ids, train_sample_ids,
    valid_data, valid_rr, valid_labels, valid_patient_ids, valid_sample_ids,
    test_data, test_rr, test_labels, test_patient_ids, test_sample_ids,
    batch_size: int = BATCH_SIZE,
    rr_option: str = None,
    num_workers: int = 4
) -> tuple:
    """
    Train/Valid/Test DataLoader 생성

    Args:
        train_*: 학습 데이터
        valid_*: 검증 데이터
        test_*: 테스트 데이터
        batch_size: 배치 크기
        rr_option: RR feature 옵션
        num_workers: DataLoader worker 수

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    train_dataset = create_dataset(
        train_data, train_rr, train_labels, train_patient_ids, train_sample_ids, rr_option
    )
    valid_dataset = create_dataset(
        valid_data, valid_rr, valid_labels, valid_patient_ids, valid_sample_ids, rr_option
    )
    test_dataset = create_dataset(
        test_data, test_rr, test_labels, test_patient_ids, test_sample_ids, rr_option
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, valid_loader, test_loader
