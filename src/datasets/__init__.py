"""
Datasets Module

새로운 Dataset 추가 방법:
1. 이 디렉토리에 새 파일 생성 (예: my_dataset.py)
2. @DATASET_REGISTRY.register("my_dataset") 데코레이터로 등록
3. config.py에서 DATASET_NAME = "my_dataset" 설정

예시:
    from src.registry import DATASET_REGISTRY

    @DATASET_REGISTRY.register("my_ecg_dataset")
    class MyECGDataset(Dataset):
        def __init__(self, data, labels, rr_features, patient_ids, sample_ids):
            ...

        def __getitem__(self, idx):
            # Returns: (ecg_input, label, rr_features, patient_id, sample_id)
            ...
"""

from .ecg_datasets import *
