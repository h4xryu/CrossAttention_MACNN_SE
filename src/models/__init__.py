"""
Models Module

새로운 모델 추가 방법:
1. 이 디렉토리에 새 파일 생성 (예: my_model.py)
2. @MODEL_REGISTRY.register("my_model") 데코레이터로 등록
3. config.py에서 MODEL_NAME = "my_model" 설정

예시:
    from src.registry import MODEL_REGISTRY

    @MODEL_REGISTRY.register("transformer_ecg")
    class TransformerECG(nn.Module):
        def __init__(self, in_channels, num_classes, **kwargs):
            super().__init__()
            ...

        def forward(self, x, rr_features=None):
            # Returns: (logits, features)
            ...
"""

from .macnn_se import *
from .blocks import *
