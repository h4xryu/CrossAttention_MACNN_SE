"""
Loss Functions Module

새로운 loss 추가 방법:
1. 이 디렉토리에 새 파일 생성 (예: my_loss.py)
2. @LOSS_REGISTRY.register("my_loss") 데코레이터로 등록
3. config.py에서 LOSS_NAME = "my_loss" 설정

예시:
    from src.registry import LOSS_REGISTRY

    @LOSS_REGISTRY.register("my_custom_loss")
    class MyCustomLoss(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            ...

        def forward(self, logits, targets):
            ...
"""

from .cross_entropy import *
from .focal_loss import *
