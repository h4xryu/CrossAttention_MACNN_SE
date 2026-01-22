"""
ECG Classification Framework

모듈화된 구조로 연구자가 쉽게 컴포넌트를 교체할 수 있습니다.

사용 예시:
    from src.registry import MODEL_REGISTRY, LOSS_REGISTRY
    from src.models import *  # 모든 모델 자동 등록
    from src.losses import *  # 모든 loss 자동 등록

    # config.py에서 설정된 이름으로 컴포넌트 생성
    model = MODEL_REGISTRY.get(config.MODEL_NAME)(**config.MODEL_CONFIG)
    criterion = LOSS_REGISTRY.get(config.LOSS_NAME)(**config.LOSS_CONFIG)
"""

from . import registry
from . import models
from . import losses
from . import datasets
from . import optimizers
from . import schedulers
