"""
Optimizers Module

새로운 Optimizer 추가 방법:
1. @OPTIMIZER_REGISTRY.register("my_optimizer") 데코레이터로 등록
2. config.py에서 OPTIMIZER_NAME = "my_optimizer" 설정

예시:
    from src.registry import OPTIMIZER_REGISTRY

    @OPTIMIZER_REGISTRY.register("my_optimizer")
    def create_my_optimizer(params, lr, **kwargs):
        return MyCustomOptimizer(params, lr=lr, **kwargs)
"""

import torch.optim as optim
from src.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("adam")
def create_adam(params, lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, **kwargs):
    """Adam optimizer"""
    return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


@OPTIMIZER_REGISTRY.register("adamw")
def create_adamw(params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, **kwargs):
    """AdamW optimizer with decoupled weight decay"""
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


@OPTIMIZER_REGISTRY.register("sgd")
def create_sgd(params, lr=0.01, weight_decay=0.0, momentum=0.9, nesterov=False, **kwargs):
    """SGD optimizer with momentum"""
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)


@OPTIMIZER_REGISTRY.register("rmsprop")
def create_rmsprop(params, lr=0.01, weight_decay=0.0, momentum=0.0, alpha=0.99, eps=1e-8, **kwargs):
    """RMSprop optimizer"""
    return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum, alpha=alpha, eps=eps)


def build_optimizer(params, name: str, **kwargs):
    """
    Build optimizer from config.

    Args:
        params: Model parameters
        name: Optimizer name in registry
        **kwargs: Optimizer arguments

    Returns:
        Optimizer instance
    """
    create_fn = OPTIMIZER_REGISTRY.get(name)
    return create_fn(params, **kwargs)
