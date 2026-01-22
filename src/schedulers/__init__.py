"""
Learning Rate Schedulers Module

새로운 Scheduler 추가 방법:
1. @SCHEDULER_REGISTRY.register("my_scheduler") 데코레이터로 등록
2. config.py에서 SCHEDULER_NAME = "my_scheduler" 설정

예시:
    from src.registry import SCHEDULER_REGISTRY

    @SCHEDULER_REGISTRY.register("my_scheduler")
    def create_my_scheduler(optimizer, **kwargs):
        return MyCustomScheduler(optimizer, **kwargs)
"""

import torch.optim.lr_scheduler as lr_scheduler
from src.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("lambda_decay")
def create_lambda_decay(optimizer, decay_rate=0.99, decay_steps=200, **kwargs):
    """
    Lambda LR decay (DAEAC style)

    lr = base_lr * decay_rate^(step/decay_steps)
    """
    return lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: decay_rate ** (step / decay_steps)
    )


@SCHEDULER_REGISTRY.register("step")
def create_step_lr(optimizer, step_size=30, gamma=0.1, **kwargs):
    """Step LR: decay by gamma every step_size epochs"""
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


@SCHEDULER_REGISTRY.register("multistep")
def create_multistep_lr(optimizer, milestones=[100, 150], gamma=0.1, **kwargs):
    """MultiStep LR: decay at specified milestones"""
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


@SCHEDULER_REGISTRY.register("exponential")
def create_exponential_lr(optimizer, gamma=0.95, **kwargs):
    """Exponential LR: decay by gamma every epoch"""
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


@SCHEDULER_REGISTRY.register("cosine")
def create_cosine_annealing(optimizer, t_max=100, eta_min=1e-6, **kwargs):
    """Cosine annealing LR"""
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


@SCHEDULER_REGISTRY.register("cosine_warmup")
def create_cosine_warmup(optimizer, t_max=100, eta_min=1e-6, warmup_epochs=5, **kwargs):
    """Cosine annealing with linear warmup"""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            import math
            progress = (epoch - warmup_epochs) / (t_max - warmup_epochs)
            return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@SCHEDULER_REGISTRY.register("reduce_on_plateau")
def create_reduce_on_plateau(optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-6, **kwargs):
    """Reduce LR on plateau (requires manual step with metric)"""
    return lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
    )


@SCHEDULER_REGISTRY.register("none")
def create_no_scheduler(optimizer, **kwargs):
    """No scheduler (constant LR)"""
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)


def build_scheduler(optimizer, name: str, **kwargs):
    """
    Build scheduler from config.

    Args:
        optimizer: Optimizer instance
        name: Scheduler name in registry
        **kwargs: Scheduler arguments

    Returns:
        Scheduler instance
    """
    create_fn = SCHEDULER_REGISTRY.get(name)
    return create_fn(optimizer, **kwargs)
