"""
Registry System for Modular Components

연구자가 새로운 모델, loss, dataset, optimizer 등을 쉽게 추가할 수 있도록
데코레이터 기반 등록 시스템을 제공합니다.

사용 예시:
    from src.registry import MODEL_REGISTRY

    @MODEL_REGISTRY.register("my_model")
    class MyModel(nn.Module):
        ...

    # 사용
    model_cls = MODEL_REGISTRY.get("my_model")
    model = model_cls(**config)
"""

from typing import Dict, Any, Callable, Optional, Type
import torch.nn as nn


class Registry:
    """
    Generic registry for components (models, losses, datasets, etc.)

    Example:
        >>> MODEL_REGISTRY = Registry("model")
        >>> @MODEL_REGISTRY.register("resnet")
        ... class ResNet(nn.Module):
        ...     pass
        >>> model_cls = MODEL_REGISTRY.get("resnet")
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """
        Register a class or function with the given name.

        Args:
            name: Unique identifier for the component

        Returns:
            Decorator function
        """
        def decorator(cls_or_fn):
            if name in self._registry:
                raise ValueError(
                    f"'{name}' is already registered in {self._name} registry. "
                    f"Existing: {self._registry[name]}, New: {cls_or_fn}"
                )
            self._registry[name] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, name: str) -> Any:
        """
        Get a registered component by name.

        Args:
            name: Identifier of the component

        Returns:
            The registered class or function

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' is not registered in {self._name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._registry.keys())})"


# =============================================================================
# Global Registries
# =============================================================================

MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
DATASET_REGISTRY = Registry("dataset")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
METRIC_REGISTRY = Registry("metric")


def build_from_config(registry: Registry, config: dict) -> Any:
    """
    Build a component from config dict.

    Config must have 'name' key and optional other kwargs.

    Example:
        config = {
            "name": "macnn_se",
            "reduction": 16,
            "dilations": [1, 6, 12, 18]
        }
        model = build_from_config(MODEL_REGISTRY, config)
    """
    config = config.copy()
    name = config.pop("name")
    cls = registry.get(name)
    return cls(**config)
