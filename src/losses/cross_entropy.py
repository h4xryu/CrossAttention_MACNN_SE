"""
Cross Entropy Loss variations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss with optional class weights and label smoothing.

    Args:
        weight: Class weights tensor (optional)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        weight=None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) model output logits
            targets: (B,) class indices

        Returns:
            Loss tensor
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )


@LOSS_REGISTRY.register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy with Label Smoothing.

    Soft targets: (1 - smoothing) for correct class, smoothing / (num_classes - 1) for others.

    Args:
        num_classes: Number of classes
        smoothing: Smoothing factor (0.0 ~ 1.0)
        weight: Class weights (optional)
    """

    def __init__(
        self,
        num_classes: int = 4,
        smoothing: float = 0.1,
        weight=None,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) model output logits
            targets: (B,) class indices

        Returns:
            Loss tensor
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Weighted loss if weights provided
        if self.weight is not None:
            # Apply class weights
            weight_expanded = self.weight[targets]
            loss = -torch.sum(smooth_labels * log_probs, dim=-1)
            loss = loss * weight_expanded
            return loss.mean()
        else:
            loss = -torch.sum(smooth_labels * log_probs, dim=-1)
            return loss.mean()


@LOSS_REGISTRY.register("weighted_ce")
class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy with automatic class weight calculation.

    Weights are computed as inverse frequency: weight[c] = N / (C * count[c])

    Args:
        num_classes: Number of classes
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        num_classes: int = 4,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = None

    def compute_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights from label distribution.

        Args:
            labels: All training labels

        Returns:
            Class weight tensor
        """
        import numpy as np
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        counts = np.bincount(labels_np.flatten().astype(np.int64), minlength=self.num_classes)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight.to(logits.device)
        else:
            weight = None

        return F.cross_entropy(
            logits, targets, weight=weight, reduction=self.reduction
        )
