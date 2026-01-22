"""
Focal Loss for imbalanced classification

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Addresses class imbalance by down-weighting easy examples.

    Args:
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        alpha: Class weights (optional). Can be:
               - None: no weighting
               - float: weight for positive class (binary)
               - List/Tensor: per-class weights
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) model output logits
            targets: (B,) class indices

        Returns:
            Focal loss tensor
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


@LOSS_REGISTRY.register("class_balanced_focal")
class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss

    Combines focal loss with effective number of samples weighting.
    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

    Args:
        gamma: Focal loss gamma parameter
        beta: Class balance beta (0.9, 0.99, 0.999, 0.9999)
        num_classes: Number of classes
        samples_per_class: List of sample counts per class
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.9999,
        num_classes: int = 4,
        samples_per_class=None,
        **kwargs
    ):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.num_classes = num_classes

        if samples_per_class is not None:
            self.samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
            self._compute_weights()
        else:
            self.samples_per_class = None
            self.class_weights = None

    def _compute_weights(self):
        """Compute class-balanced weights."""
        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_class)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        self.class_weights = weights

    def set_samples_per_class(self, samples_per_class):
        """Update sample counts and recompute weights."""
        self.samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
        self._compute_weights()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)

        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            weight_t = weights[targets]
            focal_loss = weight_t * focal_loss

        return focal_loss.mean()
