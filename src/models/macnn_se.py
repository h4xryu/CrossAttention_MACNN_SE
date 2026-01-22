"""
MACNN_SE: Multi-scale Atrous CNN with Squeeze-and-Excitation

ECG 부정맥 분류를 위한 메인 모델입니다.
"""

import torch
import torch.nn as nn
from src.registry import MODEL_REGISTRY
from .blocks import (
    SELayer, ASPP, ResidualBlock, GlobalAvgPool,
    ResidualClassifier, get_fusion_block
)


@MODEL_REGISTRY.register("macnn_se")
class MACNN_SE(nn.Module):
    """
    MACNN_SE: Multi-scale Atrous CNN with SE layers

    Architecture:
        Input → Conv → ASPP → SE → Residual → ASPP → SE → Residual
        → BN+ReLU → ASPP → SE → GAP → [Fusion] → FC → Output

    Args:
        in_channels: Number of input channels (1 or 3)
        num_classes: Number of output classes
        reduction: SE layer reduction ratio
        dilations: Tuple of dilation rates for ASPP
        dropout: Dropout probability
        act_func: Activation function name
        final_act_func: Final ASPP activation
        aspp_bn: Use BatchNorm in ASPP
        aspp_act: Use activation in ASPP
        fusion_type: Fusion type ('concat', 'concat_proj', 'mhca', or None)
        fusion_emb: Fusion embedding dimension
        fusion_expansion: Fusion MLP expansion factor
        fusion_num_heads: Number of heads for MHCA fusion
        rr_dim: RR feature dimension
        apply_residual: Use residual classifier
    """

    def __init__(
        self,
        in_channels: int = 1,   # 항상 1 (입력 형태: N, 1, lead, length)
        lead: int = 3,          # 1 (ECG only) or 3 (ECG + 2 RR ratios)
        num_classes: int = 4,
        reduction: int = 16,
        dilations=(1, 6, 12, 18),
        dropout: float = 0.0,
        act_func: str = 'tanh',
        final_act_func: str = 'tanh',
        aspp_bn: bool = True,
        aspp_act: bool = True,
        fusion_type=None,
        fusion_emb: int = 64,
        fusion_expansion: int = 2,
        fusion_num_heads: int = 1,
        rr_dim: int = 2,
        apply_residual: bool = False,
        **kwargs  # Ignore extra config parameters
    ):
        super().__init__()

        self.in_channels = in_channels
        self.lead = lead
        self.num_classes = num_classes
        self.num_dilations = len(dilations)
        self.apply_residual = apply_residual
        self.fusion_type = fusion_type

        # Initial convolution: (N, 1, lead, L) -> (N, 4, 1, L)
        # kernel (lead, 3) reduces lead dimension to 1
        self.conv1 = nn.Conv2d(
            in_channels, 4,
            kernel_size=(lead, 3), stride=1, padding=(0, 1)
        )

        # Stage 1: ASPP + SE + Residual
        self.aspp_1 = ASPP(4, 4, 3, dilations, aspp_bn, aspp_act, act_func)
        self.se_1 = SELayer(self.num_dilations * 4, reduction=4)
        self.res_1 = ResidualBlock(
            self.num_dilations * 4,
            self.num_dilations * 4,
            kernel_size=3, stride=1, dropout=dropout
        )

        # Stage 2: ASPP + SE + Residual (with downsampling)
        self.aspp_2 = ASPP(
            self.num_dilations * 4,
            self.num_dilations * 4,
            3, dilations, aspp_bn, aspp_act, act_func
        )
        self.se_2 = SELayer(self.num_dilations ** 2 * 4, reduction=8)
        self.res_2 = ResidualBlock(
            self.num_dilations ** 2 * 4,
            self.num_dilations ** 2 * 4,
            kernel_size=3, stride=2, dropout=dropout
        )

        # Stage 3: BN + ReLU + ASPP + SE
        self.bn = nn.BatchNorm2d(self.num_dilations ** 2 * 4)
        self.relu = nn.ReLU(inplace=True)
        self.aspp_3 = ASPP(
            self.num_dilations ** 2 * 4,
            self.num_dilations ** 2 * 4,
            3, dilations, aspp_bn, aspp_act, final_act_func
        )
        self.se_3 = SELayer(self.num_dilations ** 3 * 4, reduction=reduction)

        # Global Average Pooling
        self.gap = GlobalAvgPool()

        # Feature dimension after GAP
        self.feature_dim = self.num_dilations ** 3 * 4

        # Fusion block (optional)
        self.fusion_block = None
        self.feature_proj = None

        if fusion_type is not None and fusion_type.lower() != 'none':
            self.feature_proj = nn.Linear(self.feature_dim, fusion_emb)
            self.fusion_block = get_fusion_block(
                fusion_type,
                emb_dim=fusion_emb,
                expansion=fusion_expansion,
                rr_dim=rr_dim,
                num_heads=fusion_num_heads
            )
            # Output dimension after fusion
            if fusion_type == 'concat':
                fc_input_dim = fusion_emb + rr_dim
            else:
                fc_input_dim = 2 * fusion_emb
        else:
            fc_input_dim = self.feature_dim

        # Classifier
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.res_classifier = ResidualClassifier(num_classes) if apply_residual else None

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor = None):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W) or (B, 1, C, L)
            rr_features: Optional RR features (B, rr_dim)

        Returns:
            logits: (B, num_classes)
            features: (B, feature_dim) intermediate features
        """
        # Stage 1
        out = self.conv1(x)
        out = self.aspp_1(out)
        out = self.se_1(out)
        out = self.res_1(out)

        # Stage 2
        out = self.aspp_2(out)
        out = self.se_2(out)
        out = self.res_2(out)

        # Stage 3
        out = self.relu(self.bn(out))
        out = self.aspp_3(out)
        out = self.se_3(out)

        # GAP → flatten
        features = self.gap(out).view(out.size(0), -1)

        # Fusion with RR features
        if self.fusion_block is not None and rr_features is not None:
            proj_features = self.feature_proj(features)
            fused = self.fusion_block(proj_features, rr_features)
        else:
            fused = features

        # Classification
        logits = self.fc(fused)
        if self.res_classifier is not None:
            logits = self.res_classifier(logits)

        return logits, features

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate feature maps (before GAP)."""
        out = self.conv1(x)
        out = self.se_1(self.aspp_1(out))
        out = self.res_1(out)
        out = self.se_2(self.aspp_2(out))
        out = self.res_2(out)
        out = self.relu(self.bn(out))
        out = self.se_3(self.aspp_3(out))
        return out

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


@MODEL_REGISTRY.register("macnn")
class MACNN(nn.Module):
    """
    MACNN: Multi-scale Atrous CNN (without SE layers)

    Simpler version without Squeeze-and-Excitation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        lead: int = 3,
        num_classes: int = 4,
        dilations=(1, 6, 12, 18),
        dropout: float = 0.0,
        act_func: str = 'tanh',
        final_act_func: str = 'tanh',
        aspp_bn: bool = True,
        aspp_act: bool = True,
        apply_residual: bool = False,
        **kwargs
    ):
        super().__init__()

        self.lead = lead
        self.num_dilations = len(dilations)
        self.apply_residual = apply_residual

        self.conv1 = nn.Conv2d(in_channels, 4, (lead, 3), 1, (0, 1))

        self.aspp_1 = ASPP(4, 4, 3, dilations, aspp_bn, aspp_act, act_func)
        self.res_1 = ResidualBlock(
            self.num_dilations * 4, self.num_dilations * 4, 3, 1, dropout
        )

        self.aspp_2 = ASPP(
            self.num_dilations * 4, self.num_dilations * 4,
            3, dilations, aspp_bn, aspp_act, act_func
        )
        self.res_2 = ResidualBlock(
            self.num_dilations ** 2 * 4, self.num_dilations ** 2 * 4, 3, 2, dropout
        )

        self.bn = nn.BatchNorm2d(self.num_dilations ** 2 * 4)
        self.relu = nn.ReLU(inplace=True)

        self.aspp_3 = ASPP(
            self.num_dilations ** 2 * 4, self.num_dilations ** 2 * 4,
            3, dilations, aspp_bn, aspp_act, final_act_func
        )

        self.gap = GlobalAvgPool()
        self.feature_dim = self.num_dilations ** 3 * 4

        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.res_classifier = ResidualClassifier(num_classes) if apply_residual else None

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor = None):
        out = self.conv1(x)
        out = self.res_1(self.aspp_1(out))
        out = self.res_2(self.aspp_2(out))
        out = self.relu(self.bn(out))
        out = self.aspp_3(out)

        features = self.gap(out).view(out.size(0), -1)
        logits = self.fc(features)

        if self.res_classifier is not None:
            logits = self.res_classifier(logits)

        return logits, features

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


@MODEL_REGISTRY.register("acnn")
class ACNN(nn.Module):
    """
    ACNN: Atrous CNN

    Original model without intermediate SE layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        lead: int = 3,
        num_classes: int = 4,
        reduction: int = 16,
        dilations=(1, 6, 12, 18),
        dropout: float = 0.0,
        act_func: str = 'tanh',
        final_act_func: str = 'tanh',
        aspp_bn: bool = True,
        aspp_act: bool = True,
        apply_residual: bool = False,
        **kwargs
    ):
        super().__init__()

        self.lead = lead
        self.num_dilations = len(dilations)
        self.apply_residual = apply_residual

        self.conv1 = nn.Conv2d(in_channels, 4, (lead, 3), 1, (0, 1))

        self.aspp_1 = ASPP(4, 4, 3, dilations, aspp_bn, aspp_act, act_func)

        # Conv instead of direct SE
        self.conv1_1 = nn.Conv2d(4, self.num_dilations * 4, (1, 3), (1, 1), (0, 1))
        self.res_1 = ResidualBlock(
            self.num_dilations * 4, self.num_dilations * 4, 3, 1, dropout
        )

        self.conv2_2 = nn.Conv2d(
            self.num_dilations * 4, self.num_dilations ** 2 * 4,
            (1, 3), (1, 1), (0, 1)
        )
        self.res_2 = ResidualBlock(
            self.num_dilations ** 2 * 4, self.num_dilations ** 2 * 4, 3, 2, dropout
        )

        self.bn = nn.BatchNorm2d(self.num_dilations ** 2 * 4)
        self.relu = nn.ReLU(inplace=True)

        self.aspp_3 = ASPP(
            self.num_dilations ** 2 * 4, self.num_dilations ** 2 * 4,
            3, dilations, aspp_bn, aspp_act, final_act_func
        )
        self.se = SELayer(self.num_dilations ** 3 * 4, reduction)

        self.gap = GlobalAvgPool()
        self.feature_dim = self.num_dilations ** 3 * 4

        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.res_classifier = ResidualClassifier(num_classes) if apply_residual else None

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor = None):
        out = self.conv1(x)
        out = self.conv1_1(out)
        out = self.res_1(out)
        out = self.conv2_2(out)
        out = self.res_2(out)
        out = self.relu(self.bn(out))
        out = self.se(self.aspp_3(out))

        features = self.gap(out).view(out.size(0), -1)
        logits = self.fc(features)

        if self.res_classifier is not None:
            logits = self.res_classifier(logits)

        return logits, features
