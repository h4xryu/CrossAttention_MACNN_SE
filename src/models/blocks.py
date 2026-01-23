"""
Reusable Neural Network Building Blocks

다른 모델에서 재사용할 수 있는 기본 블록들입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Helpers (shape-safe)
# =============================================================================

def _ensure_2d_feat(x: torch.Tensor) -> torch.Tensor:
    """
    Accept (B, C) or (B, T, C) and return (B, C).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x)}")

    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.mean(dim=1)
    raise ValueError(f"x must be 2D or 3D, got shape={tuple(x.shape)}")


def _ensure_3d_seq(x: torch.Tensor) -> torch.Tensor:
    """
    Accept (B, C) or (B, T, C) and return (B, T, C).
    If (B, C), convert to (B, 1, C).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x)}")

    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(1)
    raise ValueError(f"x must be 2D or 3D, got shape={tuple(x.shape)}")


def get_activation(name: str, in_channels: int = None):
    """
    Get activation function by name.

    Args:
        name: Activation name ('tanh', 'relu', 'leaky_relu', 'prelu', 'elu', 'gelu')
        in_channels: Number of input channels (for PReLU variants)

    Returns:
        nn.Module activation layer
    """
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.01, inplace=True),
        'prelu': nn.PReLU(init=0.05),
        'elu': nn.ELU(inplace=True),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(inplace=True),
    }

    if name == 'cprelu' and in_channels is not None:
        return nn.PReLU(num_parameters=in_channels, init=0.05)

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")

    return activations[name]


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer

    Channel attention mechanism.
    Reference: "Squeeze-and-Excitation Networks" (Hu et al., 2018)

    Args:
        channel: Number of input channels
        reduction: Reduction ratio for bottleneck
    """

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Convolution + BatchNorm + Activation block.

    Args:
        in_channel: Input channels
        out_channel: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        dilation: Dilation rate
        groups: Groups for grouped convolution
        use_bn: Whether to use BatchNorm
        use_act: Whether to use activation
        act: Activation function name
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_bn=True,
        use_act=True,
        act='tanh'
    ):
        super().__init__()
        self.use_bn = use_bn
        self.use_act = use_act

        self.conv = nn.Conv2d(
            in_channel, out_channel,
            kernel_size, stride, padding, dilation, groups, bias
        )
        self.bn = nn.BatchNorm2d(out_channel) if use_bn else nn.Identity()
        self.act = get_activation(act, out_channel) if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block with optional downsampling.

    Args:
        in_channel: Input channels
        out_channel: Output channels
        kernel_size: Kernel size
        stride: Stride (2 for downsampling)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - stride + 1) // 2)
        )

        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, (kernel_size - 1) // 2)
        )

        self.downsample = nn.AvgPool2d(kernel_size=(1, stride)) if stride > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.dropout1(self.relu1(self.bn1(x))))
        out = self.conv2(self.dropout2(self.relu2(self.bn2(out))))

        if self.downsample is not None:
            x = self.downsample(x)

        return out + x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    Multi-scale feature extraction using dilated convolutions.
    Reference: "DeepLab" (Chen et al.)

    Args:
        in_channel: Input channels
        out_channel: Output channels per branch
        kernel_size: Kernel size
        dilations: Tuple of dilation rates
        use_bn: Whether to use BatchNorm
        use_act: Whether to use activation
        act_func: Activation function name
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        dilations=(1, 6, 12, 18),
        use_bn=True,
        use_act=True,
        act_func='tanh'
    ):
        super().__init__()
        self.num_branches = len(dilations)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            padding = dilation * (kernel_size - 1) // 2
            self.branches.append(
                ConvBlock(
                    in_channel, out_channel,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=(1, dilation),
                    use_bn=use_bn,
                    use_act=use_act,
                    act=act_func
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return torch.cat(features, dim=1)


class GlobalAvgPool(nn.Module):
    """Global Average Pooling layer."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class ResidualClassifier(nn.Module):
    """
    Classifier with residual connection.

    Args:
        num_classes: Number of output classes
        hidden_dim: Hidden dimension (default: same as num_classes)
    """

    def __init__(self, num_classes: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or num_classes

        self.fc = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


# =============================================================================
# Fusion Blocks (for RR features + ECG features)
# =============================================================================

class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.

    Applies MLP to ECG features, then concatenates with RR features.

    Args:
        emb_dim: Embedding dimension
        expansion: MLP expansion factor
    """

    def __init__(self, emb_dim: int = 64, expansion: int = 2, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, expansion * emb_dim),
            nn.LayerNorm(expansion * emb_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(expansion * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mlp(x), rr_features], dim=1)


class ConcatProjectionFusion(nn.Module):
    """
    Concatenation with RR feature projection.

    Projects both ECG and RR features to same dimension before concatenation.

    Args:
        emb_dim: Embedding dimension
        expansion: MLP expansion factor
        rr_dim: RR feature dimension
    """

    def __init__(self, emb_dim: int = 64, expansion: int = 2, rr_dim: int = 7, **kwargs):
        super().__init__()
        self.rr_proj = nn.Sequential(
            nn.Linear(rr_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, expansion * emb_dim),
            nn.LayerNorm(expansion * emb_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(expansion * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.rr_proj(rr_features), self.mlp(x)], dim=1)



class MultiHeadCrossAttentionFusion(nn.Module):
    """
    RR (Query) → ECG (Key, Value)
    Cross-Attention + FFN (Transformer-style)

    Output is aligned to 2 * emb_dim to be consistent with concat_proj fusion.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        expansion: int = 2,
        rr_dim: int = 7,
        num_heads: int = 1,
        **kwargs
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim({emb_dim}) must be divisible by num_heads({num_heads}).")

        self.emb_dim = emb_dim
        self.rr_dim = rr_dim
        self.num_heads = num_heads
        self.out_dim = 2 * emb_dim  # 🔴 핵심 수정

        # RR → Query projection
        self.rr_proj = nn.Sequential(
            nn.Linear(rr_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU()
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )

        # FFN (Position-wise)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, expansion * emb_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(expansion * emb_dim, emb_dim),
        )

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, rr_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, emb_dim) or (B, emb_dim)
            rr_features: (B, rr_dim)
        Returns:
            (B, 2 * emb_dim)
        """

        # Ensure ECG is sequence
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)          # (B, 1, emb)
            x_pool = x                      # (B, emb)
        elif x.dim() == 3:
            x_seq = x                       # (B, T, emb)
            x_pool = x.mean(dim=1)          # (B, emb)
        else:
            raise ValueError(f"x must be 2D or 3D, got {tuple(x.shape)}")

        # RR → Query
        q = self.rr_proj(rr_features).unsqueeze(1)  # (B, 1, emb)

        # Cross-Attention
        attn_out, _ = self.cross_attn(
            query=q,
            key=x_seq,
            value=x_seq
        )  # (B, 1, emb)

        attn_out = self.norm1(attn_out.squeeze(1))  # (B, emb)

        # FFN + Residual
        attn_out = self.norm2(attn_out + self.ffn(attn_out))  # (B, emb)

        return torch.cat([attn_out, x_pool], dim=1)  # (B, 2*emb)



# Fusion block factory
FUSION_BLOCKS = {
    'concat': ConcatFusion,
    'concat_proj': ConcatProjectionFusion,
    'mhca': MultiHeadCrossAttentionFusion,
}


def get_fusion_block(fusion_type: str, **kwargs):
    """
    Get fusion block by name.

    Args:
        fusion_type: 'concat', 'concat_proj', 'mhca', or None
        **kwargs: Arguments for fusion block

    Returns:
        Fusion block instance or None
    """
    if fusion_type is None or fusion_type.lower() == 'none':
        return None

    if fusion_type not in FUSION_BLOCKS:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Available: {list(FUSION_BLOCKS.keys())}")

    return FUSION_BLOCKS[fusion_type](**kwargs)
