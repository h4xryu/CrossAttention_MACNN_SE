"""
ResU.py - ResU-Net based ECG Classification Models

opt1: 1D input (single channel ECG + 7-dim RR features via late fusion)
      - Input: (B, 1, L) + (B, 7) RR features
      - Conv1d based encoder

opt3: 2D input (DAEAC 스타일: ECG + RR ratios as 3 channels)
      - Input: (B, 1, 3, L) where 3 = [ECG, pre_rr_ratio, near_pre_rr_ratio]
      - Conv2d based encoder
      - RR features already embedded in input spatially
      - Additional 2-dim RR features for cross-attention (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1D Blocks (for opt1)
# =============================================================================

class INCONV1D(nn.Module):
    """Initial Conv block for 1D ECG signals"""
    def __init__(self, in_ch, out_ch):
        super(INCONV1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_ch,
                               out_channels=out_ch,
                               kernel_size=15,
                               padding=7,
                               stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        return x


class ResU1DBlock(nn.Module):
    """Residual U-Block for 1D signals"""
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(ResU1DBlock, self).__init__()
        self.downsample = downsampling
        K, P = 9, 4

        self.conv1 = nn.Conv1d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=K,
                               padding=P,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for idx in range(layers):
            if idx == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(out_ch, mid_ch, K, stride=2, padding=P, bias=False),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(mid_ch, mid_ch, K, stride=2, padding=P, bias=False),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

            if idx == layers - 1:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(mid_ch * 2, out_ch, K, stride=2, padding=P, output_padding=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(mid_ch * 2, mid_ch, K, stride=2, padding=P, output_padding=1, bias=False),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []

        for layer in self.encoders:
            out = layer(out)
            encoder_out.append(out)

        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(out, size=skip.size(-1), mode='linear', align_corners=False)
            out = layer(torch.cat([out, skip], dim=1))

        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]

        out += x_in

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out


# =============================================================================
# 2D Blocks (for opt3)
# =============================================================================

class INCONV2D(nn.Module):
    """Initial Conv block for 2D ECG signals (multi-lead)"""
    def __init__(self, in_ch, out_ch):
        super(INCONV2D, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_ch,
                                 out_channels=out_ch,
                                 kernel_size=(3, 15),
                                 padding=(1, 7),
                                 stride=(1, 2),
                                 bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        return x


class ResU2DBlock(nn.Module):
    """Residual U-Block for 2D signals (multi-lead ECG)"""
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(ResU2DBlock, self).__init__()
        self.downsample = downsampling

        self.conv1 = nn.Conv2d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=(3, 9),
                               padding=(1, 4),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for idx in range(layers):
            if idx == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(out_ch, mid_ch, (3, 9), (1, 2), (1, 4), bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(mid_ch, mid_ch, (3, 9), (1, 2), (1, 4), bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))

            if idx == layers - 1:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(mid_ch * 2, out_ch, (3, 9), (1, 2), (1, 4), (0, 1), bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(mid_ch * 2, mid_ch, (3, 9), (1, 2), (1, 4), (0, 1), bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, (3, 9), padding=(1, 4), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU()
        )

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.idfunc_1 = nn.Conv2d(out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []

        for layer in self.encoders:
            out = layer(out)
            encoder_out.append(out)

        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            if out.size(-1) != skip.size(-1):
                padding_size = skip.size(-1) - out.size(-1)
                if padding_size > 0:
                    out = F.pad(out, (0, padding_size, 0, 0))
                elif padding_size < 0:
                    out = out[..., :skip.size(-1)]

            out = layer(torch.cat([out, skip], dim=1))

        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]

        out += x_in

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out


# =============================================================================
# Cross-Attention Fusion Module
# =============================================================================

class MultiHeadCrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion: RR features as Query, ECG features as Key/Value
    """
    def __init__(self, emb_dim, rr_dim, num_heads=1, dropout=0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.rr_dim = rr_dim
        self.num_heads = num_heads

        # RR projection to query
        self.rr_proj = nn.Linear(rr_dim, emb_dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ecg_seq, rr_features):
        """
        Args:
            ecg_seq: (B, L, emb_dim) - ECG sequence features
            rr_features: (B, rr_dim) - RR interval features

        Returns:
            fused: (B, 2*emb_dim) - Fused features
        """
        B = ecg_seq.size(0)

        # RR as query: (B, 1, emb_dim)
        rr_query = self.rr_proj(rr_features).unsqueeze(1)

        # Cross-attention: RR attends to ECG
        attn_out, _ = self.cross_attn(rr_query, ecg_seq, ecg_seq)
        attn_out = self.norm(attn_out.squeeze(1))
        attn_out = self.dropout(attn_out)

        # Global average pooling of ECG
        ecg_global = ecg_seq.mean(dim=1)

        # Concatenate
        fused = torch.cat([ecg_global, attn_out], dim=1)

        return fused


# =============================================================================
# ResU Models
# =============================================================================

class ResU_CrossAttention_1D(nn.Module):
    """
    ResU with Cross-Attention for 1D ECG (opt1)
    - Single channel 1D ECG input
    - RR features via Cross-Attention late fusion
    """
    def __init__(self, nOUT, n_pid=0,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=1):
        super().__init__()

        # 1D ECG Encoder
        self.inconv = INCONV1D(in_ch=in_channels, out_ch=out_ch)
        self.rub_0 = ResU1DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_1 = ResU1DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)
        self.dropout = nn.Dropout1d(0.3)

        # Cross-Attention Fusion
        self.fusion = MultiHeadCrossAttentionFusion(
            emb_dim=out_ch,
            rr_dim=n_rr,
            num_heads=num_heads,
            dropout=0.3
        )

        # Classifier
        self.fc = nn.Linear(2 * out_ch, nOUT)

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        """
        Args:
            ecg_signal: (B, 1, L) - Single channel ECG
            rr_features: (B, n_rr) - RR features
        """
        # ECG encoding
        x = self.inconv(ecg_signal)
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)
        x_seq = x.permute(0, 2, 1)  # (B, L, C)

        # Cross-attention fusion
        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)

        fused = self.fusion(x_seq, rr_features)
        logits = self.fc(fused)

        return logits, fused


class ResU_CrossAttention_2D(nn.Module):
    """
    ResU with Cross-Attention for 2D ECG (opt3)
    - DAEAC 스타일: (B, 1, 3, L) where 3 = [ECG, pre_rr_ratio, near_pre_rr_ratio]
    - RR features (2-dim) via Cross-Attention late fusion
    """
    def __init__(self, nOUT, n_pid=0,
                 in_channels=1, n_channels_opt3=3, out_ch=180, mid_ch=30, n_rr=2, num_heads=1, **kwargs):
        super().__init__()
        self.n_channels = n_channels_opt3

        # 2D ECG Encoder
        self.inconv = INCONV2D(in_ch=in_channels, out_ch=out_ch)
        self.rub_0 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_1 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)
        self.dropout = nn.Dropout2d(0.3)

        # Global pooling over channel dimension (ECG+RR channels)
        self.channel_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, L')

        # Cross-Attention Fusion
        self.fusion = MultiHeadCrossAttentionFusion(
            emb_dim=out_ch,
            rr_dim=n_rr,
            num_heads=num_heads,
            dropout=0.3
        )

        # Classifier
        self.fc = nn.Linear(2 * out_ch, nOUT)

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        """
        Args:
            ecg_signal: (B, 1, 3, L) - DAEAC style [ECG, pre_rr, near_pre_rr]
            rr_features: (B, 2) - RR features [pre_rr_ratio, near_pre_rr_ratio]
        """
        # ECG encoding
        x = self.inconv(ecg_signal)       # (B, out_ch, 3, L')
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)

        # Pool over channels: (B, C, 3, L') -> (B, C, 1, L') -> (B, C, L')
        x = self.channel_pool(x).squeeze(2)
        x_seq = x.permute(0, 2, 1)  # (B, L', C)

        # Cross-attention fusion
        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)

        fused = self.fusion(x_seq, rr_features)
        logits = self.fc(fused)

        return logits, fused


class ResU_Baseline_1D(nn.Module):
    """
    ResU Baseline for 1D ECG (no RR fusion)
    """
    def __init__(self, nOUT, n_pid=0,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=1):
        super().__init__()

        # 1D ECG Encoder
        self.inconv = INCONV1D(in_ch=in_channels, out_ch=out_ch)
        self.rub_0 = ResU1DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_1 = ResU1DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)
        self.dropout = nn.Dropout1d(0.3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc = nn.Linear(out_ch, nOUT)

    def forward(self, ecg_signal, rr_features=None, rr_remove_ablation=False):
        x = self.inconv(ecg_signal)
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)
        x = self.avgpool(x).squeeze(-1)

        logits = self.fc(x)
        return logits, x


class ResU_Baseline_2D(nn.Module):
    """
    ResU Baseline for 2D ECG (no RR fusion)
    - DAEAC 스타일: (B, 1, 3, L) where 3 = [ECG, pre_rr_ratio, near_pre_rr_ratio]
    - RR features는 입력에 포함되어 있지만 별도 fusion 없음
    """
    def __init__(self, nOUT, n_pid=0,
                 in_channels=1, n_channels_opt3=3, out_ch=180, mid_ch=30, n_rr=2, num_heads=1, **kwargs):
        super().__init__()

        # 2D ECG Encoder
        self.inconv = INCONV2D(in_ch=in_channels, out_ch=out_ch)
        self.rub_0 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_1 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)
        self.dropout = nn.Dropout2d(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(out_ch, nOUT)

    def forward(self, ecg_signal, rr_features=None, rr_remove_ablation=False):
        """
        Args:
            ecg_signal: (B, 1, 3, L) - DAEAC style [ECG, pre_rr, near_pre_rr]
            rr_features: (B, 2) - RR features (unused in baseline)
        """
        x = self.inconv(ecg_signal)
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)

        logits = self.fc(x)
        return logits, x


class ResU_NaiveConcatenate_1D(nn.Module):
    """
    ResU with Naive Concatenation for 1D ECG (opt1)
    - ECG features + RR features via simple concatenation
    """
    def __init__(self, nOUT, n_pid=0,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=1):
        super().__init__()

        # 1D ECG Encoder
        self.inconv = INCONV1D(in_ch=in_channels, out_ch=out_ch)
        self.rub_0 = ResU1DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_1 = ResU1DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)
        self.dropout = nn.Dropout1d(0.3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Classifier (ECG features + RR features)
        self.fc = nn.Linear(out_ch + n_rr, nOUT)
        self.n_rr = n_rr

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        x = self.inconv(ecg_signal)
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)
        ecg_feat = self.avgpool(x).squeeze(-1)

        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)

        fused = torch.cat([ecg_feat, rr_features], dim=1)
        logits = self.fc(fused)

        return logits, fused


class ResU_NaiveConcatenate_2D(nn.Module):
    """
    ResU with Naive Concatenation for 2D ECG (opt3)
    - DAEAC 스타일: (B, 1, 3, L) where 3 = [ECG, pre_rr_ratio, near_pre_rr_ratio]
    - ECG features + 2-dim RR features via simple concatenation
    """
    def __init__(self, nOUT, n_pid=0,
                 in_channels=1, n_channels_opt3=3, out_ch=180, mid_ch=30, n_rr=2, num_heads=1, **kwargs):
        super().__init__()

        # 2D ECG Encoder
        self.inconv = INCONV2D(in_ch=in_channels, out_ch=out_ch)
        self.rub_0 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_1 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)
        self.dropout = nn.Dropout2d(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier (ECG features + 2-dim RR features)
        self.fc = nn.Linear(out_ch + n_rr, nOUT)

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        """
        Args:
            ecg_signal: (B, 1, 3, L) - DAEAC style [ECG, pre_rr, near_pre_rr]
            rr_features: (B, 2) - RR features [pre_rr_ratio, near_pre_rr_ratio]
        """
        x = self.inconv(ecg_signal)
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)
        ecg_feat = self.avgpool(x).squeeze(-1).squeeze(-1)

        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)

        fused = torch.cat([ecg_feat, rr_features], dim=1)
        logits = self.fc(fused)

        return logits, fused


# =============================================================================
# Model Selection Function
# =============================================================================

def get_resu_model(model_type: str, input_mode: str, nOUT: int, n_pid: int = 0, **config):
    """
    Get ResU model based on type and input mode.

    Args:
        model_type: 'baseline', 'naive_concat', 'cross_attention'
        input_mode: 'opt1' (1D) or 'opt3' (2D)
        nOUT: Number of output classes
        n_pid: Number of patients (unused, for interface compatibility)
        **config: Model configuration (in_channels, out_ch, mid_ch, n_rr, num_heads, n_leads)

    Returns:
        model: ResU model instance
    """
    models_1d = {
        'baseline': ResU_Baseline_1D,
        'naive_concat': ResU_NaiveConcatenate_1D,
        'cross_attention': ResU_CrossAttention_1D,
    }

    models_2d = {
        'baseline': ResU_Baseline_2D,
        'naive_concat': ResU_NaiveConcatenate_2D,
        'cross_attention': ResU_CrossAttention_2D,
    }

    if input_mode == 'opt1':
        model_dict = models_1d
    elif input_mode == 'opt3':
        model_dict = models_2d
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}. Must be 'opt1' or 'opt3'")

    if model_type not in model_dict:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(model_dict.keys())}")

    model_class = model_dict[model_type]
    model = model_class(nOUT=nOUT, n_pid=n_pid, **config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_class.__name__}")
    print(f"  - Type  : {model_type}")
    print(f"  - Mode  : {input_mode}")
    print(f"  - Params: {n_params:,}")

    return model
