

class INCONV2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
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
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(ResUBlock, self).__init__()
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
            # Adjust size if needed
            if out.size(-1) != encoder_out[-1 - idx].size(-1):
                padding_size = encoder_out[-1 - idx].size(-1) - out.size(-1)
                if padding_size > 0:
                    out = F.pad(out, (0, padding_size, 0, 0))
                elif padding_size < 0:
                    out = out[..., :encoder_out[-1 - idx].size(-1)]

            out = layer(torch.cat([out, encoder_out[-1 - idx]], dim=1))

        # Ensure output size matches input
        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]

        out += x_in

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out

class ResU_CrossAttention(nn.Module):
    """
    B2: Cross-Attention without Dense Block
    RR Query attends to ECG sequence -> AvgPool -> Linear

    이거 opt1 opt2 opt3? 잘 하게 해줘 이거 main_autoexp대로
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder (no Dense Block)
        if 1Dinput : #opt 1일때임 (2개 early입력)
            self.conv = nn.Conv1d(in_channels, out_ch, 5, padding=2, stride=2, bias=False)
            self.bn = nn.BatchNorm1d(out_ch)
            self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
            self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3)
        elif 3Dinput : # opt3일때임
            #INCONV2D랑 ResU2DBlock  이런거 적절히 잘 수정해줘 forward에도 (입력구조에 맞게)
            self.inconv = INCONV2D(in_ch=lead, out_ch=out_ch)
            self.rub_0 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
            self.rub_1 = ResU2DBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)

        # Cross-Attention (Query: RR, Key/Value: ECG)
        self.fusion = MultiHeadCrossAttentionFusion(
            emb_dim=out_ch,
            rr_dim=n_rr,
            num_heads=num_heads,
            dropout=0.3
        )
        self.dropout = nn.Dropout1d(0.3)
        # Post-attention processing
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(2*out_ch, 4),
        )


    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.dropout(x)
        x_seq = x.permute(0, 2, 1)  # (B, L, C) - batch_first format
        
        # Cross-attention
        fused = self.fusion(x_seq, rr_features)

        logits = self.fc(fused)
        return logits, fused
