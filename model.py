import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_act_func(act_func, in_channels):
    if act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'leaky_relu':
        # return nn.LeakyReLU(0.05)
        return nn.LeakyReLU(0.01)
    elif act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'prelu':
        return nn.PReLU(init=0.05)
    elif act_func == 'cprelu':
        return nn.PReLU(num_parameters=in_channels, init=0.05)
    elif act_func == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualClassifier(nn.Module):

    def __init__(self, class_num):
        super(ResidualClassifier, self).__init__()
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.class_num, class_num),
            nn.ReLU(),
            nn.Linear(self.class_num, class_num)
        )

    def forward(self, x):

        return x + self.fc(x)


def _get_act_func(act_func, in_channels):
    if act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'leaky_relu':
        # return nn.LeakyReLU(0.05)
        return nn.LeakyReLU(0.01)
    elif act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'prelu':
        return nn.PReLU(init=0.05)
    elif act_func == 'cprelu':
        return nn.PReLU(num_parameters=in_channels, init=0.05)
    elif act_func == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding, dilation, groups,
                 bias=True, padding_mode='zeros',
                 use_bn=True, use_act=True,
                 act='tanh'):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_act = use_act

        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride,
                              padding, dilation,
                              groups, bias, padding_mode)
        self.bn = nn.BatchNorm2d(out_channel)
        self.tanh = _get_act_func(act, in_channel)

    def forward(self, x):

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.tanh(x)
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, p=0.0):
        super(ResidualBlock, self).__init__()

        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)

        self.conv1 = nn.Conv2d(in_channel, out_channel, (1, kernel_size),
                               stride=(1, stride), padding=(0, int((kernel_size - stride + 1) / 2)),
                               dilation=1, groups=1)

        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=p)

        self.conv2 = nn.Conv2d(out_channel, out_channel, (1, kernel_size),
                               stride=1, padding=(0, int((kernel_size - 1) / 2)),
                               dilation=1, groups=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, stride))

    def forward(self, x):

        net = self.conv1(self.relu1(self.bn1(x)))
        net = self.conv2(self.relu2(self.bn2(net)))

        if self.stride == 1:
            res = net + x
        else:
            res = net + self.avg_pool(x)

        return res


class ASPP(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size, dilations=(1, 6, 12, 18),
                 use_bn=True, use_act=True, act_func='tanh'):
        super(ASPP, self).__init__()

        self.num_scale = len(dilations)

        self.convs = nn.ModuleList()
        for dilation in dilations:
            padding = int(dilation * (kernel_size - 1) / 2.0)
            self.convs.append(ConvBlock(in_channel, out_channel,
                                        (1, kernel_size), stride=1, padding=(0, padding),
                                        dilation=(1, dilation), groups=1, use_bn=use_bn,
                                        use_act=use_act, act=act_func))

    def forward(self, x):
        feats = [self.convs[i](x) for i in range(self.num_scale)]
        res = torch.cat(feats, dim=1)
        return res


class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        return self.gap(x)


class ResidualClassifier(nn.Module):

    def __init__(self, class_num):
        super(ResidualClassifier, self).__init__()
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.class_num, class_num),
            nn.ReLU(),
            nn.Linear(self.class_num, class_num)
        )

    def forward(self, x):

        return x + self.fc(x)



class ACNN(nn.Module):

    def __init__(self,
                 reduction=16,
                 aspp_bn=True,
                 aspp_act=True,
                 lead=2,
                 p=0.0,
                 dilations=(1, 6, 12, 18),
                 act_func='tanh',
                 f_act_func='tanh',
                 apply_residual=False):

        super(ACNN, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)
        self.apply_residual = apply_residual

        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn,
                           use_act=aspp_act, act_func=act_func)

        self.conv1_1 = nn.Conv2d(4, self.dila_num * 4, kernel_size=(1, 3),
                                 stride=(1, 1), padding=(0, 1))
        # self.se_layer_1 = SELayer(self.dila_num * 4, reduction=4)

        self.residual_1 = ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p)

        self.conv2_2 = nn.Conv2d(self.dila_num * 4, self.dila_num ** 2 * 4, kernel_size=(1, 3),
                                 stride=(1, 1), padding=(0, 1))
        # self.se_layer_2 = SELayer(self.dila_num ** 2 * 4, reduction=8)

        self.residual_2 = ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4)
        self.relu = nn.ReLU()

        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func)
        self.se_layer = SELayer(self.dila_num ** 3 * 4, reduction=reduction)

        self.gap = GAP()

        self.fc = nn.Linear(self.dila_num ** 3 * 4, 4)
        self.res_transfer = ResidualClassifier(4)

    def forward(self, x):
        net = self.conv1(x)

        net = self.conv1_1(net)
        # net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.conv2_2(net)
        # net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.se_layer(self.aspp(net))).view(-1, self.dila_num ** 3 * 4)

        logits = self.fc(net)
        if self.apply_residual:
            logits = self.res_transfer(logits)

        return net, logits

class ConcatFusionBlock(nn.Module):

    def __init__(self, emb=64, expansion=2):
        super(ConcatFusionBlock, self).__init__()
        
        self.mem = nn.Sequential(
            nn.Linear(emb, expansion * emb),
            nn.LayerNorm(expansion * emb),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(expansion * emb, emb),
        )

    def forward(self, x, rr_features):
        return torch.cat([self.mem(x), rr_features], dim=1)
    
class ConcatProjectionFusionBlock(nn.Module):

    def __init__(self, emb=64, expansion=2, rr_dim=7):
        super(ConcatProjectionFusionBlock, self).__init__()
        
        self.rr_proj = nn.Sequential(
            nn.Linear(rr_dim, emb),
            nn.LayerNorm(emb),
            nn.GELU()
        )

        self.mem = nn.Sequential(
            nn.Linear(emb, expansion * emb),
            nn.LayerNorm(expansion * emb),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(expansion * emb, emb),
        )

    def forward(self, x, rr_features):
        return torch.cat([self.rr_proj(rr_features), self.mem(x)], dim=1)

    
class MultiHeadCrossAttentionFusionBlock(nn.Module):

    def __init__(self, emb=64, expansion=2, rr_dim=7, num_heads=1):
        super(MultiHeadCrossAttentionFusionBlock, self).__init__()
        
        self.rr_proj = nn.Sequential(
            nn.Linear(rr_dim, emb),
            nn.LayerNorm(emb),
            nn.GELU()
        )

        self.mem = nn.Sequential(
            nn.Linear(emb, expansion * emb),
            nn.LayerNorm(expansion * emb),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(expansion * emb, emb),
        )

        self.mhca = nn.MultiheadAttention(embed_dim=emb, num_heads=num_heads, dropout=0.2, batch_first=True)

    def forward(self, x, rr_features):
        # x: (B, emb), rr_features: (B, rr_dim)
        rr_proj = self.rr_proj(rr_features)  # (B, emb)
        x_mem = self.mem(x)  # (B, emb)
        
        # Reshape for MultiheadAttention: (B, 1, emb)
        rr_proj = rr_proj.unsqueeze(1)  # query
        x_mem = x_mem.unsqueeze(1)      # key, value
        
        # Cross-attention: rr_features attend to x
        attn_out, _ = self.mhca(rr_proj, x_mem, x_mem)  # (B, 1, emb)
        attn_out = attn_out.squeeze(1)  # (B, emb)
        
        return torch.cat([attn_out, x_mem.squeeze(1)], dim=1)  # (B, 2*emb)

    
#=========================================================================================
# main model class 
#=========================================================================================

class MACNN_SE(nn.Module): 
    def __init__(self, reduction=16, aspp_bn=True, aspp_act=True, lead=3, p=0.0, 
                 dilations=(1, 6, 12, 18), act_func='tanh', f_act_func='tanh', 
                 apply_residual=False, 
                 fusion_type=None, fusion_emb=64, fusion_expansion=2, rr_dim=7, num_heads=1): 
        super(MACNN_SE, self).__init__() 
        self.lead = lead 
        self.dila_num = len(dilations) 
        self.apply_residual = apply_residual 
        self.fusion_type = fusion_type
        
        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1), dilation=(1, 1), groups=1) 
        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=act_func) 
        self.se_layer_1 = SELayer(self.dila_num * 4, reduction=4) 
        self.residual_1 = ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p) 
        self.aspp_2 = ASPP(self.dila_num * 4, self.dila_num * 4, 3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=act_func) 
        self.se_layer_2 = SELayer(self.dila_num ** 2 * 4, reduction=8) 
        self.residual_2 = ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p) 
        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4) 
        self.relu = nn.ReLU() 
        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func) 
        self.se_layer = SELayer(self.dila_num ** 3 * 4, reduction=reduction) 
        self.gap = GAP() 
        
        # Fusion block selection
        if fusion_type == 'concat':
            self.fusion_block = ConcatFusionBlock(emb=fusion_emb, expansion=fusion_expansion)
            fc_input_dim = fusion_emb + rr_dim
        elif fusion_type == 'concat_proj':
            self.fusion_block = ConcatProjectionFusionBlock(emb=fusion_emb, expansion=fusion_expansion, rr_dim=rr_dim)
            fc_input_dim = 2 * fusion_emb
        elif fusion_type == 'mhca':
            self.fusion_block = MultiHeadCrossAttentionFusionBlock(emb=fusion_emb, expansion=fusion_expansion, rr_dim=rr_dim, num_heads=num_heads)
            fc_input_dim = 2 * fusion_emb
        else:
            self.fusion_block = None
            fc_input_dim = self.dila_num ** 3 * 4
        
        if fusion_type is not None:
            self.feature_proj = nn.Linear(self.dila_num ** 3 * 4, fusion_emb)
        else:
            self.feature_proj = None
            
        self.fc = nn.Linear(fc_input_dim, 4) 
        self.res_transfer = ResidualClassifier(4) 

    def get_model_params(self):
        """모델의 전체 파라미터 수와 학습 가능한 파라미터 수를 반환합니다."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def forward(self, x, rr_features=None): 
        net = self.conv1(x) 

        net = self.aspp_1(net) 
        net = self.se_layer_1(net) 
        net = self.residual_1(net) 

        net = self.aspp_2(net) 
        net = self.se_layer_2(net) 
        net = self.residual_2(net) 

        net = self.relu(self.bn(net)) 
        net = self.gap(self.se_layer(self.aspp(net))).view(-1, self.dila_num ** 3 * 4) 
        

        if self.fusion_block is not None and rr_features is not None:
            net = self.feature_proj(net)  
            net = self.fusion_block(net, rr_features)
        
        logits = self.fc(net) 
        if self.apply_residual: 
            logits = self.res_transfer(logits) 
        return logits, net 

    def get_feature_maps(self, x): 
        net = self.conv1(x) 

        net = self.aspp_1(net) 
        net = self.se_layer_1(net) 
        net = self.residual_1(net) 

        net = self.aspp_2(net) 
        net = self.se_layer_2(net) 
        net = self.residual_2(net) 

        net = self.relu(self.bn(net)) 
        net = self.se_layer(self.aspp(net)) 
        return net

#=======================================================================================
class MACNN_SE_m(nn.Module):

    def __init__(self,
                 reduction=16,
                 aspp_bn=True,
                 aspp_act=True,
                 lead=2,
                 p=0.0,
                 dilations=(1, 6, 12, 18),
                 act_func='tanh',
                 f_act_func='tanh',
                 apply_residual=False):

        super(MACNN_SE_m, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)
        self.apply_residual = apply_residual

        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn,
                           use_act=aspp_act, act_func=act_func)
        self.se_layer_1 = SELayer(self.dila_num * 4, reduction=4)

        self.residual_1 = nn.Sequential(
            ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p),
            ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p)
        )

        self.aspp_2 = ASPP(self.dila_num * 4, self.dila_num * 4, 3, dilations=dilations,
                           use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer_2 = SELayer(self.dila_num ** 2 * 4, reduction=8)

        self.residual_2 = nn.Sequential(
            ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 1, p=p),
            ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p)
        )

        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4)
        self.relu = nn.ReLU()

        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func)
        self.se_layer = SELayer(self.dila_num ** 3 * 4, reduction=reduction)

        self.gap = GAP()

        self.fc = nn.Linear(self.dila_num ** 3 * 4, 4)
        self.res_transfer = ResidualClassifier(4)

    def forward(self, x):
        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.se_layer(self.aspp(net))).view(-1, self.dila_num ** 3 * 4)

        logits = self.fc(net)
        if self.apply_residual:
            logits = self.res_transfer(logits)

        return net, logits


class MACNN(nn.Module):

    def __init__(self,
                 reduction=16,
                 aspp_bn=True,
                 aspp_act=True,
                 lead=2,
                 p=0.0,
                 dilations=(1, 6, 12, 18),
                 act_func='tanh',
                 f_act_func='tanh',
                 apply_residual=False):

        super(MACNN, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)
        self.apply_residual = apply_residual

        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn,
                           use_act=aspp_act, act_func=act_func)

        self.residual_1 = ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p)

        self.aspp_2 = ASPP(self.dila_num * 4, self.dila_num * 4, 3, dilations=dilations,
                           use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)

        self.residual_2 = ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4)
        self.relu = nn.ReLU()

        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func)

        self.gap = GAP()

        self.fc = nn.Linear(self.dila_num ** 3 * 4, 4)
        self.res_transfer = ResidualClassifier(4)

    def forward(self, x):
        net = self.conv1(x)
        net = self.aspp_1(net)
        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.aspp(net)).view(-1, self.dila_num ** 3 * 4)

        logits = self.fc(net)
        if self.apply_residual:
            logits = self.res_transfer(logits)

        return net, logits

    def get_feature_maps(self, x):

        net = self.conv1(x)

        net = self.aspp_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.aspp(net)

        return net


class MACNN_ResSE(nn.Module):

    def __init__(self, reduction=16, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18), act_func='tanh', f_act_func='tanh'):
        super(MACNN_ResSE, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn,
                           use_act=aspp_act, act_func=act_func)
        self.se_layer_1 = ResSELayer(self.dila_num * 4, reduction=4)

        self.residual_1 = ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p)

        self.aspp_2 = ASPP(self.dila_num * 4, self.dila_num * 4, 3, dilations=dilations,
                           use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer_2 = ResSELayer(self.dila_num ** 2 * 4, reduction=8)

        self.residual_2 = ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4)
        self.relu = nn.ReLU()

        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func)
        self.se_layer = ResSELayer(self.dila_num ** 3 * 4, reduction=reduction)

        self.gap = GAP()

        self.fc = nn.Linear(self.dila_num ** 3 * 4, 4)

    def forward(self, x):
        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.se_layer(self.aspp(net))).view(-1, self.dila_num ** 3 * 4)

        logits = self.fc(net)

        return net, logits

    def get_feature_maps(self, x):

        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = torch.abs(net)
        net = torch.sum(net, dim=1).squeeze()

        return net







    
    
class Decoder(nn.Module):
    
    def __init__(self, in_channel):
        super(Decoder, self).__init__()
        self.in_channel = in_channel

        self.deconv0 = nn.ConvTranspose1d(in_channel, 128, kernel_size=3, stride=1, padding=1)
        self.relu0 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(128)

        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(32)

        self.ups1 = nn.Upsample(scale_factor=2)

        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(16)

        self.deconv4 = nn.ConvTranspose1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(8)

        self.deconv5 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU()
        self.bn5 = nn.BatchNorm1d(4)

        self.deconv6 = nn.ConvTranspose1d(4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        net = self.bn0(self.relu0(self.deconv0(x)))
        net = self.bn1(self.relu1(self.deconv1(net)))
        net = self.bn2(self.relu2(self.deconv2(net)))

        net = self.ups1(net)

        net = self.bn3(self.relu3(self.deconv3(net)))
        net = self.bn4(self.relu4(self.deconv4(net)))
        net = self.bn5(self.relu5(self.deconv5(net)))
        net = self.deconv6(net)

        return net



if __name__ == '__main__':
    print("=" * 80)
    print("Testing MACNN_SE with different fusion configurations")
    print("=" * 80)
    
    # Pseudo inputs
    batch_size = 8
    ecg_input = torch.randn(batch_size, 1,3 ,128)  # (B, lead, 1, length)
    rr_input = torch.randn(batch_size, 2)  # (B, rr_dim)
    
    print("\n[1] MACNN_SE without fusion")
    print("-" * 80)
    model1 = MACNN_SE()
    total1 = sum([param.nelement() for param in model1.parameters()])
    print(f"Number of parameters: {total1/1e6:.2f}M")
   