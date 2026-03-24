import torch
import torch.nn as nn
import torchvision.models as models  # [新增] 引入 torchvision 以使用 ResNet
from .speaker_encoder import FrozenReDimNetB6

class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 10), time_mask_width = (0, 5)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class dual_resnet_model(torch.nn.Module):
    def __init__(self, n_mels=80, embedding_dim=192):

        super(dual_resnet_model, self).__init__()
        print("Model: Dual ResNet34 (Source: Trainable, Speaker: Frozen)")
        print("embedding_dim: {}".format(embedding_dim))
        
        self.specaug = FbankAug()
        
        # ===========================
        # [Path 1] Source Tracing Backbone (Trainable ResNet34)
        # ===========================
        # 使用 torchvision 的标准 ResNet34
        # weights=None 表示不使用预训练权重（因为 ImageNet 是 RGB 图像，频谱图性质不同，通常从头训练效果更好或持平）
        self.source_encoder = models.resnet34(weights=None)
        
        # 1. 修改第一层卷积：ResNet 默认接受 3 通道 (RGB)，我们需要改为 1 通道 (频谱图)
        self.source_encoder.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 2. 修改最后一层全连接层 (fc)：映射到我们需要的 embedding_dim
        num_ftrs = self.source_encoder.fc.in_features  # ResNet34 默认是 512
        self.source_encoder.fc = nn.Linear(num_ftrs, embedding_dim)
        
        # Speaker branch: frozen ReDimNet-B6
        self.speaker_encoder = FrozenReDimNetB6()
        
    
    def forward(self, feat, waveform_for_ecapa, aug): 
        """
        feat: Mel Spectrogram [Batch, 80, Time] or [Batch, 1, 80, Time] -> 用于 Path 1 提取 Source
        waveform_for_ecapa: Raw Waveform [Batch, Time] -> 用于 Path 2 提取 Speaker
        """
        
        # ===========================
        # Path 1: Source Embedding (Trainable ResNet34)
        # ===========================
        # 统一输入维度处理
        if feat.dim() == 4:
            feat = feat.squeeze(1) # 先统一变成 [B, 80, T]
            
        # 归一化
        feat = feat - torch.mean(feat, dim=-1, keepdim=True)
        
        if aug == True:
            feat = self.specaug(feat)
            
        # ResNet 是 2D CNN，期望输入为 [Batch, Channels, Height, Width]
        # 这里我们将 Mel 维度 (80) 视为 Height，时间维度视为 Width
        # 输入变为: [Batch, 1, 80, Time]
        x = feat.unsqueeze(1) 
        
        # 前向传播 (Conv -> BatchNorm -> ReLU -> MaxPool -> Layers -> AvgPool -> FC)
        # ResNet 内部自带 AdaptiveAvgPool2d((1, 1))，所以对变长输入 Time 也是鲁棒的
        source_embedding = self.source_encoder(x) # 输出 [Batch, embedding_dim]

        # ===========================
        # Path 2: Speaker Embedding (Frozen ECAPA)
        # ===========================
        speaker_embedding = None
        if waveform_for_ecapa is not None:
            with torch.no_grad():
                speaker_embedding = self.speaker_encoder(waveform_for_ecapa)
                
        # ===========================
        # Return Both
        # ===========================
        return source_embedding, speaker_embedding
    

def dual_resnet34_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=512, input_layer="conv2d2", pos_enc_layer_type="rel_pos"):
    # 实例化新的 ResNet 版本模型
    model = dual_resnet_model(n_mels=n_mels, embedding_dim=embedding_dim)
    return model