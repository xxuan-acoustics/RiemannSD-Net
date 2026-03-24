import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import librosa

# 保持原有的 Augmentation 模块不变
class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 10), time_mask_width=(0, 5)):
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

# -----------------------------------------------------------------------
# ResNet Components
# -----------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Conv3x3 with padding
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet34(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], block=BasicBlock, num_filters=[32, 64, 128, 256], n_mels=80, embedding_dim=192):
        super(ResNet34, self).__init__()

        self.inplanes = num_filters[0]
        self.specaug = FbankAug() # Spec augmentation

        # 初始卷积层：适应音频输入的单通道 (1 channel)
        # 这里的 stride 可以调整，stride=(1,1) 保留更多细节，stride=(2,2) 类似原版 ResNet 减少显存
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 2), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2)

        # Statistics Pooling 输出维度是 channel * 2 (Mean + Std)
        self.fc_input_dim = num_filters[3] * block.expansion * 2
        
        # 最终的全连接层 (Embedding Layer)
        self.bn_emb = nn.BatchNorm1d(self.fc_input_dim)
        self.fc_emb = nn.Linear(self.fc_input_dim, embedding_dim)
        self.bn_final = nn.BatchNorm1d(embedding_dim)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, aug):
        # x shape: [Batch, n_mels, Time]
        
        # 1. 预处理 (Mean Normalization)
        x = x - torch.mean(x, dim=-1, keepdim=True)

        # 2. SpecAugment
        if aug:
            x = self.specaug(x)

        # 3. 增加 Channel 维度: [Batch, n_mels, Time] -> [Batch, 1, n_mels, Time]
        x = x.unsqueeze(1)

        # 4. ResNet Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Output shape: [Batch, 256, Freq', Time'] (假设 filter=256)

        # 5. Statistics Pooling (Mean + Std) over (Freq, Time) dimensions
        # 将 (Batch, C, H, W) 展平为 (Batch, C, H*W) 然后求均值和方差
        x = x.view(x.size(0), x.size(1), -1) 
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        x = torch.cat((mean, std), dim=1) # [Batch, C*2]

        # 6. Embedding Layer
        x = self.bn_emb(x)
        x = self.fc_emb(x)
        x = self.bn_final(x)

        return x

# -----------------------------------------------------------------------
# Factory Functions (保持接口兼容)
# -----------------------------------------------------------------------

def resnet34(n_mels=80, embedding_dim=192):
    # 标准 ResNet34 配置: layers=[3, 4, 6, 3]
    # num_filters 可以根据显存调整，标准是 [64, 128, 256, 512]
    # 这里为了适应音频任务通常稍微减小一点 filter 或者保持标准
    model = ResNet34(layers=[3, 4, 6, 3], num_filters=[64, 128, 256, 512], n_mels=n_mels, embedding_dim=embedding_dim)
    return model




# if __name__ == "__main__":
#     # 测试代码
#     model = resnet34(n_mels=80, embedding_dim=192)
#     # 输入模拟: [Batch, n_mels, Time]
#     # dummy_input = torch.randn(2, 80, 200) 
#     # out = model(dummy_input, aug=True)
#     # print(f"Input shape: {dummy_input.shape}")
#     # print(f"Output embedding shape: {out.shape}") # Should be [2, 192]