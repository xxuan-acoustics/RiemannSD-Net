import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from .speaker_encoder import FrozenReDimNetB6
import torch.utils.checkpoint as checkpoint 



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000): 
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            padding = torch.zeros(1, seq_len - self.pe.size(1), self.pe.size(2)).to(x.device)
            self.pe = torch.cat([self.pe, padding], dim=1)
        x = x + self.pe[:, :seq_len, :]
        return x

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
        if dim == 1: mask = mask.unsqueeze(2)
        else: mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x



class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return 0.5 * x + residual 

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1, scale=1.0):
        """
        scale: 缩放因子。
        对于共享权重的 Macaron 结构，通常设为 0.5。
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                                        padding=padding, groups=d_model)
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale 

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2) 
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        # 关键：应用 0.5 的缩放
        return self.scale * x + residual



class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_head=4, num_blocks=6, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.pos_encoder = PositionalEncoding(output_dim, max_len=20000)
        self.dropout = nn.Dropout(dropout)
        
        # 使用权重共享的 Block
        self.blocks = nn.ModuleList([
            SharedConvConformerBlock(output_dim, n_head, dropout=dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for block in self.blocks:

            if self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
        return x 


class dual_conv_conformer_model(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, embedding_dim=192, input_layer="conv2d", 
            pos_enc_layer_type="rel_pos"):

        super(dual_conv_conformer_model, self).__init__()
        print("Model: Dual Conv-Conformer (Shared Weights) + Frozen ECAPA")
        print("Configuration: Parameters UNCHANGED compared to standard Conformer.")
        print(f"Num Blocks: {num_blocks}, Embedding Dim: {embedding_dim}")
        
        self.specaug = FbankAug()
        
        # --- [Path 1] Source Tracing Backbone ---
        conformer_dim = 256 
        
        self.SourceEncoder = ConformerEncoder(
            input_dim=n_mels,         
            output_dim=conformer_dim, 
            n_head=4,
            num_blocks=num_blocks,
            dropout=0.1
        )

        input_dim_for_pooling = conformer_dim 
        self.pooling = AttentiveStatisticsPooling(input_dim_for_pooling)
        self.bn = BatchNorm1d(input_size=input_dim_for_pooling * 2) 
        self.fc = torch.nn.Linear(input_dim_for_pooling * 2, embedding_dim)
        
        # Speaker branch: frozen ReDimNet-B6
        self.speaker_encoder = FrozenReDimNetB6()
        
    
    def forward(self, feat, waveform_for_ecapa, aug): 
        # Path 1: Source Embedding
        if feat.dim() == 4:
            feat = feat.squeeze(1) 

        feat = feat - torch.mean(feat, dim=-1, keepdim=True)
        if aug == True:
            feat = self.specaug(feat)
            
        x = feat.permute(0, 2, 1) # [B, T, 80]
        x = self.SourceEncoder(x) # [B, T, 256]
        
        x = x.permute(0, 2, 1) # [B, 256, T]
        x = self.pooling(x) 
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        source_embedding = x.squeeze(1) 

        # Path 2: Speaker Embedding
        speaker_embedding = None
        if waveform_for_ecapa is not None:
            with torch.no_grad():
                speaker_embedding = self.speaker_encoder(waveform_for_ecapa)
        
        return source_embedding, speaker_embedding
    
def dual_conv_conformer_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=512, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = dual_conv_conformer_model(n_mels=n_mels, num_blocks=num_blocks, embedding_dim=embedding_dim)
    return model