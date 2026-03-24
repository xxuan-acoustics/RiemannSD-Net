import torch
import torch.nn as nn
import math
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
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

# ==========================================
# [关键修改] Positional Encoding
# 将默认 max_len 从 5000 改为 20000
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
             raise RuntimeError(f"Input sequence length {seq_len} exceeds max PE length {self.pe.size(1)}. Please increase max_len.")
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# ==========================================
# Transformer Backbone
# ==========================================
class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TransformerBackbone, self).__init__()
        
        # 特征投影层
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # 位置编码 (max_len 默认为 20000)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=20000)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # [Batch, Time, Feats]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.d_model = d_model

    def forward(self, x):
        # x shape: [Batch, Time, input_dim]
        
        x = self.input_proj(x) # -> [Batch, Time, d_model]
        x = x * math.sqrt(self.d_model) # 缩放
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        return output

# ==========================================
# Dual Model (Source: Transformer, Speaker: ECAPA)
# ==========================================
class dual_transformer_model(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):

        super(dual_transformer_model, self).__init__()
        print("Model: Dual Transformer (Source) + Frozen ECAPA (Speaker)")
        print("num_blocks: {}".format(num_blocks))
        print("embedding_dim: {}".format(embedding_dim))
        
        self.specaug = FbankAug()
        
        # --- [Path 1] Source Tracing Backbone (Transformer) ---
        # 这里的 d_model 设为 80 以匹配 Mel 维度，避免额外的 Pooling 复杂度
        # 你也可以设大一点(如 256)，但需要调整 input_dim_for_pooling
        
        self.SourceEncoder = TransformerBackbone(
            input_dim=n_mels,     # 输入 80
            d_model=80,           # 内部维度
            nhead=4,
            num_layers=num_blocks, 
            dim_feedforward=320,
            dropout=0.1
        )

        # 统计池化层
        input_dim_for_pooling = 80 
        self.pooling = AttentiveStatisticsPooling(input_dim_for_pooling)
        self.bn = BatchNorm1d(input_size=input_dim_for_pooling * 2) 
        self.fc = torch.nn.Linear(input_dim_for_pooling * 2, embedding_dim)
        
        # Speaker branch: frozen ReDimNet-B6
        self.speaker_encoder = FrozenReDimNetB6()
        
    
    def forward(self, feat, waveform_for_ecapa, aug): 
        """
        feat: Mel Spectrogram [Batch, 1, 80, Time]
        waveform_for_ecapa: Raw Waveform [Batch, Time]
        """
        
        # ===========================
        # Path 1: Source Embedding (Transformer)
        # ===========================
        # feat: [Batch, 1, 80, T]
        if feat.dim() == 4:
            feat = feat.squeeze(1) # [B, 80, T]

        feat = feat - torch.mean(feat, dim=-1, keepdim=True)
        if aug == True:
            feat = self.specaug(feat)
            
        # Transformer 需要 [Batch, Time, Feats]
        feat = feat.permute(0, 2, 1) # [B, 80, T] -> [B, T, 80]
        
        # 传入 Transformer
        x = self.SourceEncoder(feat) # 输出 [B, T, 80]
        
        # 准备 Pooling
        # x: [B, T, 80] -> permute -> [B, 80, T]
        x = x.permute(0, 2, 1) 
        
        x = self.pooling(x) # Global Pooling -> [B, 160, 1]
        x = self.bn(x)
        x = x.permute(0, 2, 1) # [B, 1, 160]
        x = self.fc(x)
        source_embedding = x.squeeze(1) # [Batch, 192]

        # ===========================
        # Path 2: Speaker Embedding (Frozen ECAPA)
        # ===========================
        speaker_embedding = None
        if waveform_for_ecapa is not None:
            with torch.no_grad():
                speaker_embedding = self.speaker_encoder(waveform_for_ecapa)
        
        return source_embedding, speaker_embedding
    

def dual_transformer_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=512, input_layer="conv2d2", pos_enc_layer_type="rel_pos"):
    
    model = dual_transformer_model(n_mels=n_mels, num_blocks=num_blocks, embedding_dim=embedding_dim)
    return model