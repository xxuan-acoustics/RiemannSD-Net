import torch
import torch.nn as nn
import torch.nn.functional as F
from .speaker_encoder import FrozenReDimNetB6

class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, first=False, is_pool=True):
        super(Residual_block, self).__init__()
        self.first = first
        self.is_pool = is_pool
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=1)
        
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=1)

        if in_channels != out_channels:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             padding=0, kernel_size=1, stride=1)
        else:
            self.downsample = False
        
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        

        if self.is_pool:
            out = self.mp(out)
            
        return out



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=3, dropout=0.2):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim * heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_dim, heads)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj=None):

        B, N, _ = h.shape
        
        # Linear Transform: [B, N, in] -> [B, N, heads, out]
        Wh = torch.matmul(h, self.W) 
        Wh = Wh.view(B, N, self.heads, self.out_dim) 
        
        a1 = self.a[:self.out_dim, :]
        a2 = self.a[self.out_dim:, :]
        
        # Attention Mechanism using einsum
        Wh1 = torch.einsum('bnhd,dh->bnh', Wh, a1) 
        Wh2 = torch.einsum('bnhd,dh->bnh', Wh, a2)
        
        # Broadcast add -> [B, N, N, h]
        e = Wh1.unsqueeze(2) + Wh2.unsqueeze(1) 
        e = self.leakyrelu(e)
        
        attention = F.softmax(e, dim=2) 
        
        # Weighted sum -> [B, N, h, d]
        h_prime = torch.einsum('bnmh,bmhd->bnhd', attention, Wh)
        
        # Flatten heads
        h_prime = h_prime.contiguous().view(B, N, -1)
        
        return F.elu(h_prime)


# ==========================================
# 3. AASIST Backbone (Source Encoder)
# ==========================================
class AASIST_Backbone(nn.Module):
    def __init__(self, n_mels=80, filts=[70, 32, 32, 32, 64, 64], dims=[64, 64, 64, 64, 128, 128], gat_dims=[64, 32], output_dim=512):
        super(AASIST_Backbone, self).__init__()
        
        self.filts = filts
        
        # Front-end: 适配 [B, 80, T] 输入
        self.front_end = nn.Conv1d(in_channels=n_mels, 
                                   out_channels=filts[0], 
                                   kernel_size=3, 
                                   padding=1)
        
        self.first_bn = nn.BatchNorm1d(num_features=filts[0])
        self.selu = nn.SELU(inplace=True)
        
        # Encoder: 仅前两层开启池化 (is_pool=True)
        # 这将把时间维度从 ~300 降到 ~33，极大节省显存
        self.encoder = nn.Sequential(
            Residual_block(filts[0], filts[0], first=True, is_pool=True),
            Residual_block(filts[0], filts[1], is_pool=True),
            Residual_block(filts[1], filts[2], is_pool=False),
            Residual_block(filts[2], filts[3], is_pool=False),
            Residual_block(filts[3], filts[4], is_pool=False),
            Residual_block(filts[4], filts[5], is_pool=False)
        )
        
        self.post_enc_dim = filts[-1] # 64
        
        # Graph Attention Layers
        self.gat_layer1 = GraphAttentionLayer(self.post_enc_dim, gat_dims[0], heads=3)
        self.gat_layer2 = GraphAttentionLayer(gat_dims[0]*3, gat_dims[1], heads=3)
        
        self.pool_dim = gat_dims[1] * 3 
        
        # Projection to embedding_dim
        self.fc_final = nn.Sequential(
            nn.Linear(self.pool_dim * 2 + self.post_enc_dim * 2, 64), 
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [Batch, 80, Time]
        x = self.front_end(x) 
        x = self.first_bn(x)
        x = self.selu(x)
        
        e_x = self.encoder(x) # Output shape approx [Batch, 64, 33]
        
        graph_input = e_x.permute(0, 2, 1) # [Batch, 33, 64]
        
        gat_out1 = self.gat_layer1(graph_input) 
        gat_out2 = self.gat_layer2(gat_out1)    
        
        g_max, _ = torch.max(gat_out2, dim=1)
        g_avg = torch.mean(gat_out2, dim=1)
        e_max, _ = torch.max(e_x, dim=2)
        e_avg = torch.mean(e_x, dim=2)
        
        feat = torch.cat([g_max, g_avg, e_max, e_avg], dim=1)
        out = self.fc_final(feat)
        
        return out


# ==========================================
# 4. Dual Model Wrapper (主模型类)
# ==========================================
class DualAASIST(nn.Module):
    def __init__(self, n_mels=80, embedding_dim=512):
        super(DualAASIST, self).__init__()
        print(f"Model: Dual AASIST initialized")
        print(f"  - Path 1: Source Encoder (AASIST, Trainable, Output: {embedding_dim})")
        print(f"  - Path 2: Speaker Encoder (ECAPA-TDNN, Frozen, Output: 192)")

        # Path 1: Source (AASIST)
        self.source_encoder = AASIST_Backbone(n_mels=n_mels, output_dim=embedding_dim)

        # Path 2: Speaker (frozen ReDimNet-B6)
        self.speaker_encoder = FrozenReDimNetB6()

    def forward(self, mel, waveform_for_ecapa=None, aug=False):
        """
        参数:
          mel: [Batch, 80, Time] -> 给 AASIST
          waveform_for_ecapa: [Batch, Time] (或者是 None) -> 给 ECAPA
          aug: (未使用，保留接口兼容)
        """
        
        # 1. Source Path (AASIST)
        source_embedding = self.source_encoder(mel)

        # 2. Speaker Path (ECAPA)
        speaker_embedding = None
        
        # 修复 NoneType error: 仅当 waveform 存在时才计算
        if waveform_for_ecapa is not None:
            with torch.no_grad():
                speaker_embedding = self.speaker_encoder(waveform_for_ecapa)
        
        return source_embedding, speaker_embedding


# ==========================================
# 5. Factory Function (兼容 main.py 接口)
# ==========================================
def dual_aasist_cat(n_mels=80, num_blocks=6, output_size=256, 
                    embedding_dim=512, input_layer="conv2d2", 
                    pos_enc_layer_type="rel_pos", **kwargs):
    """
    接收 main.py 传递的所有参数（包括 num_blocks, input_layer 等），
    防止 TypeError。
    """
    model = DualAASIST(n_mels=n_mels, embedding_dim=embedding_dim)
    return model