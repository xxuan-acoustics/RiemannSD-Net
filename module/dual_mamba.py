import torch
from wenet.transformer.encoder_cat import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
import torch.nn as nn
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
    

from mamba_ssm.modules.mamba_simple import Mamba as MambaSsm

# from mamba.mamba_ssm.modules.mamba_simple import Mamba#

class PN_BiMambas_Encoder(nn.Module):
    def __init__(self, d_model, n_state):
        super(PN_BiMambas_Encoder, self).__init__()
        self.d_model = d_model
        
        self.mamba = MambaSsm(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # self.concat_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # self.feed_forward2 = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     Swish(),
        #     nn.Dropout(0),
        #     nn.Linear(d_model * 4, d_model),
        #     nn.Dropout(0)
        # )

    def forward(self, x):
        # #------------去掉双向，只剩单向----------------------
        # # Residual connection of the original input
        # residual = x

        # # #FNN
        # # mamba_in = self.norm1(x)
        # # ff_in = self.feed_forward(mamba_in)
        # # input = ff_in + residual
        
        # # Forward Mamba
        # # x_norm = self.norm1(input)#2FNN
        # x_norm = self.norm1(x)
        # mamba_out_forward = self.mamba(x_norm)

        # # # Backward Mamba
        # # x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        # # mamba_out_backward = self.mamba(x_flip)
        # # mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
        # # mamba_out = mamba_out_forward + mamba_out_backward

        # mamba_out = mamba_out_forward
        
        # mamba_out = self.norm2(mamba_out)
        # ff_out = self.feed_forward(mamba_out)

        # output = ff_out + residual


        #-=-----去掉FFN-----
        # Residual connection of the original input
        # residual = x

        
        # # Forward Mamba
        # x_norm = self.norm1(x)
        # mamba_out_forward = self.mamba(x_norm)

        # # Backward Mamba
        # x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        # mamba_out_backward = self.mamba(x_flip)
        # mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
        # mamba_out = mamba_out_forward + mamba_out_backward

        
        # mamba_out = self.norm2(mamba_out)
        # ff_out = self.feed_forward(mamba_out)
        # output = mamba_out + residual

        # #-----------去掉3个LayerNorm--------------
        # # Residual connection of the original input
        # residual = x
        
        # # Forward Mamba
        # # x_norm = self.norm1(x)
        # mamba_out_forward = self.mamba(x)

        # # # Backward Mamba
        # x_flip = torch.flip(x, dims=[1])  # Flip Sequence
        # mamba_out_backward = self.mamba(x_flip)
        # mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
        # mamba_out = mamba_out_forward + mamba_out_backward

        
        # # mamba_out = self.norm2(mamba_out)
        # ff_out = self.feed_forward(mamba_out)

        # output = ff_out + residual


        # # #----------原始 -------
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence

        # x_flip = self.norm1(x_flip)#2.14加
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # print("mamba_out_forward",mamba_out_forward.shape)#torch.Size([20, 208, 144])
        # print("mamba_out_backward",mamba_out_backward.shape)#torch.Size([20, 208, 144])


        # ADD forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        # print("add_mamba_out",mamba_out.shape) #([20, 208, 144])
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)
        # ff_out = self.feed_forward2(mamba_out)


        output = ff_out + residual
        return output

class PN_BiMambas(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(PN_BiMambas_Encoder(
                d_model = dim, 
                n_state = 16
            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x

from .speaker_encoder import FrozenReDimNetB6

class dual_mamba(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):

        super(dual_mamba, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("num_blocks: {}".format(num_blocks))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        print("embedding_dim: {}".format(embedding_dim))
        
        self.specaug = FbankAug()
        
        # --- [Path 1] Source Tracing Backbone ---
        # 这里的 PN_BiMambas 不需要知道 waveform 的存在
        self.Mamba = PN_BiMambas(dim=80, depth=12)

        input_dim_for_pooling = 80 
        self.pooling = AttentiveStatisticsPooling(input_dim_for_pooling)
        self.bn = BatchNorm1d(input_size=input_dim_for_pooling * 2) 
        self.fc = torch.nn.Linear(input_dim_for_pooling * 2, embedding_dim)
        
        # Speaker branch: frozen ReDimNet-B6
        self.speaker_encoder = FrozenReDimNetB6()
        
    
    def forward(self, feat, waveform_for_ecapa, aug): 
        """
        feat: Mel Spectrogram [Batch, 1, 80, Time] -> 用于 Mamba 提取 Source
        waveform_for_ecapa: Raw Waveform [Batch, Time] -> 用于 ECAPA 提取 Speaker
        """
        
        # ===========================
        # Path 1: Source Embedding (Mamba)
        # ===========================
        # feat: [Batch, 1, 80, T]
        feat = feat - torch.mean(feat, dim=-1, keepdim=True)
        if aug == True:
            feat = self.specaug(feat)
            
        feat = feat.permute(0, 2, 1) # [B, 80, T] -> [B, T, 80]
        
        # 注意：这里改回只传 feat！PN_BiMambas 不需要 waveform
        x = self.Mamba(feat) 
        
        x = x.permute(0, 2, 1) # [B, T, 80] -> [B, 80, T]
        x = self.pooling(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
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
    

def dual_mamba_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=512, input_layer="conv2d2", pos_enc_layer_type="rel_pos"):
    model = dual_mamba(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model

# model= conformer_cat()
# print(model)
# for i in range(6, 7):
#     print("num_blocks is {}".format(i))
#     model = conformer_cat(num_blocks=i)
#
#     import time
#     model = model.eval()
#     time1 = time.time()
#     with torch.no_grad():
#         for i in range(100):
#             data = torch.randn(1,  80, 300)
#             embedding = model(data)
#     time2 = time.time()
#     val = (time2 - time1)/100
#     rtf = val / 5
#
#     total = sum([param.nelement() for param in model.parameters()])
#     print("total param: {:.8f}M".format(total/1e6))
#     print("RTF {:.4f}".format(rtf))