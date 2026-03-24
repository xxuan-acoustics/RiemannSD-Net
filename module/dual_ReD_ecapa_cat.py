"""Dual-branch model: ECAPA-TDNN (source) + frozen ReDimNet-B6 (speaker)."""

import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from .speaker_encoder import FrozenReDimNetB6


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


class dual_ecapa_model(torch.nn.Module):
    def __init__(self, n_mels=80, embedding_dim=192):
        super(dual_ecapa_model, self).__init__()
        self.specaug = FbankAug()

        # Source branch: trainable ECAPA-TDNN
        self.source_encoder = ECAPA_TDNN(
            input_size=n_mels,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=embedding_dim,
        )

        # Speaker branch: frozen ReDimNet-B6
        self.speaker_encoder = FrozenReDimNetB6()

    def forward(self, feat, waveform_for_ecapa, aug):
        """
        Args:
            feat: Mel spectrogram [B, 80, T] or [B, 1, 80, T].
            waveform_for_ecapa: Raw waveform [B, T] for speaker encoder.
            aug: Whether to apply SpecAugment.
        Returns:
            (source_embedding, speaker_embedding)
        """
        if feat.dim() == 4:
            feat = feat.squeeze(1)
        feat = feat - torch.mean(feat, dim=-1, keepdim=True)
        if aug:
            feat = self.specaug(feat)
        x = feat.transpose(1, 2)
        source_embedding = self.source_encoder(x).squeeze(1)

        speaker_embedding = None
        if waveform_for_ecapa is not None:
            with torch.no_grad():
                speaker_embedding = self.speaker_encoder(waveform_for_ecapa)

        return source_embedding, speaker_embedding


def dual_ReD_ecapa_cat(n_mels=80, num_blocks=6, output_size=256,
                       embedding_dim=512, input_layer="conv2d2", pos_enc_layer_type="rel_pos"):
    return dual_ecapa_model(n_mels=n_mels, embedding_dim=embedding_dim)
