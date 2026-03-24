"""Frozen ReDimNet-B6 speaker encoder shared by all dual-branch models.

The speaker branch extracts a fixed speaker embedding f_spk from raw waveform,
using a pre-trained ReDimNet-B6 model (trained on VoxCeleb2). The encoder is
frozen (eval mode, no gradient) throughout training.
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio

# ─────────────────────────── ReDimNet import ───────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
redimnet_root = os.path.join(project_root, 'ReDimNet')
if redimnet_root not in sys.path:
    sys.path.append(redimnet_root)

try:
    from redimnet.model import ReDimNetWrap
except ImportError:
    class ReDimNetWrap(nn.Module):
        pass

# ─────────────────────────── Weight loading ───────────────────────────
URL_TEMPLATE = "https://github.com/IDRnD/ReDimNet/releases/download/latest/{model_name}"


def _load_redimnet(model_name='b6', train_type='ptn', dataset='vox2'):
    """Download and instantiate ReDimNet, stripping the built-in spec layer."""
    file_name = f'{model_name}-{dataset}-{train_type}.pt'
    url = URL_TEMPLATE.format(model_name=file_name)

    print(f"Loading ReDimNet: {file_name} ...")
    try:
        full_state_dict = torch.hub.load_state_dict_from_url(
            url, progress=True, map_location='cpu')
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return None

    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']

    # Remove spec-layer config keys (not needed; we supply mel externally)
    for key in ('n_mels', 'n_fft', 'win_length', 'hop_length', 'window'):
        model_config.pop(key, None)

    try:
        model = ReDimNetWrap(**model_config)
    except TypeError:
        minimal_config = {
            k: v for k, v in model_config.items()
            if k in ['C', 'm', 'n_class', 'model_name', 'block_type']
        }
        model = ReDimNetWrap(**minimal_config)

    # Remove spec.* weights from state dict
    spec_keys = [k for k in state_dict if k.startswith("spec.")]
    if spec_keys:
        print(f"Removed {len(spec_keys)} spec-layer keys from state dict.")
        for k in spec_keys:
            del state_dict[k]

    try:
        model.load_state_dict(state_dict, strict=True)
        print("ReDimNet weights loaded successfully.")
    except RuntimeError as e:
        print(f"Warning during loading: {e}")
        model.load_state_dict(state_dict, strict=False)

    return model


# ─────────────────────────── Speaker encoder module ───────────────────────────

class FrozenReDimNetB6(nn.Module):
    """Frozen ReDimNet-B6 speaker encoder.

    Accepts raw waveform, extracts 72-bin log-mel features internally,
    and produces a fixed-dimensional speaker embedding.

    Usage in dual-branch models::

        self.speaker_encoder = FrozenReDimNetB6()
        ...
        with torch.no_grad():
            spk_emb = self.speaker_encoder(waveform)  # [B, spk_dim]
    """

    def __init__(self):
        super().__init__()
        self.model = _load_redimnet()

        # 72-mel spectrogram to match B6 pre-trained weights
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            n_mels=72,
            window_fn=torch.hamming_window,
        )

        # Freeze all parameters
        if self.model is not None:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, waveform):
        """
        Args:
            waveform: Raw waveform [B, T].
        Returns:
            Speaker embedding [B, D] or None if model failed to load.
        """
        if self.model is None:
            return None

        self.model.eval()
        waveform = torch.nan_to_num(waveform)
        spk_feat = self.melspec(waveform) + 1e-6
        spk_feat = spk_feat.log()
        spk_feat = spk_feat - spk_feat.mean(dim=-1, keepdim=True)
        x = spk_feat.unsqueeze(1)  # [B, 1, 72, T]

        out = self.model(x)
        if isinstance(out, tuple):
            emb = out[0]
        else:
            emb = out
        if emb.dim() > 2:
            emb = emb.reshape(emb.size(0), -1)
        return emb

    def train(self, mode=True):
        # Always stay in eval mode
        return super().train(False)
