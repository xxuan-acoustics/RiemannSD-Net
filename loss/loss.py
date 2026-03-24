"""Loss functions.

This module contains all loss functions used for source verification training:
  - amsoftmax: Standard AM-Softmax (baseline)
  - FuzzyArcFaceLoss: Fuzzy membership ArcFace
  - AAMsoftmax: Additive Angular Margin Softmax (ArcFace)
  - ChebyAAMSoftmax: Chebyshev polynomial AAM (no speaker disentanglement)
  - ChebySDAAMSoftmax: ChebySD-AAM (proposed, Section 3.2)
  - RiemannSDAAMSoftmax: RiemannSD-AAM (proposed, Section 3.3)

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .utils import accuracy


class amsoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30, **kwargs):
        super(amsoftmax, self).__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        print('Initialised AM-Softmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        acc = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        return loss, acc




class FuzzyArcFaceLoss(nn.Module):
    """ArcFace with fuzzy membership for dynamic margin adjustment."""

    def __init__(self, in_features, out_features, s=64.0, m=0.50, tau=0.1, easy_margin=False):
        super(FuzzyArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.tau = tau
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine_yi = cosine.gather(1, label.view(-1, 1)).squeeze()

        # Fuzzy membership
        abs_cosine_yi = torch.abs(cosine_yi)
        mu = torch.where(abs_cosine_yi >= self.tau, abs_cosine_yi, torch.ones_like(abs_cosine_yi))
        m_adjusted = self.m * mu

        cosine_yi_clamped = torch.clamp(cosine_yi, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine_yi_clamped)
        final_theta = theta + m_adjusted
        phi = torch.cos(final_theta)

        if self.easy_margin:
            phi = torch.where(cosine_yi > 0, phi, cosine_yi)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = cosine + (phi - cosine_yi).view(-1, 1) * one_hot
        output *= self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


class AAMsoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace)."""

    def __init__(self, n_class, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 512), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


# ───────────────────────────── Chebyshev utilities ─────────────────────────────

def clenshaw_curtis_chebyshev_coefficients(func, degree=30, num_samples=1000, margin=0.3):
    """Fit Chebyshev coefficients for func(x, margin) on [-1, 1]."""
    j = np.arange(num_samples)
    x = np.cos(np.pi * j / (num_samples - 1))
    y = func(x, margin)
    coeffs = np.polynomial.chebyshev.chebfit(x, y, degree)
    return coeffs


class ChebyshevClenshawFunction(torch.autograd.Function):
    """Custom autograd for Clenshaw evaluation of Chebyshev series."""

    @staticmethod
    def forward(ctx, x, coeffs):
        n = coeffs.shape[0] - 1
        b_kplus1 = torch.zeros_like(x)
        b_kplus2 = torch.zeros_like(x)
        x2 = 2 * x
        for k in range(n, -1, -1):
            b_k = coeffs[k] + x2 * b_kplus1 - b_kplus2
            b_kplus2 = b_kplus1
            b_kplus1 = b_k
        result = b_k - b_kplus2 * x
        ctx.save_for_backward(x, coeffs)
        ctx.n = n
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, coeffs = ctx.saved_tensors
        n = ctx.n
        U = [torch.ones_like(x)]
        if n >= 1:
            U.append(2 * x)
            for k in range(2, n):
                U.append(2 * x * U[k - 1] - U[k - 2])
        derivative = torch.zeros_like(x)
        for k in range(1, n + 1):
            derivative = derivative + coeffs[k] * k * U[k - 1]
        return grad_output * derivative, None


class ChebyAAMSoftmax(nn.Module):
    """ChebyAAM: Chebyshev polynomial approximation of ArcFace.

    Replaces the standard cos(arccos(x) + m) with a Chebyshev series expansion,
    providing gradient stability near the boundary of [-1, 1].

    Args:
        n_class: Number of classes.
        m: Additive angular margin (default: 0.3).
        s: Cosine scale factor (default: 30).
        in_feats: Input embedding dimension (default: 512).
        chebyshev_degree: Chebyshev polynomial order (default: 30).
        num_samples: Number of samples for coefficient fitting.
        easy_margin: Whether to use easy margin mode.
        pos_squash_k: Positive squash coefficient (>1 to activate).
    """

    def __init__(self, n_class, m=0.3, s=30.0, in_feats=512,
                 chebyshev_degree=30, num_samples=1000,
                 easy_margin=False, pos_squash_k=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_feats), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = easy_margin
        self.pos_squash_k = pos_squash_k
        self.chebyshev_degree = chebyshev_degree
        self.num_samples = num_samples
        self._update_margin(m)

    def forward(self, x, label):
        """
        Args:
            x: Feature vectors [B, D], unnormalized.
            label: Class labels [B].
        Returns:
            loss: Scalar training loss.
            prec1: Top-1 accuracy (%).
        """
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if self.pos_squash_k > 1:
            idx = torch.arange(cosine.size(0), device=cosine.device)
            cosine_y = 1.0 - (1.0 - cosine[idx, label]).pow(self.pos_squash_k)
            cosine = cosine.clone()
            cosine[idx, label] = cosine_y

        # Chebyshev approximation of cos(arccos(cosine) + margin)
        phi = ChebyshevClenshawFunction.apply(cosine, self.coefficients)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1

    def update_margin(self, m):
        """Update margin and recompute Chebyshev coefficients (for margin scheduling)."""
        self._update_margin(m)

    def _update_margin(self, m):
        self.m = m
        self.th = math.cos(math.pi - m)
        self.mmm = 1.0 + math.cos(math.pi - m)

        def _target(u, margin):
            return np.cos(np.arccos(np.clip(u, -1, 1)) + margin)

        coeffs_t = torch.from_numpy(
            clenshaw_curtis_chebyshev_coefficients(
                _target, degree=self.chebyshev_degree,
                num_samples=self.num_samples, margin=m,
            )
        ).float()

        if not hasattr(self, "coefficients"):
            self.register_buffer("coefficients", coeffs_t)
        else:
            self.coefficients = coeffs_t.to(self.coefficients.device)



class ChebySDAAMSoftmax(nn.Module):
    """ChebySD-AAM: Chebyshev Speaker-Disentangled AAM-Softmax (proposed, Section 3.2).

    Combines Chebyshev polynomial approximation of cos(arccos(x) + m) for
    gradient-stable angular margin with a speaker-aware penalty term that
    discourages the source embedding from correlating with speaker identity.

    NOTE: The current implementation computes the speaker penalty via a
    batch-level speaker similarity matrix aggregated per class (H_matrix),
    whereas the paper (Eq. 2) defines M_spk = max(0, |f_src^T f_spk| - tau).

    Args:
        in_feats: Dimension of source embedding f_src.
        n_class: Number of source classes.
        m: Additive angular margin (default: 0.3).
        s: Cosine scale factor (default: 30).
        lambda_val: Speaker disentanglement coefficient (default: 0.5).
        cheby_order: Order K of the Chebyshev series (default: 10).
    """

    def __init__(self, in_feats, n_class, m, s, lambda_val=0.5, cheby_order=10):
        super(ChebySDAAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.lambda_val = lambda_val
        self.cheby_order = cheby_order
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, in_feats), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        # Pre-compute Chebyshev coefficients for cos(arccos(x) + m)
        coeffs = self._calculate_coeffs(m, cheby_order)
        self.register_buffer('coeffs', coeffs)
        print(f'Initialised ChebySD-AAM: m={self.m:.3f} s={self.s:.3f} lambda={self.lambda_val:.3f} order={self.cheby_order}')

    def _calculate_coeffs(self, m, order):
        """Compute Chebyshev series coefficients per Eq. (2)."""
        coeffs = []
        # k=0: store 0.5 * a0 for Clenshaw convenience
        a0 = -2 * math.sin(m) / math.pi
        coeffs.append(0.5 * a0)
        # k=1
        coeffs.append(math.cos(m))
        # k >= 2: odd terms are zero, even terms from closed-form
        for k in range(2, order + 1):
            if k % 2 != 0:
                coeffs.append(0.0)
            else:
                n = k // 2
                term = (1 / (2 * n - 1)) - (1 / (2 * n + 1))
                val = (2 * math.sin(m) / math.pi) * term
                coeffs.append(val)
        return torch.tensor(coeffs, dtype=torch.float32)

    def _clenshaw_recurrence(self, x, coeffs):
        """Clenshaw algorithm: numerically stable evaluation of sum(c_k * T_k(x))."""
        b_k2 = torch.zeros_like(x)
        b_k1 = torch.zeros_like(x)
        x2 = 2 * x
        for k in range(len(coeffs) - 1, 0, -1):
            c_k = coeffs[k]
            b_k = c_k + x2 * b_k1 - b_k2
            b_k2 = b_k1
            b_k1 = b_k
        val = coeffs[0] + x * b_k1 - b_k2
        return val

    def forward(self, x, speaker_emb, label=None):
        """
        Args:
            x: Source embedding f_src [B, D].
            speaker_emb: Frozen speaker embedding f_spk [B, D].
            label: Source labels [B].
        """
        # Part 1: Chebyshev-approximated angular margin
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)
        phi = self._clenshaw_recurrence(cosine, self.coeffs)

        # Part 2: Speaker disentanglement penalty (polynomial-weighted)
        spk_emb_norm = F.normalize(speaker_emb, p=2, dim=1)
        spk_sim_matrix = torch.mm(spk_emb_norm, spk_emb_norm.t())
        spk_sim_poly = torch.pow(spk_sim_matrix, 2)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        aggregated_sim = torch.mm(spk_sim_poly, one_hot)
        class_counts = one_hot.sum(dim=0).view(1, -1).clamp(min=1.0)
        H_matrix = aggregated_sim / class_counts

        # Part 3: Combine logits
        # Target: Chebyshev phi; Non-target: cosine + lambda * H
        negative_logits = cosine + (self.lambda_val * H_matrix)
        output = (one_hot * phi) + ((1.0 - one_hot) * negative_logits)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


class RiemannSDAAMSoftmax(nn.Module):
    """RiemannSD-AAM: Riemannian Speaker-Disentangled AAM-Softmax (proposed, Section 3.3).

    Uses projection-based geometry to penalise speaker information leaking
    into the source embedding.

    NOTE: The paper formulates this loss using hyperbolic (Poincare ball) distance
    with curvature c, projecting embeddings via an exponential map. The current
    implementation uses Euclidean projection energy (squared cosine coefficients).

    Args:
        in_feats: Dimension of source embedding f_src.
        n_class: Number of source classes.
        spk_feats: Dimension of speaker embedding f_spk (e.g. 192 from ECAPA).
        m: Additive angular margin (default: 0.35).
        s: Cosine scale factor (default: 30).
        lambda_val: Speaker disentanglement coefficient (default: 0.2).
    """

    def __init__(self, in_feats, n_class, spk_feats=192, m=0.35, s=30, lambda_val=0.2):
        super(RiemannSDAAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.spk_feats = spk_feats
        self.lambda_val = lambda_val

        # Dimension alignment: map speaker dim to source dim
        if in_feats != spk_feats:
            self.spk_transform = nn.Linear(spk_feats, in_feats, bias=False)
            nn.init.orthogonal_(self.spk_transform.weight)
        else:
            self.spk_transform = None

        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, in_feats), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print(f'Initialised RiemannSD-AAM: SourceDim={in_feats} SpkDim={spk_feats} m={m}')

    def forward(self, x, speaker_emb, label=None):
        """
        Args:
            x: Source embedding f_src [B, in_feats].
            speaker_emb: Speaker embedding f_spk [B, spk_feats].
            label: Source labels [B].
        """
        # Step 0: Dimension alignment
        if self.spk_transform is not None:
            speaker_emb_mapped = self.spk_transform(speaker_emb)
        else:
            speaker_emb_mapped = speaker_emb

        # Part 1: Source classification with AAM
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Part 2: Riemannian projection penalty
        spk_norm = F.normalize(speaker_emb_mapped, p=2, dim=1)
        projection_coeffs = torch.mm(x_norm, spk_norm.t())
        projection_energy = torch.pow(projection_coeffs, 2)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        aggregated_energy = torch.mm(projection_energy, one_hot)
        class_counts = one_hot.sum(dim=0).view(1, -1).clamp(min=1.0)
        H_matrix = aggregated_energy / class_counts

        # Part 3: Combine
        negative_logits = cosine + (self.lambda_val * H_matrix)
        output = (one_hot * phi) + ((1.0 - one_hot) * negative_logits)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


