import torch
import torch.nn.functional as F

class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def forward(self, logits, targets, reduction='mean'):
        # logits: (B,C) raw scores, targets: {0,1}
        x = torch.sigmoid(logits)
        if self.clip and self.clip > 0:
            x = x.clamp(self.clip, 1 - self.clip)

        xs_pos = x
        xs_neg = 1 - x

        # focal-like modulating factors (asymmetric)
        mod_pos = (1 - xs_pos).pow(self.gp)
        mod_neg = xs_pos.pow(self.gn)

        loss_pos = targets * torch.log(xs_pos.clamp_min(self.eps)) * mod_pos
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp_min(self.eps)) * mod_neg

        loss = -(loss_pos + loss_neg)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha  # shape (C,) or scalar
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets, reduction='mean'):
        p = torch.sigmoid(logits)
        # BCE with logits components
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        # convert BCE to p_t form: p_t = p for y=1, = 1-p for y=0
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha = self.alpha
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
            if alpha.ndim == 0:
                alpha = alpha
            else:
                alpha = alpha.view(1, -1)  # (1,C)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal = alpha_t * focal
        if reduction == 'mean':
            return focal.mean()
        elif reduction == 'sum':
            return focal.sum()
        return focal


def kl_align_samples_as_gauss(z1, z2, tau=1.0, reduction='mean'):
    mse = F.mse_loss(z1, z2, reduction='none').sum(dim=-1)  
    kl_sym_per_sample = (1.0 / (tau**2)) * mse       
    if reduction == 'none':
        return kl_sym_per_sample
    return kl_sym_per_sample.mean() if reduction == 'mean' else kl_sym_per_sample.sum()


def kl_latents_norm(label_latent, feat_latent, eps=1e-12):
    p = torch.relu(label_latent) + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = torch.relu(feat_latent) + eps
    q = q / q.sum(dim=-1, keepdim=True)

    logq = (q+eps).log()
    kl = (p * (p.log() - logq)).sum(dim=-1).mean()
    return kl

def kl_latents_as_logits(label_latent, feat_latent, tau=1.0):
    # tau: temperature
    p = F.softmax(label_latent / tau, dim=-1)
    log_q = F.log_softmax(feat_latent / tau, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean')
    return kl

def js_divergence(label_latent, feat_latent, tau=1.0, eps=1e-12):
    p = F.softmax(label_latent / tau, dim=-1)
    q = F.softmax(feat_latent / tau, dim=-1)
    m = 0.5 * (p + q)

    kl_pm = (p * (p.log() - (m+eps).log())).sum(-1)
    kl_qm = (q * (q.log() - (m+eps).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm).mean()