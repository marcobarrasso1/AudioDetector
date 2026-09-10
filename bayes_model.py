import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Bayes by Backprop — core idea
#
# Every weight is a Gaussian distribution N(μ, σ²) instead of a single number.
# At each forward pass we sample:  w = μ + σ * ε,   ε ~ N(0,1)
# This is the reparameterization trick — gradients flow through μ and σ.
#
# Training loss is the ELBO:
#   Loss = NLL(y | x, w)  +  β * KL( q(w|μ,σ) || p(w) )
#
# where:
#   NLL  = cross entropy  (how well the model fits the data)
#   KL   = divergence from prior N(0,1)  (regularization)
#   β    = 1/N_train  (prevents KL overwhelming likelihood)
#
# At inference: run N forward passes, each samples different weights
# → distribution over predictions → mean + uncertainty
# ---------------------------------------------------------------------------


class BayesianLinear(nn.Module):
    """Bayesian drop-in replacement for nn.Linear."""
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma  = prior_sigma
        self.sampling     = True   # False -> use posterior mean weights (mu)

        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_rho   = nn.Parameter(torch.empty(out_features))
        self._init()

    def _init(self):
        nn.init.normal_(self.weight_mu,  0.0, 0.1)
        nn.init.normal_(self.bias_mu,    0.0, 0.1)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho,   -3.0)

    @property
    def weight_sigma(self): return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self):   return F.softplus(self.bias_rho)

    def forward(self, x):
        if not self.sampling:
            return F.linear(x, self.weight_mu, self.bias_mu)
        w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu   + self.bias_sigma   * torch.randn_like(self.bias_mu)
        return F.linear(x, w, b)

    def kl_divergence(self):
        pv   = self.prior_sigma ** 2
        w_v  = self.weight_sigma ** 2
        b_v  = self.bias_sigma   ** 2
        kl_w = 0.5 * (w_v/pv + self.weight_mu**2/pv - 1 - torch.log(w_v/pv)).sum()
        kl_b = 0.5 * (b_v/pv + self.bias_mu**2/pv   - 1 - torch.log(b_v/pv)).sum()
        return kl_w + kl_b


class BayesianConv2d(nn.Module):
    """Bayesian drop-in replacement for nn.Conv2d."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, prior_sigma=1.0):
        super().__init__()
        self.stride      = stride
        self.padding     = padding
        self.prior_sigma = prior_sigma
        self.sampling    = True   # False -> use posterior mean weights (mu)

        k       = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        w_shape = (out_channels, in_channels, *k)

        self.weight_mu  = nn.Parameter(torch.empty(*w_shape))
        self.weight_rho = nn.Parameter(torch.empty(*w_shape))
        self.bias_mu    = nn.Parameter(torch.empty(out_channels))
        self.bias_rho   = nn.Parameter(torch.empty(out_channels))
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.bias_mu,    0.0, 0.1)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho,   -3.0)

    @property
    def weight_sigma(self): return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self):   return F.softplus(self.bias_rho)

    def forward(self, x):
        if not self.sampling:
            return F.conv2d(x, self.weight_mu, self.bias_mu,
                            stride=self.stride, padding=self.padding)
        w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu   + self.bias_sigma   * torch.randn_like(self.bias_mu)
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)

    def kl_divergence(self):
        pv   = self.prior_sigma ** 2
        w_v  = self.weight_sigma ** 2
        b_v  = self.bias_sigma   ** 2
        kl_w = 0.5 * (w_v/pv + self.weight_mu**2/pv - 1 - torch.log(w_v/pv)).sum()
        kl_b = 0.5 * (b_v/pv + self.bias_mu**2/pv   - 1 - torch.log(b_v/pv)).sum()
        return kl_w + kl_b


def total_kl(model: nn.Module) -> torch.Tensor:
    """Sum KL divergence from all Bayesian layers in the model."""
    kl = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d)):
            kl = kl + module.kl_divergence()
    return kl


# ---------------------------------------------------------------------------
# SE block — stays deterministic
# Reweighting channels doesn't need uncertainty and keeping it
# deterministic reduces the KL term without hurting performance
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


# ---------------------------------------------------------------------------
# Bayesian residual block
# Mirrors ResBlock from MCDC exactly, but:
#   - nn.Conv2d  → BayesianConv2d  (weights are distributions)
#   - Dropout2d removed            (uncertainty comes from sampling, not dropout)
#   - BatchNorm stays deterministic (mixing stochastic weights + BN is stable)
#   - Skip connection stays deterministic (just dimension matching)
# ---------------------------------------------------------------------------

class BayesResBlock(nn.Module):
    def __init__(self, cin, cout, stride_t=1, prior_sigma=1.0):
        super().__init__()
        self.conv1 = BayesianConv2d(cin,  cout, kernel_size=3, padding=1,
                                    stride=(1, stride_t), prior_sigma=prior_sigma)
        self.bn1   = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = BayesianConv2d(cout, cout, kernel_size=3, padding=1,
                                    prior_sigma=prior_sigma)
        self.bn2   = nn.BatchNorm2d(cout)
        self.se    = SEBlock(cout)
        self.relu  = nn.ReLU(inplace=True)

        # Skip connection — deterministic, just matches dimensions
        if cin != cout or stride_t != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=(1, stride_t), bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + self.skip(x))


# ---------------------------------------------------------------------------
# BayesNN — mirrors MCDC architecture exactly
#
# Input : [B, 1, 80, T]   — 1-channel mel spectrogram, T variable
# Output: [B, 2]           — logits for (real, fake)
#
# Architecture is identical to MCDC:
#   stem:    1  →  32
#   layer1:  32 →  64   T/2
#   layer2:  64 → 128   T/4
#   layer3: 128 → 256   T/8
#   layer4: 256 → 256   T/16
#   layer5: 256 → 256   T/16  (refine)
#   freq_pool → stats pool → FC(512→256→64→2)
#
# The only differences from MCDC:
#   1. Conv2d → BayesianConv2d (weights are N(μ,σ²) instead of point estimates)
#   2. Linear → BayesianLinear in FC head
#   3. No Dropout2d / Dropout (uncertainty comes from weight sampling)
#   4. Training uses ELBO loss instead of plain cross entropy
#   5. mc_predict uses model.eval() instead of model.train()
#
# Parameter count: ~2x MCDC because each weight stores μ AND ρ
# Effective weight count: same as MCDC (~3.8M)
# ---------------------------------------------------------------------------

class BayesNN(nn.Module):
    def __init__(self, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma

        # Deterministic stem — stable entry point before Bayesian layers
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Bayesian encoder — same channel progression as MCDC
        self.layer1 = BayesResBlock(32,  64,  stride_t=2, prior_sigma=prior_sigma)  # T/2
        self.layer2 = BayesResBlock(64,  128, stride_t=2, prior_sigma=prior_sigma)  # T/4
        self.layer3 = BayesResBlock(128, 256, stride_t=2, prior_sigma=prior_sigma)  # T/8
        self.layer4 = BayesResBlock(256, 256, stride_t=2, prior_sigma=prior_sigma)  # T/16
        self.layer5 = BayesResBlock(256, 256, stride_t=1, prior_sigma=prior_sigma)  # refine

        # Collapse frequency axis → [B, 256, T/16]
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Bayesian FC head — mirrors MCDC fc block but with BayesianLinear
        # No Dropout here — uncertainty comes from weight sampling
        self.fc = nn.Sequential(
            BayesianLinear(256 * 2, 256, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),
            BayesianLinear(256, 64, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),
            BayesianLinear(64, 2, prior_sigma=prior_sigma),
        )

    def forward(self, x):
        # x: [B, 1, 80, T]
        x  = self.stem(x)                       # [B,  32, 80, T]
        x  = self.layer1(x)                     # [B,  64, 80, T/2]
        x  = self.layer2(x)                     # [B, 128, 80, T/4]
        x  = self.layer3(x)                     # [B, 256, 80, T/8]
        x  = self.layer4(x)                     # [B, 256, 80, T/16]
        x  = self.layer5(x)                     # [B, 256, 80, T/16]
        x  = self.freq_pool(x).squeeze(2)       # [B, 256, T/16]

        # Stats pooling — same as MCDC
        mu = x.mean(dim=-1)                     # [B, 256]
        sd = x.std(dim=-1).clamp_min(1e-5)      # [B, 256]
        z  = torch.cat([mu, sd], dim=1)         # [B, 512]

        return self.fc(z)                       # [B, 2]

    def set_sampling(self, flag: bool):
        """Toggle weight sampling in every Bayesian layer.
        False = deterministic forward with the posterior means (mu) —
        the point-estimate baseline to compare against the full posterior."""
        for m in self.modules():
            if isinstance(m, (BayesianLinear, BayesianConv2d)):
                m.sampling = flag

    def elbo_loss(self, logits, targets, class_weights, n_train_samples):
        """
        ELBO loss = NLL + β * KL

        Args:
            logits          : [B, 2]  raw model output
            targets         : [B]     ground truth labels
            class_weights   : [2]     inverse frequency weights for imbalance
            n_train_samples : int     total training samples (β = 1/N)

        Returns:
            loss : scalar ELBO
            nll  : cross entropy component
            kl   : KL divergence component (before scaling)
        """
        nll  = F.cross_entropy(logits, targets, weight=class_weights)
        kl   = total_kl(self)
        beta = 1.0 / n_train_samples
        loss = nll + beta * kl
        return loss, nll, kl

    @torch.no_grad()
    def mc_predict(self, x, n_samples=50):
        """
        Bayesian inference via weight sampling.
        Each forward pass samples a different set of weights from N(μ,σ²).

        NOTE: unlike MCDC which calls model.train() to keep dropout active,
        here we call model.eval() — stochasticity comes from weight sampling
        which happens regardless of train/eval mode. BatchNorm correctly uses
        running statistics in eval mode.

        Args:
            x         : [B, 1, 80, T]
            n_samples : number of weight samples (30-50 is sufficient)

        Returns:
            mean_prob : [B, 2]  mean softmax — use for EER / decisions
            variance  : [B, 2]  predictive variance
            bald      : [B]     BALD epistemic uncertainty score
        """
        self.eval()             # BN uses running stats
        self.set_sampling(True) # make sure weight sampling is on

        preds = torch.stack([
            F.softmax(self(x), dim=-1) for _ in range(n_samples)
        ])              # [N, B, 2]

        mean_prob = preds.mean(dim=0)
        variance  = preds.var(dim=0)

        eps    = 1e-8
        H_mean = -(mean_prob * (mean_prob + eps).log()).sum(-1)
        mean_H = -(preds * (preds + eps).log()).sum(-1).mean(0)
        bald   = H_mean - mean_H

        return mean_prob, variance, bald


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BayesNN(prior_sigma=1.0).to(device)

    # total stored params (μ + ρ pairs)
    n_stored = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # effective weight count (just μ)
    n_weights = sum(
        m.weight_mu.numel() + m.bias_mu.numel()
        for m in model.modules()
        if isinstance(m, (BayesianLinear, BayesianConv2d))
    )
    print(f"Stored parameters (μ+ρ) : {n_stored/1e6:.2f}M")
    print(f"Effective weights  (μ)  : {n_weights/1e6:.2f}M  ← compare with MCDC ~3.8M")

    # training pass
    x = torch.randn(8, 1, 80, 800).to(device)
    y = torch.randint(0, 2, (8,)).to(device)
    w = torch.tensor([8.84, 1.0]).to(device)

    model.train()
    logits        = model(x)
    loss, nll, kl = model.elbo_loss(logits, y, w, n_train_samples=25380)
    print(f"\nTraining:")
    print(f"  logits : {logits.shape}")
    print(f"  ELBO   : {loss.item():.4f}  (NLL={nll.item():.4f}  KL={kl.item():.1f})")

    # mc inference
    mean_prob, variance, bald = model.mc_predict(x, n_samples=30)
    print(f"\nMC inference (30 samples):")
    print(f"  mean_prob : {mean_prob.shape}")
    print(f"  bald mean : {bald.mean().item():.4f}")