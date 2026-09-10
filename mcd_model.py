import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns to reweight each frequency channel — cheap but effective
    on mel spectrograms where different frequency bands carry
    different amounts of spoofing information.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                        # [B, C, 1, 1]
            nn.Flatten(),                                   # [B, C]
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class ResBlock(nn.Module):
    """
    Residual block with SE and Dropout2d.

    Dropout2d drops entire feature maps (channels) rather than individual
    pixels — this is the right choice for conv layers in MC Dropout because:
      1. It creates more meaningful diversity between forward passes
      2. Avoids the spatial correlation issue of per-pixel dropout on feature maps

    stride_t controls downsampling along the time axis only.
    """
    def __init__(self, cin, cout, stride_t=1, p_drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, stride=(1, stride_t)),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),           # MC Dropout — spatial
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
        )
        self.se   = SEBlock(cout)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection — matches dimensions when cin != cout or stride > 1
        if cin != cout or stride_t != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=(1, stride_t), bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.skip(x))


class MCDC(nn.Module):
    """
    Monte Carlo Dropout CNN countermeasure (MCDC).

    Input : [B, 1, 80, T]   — 1-channel mel spectrogram, T variable
    Output: [B, 2]           — logits for (real, fake)

    Training  : call model.train() and use forward() normally
    Inference : call model.mc_predict() to get mean prob + uncertainty
    """
    def __init__(self, p_drop=0.3):
        super().__init__()

        # Stem — initial projection, no downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Encoder — 4 stages, each halving the time dimension
        # Frequency axis stays at 80 throughout
        #
        #   Input:   [B,  1, 80,  T]
        #   layer1:  [B, 64, 80, T/2]
        #   layer2:  [B,128, 80, T/4]
        #   layer3:  [B,256, 80, T/8]
        #   layer4:  [B,256, 80,T/16]   <- refinement, no stride
        self.layer1 = ResBlock(32,  64,  stride_t=2, p_drop=p_drop)
        self.layer2 = ResBlock(64,  128, stride_t=2, p_drop=p_drop)
        self.layer3 = ResBlock(128, 256, stride_t=2, p_drop=p_drop)
        self.layer4 = ResBlock(256, 256, stride_t=2, p_drop=p_drop)
        self.layer5 = ResBlock(256, 256, stride_t=1, p_drop=p_drop)   # refine

        # Collapse frequency axis → [B, 256, T/16]
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Stats pooling over time → fixed [B, 512] regardless of T
        # Concatenating mean and std gives the classifier information about
        # both the average spectral pattern and its variability over time
        # (same principle as x-vectors in speaker recognition)

        # Classifier head — two FC layers with MC Dropout
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),             # MC Dropout
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),             # MC Dropout
            nn.Linear(64, 2),
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

        # Stats pooling
        mu = x.mean(dim=-1)                     # [B, 256]
        sd = x.std(dim=-1).clamp_min(1e-5)      # [B, 256]
        z  = torch.cat([mu, sd], dim=1)         # [B, 512]

        return self.fc(z)                       # [B, 2]

    def enable_mc_dropout(self):
        """
        eval() everything (BatchNorm uses running stats — predictions no longer
        depend on batch composition), then switch ONLY the dropout layers back
        to train mode so they keep sampling masks.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    @torch.no_grad()
    def mc_predict(self, x, n_samples=50):
        """
        Monte Carlo Dropout inference.
        Keeps dropout active and runs n_samples stochastic forward passes
        to approximate the posterior predictive distribution.

        Args:
            x         : [B, 1, 80, T]
            n_samples : number of stochastic forward passes (30-50 is enough)

        Returns:
            mean_prob : [B, 2]  mean softmax — use this for EER / decisions
            variance  : [B, 2]  predictive variance — simple uncertainty measure
            bald      : [B]     BALD score — epistemic uncertainty (best for report)
        """
        self.enable_mc_dropout()    # dropout ON, BatchNorm frozen

        preds = torch.stack([
            F.softmax(self(x), dim=-1) for _ in range(n_samples)
        ])              # [N, B, 2]

        mean_prob = preds.mean(dim=0)       # [B, 2]
        variance  = preds.var(dim=0)        # [B, 2]

        # BALD = H[E[p]] - E[H[p]]
        # Measures how much the model disagrees with itself across dropout masks.
        # High BALD = the model is genuinely uncertain (epistemic uncertainty),
        # not just predicting a borderline probability (aleatoric uncertainty).
        eps    = 1e-8
        H_mean = -(mean_prob * (mean_prob + eps).log()).sum(-1)     # [B]
        mean_H = -(preds * (preds + eps).log()).sum(-1).mean(0)     # [B]
        bald   = H_mean - mean_H                                    # [B]

        return mean_prob, variance, bald
