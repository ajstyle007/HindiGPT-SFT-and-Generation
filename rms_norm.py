import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        RMS_norm = x / (rms + self.eps) * self.weight
        return RMS_norm
    


# ✅ LayerNorm vs RMSNorm — Why RMSNorm is preferred in LLMs
# 1. LayerNorm formula

# LN normalize both mean and variance:

# LN(𝑥) = ((𝑥−𝜇) / (𝜎^2 + 𝜖)^0.5) * 𝛾 + 𝛽

# Mean subtract hoti hai → zero-centered output
# Variance normalize hoti hai
# Bias vector (beta) bhi learn hota hai

# 2. RMSNorm formula
# RMSNorm mean subtract nahi karta:

# RMSNorm(𝑥) = (𝑥 / (((1/𝑑) * ∑𝑥^2) + 𝜖)) * 𝛾

# No mean-centering
# Sirf RMS scale normalize hota hai (variance part)