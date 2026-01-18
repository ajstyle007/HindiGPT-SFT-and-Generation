import torch

def rotate_half(x):
    # Split last dimension into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Rotate 2D components: (x1, x2) -> (-x2, x1)
    return torch.cat([-x2, x1], dim=-1)


# sinusoidal freq (ROPE angles) precompute
# Har position ke liye ROPE frequencies banti hain:

def build_rope_frequencies(seq_len, head_dim, base=10000):
    """
    Returns rope frequencies shape: (seq_len, head_dim/2)
    """

    half_dim = head_dim // 2
    freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

    #get positions [0..seq_len]
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

    # 0(theta) = positions * freq 
    angles = positions * freq   # shape (seq_len, half_dim)

    # cos , sine for rotation
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """
    x: (batch, heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim/2)
    """

    cos = cos.to(x.device).float()
    sin = sin.to(x.device).float()

    # reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., : x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]

    # Apply rotation
    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x1 * sin + x2 * cos

    return torch.cat([rotated_1, rotated_2], dim=-1)

