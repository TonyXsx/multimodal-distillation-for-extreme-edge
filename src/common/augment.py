"""specaugment."""

import torch


def spec_augment(x, n_freq=2, n_time=2, f_max=12, t_max=40):
    """time/freq masking, training only. x is [B,1,T,F] normalised log-mel."""
    B, _, T, F_ = x.shape
    x = x.clone()
    for _ in range(n_freq):
        f = int(torch.randint(0, f_max + 1, (1,)))
        if f > 0:
            f0 = int(torch.randint(0, max(1, F_ - f), (1,)))
            x[:, :, :, f0:f0 + f] = 0.0
    for _ in range(n_time):
        t = int(torch.randint(0, t_max + 1, (1,)))
        if t > 0:
            t0 = int(torch.randint(0, max(1, T - t), (1,)))
            x[:, :, t0:t0 + t, :] = 0.0
    return x
