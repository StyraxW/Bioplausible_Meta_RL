"""Slow-memory front ends: modules that turn the input stream x into extra RNN input z.

State lives in buffers outside autograd and advances once per real timestep,
never once per TBPTT window recomputation. The training loop calls:

    reset()            at the start of a run
    run(X) -> Z        once per stretch of sessions without a break
    overnight(T)       at each overnight break (T in seconds)

Only feedforward modules (z depends on x alone, e.g. Model 1, B4 traces) are
supported by the loop so far: their whole Z can be computed before the RNN
pass. Feedback modules (Models 2/3: z depends on h) will need per-step stepping
inside the loop.
"""
import math

import torch
import torch.nn as nn

from .timing import decay_over, lam


class SlowMemory(nn.Module):
    feedforward = True
    n_out: int

    def reset(self):
        raise NotImplementedError

    @torch.no_grad()
    def run(self, X):
        """X: (T, 1, n_in) -> Z: (T, 1, n_out). Advances state once per step."""
        raise NotImplementedError

    @torch.no_grad()
    def overnight(self, duration):
        raise NotImplementedError


class TraceBank(SlowMemory):
    """Leaky traces of the input at several time constants.

    averaging form: s(t) = λ s(t-1) + (1-λ) x(t)
    jump form:      s(t) = λ s(t-1) + x(t)
    with λ = exp(-dt/τ). Output z = traces for every τ, concatenated (n_in * len(taus)).
    Overnight: s *= exp(-T/τ).
    """
    def __init__(self, n_in, taus, dt, form='averaging'):
        super().__init__()
        if form not in ('averaging', 'jump'):
            raise ValueError(f'form must be averaging or jump, got {form!r}')
        self.n_in, self.taus, self.dt, self.form = n_in, tuple(taus), dt, form
        self.n_out = n_in * len(self.taus)
        self.register_buffer('lams', torch.tensor([lam(t, dt) for t in self.taus]))
        self.register_buffer('s', torch.zeros(len(self.taus), n_in))

    def reset(self):
        self.s.zero_()

    @torch.no_grad()
    def run(self, X):
        lams = self.lams[:, None]
        gain = (1 - lams) if self.form == 'averaging' else torch.ones_like(lams)
        out = torch.empty(len(X), 1, self.n_out, dtype=X.dtype, device=X.device)
        s = self.s
        for t in range(len(X)):
            s = lams * s + gain * X[t, 0]
            out[t, 0] = s.reshape(-1)
        self.s.copy_(s)
        return out

    @torch.no_grad()
    def overnight(self, duration):
        self.s *= torch.tensor([decay_over(t, duration) for t in self.taus],
                               dtype=self.s.dtype, device=self.s.device)[:, None]


def log_spaced_taus(tau_min, tau_max, n):
    """n time constants log-spaced from tau_min to tau_max (seconds)."""
    if n == 1:
        return (float(tau_min),)
    r = math.log(tau_max / tau_min) / (n - 1)
    return tuple(tau_min * math.exp(r * k) for k in range(n))
