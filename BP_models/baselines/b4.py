"""B4 (in progress): leaky traces of the input at several time constants.

So far only the trace bank; the linear readout is not built yet. Also used in
the tests as a second x-driven slow memory for the training loop.

    averaging form: s(t) = λ s(t-1) + (1-λ) x(t)
    jump form:      s(t) = λ s(t-1) + x(t)
    λ = exp(-dt/τ); output = traces for every τ, concatenated (n_in * len(taus)).
    Overnight: s *= exp(-T/τ).
"""
import torch
import torch.nn as nn

from ..timing import decay_over, lam


class TraceBank(nn.Module):
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

    def reset_traces(self):  # no learned weights: same as reset
        self.s.zero_()

    @torch.no_grad()
    def advance(self, X):
        """Step through X of shape (T, 1, n_in); returns (T, 1, n_out)."""
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
