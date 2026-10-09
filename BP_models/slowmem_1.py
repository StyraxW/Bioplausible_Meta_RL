"""Model 1 slow memory: trial-type layer -> Hebbian binding layer -> z.

Implements docs/model_schematics.md (revised spec), layers 1-3:

    e(t)   = λe e(t-1) + (1-λe) [cA(t), cB(t)]                 cue trace, averaging form
    off(t) = max(0, c(t-1) - c(t)),  c = cA + cB              cue-offset gate (= outcome step)
    y(t)   = φ(R [eA, eB, r]ᵀ + b) · off(t)                   trial-type units, φ = ReLU
    ΔWij   = ηb ( yi(t) ȳj(t-1) - yi(t)² Wij ),  Wii = 0       Hebbian + Oja row decay
    ȳ(t)   = λb ȳ(t-1) + y(t)                                  event trace, jump form
    z(t)   = Wb ȳ(t)

Order within a step: e, off, y; then the Wb update (uses ȳ(t-1)); then ȳ(t);
then z(t) with the updated Wb. y is nonzero only on outcome steps, so Wb changes
only there. At the fixed point Wij = E[yi ȳj] / E[yi²]; for one-hot y this is
E[ȳj | unit i fires]: row i = the typical recent history before trial type i.

Everything here is outside autograd (buffers, no_grad). z enters the core RNN
detached, so TD gradients train Wz but never Wb. Nothing depends on h or δ, so
z for a whole stretch of sessions can be computed before the RNN pass.

trial_types:
  'random' : R (Ny x 3) and b fixed random, seeded (the spec)
  'onehot' : four units A+, A-, B+, B- (sanity-check variant); index 2*cue + (0 if rewarded else 1)
"""
import torch
import torch.nn as nn

from .timing import decay_over, lam

ONEHOT_NAMES = ('A+', 'A-', 'B+', 'B-')


class SlowMem1(nn.Module):
    def __init__(self, dt, tau_e=1.5, tau_b=15.0, eta_b=0.05, trial_types='random',
                 n_y=50, r_scale=1.0, b_scale=1.0, seed=0):
        super().__init__()
        if trial_types not in ('random', 'onehot'):
            raise ValueError(f"trial_types must be 'random' or 'onehot', got {trial_types!r}")
        self.dt, self.tau_e, self.tau_b, self.eta_b = dt, tau_e, tau_b, eta_b
        self.trial_types = trial_types
        self.n_y = 4 if trial_types == 'onehot' else n_y
        self.n_out = self.n_y
        self.lam_e, self.lam_b = lam(tau_e, dt), lam(tau_b, dt)

        gen = torch.Generator().manual_seed(int(seed))
        self.register_buffer('R', r_scale * torch.randn(self.n_y, 3, generator=gen))
        self.register_buffer('b', b_scale * torch.randn(self.n_y, generator=gen))
        self.register_buffer('e', torch.zeros(2))
        self.register_buffer('c_prev', torch.zeros(()))
        self.register_buffer('ybar', torch.zeros(self.n_y))
        self.register_buffer('Wb', torch.zeros(self.n_y, self.n_y))
        self.learn_wb = True  # False freezes Wb (frozen-weight test); traces still evolve

    def reset(self):
        """Clear traces and Wb (start of a run)."""
        self.reset_traces()
        self.Wb.zero_()

    def reset_traces(self):
        """Clear activity (e, ȳ, c_prev) but keep the learned Wb."""
        for buf in (self.e, self.c_prev, self.ybar):
            buf.zero_()

    @torch.no_grad()
    def trial_type(self, e, r, off):
        """y(t) from the cue trace e = [eA, eB], reward r and gate off."""
        if self.trial_types == 'onehot':
            y = torch.zeros(4, device=e.device, dtype=e.dtype)
            if off > 0:
                cue = int(torch.argmax(e))
                y[2 * cue + (0 if r > 0 else 1)] = off
            return y
        return torch.relu(self.R @ torch.stack([e[0], e[1], r]) + self.b) * off

    @torch.no_grad()
    def step(self, x):
        """Advance one timestep. x = [cA, cB, r]. Returns z(t) and the step's internals."""
        c_vec, r = x[:2], x[2]
        self.e.mul_(self.lam_e).add_((1 - self.lam_e) * c_vec)
        c = c_vec.sum()
        off = torch.clamp(self.c_prev - c, min=0.0)
        self.c_prev.copy_(c)
        y = self.trial_type(self.e, r, off)
        if off > 0 and self.learn_wb:  # y is zero otherwise, so Wb would not change
            self.Wb.add_(self.eta_b * (torch.outer(y, self.ybar) - (y ** 2)[:, None] * self.Wb))
            self.Wb.fill_diagonal_(0.0)
        self.ybar.mul_(self.lam_b).add_(y)
        return self.Wb @ self.ybar, y, off

    @torch.no_grad()
    def advance(self, X, record=False):
        """Step through X of shape (T, 1, 3) in order; returns Z of shape (T, 1, n_y).

        record=True also returns per-step e, off, y and ȳ (numpy) for analysis.
        """
        Z = torch.empty(len(X), 1, self.n_y, dtype=X.dtype, device=X.device)
        rec = {k: [] for k in ('e', 'off', 'y', 'ybar')} if record else None
        for t in range(len(X)):
            z, y, off = self.step(X[t, 0])
            Z[t, 0] = z
            if record:
                rec['e'].append(self.e.clone())
                rec['off'].append(off.clone())
                rec['y'].append(y)
                rec['ybar'].append(self.ybar.clone())
        if record:
            return Z, {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
        return Z

    @torch.no_grad()
    def overnight(self, duration):
        """Traces decay by exp(-T/τ); Wb persists; no cue is on across the break."""
        self.e.mul_(decay_over(self.tau_e, duration))
        self.ybar.mul_(decay_over(self.tau_b, duration))
        self.c_prev.zero_()
