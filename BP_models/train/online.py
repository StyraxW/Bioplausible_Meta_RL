"""Online TD training with sliding-window truncated BPTT over a session stream.

Window semantics are those of valuernn's train_model_TBPTT (train_bptt.py),
reproduced step for step (tests/test_online.py checks equality):

- a window of W steps ends at every `stride`-th step; each window runs the
  core over its W inputs from the detached state h0, computes the mean squared
  TD error over its W-1 transitions (semi-gradient: target V(t+1) detached),
  and takes one optimizer step;
- h0 for the next window is the state `stride` steps into the current window,
  computed with the pre-update weights.

Additions:
- Sessions without a break between them form one continuous segment; windows
  slide straight across those session boundaries. A Break ends the segment:
  the next segment's windows start fresh (window cut).
- Break: no learning; slow memory decays analytically (memory.overnight);
  h follows the break's h_policy ('reset' noise from a dedicated generator,
  'relax' silent steps, 'carry').
- Optional feedforward slow memory: Z = memory.run(X) is computed once per
  segment, before the RNN pass, so slow state advances once per real step.
  The core's input is [x, z].
- Recording is online: V(t) and h(t) are taken from the first window in which
  t is the newest step (before any update that used data after t); delta(t)
  = r(t+1) + γV(t+1) - V(t) from the window that first contains t+1. This
  differs from valuernn's data_saver, which keeps the whole first window and
  then the last `stride` steps of V_hat.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from ..tasks.sessions import Break, Session

RELAX_CHUNK = 16384  # cuDNN rejects a single GRU sequence somewhere in 32768-65536 steps
RELAX_PROBE_STEP = 300  # report drift of h after this step during 'relax'


class OnlineTrainer:
    def __init__(self, model, memory=None, window_size=50, stride=1, lr=0.003,
                 dt=0.5, device='cpu', break_seed=0, optimizer=None,
                 record_hidden=False, log=print):
        if window_size < 2:
            raise ValueError(f'{window_size=} must be >= 2')
        if not 1 <= stride < window_size:
            raise ValueError(f'{stride=} must satisfy 1 <= stride < window_size')
        if memory is not None and not memory.feedforward:
            raise NotImplementedError('feedback slow memory (Models 2/3) needs per-step stepping; '
                                      'not implemented yet')
        if model.recurrent_cell.lower() not in ('gru', 'rnn'):
            raise ValueError('core must be GRU or RNN (LSTM state is a tuple)')
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.memory = memory.to(self.device) if memory is not None else None
        self.W, self.stride, self.dt = window_size, stride, dt
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.break_gen = torch.Generator(device=self.device).manual_seed(int(break_seed))
        self.record_hidden = record_hidden
        self.log = log

    # ------------------------------------------------------------------ run
    def run(self, schedule):
        """Train on a SessionSchedule. Returns a dict of numpy arrays and logs."""
        self.model.train()
        if self.memory is not None:
            self.memory.reset()
        h = None  # zeros at the start of the run, as in valuernn
        rec = dict(V=[], delta=[], hidden=[], window_loss=[], window_session=[])
        sessions, offsets, breaks, segment = [], [], [], []
        t0, n_steps = time.perf_counter(), 0

        def flush(h):
            nonlocal n_steps
            if not segment:
                return h
            h = self._train_segment(segment, h, rec)
            for s in segment:
                sessions.append(s)
                offsets.append(n_steps)
                n_steps += len(s.X)
            segment.clear()
            return h

        for event in schedule:
            if isinstance(event, Session):
                segment.append(event)
            elif isinstance(event, Break):
                h = flush(h)
                h, info = self._apply_break(event, h)
                breaks.append(info)
            else:
                raise TypeError(f'unexpected schedule event {event!r}')
        h = flush(h)

        out = dict(
            V=np.concatenate(rec['V']),
            delta=np.concatenate(rec['delta']),
            step_session=np.concatenate([np.full(len(s.X), s.index) for s in sessions]),
            window_loss=np.asarray(rec['window_loss']),
            window_session=np.asarray(rec['window_session']),
            session_offsets=np.asarray(offsets),
            breaks=breaks,
            sessions=sessions,
            runtime_s=time.perf_counter() - t0,
        )
        if self.record_hidden:
            out['hidden'] = np.concatenate(rec['hidden'])
        return out

    # -------------------------------------------------------------- segment
    def _train_segment(self, segment, h0, rec):
        X = torch.cat([s.X for s in segment]).to(self.device)
        y = torch.cat([s.y for s in segment]).to(self.device)
        step_session = torch.cat([torch.full((len(s.X),), s.index) for s in segment])
        if self.memory is not None:
            Z = self.memory.run(X)  # once per real step; z enters the core detached
            X = torch.cat([X, Z], dim=-1)
        T, W, s = len(X), self.W, self.stride
        gamma = self.model.gamma

        V_rec = np.full(T, np.nan)
        d_rec = np.full(T, np.nan)
        h_rec = np.full((T, self.model.hidden_size), np.nan) if self.record_hidden else None

        ends = list(range(min(W, T), T + 1, s))
        self.optimizer.zero_grad()
        h_last, end_last = h0, 0
        for c, i in enumerate(ends):
            start = max(0, i - W)
            Xw, yw = X[start:i], y[start:i]
            V, hs = self.model(Xw, h0=h0, return_hiddens=True)
            V_hat = V[:-1]
            V_target = yw[1:] + gamma * V[1:].detach()
            h0 = hs[s - 1].detach().unsqueeze(1)
            loss = self.loss_fn(V_hat, V_target) / len(V_hat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # online records: newest steps of this window (all steps for the first window)
            new = (i - start) if c == 0 else s
            V_np = V.detach()[:, 0, 0].cpu().numpy()
            V_rec[i - new:i] = V_np[-new:]
            d_np = (V_target - V_hat).detach()[:, 0, 0].cpu().numpy()
            n_d = min(new, len(d_np))
            d_rec[i - 1 - n_d:i - 1] = d_np[-n_d:]
            if h_rec is not None:
                h_rec[i - new:i] = hs.detach()[-new:, 0].cpu().numpy()
            rec['window_loss'].append(loss.item())
            rec['window_session'].append(int(step_session[i - 1]))
            h_last, end_last = hs[-1].detach().view(1, 1, -1), i

        if end_last < T:  # tail not covered by a window (stride > 1): run it without learning
            with torch.no_grad():
                V, hs = self.model(X[end_last:], h0=h_last, return_hiddens=True)
            V_rec[end_last:] = V[:, 0, 0].cpu().numpy()
            if h_rec is not None:
                h_rec[end_last:] = hs[:, 0].cpu().numpy()
            h_last = hs[-1].view(1, 1, -1)

        rec['V'].append(V_rec)
        rec['delta'].append(d_rec)
        if h_rec is not None:
            rec['hidden'].append(h_rec)
        if self.log is not None:
            losses = np.asarray(rec['window_loss'][-len(ends):])
            win_sess = np.asarray(rec['window_session'][-len(ends):])
            for sg in segment:
                m = win_sess == sg.index
                if m.any():
                    self.log(f'session {sg.index}: {m.sum()} windows, mean loss {losses[m].mean():.4f}')
        return h_last

    # ---------------------------------------------------------------- break
    @torch.no_grad()
    def _apply_break(self, brk, h):
        spec = brk.spec
        if spec.replay:
            raise NotImplementedError('replay during breaks is not implemented yet')
        if self.memory is not None:
            self.memory.overnight(spec.duration)
        if h is None:
            h = torch.zeros(1, 1, self.model.hidden_size, device=self.device)
        info = dict(after_session=brk.after_session, h_policy=spec.h_policy,
                    duration=spec.duration, h_norm_before=h.norm().item())
        if spec.h_policy == 'carry':
            h_new = h
        elif spec.h_policy == 'reset':
            h_new = spec.reset_state(h, self.break_gen)
            info['reset_sigma'] = spec.reset_sigma
        elif spec.h_policy == 'relax':
            h_new, relax_info = self._relax(h, spec.n_relax_steps(self.dt))
            info.update(relax_info)
        else:
            raise ValueError(spec.h_policy)
        info['h_norm_after'] = h_new.norm().item()
        if self.log is not None:
            extra = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()
                     if k not in ('after_session', 'duration')}
            self.log(f'break after session {brk.after_session}: {extra}')
        return h_new, info

    @torch.no_grad()
    def _relax(self, h, n_steps):
        """Run the core on silent input (x = 0 and z = 0) for n_steps, no learning.

        z = 0 because slow traces decay by exp(-T/τ) ≈ 0 over hours for τ up to minutes.
        """
        info = dict(relax_steps=n_steps)
        if n_steps == 0:
            return h, info
        # model.input_size = n_x + n_z, so this silences both x and z
        zeros = torch.zeros(min(n_steps, RELAX_CHUNK), 1, self.model.input_size, device=self.device)
        done, h_probe, prev_last = 0, None, h
        while done < n_steps:
            n = min(RELAX_CHUNK, n_steps - done)
            out, _ = self.model.rnn(zeros[:n], h)
            if h_probe is None and done + n >= RELAX_PROBE_STEP:
                h_probe = out[RELAX_PROBE_STEP - done - 1]
            prev_last = out[-2] if n >= 2 else h[0]
            h = out[-1:].contiguous()
            done += n
        info['final_step_change'] = (h[0] - prev_last).norm().item()
        info['drift_after_step_300'] = ((h[0] - h_probe).norm().item()
                                        if h_probe is not None else float('nan'))
        return h, info
