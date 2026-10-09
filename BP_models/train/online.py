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
- Break: no learning; slow memory decays analytically (slowmem.overnight);
  h follows the break's h_policy ('reset' noise from a dedicated generator,
  'relax' silent steps, 'carry').
- Optional slow memory, implemented separately per model. Supported so far:
  x-driven ones whose z depends on the input alone (Model 1's SlowMem1, B4's
  TraceBank). Their Z = slowmem.advance(X) is computed session by session
  before the RNN pass, so slow state advances once per real step; the core's
  input is [x, z]. Models 2/3 (z depends on h) will need their own branch
  stepping inside the window pass.
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

from ..baselines.b4 import TraceBank
from ..slowmem_1 import SlowMem1
from ..tasks.sessions import Break, Session

X_DRIVEN_SLOWMEMS = (SlowMem1, TraceBank)  # z from x alone: precomputed per session
RELAX_CHUNK = 16384  # cuDNN rejects a single GRU sequence somewhere in 32768-65536 steps
RELAX_PROBE_STEP = 300  # report drift of h after this step during 'relax'


class OnlineTrainer:
    def __init__(self, model, slowmem=None, window_size=50, stride=1, lr=0.003,
                 dt=0.5, device='cpu', break_seed=0, optimizer=None,
                 record_hidden=False, log=print):
        if window_size < 2:
            raise ValueError(f'{window_size=} must be >= 2')
        if not 1 <= stride < window_size:
            raise ValueError(f'{stride=} must satisfy 1 <= stride < window_size')
        if slowmem is not None and not isinstance(slowmem, X_DRIVEN_SLOWMEMS):
            raise NotImplementedError(f'{type(slowmem).__name__}: only x-driven slow memories '
                                      '(SlowMem1, TraceBank) are supported so far')
        if model.recurrent_cell.lower() not in ('gru', 'rnn'):
            raise ValueError('core must be GRU or RNN (LSTM state is a tuple)')
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.slowmem = slowmem.to(self.device) if slowmem is not None else None
        self.W, self.stride, self.dt = window_size, stride, dt
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.break_gen = torch.Generator(device=self.device).manual_seed(int(break_seed))
        self.record_hidden = record_hidden
        self.log = log

    # ------------------------------------------------------------------ run
    def run(self, schedule, h0=None, reset_slowmem=True):
        """Train on a SessionSchedule. Returns a dict of numpy arrays and logs.

        h0: starting state (None = zeros, as in valuernn).
        reset_slowmem: False continues the slow memory as it is (traces and
        weights), e.g. to continue the stream after training.
        The final state is left in self.final_h.
        """
        self.model.train()
        if self.slowmem is not None and reset_slowmem:
            self.slowmem.reset()
        h = h0
        rec = dict(V=[], delta=[], hidden=[], window_loss=[], window_session=[], Wb=[])
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
        self.final_h = h

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
        if rec['Wb']:
            out['Wb'] = np.stack(rec['Wb'])  # (n_sessions, n_y, n_y): Wb at the end of each session
        return out

    def run_continuous(self, sessions, stop_fn=None):
        """Train on `sessions` back to back as one uninterrupted stream (no breaks),
        one session at a time, so training can stop early. Exactly equivalent to
        run() on the same sessions with overnight=None (tests/test_online.py).

        After each session, stop_fn(sessions_done, offsets, delta) -> True stops.
        Returns the same dict as run(), plus 'stopped_early'.
        """
        self.model.train()
        if self.slowmem is not None:
            self.slowmem.reset()
        rec = dict(window_loss=[], window_session=[], Wb=[])
        V, D, H, done, offsets = [], [], [], [], []
        t0, n, tail, h_last, stopped = time.perf_counter(), 0, None, None, False
        pending = []  # sessions held back until the stream reaches W steps (see below)

        def train_pending():
            nonlocal tail, h_last, n
            X = torch.cat([p[1] for p in pending])
            y = torch.cat([p[2] for p in pending])
            sess_idx = torch.cat([torch.full((len(p[1]),), p[0].index) for p in pending])
            if tail is None:
                X_all, y_all, h0, n_ctx = X, y, None, 0
            else:  # continue the stream: windows reach back into the previous steps
                X_all, y_all = torch.cat([tail['X'], X]), torch.cat([tail['y'], y])
                sess_idx = torch.cat([torch.full((n_ctx_prev := len(tail['X']),), -1), sess_idx])
                h0, n_ctx = tail['h0'], n_ctx_prev
            chunk = dict(V=[], delta=[], hidden=[], window_loss=rec['window_loss'],
                         window_session=rec['window_session'])
            h_last, n_win = self.train_on_inputs(X_all, y_all, sess_idx, h0, chunk)
            v, d = chunk['V'][0], chunk['delta'][0]
            if n_ctx and D:
                D[-1][-1] = d[n_ctx - 1]  # the previous chunk's last transition completes here
            V.append(v[n_ctx:])
            D.append(d[n_ctx:].copy())
            if self.record_hidden:
                H.append(chunk['hidden'][0][n_ctx:])
            tail = self.tail
            for sess, Xs, _ in pending:
                done.append(sess)
                offsets.append(n)
                n += len(Xs)
                if self.log is not None:
                    self.log(f'session {sess.index}: trained ({n_win} windows in this chunk)')
            pending.clear()

        for sess in sessions:
            X, y = sess.X.to(self.device), sess.y.to(self.device)
            if self.slowmem is not None:
                X = torch.cat([X, self.slowmem.advance(X)], dim=-1)  # once per real step
                if isinstance(self.slowmem, SlowMem1):
                    rec['Wb'].append(self.slowmem.Wb.cpu().numpy().copy())
            pending.append((sess, X, y))
            # An uninterrupted run places its first window at step W; train nothing
            # until the stream has W steps, so no extra early windows appear.
            if tail is None and sum(len(p[1]) for p in pending) < self.W:
                continue
            train_pending()
            if stop_fn is not None and stop_fn(done, offsets, np.concatenate(D)):
                stopped = True
                break
        if pending:  # stream ended before reaching W steps: one window over all of it
            train_pending()
        self.final_h = h_last
        out = dict(V=np.concatenate(V), delta=np.concatenate(D),
                   step_session=np.concatenate([np.full(len(s.X), s.index) for s in done]),
                   window_loss=np.asarray(rec['window_loss']),
                   window_session=np.asarray(rec['window_session']),
                   session_offsets=np.asarray(offsets), breaks=[], sessions=done,
                   runtime_s=time.perf_counter() - t0, stopped_early=stopped)
        if self.record_hidden:
            out['hidden'] = np.concatenate(H)
        if rec['Wb']:
            out['Wb'] = np.stack(rec['Wb'])
        return out

    # -------------------------------------------------------------- segment
    def _train_segment(self, segment, h0, rec):
        X = torch.cat([s.X for s in segment]).to(self.device)
        y = torch.cat([s.y for s in segment]).to(self.device)
        step_session = torch.cat([torch.full((len(s.X),), s.index) for s in segment])
        if self.slowmem is not None:
            # x-driven: advance once per real step, session by session; z enters the core detached
            Z = []
            for sess in segment:
                Z.append(self.slowmem.advance(sess.X.to(self.device)))
                if isinstance(self.slowmem, SlowMem1):
                    rec['Wb'].append(self.slowmem.Wb.cpu().numpy().copy())
            X = torch.cat([X, torch.cat(Z)], dim=-1)
        h_last, n_windows = self.train_on_inputs(X, y, step_session, h0, rec)
        if self.log is not None:
            losses = np.asarray(rec['window_loss'][-n_windows:])
            win_sess = np.asarray(rec['window_session'][-n_windows:])
            for sg in segment:
                m = win_sess == sg.index
                if m.any():
                    self.log(f'session {sg.index}: {m.sum()} windows, mean loss {losses[m].mean():.4f}')
        return h_last

    def train_on_inputs(self, X, y, step_session, h0, rec):
        """Sliding-window TBPTT over core inputs X (already [x, z]) and rewards y.

        Appends per-step V / delta (and h if record_hidden) to rec. Keeps
        self.tail = the steps the next window would reach back into (the last
        W - stride steps of X and y) plus that window's starting state, so the
        stream can be continued later exactly as if uninterrupted
        (continue-protocol test with learning on).
        Returns (final state, number of windows).
        """
        T, W, s = len(X), self.W, self.stride
        gamma = self.model.gamma
        h_start = h0

        V_rec = np.full(T, np.nan)
        d_rec = np.full(T, np.nan)
        h_all = torch.full((T, self.model.hidden_size), float('nan'), device=self.device)

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
            h_all[i - new:i] = hs.detach()[-new:, 0]
            rec['window_loss'].append(loss.item())
            rec['window_session'].append(int(step_session[i - 1]))
            h_last, end_last = hs[-1].detach().view(1, 1, -1), i

        if end_last < T:  # tail not covered by a window (stride > 1): run it without learning
            with torch.no_grad():
                V, hs = self.model(X[end_last:], h0=h_last, return_hiddens=True)
            V_rec[end_last:] = V[:, 0, 0].cpu().numpy()
            h_all[end_last:] = hs[:, 0]
            h_last = hs[-1].view(1, 1, -1)

        rec['V'].append(V_rec)
        rec['delta'].append(d_rec)
        if self.record_hidden:
            rec['hidden'].append(h_all.cpu().numpy())
        # The next window of an uninterrupted stream would end at end_last + s and start
        # at next_start, from the state the last window computed for that point (h0).
        next_start = max(0, end_last + s - W)
        self.tail = dict(X=X[next_start:], y=y[next_start:],
                         h0=h_start if next_start == 0 else h0)
        return h_last, len(ends)

    # ---------------------------------------------------------------- break
    @torch.no_grad()
    def _apply_break(self, brk, h):
        spec = brk.spec
        if spec.replay:
            raise NotImplementedError('replay during breaks is not implemented yet')
        if self.slowmem is not None:
            self.slowmem.overnight(spec.duration)
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
