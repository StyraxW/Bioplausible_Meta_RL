"""Plasticity manipulation test after training (Lee et al. Methods; Extended Data
window-size figure, panels f-g). Two protocols, chosen by `protocol`:

'continue' (the paper's, protocol details from the authors):
    The stream continues, nothing reset, into ONE more block, the next one,
    whose contingency is the reverse of the last training block. The RNN state h
    and, for Model 1, the traces (e, ȳ, c_prev) carry over from the end of
    training. A network that cannot reverse without plasticity keeps the old
    contingency -> AUROC ≈ 0.

'reset' (ours):
    h reset to 0 and slow-memory traces reset (learned slow weights kept); a new
    episode of several alternating blocks (default 4). Tests inference from
    scratch through several reversals. A network stuck on one contingency
    scores ≈ 0.5 (right in half the blocks, wrong in the other half).

Conditions in both:
- frozen:  all weights fixed: core RNN and, for Model 1, Wb. 'continue': the
           stream goes on with learning rate 0 (windows keep recomputing from
           their start states, as in the paper). 'reset': one forward pass.
- plastic: learning on (same window, optimizer state carried over). 'continue':
           the first windows reach back into the last W-1 training steps, as in
           the uninterrupted stream (needed when the block is shorter than W).
           'reset': windows start at the test boundary.
- frozen_rnn_only (Model 1 only): core frozen, Wb keeps learning.

AUROC: how well cue-onset values separate CS+ trials (cue rewarded in the
current block) from CS- trials, over all test trials (and per block).
"""
import copy

import numpy as np
import torch

from ..slowmem_1 import SlowMem1
from ..tasks.sessions import SessionSchedule
from ..train.online import OnlineTrainer
from .metrics import trial_table

PROTOCOLS = ('continue', 'reset')


def auroc(pos, neg):
    """P(value of a CS+ trial > value of a CS- trial), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _copy_slowmem(slowmem, device, reset_traces):
    if slowmem is None:
        return None
    sm = copy.deepcopy(slowmem).to(device)
    if reset_traces:
        sm.reset_traces()
    return sm


@torch.no_grad()
def frozen_values(model, slowmem, session, h0, device, freeze_slow_weights=True,
                  reset_traces=False):
    """Per-step values on `session` with weights fixed, starting from state h0
    (None = zeros) and the slow memory's state (traces reset if reset_traces).
    Works on copies; the originals are untouched."""
    model = copy.deepcopy(model).to(device).eval()
    X = session.X.to(device)
    sm = _copy_slowmem(slowmem, device, reset_traces)
    if sm is not None:
        if isinstance(sm, SlowMem1):
            sm.learn_wb = not freeze_slow_weights
        X = torch.cat([X, sm.advance(X)], dim=-1)
    V, _ = model(X, h0=None if h0 is None else h0.to(device))
    return V[:, 0, 0].cpu().numpy()


def plastic_values(model, slowmem, optimizer, test_task, h0, window_size, stride, lr, dt, device,
                   reset_traces=False, context=None, freeze_slow_weights=False):
    """Per-step values on the test episode with learning on (copies; originals untouched).

    context: the trainer's `tail` (last W-1 training inputs incl. z, rewards, and
    the state before them). Given, the test continues the training stream: its
    first windows reach back into those steps, as they would without a test
    boundary. Without it, windows start at the test boundary from h0.
    lr = 0 with context is the paper's frozen test: the stream continues with the
    learning rate set to zero.
    """
    model = copy.deepcopy(model)
    sm = _copy_slowmem(slowmem, device, reset_traces)
    if isinstance(sm, SlowMem1):
        sm.learn_wb = not freeze_slow_weights
    opt = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    opt.load_state_dict(optimizer.state_dict())  # continue with the trained optimizer state
    for group in opt.param_groups:  # load_state_dict restores the saved lr; use ours
        group['lr'] = lr
    trainer = OnlineTrainer(model, slowmem=sm, window_size=window_size, stride=stride, dt=dt,
                            device=device, optimizer=opt, log=None)
    if context is None:
        res = trainer.run(SessionSchedule(test_task, overnight=None), h0=h0, reset_slowmem=False)
        return res['V']
    session = test_task.session(0)
    X, y = session.X.to(device), session.y.to(device)
    if sm is not None:
        X = torch.cat([X, sm.advance(X)], dim=-1)  # Wb keeps learning: all plasticity on
    n_ctx = len(context['X'])
    X_all = torch.cat([context['X'].to(device), X])
    y_all = torch.cat([context['y'].to(device), y])
    h_ctx = None if context['h0'] is None else context['h0'].to(device)
    rec = dict(V=[], delta=[], hidden=[], window_loss=[], window_session=[])
    trainer.train_on_inputs(X_all, y_all, torch.zeros(len(X_all)), h_ctx, rec)
    return rec['V'][0][n_ctx:]


def evaluate_trained(model, slowmem, optimizer, h_final, test_task, window_size, stride, lr, dt,
                     device, protocol='continue', context=None):
    """AUROC on the test episode (one session of test_task).

    protocol 'continue': start from h_final and the slow memory's end-of-training
    traces; test_task should be the next block (reversed contingency). With
    context (the trainer's tail), learning-on windows reach back into training.
    protocol 'reset': start from h = 0 with slow-memory traces reset.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f'protocol must be one of {PROTOCOLS}, got {protocol!r}')
    reset = protocol == 'reset'
    h0 = None if reset else h_final
    session = test_task.session(0)
    probs = test_task.reward_probs_per_block

    def summary(V):
        table = trial_table([session], probs, [0], V)
        good, v, blk = table['good'], table['v_cue'], table['block_pos']
        return dict(auroc=auroc(v[good], v[~good]),
                    auroc_by_block=[auroc(v[good & (blk == b)], v[~good & (blk == b)])
                                    for b in np.unique(blk)],
                    # per-trial values and CS+ labels kept for plotting ROC curves
                    v_cue=v.tolist(), good=good.tolist(), block_pos=blk.tolist())

    out = dict(protocol=protocol, first_block=int(session.trials[0].block_index))
    if not reset and context is not None:
        # paper: the stream continues with the learning rate set to 0
        def run(lr_, freeze_slow):
            return plastic_values(model, slowmem, optimizer, test_task, h0, window_size, stride,
                                  lr_, dt, device, context=context, freeze_slow_weights=freeze_slow)
        out['frozen'] = summary(run(0.0, True))
        if isinstance(slowmem, SlowMem1):
            out['frozen_rnn_only'] = summary(run(0.0, False))
        out['plastic'] = summary(run(lr, False))
        return out
    out['frozen'] = summary(frozen_values(model, slowmem, session, h0, device,
                                          reset_traces=reset))
    if isinstance(slowmem, SlowMem1):
        out['frozen_rnn_only'] = summary(frozen_values(model, slowmem, session, h0, device,
                                                       freeze_slow_weights=False,
                                                       reset_traces=reset))
    out['plastic'] = summary(plastic_values(model, slowmem, optimizer, test_task, h0, window_size,
                                            stride, lr, dt, device, reset_traces=reset))
    return out
