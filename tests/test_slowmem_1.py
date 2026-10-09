import math

import numpy as np
import pytest
import torch

import BP_models  # noqa: F401
from BP_models.slowmem_1 import ONEHOT_NAMES, SlowMem1
from BP_models.tasks.sessions import SessionTask
from BP_models.tasks.trials import CueOffsetTrial, cue_offset

DT = 0.5
A_PLUS, A_MINUS, B_PLUS, B_MINUS = range(4)


def stream(cues, rewards, cue_steps=3, iti=6):
    X = np.vstack([CueOffsetTrial(c, iti, r, 2, cue_steps=cue_steps).X for c, r in zip(cues, rewards)])
    return torch.tensor(X, dtype=torch.float32)[:, None, :]


def context_stream(n, context, rng, cue_steps=3, iti=6):
    """Context 0: A+ B-. Context 1: A- B+."""
    cues = rng.integers(0, 2, n)
    rewards = (cues == context).astype(float)
    return stream(cues, rewards, cue_steps, iti), cues, rewards


def test_cue_trace_and_gate():
    m = SlowMem1(DT, tau_e=1.5, trial_types='onehot')
    X = stream([0, 1], [1.0, 0.0], cue_steps=3, iti=4)
    _, rec = m.advance(X, record=True)
    lam_e = math.exp(-DT / 1.5)
    # gate fires exactly at the outcome steps (one step after cue offset)
    assert np.array_equal(rec['off'], cue_offset(X[:, 0, :2].numpy()))
    outcome = np.flatnonzero(rec['off'])
    assert list(outcome) == [7, 15]
    # averaging form: after 3 cue steps e = 1 - λ^3, one more silent step at the gate
    assert rec['e'][outcome[0], 0] == pytest.approx((1 - lam_e ** 3) * lam_e, rel=1e-5)
    assert rec['e'][outcome[0], 1] == 0
    # y only on gate steps
    assert (rec['y'][rec['off'] == 0] == 0).all()


def test_onehot_trial_types():
    m = SlowMem1(DT, trial_types='onehot')
    X = stream([0, 0, 1, 1], [1.0, 0.0, 1.0, 0.0])
    _, rec = m.advance(X, record=True)
    fired = rec['y'][rec['off'] > 0].argmax(axis=1)
    assert [ONEHOT_NAMES[i] for i in fired] == ['A+', 'A-', 'B+', 'B-']
    assert (rec['y'].sum(axis=1)[rec['off'] > 0] == 1).all()


def test_random_trial_types_separate_conjunctions():
    m = SlowMem1(DT, trial_types='random', n_y=50, seed=3)
    X = stream([0, 0, 1, 1], [1.0, 0.0, 1.0, 0.0])
    _, rec = m.advance(X, record=True)
    Y = rec['y'][rec['off'] > 0]                      # 4 x 50, one row per trial type
    assert Y.shape == (4, 50) and (Y >= 0).all()
    assert np.linalg.matrix_rank(Y) == 4              # the four conjunctions are distinct
    # XOR is linearly readable: context 0 = {A+, B-}, context 1 = {A-, B+}
    target = np.array([1.0, -1.0, -1.0, 1.0])
    w, *_ = np.linalg.lstsq(np.c_[Y, np.ones(4)], target, rcond=None)
    assert np.allclose(np.c_[Y, np.ones(4)] @ w, target, atol=1e-6)


def test_wb_reaches_hebbian_fixed_point():
    # one-hot, a single context (A+ / B-): Wij -> E[ȳj(t-1) | type i fires]
    rng = np.random.default_rng(0)
    X, cues, rewards = context_stream(3000, 0, rng)
    m = SlowMem1(DT, tau_b=15.0, eta_b=0.01, trial_types='onehot')
    _, rec = m.advance(X, record=True)
    W = m.Wb.numpy()
    outcome = np.flatnonzero(rec['off'])
    types = rec['y'][outcome].argmax(axis=1)
    ybar_prev = rec['ybar'][outcome - 1]
    late = np.arange(len(outcome)) >= len(outcome) // 2
    for i in (A_PLUS, B_MINUS):
        expected = ybar_prev[late & (types == i)].mean(axis=0)
        for j in (A_PLUS, B_MINUS):
            if i != j:
                assert W[i, j] == pytest.approx(expected[j], rel=0.1)
    # trial types never seen in this context get no weights
    assert np.all(W[[A_MINUS, B_PLUS]] == 0) and np.all(W[:, [A_MINUS, B_PLUS]] == 0)
    assert np.all(np.diag(W) == 0)


def test_wb_structure_after_reversals():
    # sessions with one reversal each: same-context links (A+ <-> B-, A- <-> B+) dominate;
    # cross-context links (A+ <-> A-) are faint and come from trials right after reversals
    task = SessionTask(nsessions=6, ntrials_per_block=50, cue_steps=3, seed=1)
    m = SlowMem1(DT, eta_b=0.05, trial_types='onehot')
    for i in range(len(task)):
        m.advance(task.session(i).X)
        m.overnight(16 * 3600)
    W = m.Wb.numpy()
    same = [W[A_PLUS, B_MINUS], W[B_MINUS, A_PLUS], W[A_MINUS, B_PLUS], W[B_PLUS, A_MINUS]]
    cross = [W[A_PLUS, A_MINUS], W[A_MINUS, A_PLUS], W[B_PLUS, B_MINUS], W[B_MINUS, B_PLUS]]
    assert min(same) > 0 and max(cross) > 0
    assert min(same) > 3 * max(cross)


def test_overnight_decays_traces_keeps_wb():
    m = SlowMem1(DT, tau_e=1.5, tau_b=15.0, trial_types='onehot')
    m.advance(stream([0, 1, 0], [1.0, 0.0, 1.0], iti=2)[:-1])  # stop mid-trial, cue just ended
    e, ybar, W = m.e.clone(), m.ybar.clone(), m.Wb.clone()
    m.overnight(30.0)
    assert torch.allclose(m.e, e * math.exp(-30 / 1.5))
    assert torch.allclose(m.ybar, ybar * math.exp(-30 / 15.0))
    assert torch.equal(m.Wb, W) and m.c_prev.item() == 0


def test_advance_is_resumable():
    # stepping through a stream in pieces = stepping through it at once
    X = context_stream(40, 0, np.random.default_rng(2))[0]
    a, b = SlowMem1(DT, seed=5), SlowMem1(DT, seed=5)
    Za = a.advance(X)
    Zb = torch.cat([b.advance(X[:123]), b.advance(X[123:])])
    assert torch.equal(Za, Zb) and torch.equal(a.Wb, b.Wb)


def test_reset_clears_state_keeps_fixed_weights():
    m = SlowMem1(DT, seed=4)
    R, b = m.R.clone(), m.b.clone()
    m.advance(context_stream(20, 1, np.random.default_rng(3))[0])
    assert m.Wb.abs().sum() > 0
    m.reset()
    assert m.Wb.abs().sum() == 0 and m.ybar.abs().sum() == 0 and m.e.abs().sum() == 0
    assert torch.equal(m.R, R) and torch.equal(m.b, b)
