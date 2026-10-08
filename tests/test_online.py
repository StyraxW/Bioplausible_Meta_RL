import copy
import math

import numpy as np
import pytest
import torch

import BP_models  # noqa: F401
from model import ValueRNN
from train_bptt import train_model_TBPTT
from BP_models.memory import TraceBank
from BP_models.tasks.sessions import OvernightBreak, SessionSchedule, SessionTask
from BP_models.train.online import OnlineTrainer


class ListLoader:
    """Minimal stand-in for valuernn's dataloader (batch_size 1)."""
    batch_size = 1

    def __init__(self, episodes):
        self.episodes = episodes

    def __iter__(self):
        for X, y in self.episodes:
            yield X, y, [len(X)], None, None


def make_model(seed=0, input_size=3, hidden=16):
    torch.manual_seed(seed)
    m = ValueRNN(input_size=input_size, output_size=1, hidden_size=hidden, gamma=0.9)
    m.reset(seed=seed, initialization_gain=1.0)
    return m


def assert_same_weights(a, b, atol=1e-6):
    for (k, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.allclose(va, vb, atol=atol), k


def fork_train(model, episodes, W, stride):
    opt = torch.optim.Adam(model.parameters(), lr=0.003, amsgrad=True)
    losses, _, _ = train_model_TBPTT(model, ListLoader(episodes), epochs=1, optimizer=opt,
                                     window_size=W, stride_size=stride, print_every=10**9)
    return losses


def ours_train(model, task, overnight, W, stride, **kw):
    opt = torch.optim.Adam(model.parameters(), lr=0.003, amsgrad=True)
    tr = OnlineTrainer(model, window_size=W, stride=stride, optimizer=opt, log=None, **kw)
    return tr.run(SessionSchedule(task, overnight=overnight))


@pytest.mark.parametrize('W,stride', [(20, 1), (30, 4)])
def test_matches_fork_single_session(W, stride):
    task = SessionTask(nsessions=1, ntrials_per_block=8, seed=3)
    a, b = make_model(), make_model()
    s = task.session(0)
    fork_losses = fork_train(a, [(s.X, s.y)], W, stride)
    res = ours_train(b, task, None, W, stride)
    assert_same_weights(a, b)
    moved = max((p - q).abs().max().item() for p, q in zip(b.parameters(), make_model().parameters()))
    assert moved > 1e-3  # training actually changed the weights
    assert np.isclose(fork_losses[-1], res['window_loss'].mean(), atol=1e-6)


def test_continuous_sessions_match_fork_on_concatenation():
    task = SessionTask(nsessions=3, ntrials_per_block=5, seed=4)
    a, b = make_model(), make_model()
    X = torch.cat([task.session(i).X for i in range(3)])
    y = torch.cat([task.session(i).y for i in range(3)])
    fork_train(a, [(X, y)], 25, 1)
    res = ours_train(b, task, None, 25, 1)
    assert_same_weights(a, b)
    assert res['breaks'] == []


def test_breaks_cut_windows_like_fork_episodes():
    # reset with sigma 0 sets h = 0 at each session start, which is what the fork
    # does between episodes; so breaks must reproduce one-episode-per-session training
    task = SessionTask(nsessions=3, ntrials_per_block=5, seed=5)
    a, b = make_model(), make_model()
    fork_train(a, [(task.session(i).X, task.session(i).y) for i in range(3)], 25, 1)
    res = ours_train(b, task, OvernightBreak(h_policy='reset', reset_sigma=0.0), 25, 1)
    assert_same_weights(a, b)
    assert [br['after_session'] for br in res['breaks']] == [0, 1]
    assert all(br['h_norm_after'] == 0 for br in res['breaks'])


def test_online_records_cover_every_step():
    task = SessionTask(nsessions=2, ntrials_per_block=5, seed=6)
    res = ours_train(make_model(), task, OvernightBreak(), 20, 3, record_hidden=True)
    T = sum(len(task.session(i).X) for i in range(2))
    assert res['V'].shape == (T,) and not np.isnan(res['V']).any()
    assert res['hidden'].shape == (T, 16) and not np.isnan(res['hidden']).any()
    # delta is defined for every step except the last of each segment
    for k, off in enumerate(res['session_offsets']):
        seg_end = off + len(task.session(k).X) - 1
        assert np.isnan(res['delta'][seg_end])
    assert np.isnan(res['delta']).sum() == 2


def test_break_policies():
    task = SessionTask(nsessions=2, ntrials_per_block=5, seed=7)
    carry = ours_train(make_model(), task, OvernightBreak(h_policy='carry'), 20, 1)['breaks'][0]
    assert carry['h_norm_after'] == carry['h_norm_before']

    reset = ours_train(make_model(), task, OvernightBreak(h_policy='reset', reset_sigma=0.1), 20, 1,
                       break_seed=1)['breaks'][0]
    assert abs(reset['h_norm_after'] - 0.1 * math.sqrt(16)) < 0.2

    relax = ours_train(make_model(), task, OvernightBreak(h_policy='relax', relax_steps=20000),
                       20, 1)['breaks'][0]
    assert relax['relax_steps'] == 20000
    assert np.isfinite(relax['final_step_change']) and np.isfinite(relax['drift_after_step_300'])


def test_relax_default_runs_full_duration():
    task = SessionTask(nsessions=2, ntrials_per_block=3, seed=8)
    spec = OvernightBreak(duration=3600, h_policy='relax')
    br = ours_train(make_model(), task, spec, 10, 1, dt=0.5)['breaks'][0]
    assert br['relax_steps'] == 7200


def test_reset_noise_independent_of_task_and_init():
    task = SessionTask(nsessions=2, ntrials_per_block=3, seed=9)
    r1 = ours_train(make_model(seed=0), task, OvernightBreak(), 10, 1, break_seed=5)['breaks'][0]
    r2 = ours_train(make_model(seed=1), task, OvernightBreak(), 10, 1, break_seed=5)['breaks'][0]
    assert r1['h_norm_after'] == r2['h_norm_after']


def test_memory_advances_once_per_step_and_decays_overnight():
    task = SessionTask(nsessions=2, ntrials_per_block=4, seed=10)
    dt, taus = 0.5, (1.0, 20.0)
    mem = TraceBank(3, taus, dt)
    model = make_model(input_size=3 + mem.n_out)
    spec = OvernightBreak(duration=10.0, h_policy='carry')
    ours_train(model, task, spec, 15, 1, memory=mem, dt=dt)

    # reference: the same traces computed directly on the stream
    s = np.zeros((2, 3))
    lams = np.exp(-dt / np.array(taus))[:, None]
    for k in range(2):
        for x in task.session(k).X[:, 0].numpy():
            s = lams * s + (1 - lams) * x
        if k == 0:
            s = s * np.exp(-10.0 / np.array(taus))[:, None]
    assert np.allclose(mem.s.numpy(), s, atol=1e-5)


def test_feedback_memory_rejected():
    mem = TraceBank(3, (1.0,), 0.5)
    mem.feedforward = False
    with pytest.raises(NotImplementedError):
        OnlineTrainer(make_model(input_size=6), memory=mem, log=None)
