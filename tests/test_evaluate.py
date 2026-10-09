import copy
import json
import os

import numpy as np
import pytest
import torch

import BP_models  # noqa: F401
from model import ValueRNN
from BP_models.analysis.evaluate import auroc, evaluate_trained, frozen_values, plastic_values
from BP_models.config import RunConfig, set_override
from BP_models.run import run_one
from BP_models.slowmem_1 import SlowMem1
from BP_models.tasks.sessions import SessionSchedule, SessionTask
from BP_models.train.online import OnlineTrainer


def test_auroc():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert np.isnan(auroc([], [1]))


def trained(slowmem=None, W=10):
    task = SessionTask(nsessions=2, ntrials_per_block=10, cue_steps=1, seed=1)
    n_z = slowmem.n_out if slowmem is not None else 0
    model = ValueRNN(input_size=3 + n_z, output_size=1, hidden_size=8, gamma=0.8)
    model.reset(seed=0)
    tr = OnlineTrainer(model, slowmem=slowmem, window_size=W, dt=0.5, log=None)
    res = tr.run(SessionSchedule(task, overnight=None))
    return model, slowmem, tr, res


def make_block(first_block, seed=2):
    return SessionTask(nsessions=1, blocks_per_session=1, ntrials_per_block=10, cue_steps=1,
                       first_block=first_block, seed=seed)


def test_trainer_keeps_final_state_and_can_continue():
    model, _, tr, res = trained()
    assert tr.final_h.shape == (1, 1, 8) and tr.final_h.abs().sum() > 0
    # continuing from final_h without resetting gives a different start than from zeros
    block = make_block(0).session(0)
    v_cont = frozen_values(model, None, block, tr.final_h, 'cpu')
    v_zero = frozen_values(model, None, block, None, 'cpu')
    assert not np.allclose(v_cont[:5], v_zero[:5])


def test_evaluation_leaves_trained_model_untouched():
    sm = SlowMem1(0.5, trial_types='onehot')
    model, sm, tr, res = trained(sm)
    before = copy.deepcopy(model.state_dict()), sm.Wb.clone(), sm.ybar.clone(), tr.final_h.clone()
    last = res['sessions'][-1].trials[-1].block_index
    four_blocks = SessionTask(nsessions=1, blocks_per_session=4, ntrials_per_block=10,
                              cue_steps=1, seed=4)
    for protocol, task in (('continue', make_block(1 - last)), ('reset', four_blocks)):
        out = evaluate_trained(model, sm, tr.optimizer, tr.final_h, task,
                               10, 1, 0.0005, 0.5, 'cpu', protocol=protocol)
        assert set(out) == {'protocol', 'first_block', 'frozen', 'frozen_rnn_only', 'plastic'}
        n = len(task.session(0).trials)
        assert all(0 <= out[k]['auroc'] <= 1 and len(out[k]['v_cue']) == n
                   for k in ('frozen', 'frozen_rnn_only', 'plastic'))
        for k, v in before[0].items():
            assert torch.equal(model.state_dict()[k], v)
        assert torch.equal(sm.Wb, before[1]) and torch.equal(sm.ybar, before[2])
        assert torch.equal(tr.final_h, before[3])
    assert len(out['frozen']['auroc_by_block']) == 4


def test_frozen_test_carries_slow_traces_and_freezes_wb():
    sm = SlowMem1(0.5, trial_types='onehot')
    model, sm, tr, _ = trained(sm)
    block = make_block(0, seed=3).session(0)
    v1 = frozen_values(model, sm, block, tr.final_h, 'cpu')
    # the slow memory's carried-over traces matter (nothing is reset)
    sm2 = copy.deepcopy(sm)
    sm2.ybar.fill_(5.0)
    assert not np.allclose(v1, frozen_values(model, sm2, block, tr.final_h, 'cpu'))
    # letting Wb learn changes the values
    v3 = frozen_values(model, sm, block, tr.final_h, 'cpu', freeze_slow_weights=False)
    assert not np.allclose(v1, v3)


def test_reset_protocol_ignores_carried_state():
    sm = SlowMem1(0.5, trial_types='onehot')
    model, sm, tr, _ = trained(sm)
    block = make_block(0, seed=5).session(0)
    v1 = frozen_values(model, sm, block, None, 'cpu', reset_traces=True)
    sm2 = copy.deepcopy(sm)
    sm2.ybar.fill_(5.0)
    v2 = frozen_values(model, sm2, block, None, 'cpu', reset_traces=True)
    assert np.allclose(v1, v2)


class OneSession:
    """test_task stand-in exposing one session of another task as session(0)."""
    def __init__(self, task, index):
        self.task, self.index = task, index
        self.reward_probs_per_block = task.reward_probs_per_block

    def session(self, i):
        assert i == 0
        return self.task.session(self.index)


@pytest.mark.parametrize('use_slowmem,stride', [(False, 1), (True, 1), (False, 5), (True, 5)])
def test_continue_with_learning_equals_uninterrupted_stream(use_slowmem, stride):
    # window longer than a session, so the test's first windows must reach back
    # into training steps to match an uninterrupted run
    W = 120
    task = SessionTask(nsessions=3, ntrials_per_block=6, cue_steps=1, seed=7)
    assert len(task.session(2).X) < W

    def fresh():
        sm = SlowMem1(0.5, trial_types='onehot') if use_slowmem else None
        m = ValueRNN(input_size=3 + (sm.n_out if sm else 0), output_size=1, hidden_size=8, gamma=0.8)
        m.reset(seed=0)
        return m, sm

    class FirstTwo:  # schedule over sessions 0-1 only
        def __iter__(self):
            yield task.session(0)
            yield task.session(1)

    lr = 0.0005
    m, sm = fresh()
    tr = OnlineTrainer(m, slowmem=sm, window_size=W, stride=stride, lr=lr, dt=0.5, log=None)
    tr.run(FirstTwo())
    v_test = plastic_values(m, sm, tr.optimizer, OneSession(task, 2), tr.final_h, W, stride, lr,
                            0.5, 'cpu', context=tr.tail)

    m2, sm2 = fresh()
    tr2 = OnlineTrainer(m2, slowmem=sm2, window_size=W, stride=stride, lr=lr, dt=0.5, log=None)
    res = tr2.run(SessionSchedule(task, overnight=None))
    v_ref = res['V'][res['session_offsets'][2]:]
    assert np.allclose(v_test, v_ref, atol=1e-6)


def test_window_of_two_trains():
    model, _, _, _ = trained(W=2)
    fresh = ValueRNN(input_size=3, output_size=1, hidden_size=8, gamma=0.8)
    fresh.reset(seed=0)
    moved = max((p - q).abs().max().item() for p, q in zip(model.parameters(), fresh.parameters()))
    assert moved > 1e-4


def test_harness_tests_reversed_next_block(tmp_path):
    cfg = RunConfig(name='t')
    cfg.task.nsessions, cfg.task.ntrials_per_block = 2, 8
    cfg.model.hidden_size = 8
    cfg.overnight.enabled = False
    set_override(cfg, 'train.window_size', 2)
    set_override(cfg, 'model.slowmem', 'slowmem_1')
    set_override(cfg, 'model.slowmem_kwargs', {'trial_types': 'onehot'})
    d = run_one(cfg.to_dict(), 0, str(tmp_path), verbose=False)
    m = json.load(open(os.path.join(d, 'metrics.json')))
    a = np.load(os.path.join(d, 'arrays.npz'))
    last_block = a['trial_block'][-1]
    assert m['test']['protocol'] == 'continue' and m['test']['first_block'] == 1 - last_block
    assert len(m['test']['frozen']['v_cue']) == 8

    set_override(cfg, 'test.protocol', 'reset')
    d = run_one(cfg.to_dict(), 0, str(tmp_path / 'reset'), verbose=False)
    m = json.load(open(os.path.join(d, 'metrics.json')))
    assert m['test']['protocol'] == 'reset' and len(m['test']['frozen']['v_cue']) == 4 * 8
