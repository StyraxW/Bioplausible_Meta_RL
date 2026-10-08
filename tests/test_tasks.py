import numpy as np
import pytest
import torch

import BP_models  # noqa: F401  (puts valuernn on sys.path)
from BP_models.tasks.trials import CueOffsetTrial, cue_offset
from BP_models.tasks.sessions import (Break, OvernightBreak, Session, SessionSchedule,
                                      SessionTask)


@pytest.mark.parametrize('cue_steps', [1, 2, 4])
@pytest.mark.parametrize('reward', [1.0, 0])
def test_trial_layout(cue_steps, reward):
    iti = 5
    t = CueOffsetTrial(cue=1, iti=iti, reward_size=reward, ncues=2, cue_steps=cue_steps)
    assert len(t) == iti + cue_steps + 1
    c = t.X[:, :2]
    assert c[:iti].sum() == 0
    assert (c[iti:iti + cue_steps, 1] == 1).all() and c[:, 0].sum() == 0
    assert c[t.outcome_index].sum() == 0
    assert t.y[t.outcome_index, 0] == reward and t.y.sum() == reward
    assert (t.X[:, -1] == t.y[:, 0]).all()  # reward input column


@pytest.mark.parametrize('reward', [1.0, 0])
def test_trial_null_input(reward):
    t = CueOffsetTrial(cue=0, iti=3, reward_size=reward, ncues=2, cue_steps=3,
                       include_null_input=True)
    assert (t.X.sum(axis=1) == 1).all()


def test_off_fires_only_at_outcome():
    task = SessionTask(nsessions=2, cue_steps=3, ntrials_per_block=10, seed=1)
    s = task.session(0)
    X = s.X[:, 0, :].numpy()
    off = cue_offset(X[:, :2])
    starts = np.cumsum([0] + [len(t) for t in s.trials[:-1]])
    expected = np.zeros(len(X))
    for start, trial in zip(starts, s.trials):
        expected[start + trial.outcome_index] = 1
    assert (off == expected).all()
    # omitted trials get the pulse too
    assert any(t.y.sum() == 0 for t in s.trials) and any(t.y.sum() > 0 for t in s.trials)


def test_session_structure_anticorrelated():
    task = SessionTask(nsessions=4, ntrials_per_block=20, seed=2)
    for i in range(4):
        s = task.session(i)
        blocks = [t.block_index for t in s.trials]
        assert len(s.trials) == 40
        assert blocks[:20] == [blocks[0]] * 20 and blocks[20:] == [1 - blocks[0]] * 20  # one reversal
        for t in s.trials:
            # block 0: A+ B-, block 1: A- B+
            assert (t.y.sum() > 0) == (t.cue == t.block_index)
        assert s.X.shape == (sum(len(t) for t in s.trials), 1, 3)


def test_seed_reproducible_and_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = SessionTask(nsessions=3, seed=7)
    assert (np.random.get_state()[1] == before).all()
    b = SessionTask(nsessions=3, seed=7)
    c = SessionTask(nsessions=3, seed=8)
    for i in range(3):
        assert np.array_equal(a.session(i).X, b.session(i).X)
    assert not all(np.array_equal(a.session(i).X, c.session(i).X) for i in range(3))


def test_first_block_fixed():
    task = SessionTask(nsessions=5, ntrials_per_block=5, first_block=1, seed=0)
    assert all(task.session(i).trials[0].block_index == 1 for i in range(5))


def test_schedule_with_breaks():
    task = SessionTask(nsessions=3, ntrials_per_block=5, seed=0)
    spec = OvernightBreak(duration=8 * 3600, h_policy='reset')
    events = list(SessionSchedule(task, overnight=spec))
    kinds = [type(e) for e in events]
    assert kinds == [Session, Break, Session, Break, Session]
    assert [e.after_session for e in events if isinstance(e, Break)] == [0, 1]
    assert events[1].spec is spec


def test_schedule_without_breaks():
    task = SessionTask(nsessions=3, ntrials_per_block=5, seed=0)
    events = list(SessionSchedule(task, overnight=None))
    assert all(isinstance(e, Session) for e in events) and len(events) == 3


def test_relax_steps_default_is_full_duration():
    assert OvernightBreak().n_relax_steps(dt=0.75) == 76800
    assert OvernightBreak(duration=3600).n_relax_steps(dt=0.5) == 7200
    assert OvernightBreak(relax_steps=300).n_relax_steps(dt=0.75) == 300


def test_reset_state_is_seeded_low_power_noise():
    spec = OvernightBreak()
    assert spec.h_policy == 'reset'
    h = torch.ones(1, 1, 2000)
    a = spec.reset_state(h, torch.Generator().manual_seed(0))
    b = spec.reset_state(h, torch.Generator().manual_seed(0))
    c = spec.reset_state(h, torch.Generator().manual_seed(1))
    assert a.shape == h.shape and torch.equal(a, b) and not torch.equal(a, c)
    assert abs(a.std().item() - 0.1) < 0.01 and abs(a.mean().item()) < 0.01
    assert torch.equal(OvernightBreak(reset_sigma=0).reset_state(h, torch.Generator()), torch.zeros_like(h))


def test_invalid_break_and_fixed_kwargs():
    with pytest.raises(ValueError):
        OvernightBreak(h_policy='freeze')
    with pytest.raises(TypeError):
        SessionTask(nblocks_per_episode=4)
