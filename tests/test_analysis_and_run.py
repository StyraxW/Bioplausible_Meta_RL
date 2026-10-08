import json
import os

import numpy as np
import pytest

import BP_models  # noqa: F401
from BP_models.analysis.metrics import (TAU_STEP, discrimination, fit_tau, reversal_tau, session_start_discrimination,
                                        summarize, trial_table)
from BP_models.baselines.ideal_observer import ideal_observer_values
from BP_models.config import RunConfig, derive_seeds, set_override
from BP_models.run import main, run_one
from BP_models.tasks.sessions import SessionTask

PROBS = {0: (1, 0), 1: (0, 1)}


def sessions_of(task):
    return [task.session(i) for i in range(len(task))]


def test_trial_table_onsets_point_at_cue_steps():
    task = SessionTask(nsessions=2, ntrials_per_block=6, cue_steps=3, seed=0)
    ss = sessions_of(task)
    offsets = np.cumsum([0] + [len(s.X) for s in ss[:-1]])
    X = np.concatenate([s.X[:, 0].numpy() for s in ss])
    V = np.arange(len(X), dtype=float)
    tab = trial_table(ss, PROBS, offsets, V)
    assert len(tab['cue']) == 24
    assert (X[tab['onset_step'], tab['cue']] == 1).all()          # cue on at onset
    assert (X[tab['onset_step'] - 1, :2].sum(axis=1) == 0).all()  # silent just before
    assert (tab['v_cue'] == tab['onset_step']).all()
    assert (tab['good'] == (tab['cue'] == tab['block'])).all()
    assert (tab['rewarded'] == tab['good']).all()  # deterministic task


def test_ideal_observer_reverses_after_one_trial():
    task = SessionTask(nsessions=6, ntrials_per_block=30, seed=1)
    ss = sessions_of(task)
    tab = trial_table(ss, PROBS)
    tab['v_cue'] = ideal_observer_values(ss, PROBS, hazard=1 / 30)
    # session start: first trial is at chance, discrimination is positive overall
    first = np.flatnonzero(tab['trial'] == 0)
    assert np.allclose(tab['v_cue'][first], 0.5)
    assert np.all(session_start_discrimination(tab) > 0.5)
    # after the reversal, the first trial is wrong, later trials are near-correct
    rev = (tab['block_pos'] == 1) & (tab['rel_trial'] >= 2)
    assert discrimination(tab, rev) > 0.9
    taus = reversal_tau(tab)
    assert taus['tau_bad'] < 1.5 and taus['tau_good'] < 1.5


def test_fit_tau_recovers_known_constant():
    k = np.arange(40)
    rng = np.random.default_rng(0)
    v = 1 - 0.8 * np.exp(-k / 5.0) + 0.01 * rng.standard_normal(40)
    assert abs(fit_tau(k, v) - 5.0) < 0.5
    assert np.isnan(fit_tau([0, 1], [0, 1]))
    # step at k=1: complete within one trial -> TAU_STEP
    assert fit_tau(k, np.where(k == 0, 1.0, 0.0)) == pytest.approx(TAU_STEP)
    # pure noise or a flat series: not identifiable
    assert np.isnan(fit_tau(k, 0.01 * rng.standard_normal(40)))
    assert np.isnan(fit_tau(k, np.full(40, 1 / 30)))


def test_config_roundtrip_and_overrides():
    cfg = RunConfig()
    assert cfg.dt == 0.5 and cfg.cue_steps == 3 and cfg.model.cell == 'GRU'
    set_override(cfg, 'train.window_size', 20)
    set_override(cfg, 'overnight.h_policy', 'relax')
    cfg2 = RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert cfg2 == cfg and cfg2.train.window_size == 20
    with pytest.raises(KeyError):
        set_override(cfg, 'train.windowsize', 3)
    with pytest.raises(KeyError):
        RunConfig.from_dict({'task': {'n_sessions': 3}})
    s = derive_seeds(0)
    assert len(set(s.values())) == 3 and s == derive_seeds(0) and s != derive_seeds(1)


def small_cfg(**over):
    cfg = RunConfig(name='smoke')
    cfg.task.nsessions, cfg.task.ntrials_per_block = 3, 8
    cfg.train.window_size = 15
    cfg.model.hidden_size = 8
    cfg.overnight.duration = 600.0
    for k, v in over.items():
        set_override(cfg, k, v)
    return cfg.to_dict()


def test_run_one_writes_reproducible_results(tmp_path):
    d1 = run_one(small_cfg(), 4, str(tmp_path / 'a'), verbose=False)
    d2 = run_one(small_cfg(), 4, str(tmp_path / 'b'), verbose=False)
    for f in ('config.json', 'metrics.json', 'arrays.npz', 'weights.pt', 'log.txt'):
        assert os.path.exists(os.path.join(d1, f))
    a, b = np.load(os.path.join(d1, 'arrays.npz')), np.load(os.path.join(d2, 'arrays.npz'))
    assert np.array_equal(a['V'], b['V'])  # same config + seed -> same run
    m = json.load(open(os.path.join(d1, 'metrics.json')))
    assert len(m['model']['session']) == 3 and len(m['breaks']) == 2
    assert m['breaks'][0]['h_policy'] == 'reset'


def test_run_with_trace_bank_and_relax(tmp_path):
    cfg = small_cfg(**{'model.memory': 'trace_bank', 'model.memory_kwargs': {'taus': [1.0, 15.0]},
                       'overnight.h_policy': 'relax'})
    d = run_one(cfg, 0, str(tmp_path), verbose=False)
    m = json.load(open(os.path.join(d, 'metrics.json')))
    assert m['breaks'][0]['relax_steps'] == 1200  # 600 s at dt = 0.5


def test_cli_parallel_seeds(tmp_path):
    cfg_file = tmp_path / 'cfg.json'
    cfg_file.write_text(json.dumps(small_cfg()))
    main(['--config', str(cfg_file), '--seeds', '0-1', '--workers', '2', '--out', str(tmp_path)])
    assert sorted(os.listdir(tmp_path / 'smoke')) == ['seed_000', 'seed_001']
