"""Behavioral metrics from per-trial values.

A trial table is a dict of equal-length numpy arrays, one entry per trial, in
stream order. `v_cue` is the value at cue onset (the step the cue first
appears), the quantity plotted in valuernn's scripts.

"good" = the cue is rewarded with probability > 0.5 in the current block.
Discrimination = mean v_cue(good) - mean v_cue(bad).
"""
import numpy as np
from scipy.optimize import curve_fit


def trial_table(sessions, reward_probs_per_block, offsets=None, V=None):
    """One row per trial. With offsets and V (per-step values from OnlineTrainer),
    adds v_cue (value at cue onset) and v_pre (value on the step before the cue)."""
    cols = {k: [] for k in ('session', 'trial', 'block', 'block_pos', 'rel_trial',
                            'cue', 'rewarded', 'good', 'onset_step', 'start_step', 'length')}
    for k, s in enumerate(sessions):
        start = 0 if offsets is None else int(offsets[k])
        for j, t in enumerate(s.trials):
            cols['session'].append(s.index)
            cols['trial'].append(j)
            cols['block'].append(t.block_index)
            cols['block_pos'].append(t.block_index_in_episode)
            cols['rel_trial'].append(t.rel_trial_index)
            cols['cue'].append(t.cue)
            cols['rewarded'].append(t.y.sum() > 0)
            cols['good'].append(reward_probs_per_block[t.cue][t.block_index] > 0.5)
            cols['onset_step'].append(start + t.iti)
            cols['start_step'].append(start)
            cols['length'].append(len(t))
            start += len(t)
    table = {k: np.asarray(v) for k, v in cols.items()}
    if V is not None:
        table['v_cue'] = V[table['onset_step']]
        table['v_pre'] = V[table['onset_step'] - 1]
    return table


CONVERGENCE_THRESHOLDS = (0.0005, 0.005)  # Lee et al.: Methods text / Extended Data Fig. 1


def convergence_loss(table, delta, n_blocks=4, n_trials=20, exclude_cue=False):
    """Lee et al. convergence criterion: mean squared RPE over the last n_trials
    trials of each of the last n_blocks blocks of training.

    exclude_cue=False: all steps of those trials. Includes the cue-onset RPE, which
    no model can predict (cue identity and onset are random), so it has a floor of
    ≈ 0.02 in this task.
    exclude_cue=True: all steps except the transition into the cue (delta at
    onset - 1) — our reconstruction of the paper's measure (typical values ≈ 0.002).
    """
    blocks = list(dict.fromkeys(zip(table['session'], table['block_pos'])))  # in stream order
    sq = []
    for s, b in blocks[-n_blocks:]:
        idx = np.flatnonzero((table['session'] == s) & (table['block_pos'] == b))[-n_trials:]
        for i in idx:
            steps = np.arange(table['start_step'][i], table['start_step'][i] + table['length'][i])
            if exclude_cue:
                steps = steps[steps != table['onset_step'][i] - 1]
            d = delta[steps]
            sq.append(d[np.isfinite(d)] ** 2)
    return float(np.concatenate(sq).mean()) if sq else float('nan')


def discrimination(table, mask=None, key='v_cue'):
    m = np.ones(len(table['cue']), bool) if mask is None else mask
    good, bad = m & table['good'], m & ~table['good']
    if not good.any() or not bad.any():
        return np.nan
    return table[key][good].mean() - table[key][bad].mean()


def session_start_discrimination(table, n_trials=10):
    """Per session: discrimination over the first n_trials of the first block."""
    out = []
    for s in np.unique(table['session']):
        m = (table['session'] == s) & (table['block_pos'] == 0) & (table['rel_trial'] < n_trials)
        out.append(discrimination(table, m))
    return np.asarray(out)


def reversal_curve(table, sessions=None, max_k=None):
    """Discrimination at each trial k after the reversal (second block), pooled
    over `sessions` (default: all). Returns (k, d)."""
    m = table['block_pos'] == 1
    if sessions is not None:
        m &= np.isin(table['session'], sessions)
    max_k = int(table['rel_trial'][m].max()) + 1 if max_k is None else max_k
    ks = np.arange(max_k)
    d = np.array([discrimination(table, m & (table['rel_trial'] == k)) for k in ks])
    return ks, d


def _exp(k, a, b, tau):
    return a + b * np.exp(-k / tau)


TAU_BOUNDS = (1e-2, 1e3)
# below this τ, > 99% of the change is done after one trial: indistinguishable from a step
TAU_STEP = 1 / np.log(100)


def fit_tau(k, v):
    """Fit v(k) = a + b exp(-k/τ); returns τ in units of k.

    A step-like switch (> 99% complete after one trial) returns TAU_STEP
    (≈ 0.22), meaning "within one trial"; smaller fitted τ are not resolvable.
    Returns nan when τ is not identifiable: fewer than 4 points, the fit fails,
    the amplitude b is not significant (|b| <= 2 SE; flat or pure noise),
    τ ends at the upper bound, or its standard error is undefined or larger
    than τ.
    """
    k, v = np.asarray(k, float), np.asarray(v, float)
    ok = np.isfinite(v)
    k, v = k[ok], v[ok]
    if len(k) < 4:
        return np.nan
    a0 = v[k >= np.percentile(k, 75)].mean()
    lo, hi = TAU_BOUNDS
    try:
        (a, b, tau), pcov = curve_fit(_exp, k, v, p0=(a0, v[0] - a0, 3.0),
                                      bounds=([-np.inf, -np.inf, lo], [np.inf, np.inf, hi]),
                                      maxfev=10000)
    except (RuntimeError, ValueError):
        return np.nan
    se_b, se_tau = (np.sqrt(pcov[i, i]) if np.isfinite(pcov[i, i]) else np.inf for i in (1, 2))
    if not abs(b) > 2 * se_b:  # no significant change (flat or noise): τ meaningless
        return np.nan
    if tau <= TAU_STEP:
        return float(TAU_STEP)
    if tau >= hi * 0.99 or not se_tau < tau:
        return np.nan
    return float(tau)


def reversal_tau(table, sessions=None):
    """Reversal time constants in trials (counting trials of both cues since the
    reversal), fitted to the pooled per-trial values of each cue in the second
    block: the newly good cue (rising) and the newly bad cue (falling)."""
    m = table['block_pos'] == 1
    if sessions is not None:
        m &= np.isin(table['session'], sessions)
    good, bad = m & table['good'], m & ~table['good']
    return dict(tau_good=fit_tau(table['rel_trial'][good], table['v_cue'][good]),
                tau_bad=fit_tau(table['rel_trial'][bad], table['v_cue'][bad]))


def session_end_discrimination(table, n_trials=10):
    """Per session: discrimination over the last n_trials of the second block."""
    out = []
    for s in np.unique(table['session']):
        m = (table['session'] == s) & (table['block_pos'] == 1)
        if m.any():
            m &= table['rel_trial'] >= table['rel_trial'][m].max() - (n_trials - 1)
        out.append(discrimination(table, m))
    return np.asarray(out)


def summarize(table, pool=4):
    """Summary used by the run harness.

    Per-session τ fits use one point per trial and are often not identifiable
    (nan); `tau_pooled` fits τ over consecutive groups of `pool` sessions.
    """
    sessions = np.unique(table['session'])
    taus = [reversal_tau(table, [s]) for s in sessions]
    groups = [sessions[i:i + pool] for i in range(0, len(sessions), pool)]
    pooled = [reversal_tau(table, g) for g in groups]
    return dict(
        session=sessions.tolist(),
        start_discrimination=session_start_discrimination(table).tolist(),
        end_discrimination=session_end_discrimination(table).tolist(),
        tau_good=[t['tau_good'] for t in taus],
        tau_bad=[t['tau_bad'] for t in taus],
        tau_pooled=dict(sessions=[g.tolist() for g in groups],
                        tau_good=[t['tau_good'] for t in pooled],
                        tau_bad=[t['tau_bad'] for t in pooled]),
    )
