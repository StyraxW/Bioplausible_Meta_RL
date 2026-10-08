"""B3: Bayesian ideal observer over the hidden block (context).

Belief b over blocks is updated once per trial:
    predict:  b <- (1-h) b + h (1 - b)          (two blocks, hazard h per trial)
    value:    v = sum_k b_k p(reward | cue, k)   (expected reward at cue onset)
    update:   b <- b * p(outcome | cue, k), normalized
The belief resets to `prior` at each session start when reset_each_session
(the first block of a session is random, so nothing carries over).

v is an expected reward, not a discounted TD value: compare it with the RNNs
through scale-free metrics (reversal τ, sign of discrimination), not raw values.
"""
import numpy as np


def ideal_observer_values(sessions, reward_probs_per_block, hazard, prior=(0.5, 0.5),
                          reset_each_session=True):
    """Per-trial expected reward at cue onset, in stream order (aligned with trial_table)."""
    prior = np.asarray(prior, float)
    b = prior.copy()
    values = []
    for s in sessions:
        if reset_each_session:
            b = prior.copy()
        for t in s.trials:
            b = (1 - hazard) * b + hazard * b[::-1]
            p = np.asarray(reward_probs_per_block[t.cue], float)
            values.append(float(b @ p))
            lik = p if t.y.sum() > 0 else 1 - p
            post = b * lik
            b = post / post.sum() if post.sum() > 0 else prior.copy()
    return np.asarray(values)
