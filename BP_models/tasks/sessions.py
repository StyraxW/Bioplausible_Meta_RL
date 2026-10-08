"""Session-level task stream: sessions of two blocks (one reversal), optionally
separated by overnight breaks.

An overnight break is a separate event in the stream, not a long ITI. The
training loop handles it: no TD learning, the TBPTT window is cut, traces decay
analytically by exp(-T/tau), weights persist, and the RNN state follows
`h_policy`. In-session long-ITI probes, by contrast, are real silent steps.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from tasks.inference import ValueInference

from ..timing import to_steps
from .trials import CueOffsetTrial

H_POLICIES = ('relax', 'reset', 'carry')


@dataclass(frozen=True)
class OvernightBreak:
    """How the training loop should treat the gap between two sessions.

    duration:    gap length in seconds (used for analytic trace decay)
    h_policy:    'reset' (default) = new random low-power state, h = reset_sigma * N(0, I),
                 independent of the weights (the silent-input dynamics need not
                 have an attracting fixed point, so no "reset to rest");
                 'relax' = run silent steps with learning off;
                 'carry' = keep h unchanged
    reset_sigma: s.d. of h after 'reset' (GRU states lie in [-1, 1])
    relax_steps: silent steps for h_policy='relax'; None (default) = the full
                 duration, round(duration / dt) steps (16 h at 0.75 s = 76800)
    replay:      run offline replay during the break (separate condition)
    """
    duration: float = 16 * 3600.0
    h_policy: str = 'reset'
    reset_sigma: float = 0.1
    relax_steps: int | None = None
    replay: bool = False

    def __post_init__(self):
        if self.h_policy not in H_POLICIES:
            raise ValueError(f'h_policy must be one of {H_POLICIES}, got {self.h_policy!r}')
        if (self.duration < 0 or self.reset_sigma < 0
                or (self.relax_steps is not None and self.relax_steps < 0)):
            raise ValueError('duration, reset_sigma and relax_steps must be non-negative')

    def n_relax_steps(self, dt):
        """Silent steps to run for h_policy='relax', given dt in seconds per step."""
        return to_steps(self.duration, dt) if self.relax_steps is None else self.relax_steps

    def reset_state(self, like, generator):
        """h after h_policy='reset': reset_sigma * N(0, I), same shape/dtype/device as `like`.

        generator: a torch.Generator on like.device, seeded from the run's seed.
        """
        noise = torch.randn(like.shape, generator=generator, dtype=like.dtype, device=like.device)
        return self.reset_sigma * noise


@dataclass
class Session:
    """One session. X and y have shape (T, 1, n) like valuernn's dataloader (batch of 1)."""
    index: int
    X: torch.Tensor
    y: torch.Tensor
    trials: list = field(repr=False)


@dataclass(frozen=True)
class Break:
    """Overnight break between session `after_session` and the next one."""
    after_session: int
    spec: OvernightBreak


class SessionTask(ValueInference):
    """valuernn's ValueInference with one episode per session, two blocks per
    session, and CueOffsetTrial trials (k-step cue, outcome on the next step).

    ITIs are in steps, as in valuernn (iti_min + geometric(iti_p) - 1).
    The default reward_probs_per_block is the anti-correlated task:
    block 0 = A+ B-, block 1 = A- B+.

    first_block: 'random' (each session starts in a random block) or a block
    index used for every session.
    """
    def __init__(self, nsessions=12, cue_steps=2,
                 ntrials_per_block=50, ntrials_per_block_jitter=0,
                 reward_probs_per_block={0: (1, 0), 1: (0, 1)},
                 iti_min=6, iti_p=0.5, first_block='random', seed=None, **kwargs):
        for k in ('nepisodes', 'nblocks', 'nblocks_per_episode', 'reward_times_per_block',
                  'jitter', 'is_trial_level', 'first_block_is_random', 'first_block_identity'):
            if k in kwargs:
                raise TypeError(f'{k} is fixed by SessionTask')
        self.cue_steps = cue_steps  # set before super().__init__, which builds the trials
        self.seed = seed
        super().__init__(nepisodes=nsessions, nblocks=2, nblocks_per_episode=2,
                         ntrials_per_block=ntrials_per_block,
                         ntrials_per_block_jitter=ntrials_per_block_jitter,
                         reward_times_per_block=(cue_steps, cue_steps),
                         reward_probs_per_block=reward_probs_per_block,
                         iti_min=iti_min, iti_p=iti_p,
                         first_block_is_random=(first_block == 'random'),
                         first_block_identity=None if first_block == 'random' else int(first_block),
                         seed=seed, **kwargs)

    def make_trial(self, cue, block_index, iti):
        rew_prob = self.reward_probs_per_block[cue][block_index]
        rew_size = self.reward_sizes_per_block[cue][block_index]
        rew = rew_size if self.rng.random() <= rew_prob else 0
        return CueOffsetTrial(cue, iti, rew, self.ncues, cue_steps=self.cue_steps,
                              t_padding=self.t_padding, include_reward=self.include_reward,
                              include_null_input=self.include_null_input)

    def make_trials(self, seed=None):
        # valuernn draws ITIs from the global np.random (tasks/trial.py:get_itis),
        # so seed it too; restore the caller's global state afterwards.
        if seed is None:
            return super().make_trials(seed=seed)
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            super().make_trials(seed=seed)
        finally:
            np.random.set_state(state)

    def session(self, index):
        X, y, _, trials = self[index]
        return Session(index, X.float()[:, None, :], y.float()[:, None, :], trials)


class SessionSchedule:
    """Iterates the training stream: Session, Break, Session, ..., Session.

    overnight=None puts sessions back to back with no break event, so the
    training loop carries state and windows straight across the boundary.
    """
    def __init__(self, task, overnight=OvernightBreak()):
        self.task = task
        self.overnight = overnight

    def __iter__(self):
        n = len(self.task)
        for i in range(n):
            yield self.task.session(i)
            if self.overnight is not None and i < n - 1:
                yield Break(after_session=i, spec=self.overnight)
