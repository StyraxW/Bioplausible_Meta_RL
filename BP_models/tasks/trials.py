"""Trials whose outcome comes on the step right after a k-step cue.

Reuses valuernn's trace-conditioning Trial (cue as a one-step pulse at row
`iti`, reward at row `iti + isi`) and extends the cue to `cue_steps` rows.
With isi = cue_steps, rows are:

    0 .. iti-1                 ITI (silent)
    iti .. iti+cue_steps-1     cue
    iti+cue_steps              outcome (reward or omission); cue is off

so the cue-offset pulse off(t) = max(0, c(t-1) - c(t)) fires exactly on the
outcome row, on rewarded and omitted trials alike.
"""
import numpy as np
from tasks.trial import Trial


def cue_offset(c):
    """off(t) = max(0, c(t-1) - c(t)) per cue channel, with c(-1) = 0.

    c: array (T, ncues). Returns (T,): 1 on the step after any cue turns off.
    """
    c = np.asarray(c, dtype=float)
    prev = np.vstack([np.zeros((1, c.shape[1])), c[:-1]])
    return np.maximum(0.0, prev - c).sum(axis=1)


class CueOffsetTrial(Trial):
    def __init__(self, cue, iti, reward_size, ncues, cue_steps=2,
                 t_padding=0, include_reward=True, include_null_input=False):
        if cue_steps < 1:
            raise ValueError(f'{cue_steps=} must be >= 1')
        self.cue_steps = cue_steps  # set before super().__init__, which calls make()
        super().__init__(cue, iti, cue_steps, reward_size, True, ncues,
                         t_padding=t_padding, include_reward=include_reward,
                         include_null_input=include_null_input,
                         do_trace_conditioning=True)

    def make(self):
        super().make()  # trace mode: cue at row iti only
        cue_rows = slice(self.iti, self.iti + self.cue_steps)
        self.trial[cue_rows, self.cue] = 1.0
        self.X[cue_rows, self.cue] = 1.0
        if self.include_null_input:
            self.X[cue_rows, -1] = 0.0  # null input is off while the cue is on
            assert (self.X.sum(axis=1) == 1).all()

    @property
    def outcome_index(self):
        """Row of the outcome step (= cue offset) within the trial."""
        return self.iti + self.cue_steps
