"""Run configuration. A run is fully determined by a RunConfig plus a seed.

Times are in seconds and converted with dt; ITIs stay in steps, as in valuernn.
From the run seed, independent seeds are derived for the task, the model
initialization, the break noise and the slow memory's fixed random weights,
so changing one part (e.g. h_policy) does not change the others.
"""
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass

import numpy as np

from .timing import to_steps


@dataclass
class TaskConfig:
    nsessions: int = 12
    ntrials_per_block: int = 50
    ntrials_per_block_jitter: int = 0
    iti_min: int = 5            # steps; Lee et al. code (main_jaeeon.py: valuernn default 5)
    iti_p: float = 0.8          # geometric ITI parameter; Lee et al. code -> mean ITI ≈ 5.25 steps
    cue_duration: float = 0.5   # seconds; 1 step at dt = 0.5, as in Lee et al. code (cue pulse,
                                # outcome on the next step); trial ≈ 7.25 steps, W = 720 ≈ 100 trials
    # first block of every session. A fixed block gives strict alternation in a
    # continuous stream (2 blocks per session): a reversal every 50 trials, as in the
    # paper. 'random' draws it per session, so ~half the session boundaries have no
    # reversal (the bug in rounds 6-9). The mouse protocol with breaks (session starts
    # with the previous session's contingency) is not implemented yet.
    first_block: int | str = 0


@dataclass
class BreakConfig:
    enabled: bool = True
    duration: float = 16 * 3600.0
    h_policy: str = 'reset'
    reset_sigma: float = 0.1
    relax_steps: int | None = None
    replay: bool = False


@dataclass
class ModelConfig:
    cell: str = 'GRU'
    hidden_size: int = 20       # Lee et al. Methods (GRU, N = 20)
    gamma: float = 0.2          # Lee et al. Methods ("γ was set to γ=0.2"; main_jaeeon.py has 0.8)
    init_gain: float = 0.0      # 0 = PyTorch default init (Lee et al. Methods); >0 = valuernn TF-style
    slowmem: str = 'none'       # key in BP_models.run.SLOWMEMS: 'none' (B1/B2), 'slowmem_1', 'trace_bank'
    slowmem_kwargs: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    window_size: int = 720      # steps in a window; 2 = one TD transition (1-step BPTT, "W = 1")
    stride: int = 1
    lr: float = 0.0005          # Lee et al. Methods (Adam, amsgrad)
    record_hidden: bool = False
    # train to criterion (continuous stream only): after each session, stop once the
    # mean RPE² over the last 20 trials of each of the last 4 blocks is below
    # conv_threshold; task.nsessions is ignored, at most max_blocks blocks
    until_converged: bool = False
    conv_threshold: float = 0.005   # Lee et al. ED Fig. 2a-b (Methods text says 0.0005)
    conv_exclude_cue: bool = False  # all steps, as in the paper; see analysis.metrics.convergence_loss
    max_blocks: int = 40


@dataclass
class TestConfig:
    """Plasticity test after training (see analysis/evaluate.py).

    protocol 'continue' (Lee et al.): state carried over, the next block (reversed).
    protocol 'reset': h and slow traces reset, a new episode of alternating blocks.
    nblocks None = the protocol's default (continue: 1, reset: 4).
    """
    enabled: bool = True
    protocol: str = 'continue'
    nblocks: int | None = None

    def n_blocks(self):
        if self.nblocks is not None:
            return self.nblocks
        return 1 if self.protocol == 'continue' else 4


@dataclass
class RunConfig:
    name: str = 'run'
    dt: float = 0.5
    device: str = 'cpu'
    task: TaskConfig = field(default_factory=TaskConfig)
    overnight: BreakConfig = field(default_factory=BreakConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    @property
    def cue_steps(self):
        return to_steps(self.task.cue_duration, self.dt)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)


def _from_dict(cls, d):
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f'unknown config key {cls.__name__}.{k}')
        sub = names[k].default_factory
        if isinstance(v, dict) and sub is not MISSING and is_dataclass(sub):
            v = _from_dict(sub, v)
        kwargs[k] = v
    return cls(**kwargs)


def set_override(cfg, dotted, value):
    """Set e.g. 'train.window_size' on a RunConfig in place."""
    *path, last = dotted.split('.')
    obj = cfg
    for p in path:
        obj = getattr(obj, p)
    if not hasattr(obj, last):
        raise KeyError(f'unknown config key {dotted}')
    setattr(obj, last, value)


def derive_seeds(seed):
    """Independent integer seeds for task, model init, break noise, slow memory
    and the test episode.

    SeedSequence children depend only on their index, so adding a child later
    leaves the earlier seeds unchanged.
    """
    children = np.random.SeedSequence(seed).spawn(5)
    task, model, brk, slowmem, test = (int(c.generate_state(1)[0]) for c in children)
    return dict(task=task, model=model, breaks=brk, slowmem=slowmem, test=test)
