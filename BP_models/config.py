"""Run configuration. A run is fully determined by a RunConfig plus a seed.

Times are in seconds and converted with dt; ITIs stay in steps, as in valuernn.
From the run seed, independent seeds are derived for the task, the model
initialization and the break noise, so changing one part (e.g. h_policy)
does not change the trials or the initial weights.
"""
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass

import numpy as np

from .timing import to_steps


@dataclass
class TaskConfig:
    nsessions: int = 12
    ntrials_per_block: int = 50
    ntrials_per_block_jitter: int = 0
    iti_min: int = 6            # steps
    iti_p: float = 0.5          # geometric ITI parameter (valuernn)
    cue_duration: float = 1.5   # seconds (paper's odor duration); 3 steps at dt = 0.5
    first_block: str = 'random'


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
    hidden_size: int = 50
    gamma: float = 0.9          # PROVISIONAL: confirm from Lee et al. Methods
    init_gain: float = 1.0
    memory: str = 'none'        # key in BP_models.run.MEMORIES
    memory_kwargs: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    window_size: int = 720
    stride: int = 1
    lr: float = 0.003
    record_hidden: bool = False


@dataclass
class RunConfig:
    name: str = 'run'
    dt: float = 0.5
    device: str = 'cpu'
    task: TaskConfig = field(default_factory=TaskConfig)
    overnight: BreakConfig = field(default_factory=BreakConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

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
    """Independent integer seeds for task, model init and break noise."""
    children = np.random.SeedSequence(seed).spawn(3)
    task, model, brk = (int(c.generate_state(1)[0]) for c in children)
    return dict(task=task, model=model, breaks=brk)
