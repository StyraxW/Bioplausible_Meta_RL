"""Run harness: config + seed -> results directory.

    python -m BP_models.run --seeds 0-19 --workers 10 --set train.window_size=20
    python -m BP_models.run --config my_cfg.json --seeds 3 --device cuda

Writes <out>/<name>/seed_<k>/ with config.json (config, seeds, versions),
metrics.json, arrays.npz (per-step V/delta, per-window loss, trial table) and
weights.pt. CPU runs use one torch thread per process (measured fastest for
these small RNNs); parallelize over seeds with --workers.
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from . import VALUERNN_DIR
from .analysis.evaluate import evaluate_trained
from .analysis.metrics import CONVERGENCE_THRESHOLDS, convergence_loss, summarize, trial_table
from .baselines.ideal_observer import ideal_observer_values
from .config import RunConfig, derive_seeds, set_override
from .baselines.b4 import TraceBank
from .slowmem_1 import SlowMem1
from .tasks.sessions import OvernightBreak, SessionSchedule, SessionTask
from .train.online import OnlineTrainer
from model import ValueRNN  # valuernn (path set up by BP_models/__init__.py)

REPO_DIR = os.path.dirname(VALUERNN_DIR)
N_X = 3  # x = [cA, cB, r]

# model.slowmem -> constructor(cfg, seed, **model.slowmem_kwargs)
SLOWMEMS = {
    'none': None,
    'slowmem_1': lambda cfg, seed, **kw: SlowMem1(cfg.dt, seed=seed, **kw),
    'trace_bank': lambda cfg, seed, **kw: TraceBank(N_X, dt=cfg.dt, **kw),
}


def build(cfg, seeds):
    t = cfg.task
    # train to criterion: generate the longest possible stream; training stops early
    nsessions = cfg.train.max_blocks // 2 if cfg.train.until_converged else t.nsessions
    task = SessionTask(nsessions=nsessions, cue_steps=cfg.cue_steps,
                       ntrials_per_block=t.ntrials_per_block,
                       ntrials_per_block_jitter=t.ntrials_per_block_jitter,
                       iti_min=t.iti_min, iti_p=t.iti_p, first_block=t.first_block,
                       seed=seeds['task'])
    o = cfg.overnight
    overnight = (OvernightBreak(duration=o.duration, h_policy=o.h_policy, reset_sigma=o.reset_sigma,
                                relax_steps=o.relax_steps, replay=o.replay)
                 if o.enabled else None)
    schedule = SessionSchedule(task, overnight=overnight)

    make_slowmem = SLOWMEMS[cfg.model.slowmem]
    slowmem = (make_slowmem(cfg, seeds['slowmem'], **cfg.model.slowmem_kwargs)
               if make_slowmem else None)
    n_z = slowmem.n_out if slowmem is not None else 0
    # core input [x, z]: Wx and Wz are two blocks of one input matrix
    model = ValueRNN(input_size=N_X + n_z, output_size=1, hidden_size=cfg.model.hidden_size,
                     gamma=cfg.model.gamma, recurrent_cell=cfg.model.cell)
    model.reset(seed=seeds['model'], initialization_gain=cfg.model.init_gain)
    return task, schedule, model, slowmem


def _git_commit(path):
    try:
        rev = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', path, 'status', '--porcelain'], capture_output=True,
                               text=True, check=True).stdout.strip() != ''
        return rev + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def run_one(cfg_dict, seed, out_root, threads=1, verbose=True):
    cfg = RunConfig.from_dict(cfg_dict)
    if threads:
        torch.set_num_threads(threads)
    out_dir = os.path.join(out_root, cfg.name, f'seed_{seed:03d}')
    os.makedirs(out_dir, exist_ok=True)
    seeds = derive_seeds(seed)
    log_file = open(os.path.join(out_dir, 'log.txt'), 'w')

    def log(msg):
        log_file.write(msg + '\n')
        log_file.flush()
        if verbose:
            print(f'[{cfg.name} seed {seed}] {msg}', flush=True)

    meta = dict(config=cfg.to_dict(), seed=seed, derived_seeds=seeds, cue_steps=cfg.cue_steps,
                started=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.version.split()[0],
                torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                repo_commit=_git_commit(REPO_DIR), valuernn_commit=_git_commit(VALUERNN_DIR),
                torch_threads=torch.get_num_threads())
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    task, schedule, model, slowmem = build(cfg, seeds)
    trainer = OnlineTrainer(model, slowmem=slowmem, window_size=cfg.train.window_size,
                            stride=cfg.train.stride, lr=cfg.train.lr, dt=cfg.dt,
                            device=cfg.device, break_seed=seeds['breaks'],
                            record_hidden=cfg.train.record_hidden, log=log)
    probs = task.reward_probs_per_block
    if cfg.train.until_converged:
        if cfg.overnight.enabled:
            raise ValueError('train.until_converged needs a continuous stream (overnight.enabled=false)')

        def stop_fn(done, offsets, delta):
            if len(done) < 2:  # the criterion needs 4 blocks = 2 sessions
                return False
            last = trial_table(done[-2:], probs, offsets[-2:])
            loss = convergence_loss(last, delta, exclude_cue=cfg.train.conv_exclude_cue)
            log(f'after {2 * len(done)} blocks: convergence loss {loss:.5f}')
            return loss < cfg.train.conv_threshold

        res = trainer.run_continuous((task.session(i) for i in range(len(task))), stop_fn)
    else:
        res = trainer.run(schedule)
    table = trial_table(res['sessions'], probs, res['session_offsets'], res['V'])
    io_table = trial_table(res['sessions'], probs)
    io_table['v_cue'] = ideal_observer_values(res['sessions'], probs,
                                              hazard=1.0 / cfg.task.ntrials_per_block)
    session_loss = [float(res['window_loss'][res['window_session'] == s].mean())
                    for s in np.unique(res['window_session'])]
    conv = convergence_loss(table, res['delta'])
    conv_nc = convergence_loss(table, res['delta'], exclude_cue=True)
    metrics = dict(model=summarize(table), ideal_observer=summarize(io_table),
                   blocks_trained=2 * len(res['sessions']),
                   stopped_at_criterion=bool(res.get('stopped_early', False)),
                   convergence_loss=conv, convergence_loss_no_cue=conv_nc,
                   converged={str(th): conv < th for th in CONVERGENCE_THRESHOLDS},
                   converged_no_cue={str(th): conv_nc < th for th in CONVERGENCE_THRESHOLDS},
                   session_mean_loss=session_loss, breaks=res['breaks'],
                   runtime_s=res['runtime_s'])

    if cfg.test.enabled:
        t, p = cfg.task, cfg.test.protocol
        # 'continue': the stream goes on into the next block, the reverse of the last
        # training block; 'reset': a new episode starting in a random block
        last_block = res['sessions'][-1].trials[-1].block_index
        first = 1 - last_block if p == 'continue' else t.first_block
        test_task = SessionTask(nsessions=1, blocks_per_session=cfg.test.n_blocks(),
                                cue_steps=cfg.cue_steps, ntrials_per_block=t.ntrials_per_block,
                                ntrials_per_block_jitter=t.ntrials_per_block_jitter,
                                iti_min=t.iti_min, iti_p=t.iti_p, first_block=first,
                                seed=seeds['test'])
        metrics['test'] = evaluate_trained(model, slowmem, trainer.optimizer, trainer.final_h,
                                           test_task, cfg.train.window_size, cfg.train.stride,
                                           cfg.train.lr, cfg.dt, cfg.device, protocol=p,
                                           context=trainer.tail)
        log('test AUROC: ' + ', '.join(f'{k} {v["auroc"]:.3f}' for k, v in metrics['test'].items()
                                       if isinstance(v, dict)))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)

    arrays = {k: res[k] for k in ('V', 'delta', 'step_session', 'window_loss', 'window_session',
                                  'session_offsets')}
    for k in ('hidden', 'Wb'):
        if k in res:
            arrays[k] = res[k]
    arrays.update({f'trial_{k}': v for k, v in table.items()})
    arrays['trial_v_ideal_observer'] = io_table['v_cue']
    np.savez_compressed(os.path.join(out_dir, 'arrays.npz'), **arrays)
    torch.save(dict(model=model.state_dict(),
                    slowmem=slowmem.state_dict() if slowmem is not None else None),
               os.path.join(out_dir, 'weights.pt'))
    log(f'done in {res["runtime_s"]:.1f}s -> {out_dir}')
    log_file.close()
    return out_dir


def _parse_seeds(tokens):
    seeds = []
    for tok in tokens:
        if '-' in tok:
            a, b = tok.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(tok))
    return seeds


def _parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', help='JSON file with (part of) a RunConfig')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                   help='overrides, e.g. train.window_size=20 overnight.h_policy=relax')
    p.add_argument('--seeds', nargs='+', default=['0'], help='e.g. 0 1 2 or 0-19')
    p.add_argument('--workers', type=int, default=1, help='parallel processes (one seed each)')
    p.add_argument('--threads', type=int, default=1, help='torch threads per process')
    p.add_argument('--device', help='override config device (cpu, cuda)')
    p.add_argument('--out', default='results')
    args = p.parse_args(argv)

    cfg = RunConfig()
    if args.config:
        with open(args.config) as f:
            cfg = RunConfig.from_dict(json.load(f))
    for item in args.set:
        k, v = item.split('=', 1)
        set_override(cfg, k, _parse_value(v))
    if args.device:
        cfg.device = args.device
    cfg_dict = cfg.to_dict()
    seeds = _parse_seeds(args.seeds)

    if args.workers <= 1 or len(seeds) == 1:
        for s in seeds:
            run_one(cfg_dict, s, args.out, threads=args.threads)
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
        futures = {ex.submit(run_one, cfg_dict, s, args.out, args.threads, False): s for s in seeds}
        for fut in futures:
            print(f'seed {futures[fut]} -> {fut.result()}', flush=True)


if __name__ == '__main__':
    main()
