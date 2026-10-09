# CLAUDE.md — Bioplausible_Meta_RL

Standing brief for Claude Code. Read this first; detailed specs are in `docs/`.

## Project in one paragraph

Lee, Hennig, Frelih, Gershman & Uchida (bioRxiv, 10.64898/2025.11.30.691382) show that an RNN trained online with TD + truncated BPTT shifts from plasticity-based to dynamics-based value updating (meta-RL), but only with long windows (W = 720 steps). That is biologically implausible and cannot span the overnight breaks mice experience (~1 reversal/session, expertise in ~12 sessions). This project tests whether a few biologically motivated inductive biases (temporal contiguity, mixed selectivity, multiple memory timescales, separating slow structure from fast state, offline replay) can replace long-horizon credit assignment. Three model families are compared against baselines B1–B4.

## Documents (source of truth)

- `docs/research_plan.md` — research plan: models, baselines, evaluation, phases, risks. Mirrors the project Google Doc; if the user says the doc changed, ask them to re-export it.
- `docs/model_schematics.md` — layer-by-layer equations for Models 1–3. For Model 1 it follows the *revised* spec and overrides `research_plan.md` §4.1 (differences listed at its top).
- `docs/2025.11.30.691382v3.full.pdf` — the Lee et al. paper. RNN Methods on p. 26–27, Extended Data Fig. 2 legend p. 46.

## Current phase

**Phase 1: Model 1.** Model order 1 → 3 → 2 (Model 2 = Model 3's architecture with a Hebbian rule).

Done: session task stream; online TBPTT loop (reproduces valuernn's `train_model_TBPTT` exactly); overnight breaks; Model 1 slow memory (`slowmem_1`); plasticity test (both protocols); train-to-criterion; metrics; B3 ideal observer; config + seed run harness. `python -m pytest tests` (63 pass).

Main result so far: **Model 1 with one-hot trial-type units at W = 1 learns to reverse without plasticity** (paper settings: converges in 16–18 blocks on 5/5 seeds, frozen AUROC 0.90–0.99), where the no-slow-memory network at short W cannot (frozen AUROC 0.00).

Open:
1. Random trial-type units (the spec) fail: z is huge (|z| ≈ 200) and saturates the GRU. Spec's rule has fixed point Wij = E[yi ȳj]/E[yi²] and y is dense. Tested fixes (no RNN): y scaled to unit length kills context information; binary k-WTA (top-k units = 1) fixes the scale but separates contexts much worse than one-hot (0.3–1.8 vs ≈ 3) and varies by seed. Next: larger pool + small k, measure trial-type overlap.
2. Our B1 (W = 720) converges slower than the paper's (4/20 by block 18 vs ≈ 75%; block-1 RPE² 0.065 vs ≈ 0.022). **Accepted for now (user, 2026-10-08)** — whenever a network converges, results match the paper. Matters later for the "sessions to expertise" comparison; likely an ITI or averaging difference we can't resolve without the authors' code.
3. Model 1's inference depends on lr: at lr 0.003 it switches to a plasticity-based solution (frozen 0.85 → 0.17).
4. Break runs need the mouse protocol: each session starts with the previous session's contingency (not implemented; `first_block` currently fixed).
5. Not built: long-ITI probes, B4 readout, Models 2/3 (per-step stepping), replay, e-prop variant.

Run: `python -m BP_models.run --config cfg.json --seeds 0-19 --workers 10 --out results` (`--set key=value` overrides; on PowerShell pass JSON-valued overrides via a config file — it strips inner quotes). `--device cuda` for W = 720.

## Notes on what was tried (condensed)

| What | Outcome |
|---|---|
| First runs: γ 0.9, lr 0.003, TF-style init, 3-step cue, ITI 6 + geom(0.5), 12 sessions with breaks | Superseded by paper settings. Model 1 random: z saturates GRU. Discrimination during training can't tell plasticity from inference → use the frozen test. |
| Frozen test, `reset` protocol (h reset, 4 new blocks) | Stuck networks score ≈ 0.5 (paper ≈ 0); paper uses `continue` (one reversed block, state carried). Both kept. |
| γ 0.8, lr 0.0005 (paper lr), 12 sessions | Model 1 one-hot W=1 frozen 0.97; baselines undertrained. Nothing met convergence: with γ 0.8 the unavoidable cue-onset RPE gives an all-steps floor ≈ 0.023. |
| B1 breaks vs no breaks, 12 and 48 sessions (old timing, 5 seeds) | Suggestive at 12 sessions (3/5 vs 1/5 infer), gone at 48; inconclusive. Redo with current setup and ~20 seeds. |
| Paper task timing, 120 / 14 blocks continuous, `continue` test | B2 frozen = 0.00 (matches paper). Model 1 one-hot W=1 0.85–0.97. Learning-on test needed windows reaching back into training (fixed; exact-continuation test). |
| lr 0.003, stride 5 | Model 1 one-hot W=1 becomes plasticity-based (frozen 0.17). |
| γ 0.2, 20 units (paper), train to RPE² < 0.005, max 40 blocks, stride 5 | Model 1 converges 16–18 blocks, frozen 0.90–0.99. B1/B2 never converge at stride 5 (5× fewer updates); their AUROC ≈ 1 was chance (see below). |
| Block-order bug: `first_block='random'` per 2-block session | ~25% of block boundaries had no reversal in continuous runs (affected the three rows above). Fixed: default `first_block=0` = strict alternation. |
| ED Fig. 2d/e replication (paper settings, stride 1, 18 blocks, 20 seeds) | Qualitative match (W=720 RPE² falls, W=20 flat; W=20 frozen 0.00 all seeds, value gap −0.33; converged W=720 reverse frozen 0.74–0.99). Quantitatively slower (open item 2). |

## Repository layout

```
Bioplausible_Meta_RL/
  CLAUDE.md
  docs/                  # research plan, model schematics, Lee et al. PDF
  valuernn/              # fork of mobeets/valuernn (github.com/StyraxW/valuernn), git submodule
  BP_models/             # our package: shared infrastructure + all models
    tasks/trials.py      # CueOffsetTrial (k-step cue, outcome next step); cue_offset()
    tasks/sessions.py    # SessionTask, SessionSchedule, OvernightBreak
    timing.py            # λ = exp(−dt/τ), overnight decay, seconds → steps, log-spaced τ
    slowmem_1.py         # Model 1 slow memory: trial-type layer + Hebbian binding -> z
    train/online.py      # OnlineTrainer: run (sessions + breaks), run_continuous (stop early)
    analysis/metrics.py  # trial table, discrimination, reversal τ, convergence loss
    analysis/evaluate.py # plasticity test ('continue' / 'reset'), AUROC
    baselines/ideal_observer.py  # B3
    baselines/b4.py      # TraceBank (B4 readout not built yet)
    config.py            # RunConfig (task/overnight/model/train/test), derived seeds
    run.py               # CLI harness, SLOWMEMS registry
  tests/                 # pytest, run from repo root
  results/               # run outputs (git-ignored)
```

`BP_models` imports the fork's modules as top-level names (`tasks.inference`, `model`, `train_bptt`); `BP_models/__init__.py` puts `valuernn/` on `sys.path`. Run code and tests from the repo root.

## Current defaults (`config.py`) = Lee et al. settings

- **Core:** valuernn `ValueRNN`, GRU, **20 units**, PyTorch default init (`init_gain=0`), V = wᵀh + b0. `model.cell='RNN'` gives the spec's vanilla RNN.
- **Learning:** semi-gradient TD(0), **γ = 0.2** (paper Methods; `main_jaeeon.py` has 0.8), Adam amsgrad **lr 0.0005**, sliding window W, stride 1 (paper: every step). **"W = 1" = `window_size=2`** (one TD transition; stride must be < window).
- **Task (paper code):** cue 1 step, outcome on the next step, ITI 5 + geometric(0.8) − 1 steps (trial ≈ 7.25 steps; W = 720 ≈ 100 trials), 50-trial blocks, anti-correlated deterministic rewards, `first_block=0` (strict alternation in a continuous stream). dt = 0.5 s (only matters for slow-memory time constants and breaks).
- **Paper training:** 18 blocks continuous (`task.nsessions=9`, `overnight.enabled=false`). Train-to-criterion: `train.until_converged` — stop when mean RPE² (all steps, last 20 trials of each of the last 4 blocks) < 0.005 (ED Fig. 2; the Methods text's 0.0005 looks like a typo), max `train.max_blocks` = 40.
- **Plasticity test** (`test.protocol`): `'continue'` (paper, default) — stream continues into one reversed block; frozen = learning rate 0, windows keep recomputing; learning on = windows reach back into training (exact continuation). `'reset'` — h and slow traces reset, 4 new blocks. Reports frozen, learning-on, and (Model 1) RNN-only-frozen AUROC of cue-onset value, CS+ vs CS−, plus per-trial values for ROC plots. **Only interpret AUROC for converged networks**: an unlearned network with a tiny fixed cue bias scores 0 or 1 on one block by chance; check the value gap too.
- **Model 1 (`slowmem_1`):** τe 1.5 s, τb 15 s, ηb 0.05; `trial_types='onehot'` (4 units) or `'random'` (50 units, spec). Wb frozen in the frozen test.

## Findings about valuernn (from inspection)

- `train_model_TBPTT` slides a window ending at every `stride`-th step; each call runs `nn.GRU` over the window (cuDNN), loss = mean TD error over the window, one Adam step. h0 for the next window = `hs[stride-1].detach()` (pre-update weights); h0 = 0 each episode.
- Target `y[t+1] + γ·V(t+1).detach()` = spec's δ shifted by one index.
- Task classes return CPU tensors; `quick_train.py` crashes on GPU. `get_itis` uses the global `np.random` (`SessionTask` seeds it and restores the state).

## Design decisions (agreed)

- **valuernn fork stays untouched** except compatibility fixes (ask first). New code in `BP_models/`, reuse by subclassing.
- **Model 1 = front end + existing core:** core input `[x, z]` (Wx and Wz are blocks of one matrix); z enters detached (TD trains Wz, never Wb).
- **Slow memory is separate per model** (`slowmem_1.py`, `slowmem_2.py`, `slowmem_3.py`); the loop has a model-specific branch. Slow state advances **once per real step**, never per window recomputation. x-driven memories (SlowMem1, TraceBank) expose `advance(X)`; z precomputed per session.
- **Overnight break = separate stream event, not a long ITI:** no learning, window cut, traces decay by exp(−T/τ), weights persist. h policy (default `reset` = h = 0.1·N(0, I), seeded from its own seed; `relax` = silent input for the full duration, chunked ≤16384 steps on GPU (cuDNN limit); `carry`). Long-ITI probes are in-session real silent steps.
- **Seeds:** run seed → independent task / model / break / slow-memory / test seeds (`config.derive_seeds`).
- **Recording is online:** V(t), h(t) from the first window where t is newest.

## Environment

- Windows 11, PowerShell, VS Code. Conda env **`MetaRL`** (Python 3.12) at `C:\Users\wangx\.conda\envs\MetaRL\python.exe`; conda not on PATH.
- PyTorch **2.14.1+cu130** on an **RTX 5070 Ti** (Blackwell, sm_120). Don't install CUDA 12.x builds.
- numpy/scipy use **OpenBLAS** (MKL's `libiomp5md.dll` clashed with torch's). Don't reinstall MKL numpy.
- Install with `python -m pip ...`. Torch from `--index-url https://download.pytorch.org/whl/cu130`.
- Device-agnostic code. GPU for W = 720; short windows as single-thread CPU processes (`--workers`), seeds in parallel.
- Evaluation needs ≥ 20 seeds per model.

## Conventions

- Every run is reproducible from a config + seed; configs and git commits are logged with results.
- Time constants in seconds, converted with dt: `λ = exp(−dt/τ)`.
- Report all tasks/conditions tried, including failures (research_plan §6.5).
