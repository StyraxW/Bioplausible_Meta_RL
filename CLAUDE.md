# CLAUDE.md — Bioplausible_Meta_RL

Standing brief for Claude Code. Read this first; detailed specs are in `docs/`.

## Project in one paragraph

Lee, Hennig, Frelih, Gershman & Uchida (bioRxiv, 10.64898/2025.11.30.691382) show that an RNN trained online with TD + truncated BPTT shifts from plasticity-based to dynamics-based value updating (meta-RL), but only with long windows (W = 720 steps). That is biologically implausible and cannot span the overnight breaks mice experience (~1 reversal/session, expertise in ~12 sessions). This project tests whether a few biologically motivated inductive biases (temporal contiguity, mixed selectivity, multiple memory timescales, separating slow structure from fast state, offline replay) can replace long-horizon credit assignment. Three model families are compared against baselines B1–B4.

## Documents (source of truth)

- `docs/research_plan.md` — full research plan: models, baselines, evaluation, phases, risks. Mirrors the project Google Doc; if the user says the doc changed, ask them to re-export it.
- `docs/model_schematics.md` — layer-by-layer equations for Models 1–3.
- **Precedence:** for Model 1, `model_schematics.md` follows the *revised* spec (cue-offset gate, jump-form event trace, explicit Oja decay) and overrides `research_plan.md` §4.1 where they differ. Differences are listed at the top of `model_schematics.md`.

## Current phase

**Phase 1: Model 1.** Model order: 1 → 3 → 2 (Model 2 = Model 3's architecture with a Hebbian rule). B1/B2 are already characterized in Lee et al.; the user decided not to rerun them now.

Done (infrastructure): session task stream; online TBPTT loop that reproduces `train_model_TBPTT` exactly (tests); overnight breaks; feedforward slow-memory interface + trace bank; metrics; B3 ideal observer; config + seed run harness. 34 tests pass (`python -m pytest tests`).

Next:
1. Model 1 front end (`BP_models/layers/`): trial-type layer (e, off, R, y), binding layer (ȳ, Wb Hebbian + Oja, z) as a `SlowMemory`; unit-test the Wb fixed point before connecting it.
2. Model 1 end to end: register in `run.MEMORIES`, compare with B2 (same loop, short W).
3. Still missing, add when needed: long-ITI probes, B4, feedback memory stepping (Models 2/3), replay.

Run: `python -m BP_models.run --seeds 0-19 --workers 10 --set train.window_size=20 name='"b2_w20"'` (results in `results/<name>/seed_<k>/`). Use `--device cuda` for W = 720.

## Repository layout

```
Bioplausible_Meta_RL/
  CLAUDE.md
  docs/
    research_plan.md
    model_schematics.md
  valuernn/              # fork of mobeets/valuernn (github.com/StyraxW/valuernn), git submodule
  BP_models/             # our package: shared infrastructure + all models
    tasks/trials.py      # CueOffsetTrial: k-step cue, outcome on the next step; cue_offset()
    tasks/sessions.py    # SessionTask, SessionSchedule, OvernightBreak
    timing.py            # λ = exp(−dt/τ), overnight decay, seconds → steps
    memory.py            # SlowMemory interface (reset/run/overnight), TraceBank
    train/online.py      # OnlineTrainer: sliding-window TBPTT over sessions + breaks
    analysis/metrics.py  # trial table, discrimination, reversal τ
    baselines/ideal_observer.py  # B3
    config.py            # RunConfig (task/overnight/model/train), derived seeds
    run.py               # CLI harness, MEMORIES registry
    layers/, models/     # Model 1–3 components (planned)
  tests/                 # pytest, run from repo root
  results/               # run outputs (git-ignored)
```

`BP_models` imports the fork's modules as top-level names (`tasks.inference`, `model`, `train_bptt`); `BP_models/__init__.py` puts `valuernn/` on `sys.path`. Run code and tests from the repo root.

## Findings about valuernn (from inspection)

- `train_model_TBPTT` slides a window ending at every step (stride 1): each call runs `nn.GRU` over the whole window (cuDNN), loss = mean TD error over the window, one Adam step. Each transition is trained ~W times; cost ∝ W per step.
- h is detached at `train_bptt.py:120` (`hs[stride-1].detach()`), i.e. the state one stride into the window, computed with pre-update weights; h0 resets to zeros each episode.
- Core is `nn.GRU` by default; `recurrent_cell='RNN'` gives the spec's vanilla tanh RNN with no fork edits.
- `V = wᵀh + b0`; target `y[t+1] + γ·V(t+1).detach()` (semi-gradient TD(0)) = spec's δ shifted by one index. γ is not fixed in the code (0.8–0.93 in scripts).
- No `.cuda()`, but task classes always return CPU tensors and `quick_train.py` crashes on GPU; move batches to the device in our code.
- `get_itis` uses the global `np.random`, so `ValueInference(seed=)` does not fix ITIs. `SessionTask` seeds it (state saved and restored).

## Design decisions (already agreed)

- **The valuernn fork stays untouched** except for compatibility/bug fixes. It provides baselines B1 (W = 720) and B2 (short window, no slow memory); editing its internals would break those comparisons. New code goes in `BP_models/` and imports from `valuernn`.
- **Reuse** valuernn's task generation and TBPTT semantics; extend by subclassing, don't rewrite.
- **Model 1 = front end + existing core.** The trial-type and binding layers turn x(t) into z(t). Feed the core RNN `[x, z]` as its input, so Wx and Wz are two blocks of one input matrix. Model 1 vs B2 then differs only by the front end.
- **Hebbian state lives outside autograd.** `e`, `ȳ`, `Wb` are registered buffers updated under `torch.no_grad()`. They persist across TBPTT cuts and across sessions (overnight: traces decay analytically by `exp(−T/τ)`; `Wb` persists). `z` enters the RNN **detached**: gradients train Wz, never Wb.
- Slow state (Hebbian, traces, eligibility) advances **once per real timestep**, never once per window recomputation. For Model 1 the front end does not depend on h, so z can be computed in one streaming per-step pass and the windows slice `[x, z]` (no Wb lag).
- **Core: GRU** (comparable to B1/B2). Vanilla RNN available as a variant (`model.cell='RNN'`).
- **dt = 0.5 s** (configurable). Cue = 1.5 s = 3 steps. ITIs stay in steps (valuernn: `iti_min=6`, `iti_p=0.5`). Sessions ≈ 1,100 steps; W = 720 gives ~380 windows per session vs ~1,100 for short W (fewer updates for B1 under session cuts).
- **γ = 0.9 is PROVISIONAL** (from `jaeeon_small.py`); bioRxiv was unreachable. Confirm from the paper's Methods.
- **Seeds:** run seed → independent task / model-init / break-noise seeds (`config.derive_seeds`).
- **Recording is online:** V(t), h(t) from the first window where t is newest; δ(t) from the window that first contains t+1.
- **Reversal τ:** per-session fits use one point per trial and are often nan; use `tau_pooled` (groups of 4 sessions). τ ≈ 0.22 (`TAU_STEP`) means "switch within one trial".

## Task stream (shared by all models)

- Inputs `x(t) = [cA(t), cB(t), r(t)]`; trial = ITI → cue (`task.cue_duration` = 1.5 s, the paper's odor; 3 steps at dt = 0.5) → outcome on the step right after cue offset.
- `off(t) = max(0, c(t−1) − c(t))`, the cue-offset pulse, computed from c. It marks the outcome step on every trial (rewarded or omitted).
- Sessions: two blocks, one reversal. ~12 sessions.
- **Overnight break = a separate stream event, not a long ITI.** No TD learning during it; TBPTT window cut; traces decay analytically by `exp(−T/τ)`; weights persist; RNN state (default `reset`): `relax` (silent steps for the full duration, `round(T/dt)` ≈ 76,800 at 16 h / 0.75 s, no learning; log the final ‖Δh‖ per step and the drift after step 300; cost ~1 s CPU, ~0.03 s GPU; on GPU run it in chunks of ≤16384 steps, since cuDNN rejects a single sequence somewhere between 32,768 and 65,536 steps), `reset` (new random low-power state h = σ·N(0, I), σ = 0.1, seeded; not a fixed point — silent-input dynamics need not have an attracting one), or `carry` — log which. Breaks are optional (`overnight=None` → sessions back to back, no cut). Optional replay during breaks is a separate condition.
- Long-ITI-break probes (37.5–300 s) are in-session: real silent steps, learning on.

## Environment

- Windows 11, PowerShell, VS Code. Conda env **`MetaRL`**, Python **3.12**.
- PyTorch **2.14.1+cu130** on an **RTX 5070 Ti** (16 GB, Blackwell, sm_120). Do not install CUDA 12.x builds; they lack Blackwell kernels.
- numpy/scipy use **OpenBLAS** (switched from MKL: MKL's `libiomp5md.dll` clashed with torch's). Don't reinstall MKL numpy.
- Install with `python -m pip ...` (never bare `pip`). Torch came from `--index-url https://download.pytorch.org/whl/cu130`.
- Code must be **device-agnostic** (`device` argument, no hard-coded `.cuda()`).
- Measured (H = 50, ms per TBPTT window): GPU 3.2 (W = 50) / 3.5 (W = 720); CPU 1 thread ~5.5 / ~55. Use the GPU for B1; run short-window models as single-thread CPU processes (`torch.set_num_threads(1)`), seeds in parallel.
- Evaluation needs ≥20 seeds per model: design runs around a seed argument from the start.

## Conventions

- Every run is reproducible from a config + seed. Log hyperparameters with results.
- Keep time constants in seconds and convert with dt: `λ = exp(−dt/τ)`.
- Report all tasks/conditions tried, including failures (fairness rule, research_plan §6.5).
- Ask before changing anything in `valuernn/` beyond compatibility fixes.
