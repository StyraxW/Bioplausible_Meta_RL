# CLAUDE.md — Bioplausible_Meta_RL

Standing brief for Claude Code. Read this first; detailed specs are in `docs/`.

## Project in one paragraph

Lee, Hennig, Frelih, Gershman & Uchida (bioRxiv, 10.64898/2025.11.30.691382) show that an RNN trained online with TD + truncated BPTT shifts from plasticity-based to dynamics-based value updating (meta-RL), but only with long windows (W = 720 steps). That is biologically implausible and cannot span the overnight breaks mice experience (~1 reversal/session, expertise in ~12 sessions). This project tests whether a few biologically motivated inductive biases (temporal contiguity, mixed selectivity, multiple memory timescales, separating slow structure from fast state, offline replay) can replace long-horizon credit assignment. Three model families are compared against baselines B1–B4.

## Documents (source of truth)

- `docs/research_plan.md` — full research plan: models, baselines, evaluation, phases, risks. Mirrors the project Google Doc; if the user says the doc changed, ask them to re-export it.
- `docs/model_schematics.md` — layer-by-layer equations for Models 1–3.
- **Precedence:** for Model 1, `model_schematics.md` follows the *revised* spec (cue-offset gate, jump-form event trace, explicit Oja decay) and overrides `research_plan.md` §4.1 where they differ. Differences are listed at the top of `model_schematics.md`.

## Current phase

**Phase 0 → Phase 1.** Infrastructure, then Model 1 (trial-type layer + Hebbian binding layer + core RNN).

Immediate next steps, in order:
1. Run `valuernn/quick_train.py` (or `example.py`) **unchanged**. Fix only Python 3.12 / numpy compatibility breakages, as separate commits in the fork.
2. Time one run on CPU and one on GPU. Report both.
3. Inspect `valuernn/train_bptt.py` and `valuernn/model.py` and report, before writing Model 1:
   - Does training step the RNN one timestep at a time, or run a whole window per call (e.g. `nn.GRU` over a sequence)?
   - Where and how is hidden state detached between truncation windows?
   - Which cell is the core (GRU vs vanilla RNN)? Spec writes vanilla: `h = f(Wh·h + Wx·x + Wz·z + b)`.
   - How V and δ are computed; does δ match `δ(t) = r(t) + γ·V(t) − V(t−1)`? What γ?
   - Does the code hard-code `.cuda()`?
4. Build the session-level task wrapper (Section "Task stream" below).
5. Implement Model 1 front end, then Model 1 end to end.

## Repository layout

```
Bioplausible_Meta_RL/
  CLAUDE.md
  docs/
    research_plan.md
    model_schematics.md
  valuernn/              # fork of mobeets/valuernn (github.com/StyraxW/valuernn)
  slowmem/               # our package (pip install -e .)
    tasks/sessions.py    # sessions, reversals, overnight breaks, long-ITI probes; wraps valuernn tasks
    layers/trial_type.py # cue trace e(t), off(t), fixed expansion R, y(t)
    layers/binding.py    # event trace ȳ(t), Wb Hebbian + Oja update, z(t)
    models/model1.py     # front end + core RNN
    train/online.py      # thin wrapper around valuernn's TBPTT step
  experiments/
    phase0_reproduce.py
    phase1_model1.py
  environment.yml
  requirements.txt
```

Create `slowmem/` and `experiments/` as needed; this layout is the plan, not yet built.

## Design decisions (already agreed)

- **The valuernn fork stays untouched** except for compatibility/bug fixes. It provides baselines B1 (W = 720) and B2 (short window, no slow memory); editing its internals would break those comparisons. New code goes in `slowmem/` and imports from `valuernn`.
- **Reuse** valuernn's task generation and TBPTT training; extend, don't rewrite.
- **Model 1 = front end + existing core.** The trial-type and binding layers turn x(t) into z(t). Feed the core RNN `[x, z]` as its input, so Wx and Wz are two blocks of one input matrix. Model 1 vs B2 then differs only by the front end.
- **Hebbian state lives outside autograd.** `e`, `ȳ`, `Wb` are registered buffers updated under `torch.no_grad()`. They persist across TBPTT cuts and across sessions (overnight: traces decay analytically by `exp(−T/τ)`; `Wb` persists). `z` enters the RNN **detached**: gradients train Wz, never Wb.
- The front end does not depend on h, so y, ȳ, z can be precomputed per window before the RNN pass (Wb then lags by at most one window). Per-step computation is also fine. Choose based on what `train_bptt.py` does; state the choice.
- If the valuernn core is a GRU, flag it: decide with the user whether Model 1 keeps the GRU (comparable to B1) or uses the spec's vanilla RNN.

## Task stream (shared by all models)

- Inputs `x(t) = [cA(t), cB(t), r(t)]`; trial = cue → ITI → outcome; **outcome follows cue offset directly, no delay.**
- `off(t) = max(0, c(t−1) − c(t))`, the cue-offset pulse, computed from c. It marks the outcome step on every trial (rewarded or omitted).
- Sessions: two blocks, one reversal. ~12 sessions. Overnight breaks between sessions: traces decay by `exp(−T/τ)`; RNN state relaxes through silent steps **or** resets to its ITI fixed point (log which); TBPTT window is cut at the break; weights persist. Optional replay during breaks is a separate condition.
- Long-ITI-break probes (37.5–300 s) are simulated with real silent steps.

## Environment

- Windows 11, PowerShell, VS Code. Conda env **`MetaRL`**, Python **3.12**.
- PyTorch **2.14.1+cu130** on an **RTX 5070 Ti** (16 GB, Blackwell, sm_120). Do not install CUDA 12.x builds; they lack Blackwell kernels.
- Install with `python -m pip ...` (never bare `pip`). Torch came from `--index-url https://download.pytorch.org/whl/cu130`.
- Code must be **device-agnostic** (`device` argument, no hard-coded `.cuda()`). Small online RNNs with per-step Hebbian updates may run faster on CPU; parallelize seeds across processes. GPU is most likely to help B1 (W = 720). Decide from measured timings.
- Evaluation needs ≥20 seeds per model: design runs around a seed argument from the start.

## Conventions

- Every run is reproducible from a config + seed. Log hyperparameters with results.
- Keep time constants in seconds and convert with dt: `λ = exp(−dt/τ)`.
- Report all tasks/conditions tried, including failures (fairness rule, research_plan §6.5).
- Ask before changing anything in `valuernn/` beyond compatibility fixes.
