# Model schematics: layer-by-layer spec for Models 1–3

Converted from `model_schematics.pptx` (working draft, October 2026). Model 1 follows the **revised spec**: cue-offset gate, jump-form event trace, explicit Oja decay.

## Differences from `research_plan.md` §4.1 (this file wins for Model 1)

| Item | research_plan.md §4.1 | This spec (revised) |
|---|---|---|
| Trial-type bias | `y = φ(R·[e; r])·off` | `y = φ(R·[eA, eB, r]ᵀ + b)·g`, fixed random bias b |
| Event trace ȳ | Averaging leaky integrator (§3.3) | **Jump form**: `ȳ(t) = λb·ȳ(t−1) + y(t)` (no `1 − λ` factor) |
| Oja term | Informal ("y·w²") | Explicit row decay: `− yi(t)²·Wij` |
| Cue trace e | Not specified | Averaging form, two channels `[eA, eB]` |

Open note: the overview table in the plan calls Model 1's rule "symmetric Hebbian", but the update `yi(t)·ȳj(t−1)` is lagged (j before i), so W is not symmetric by construction. It becomes approximately symmetric when trial types within a context alternate. Confirm with the user before enforcing symmetry.

---

## Shared framework

All three models receive the same input stream.

- `x(t) = [cA, cB, r]`: cue A, cue B, reward. **The only external input.**
- One trial: ITI → cue → outcome → ITI. The outcome comes directly at cue offset (no delay).
- `off(t) = max(0, c(t−1) − c(t))`: cue-offset pulse computed from c. Since there is no delay, it marks the outcome step on every trial. This is Model 1's gate.
- Sessions: two blocks, one reversal; overnight break between sessions.

How each part learns (the slides' color key):

| Category | Meaning |
|---|---|
| External input | x(t) |
| Fixed | Not learned (random expansion R, b, time constants) |
| Trace / slow memory | Leaky traces (e, ȳ, s, ε) |
| Hebbian-learned | Wb in Models 1 and 2 |
| Gradient-learned (TD) | Core RNN weights; U in Model 3 |
| Ignored path / open problem | Approximations and unresolved choices |

---

## Model 1: designed trial-type units → slow Hebbian binding → core RNN

Pipeline: `x(t) → e(t) → y(t) → ȳ(t) → Wb → z(t) → h(t) → V, δ`. Raw x also enters the RNN directly through Wx.

- **Fixed:** random expansion R, b; time constants τe, τb.
- **Hebbian, slow:** Wb. Carries structure across sessions.
- **TD + short BPTT:** Wh, Wx, Wz, w.
- Learning signals: δ trains the RNN; `y(t) × ȳ(t−1)` updates Wb.

### Layer 1 of 4: cue trace

A short trace carries cue identity into the outcome step, after the cue has switched off.

```
e(t) = λe·e(t−1) + (1 − λe)·[cA(t), cB(t)]
λe = exp(−dt/τe),   τe ≈ 1–2 s
```

- Two channels, one per odor: `e = [eA, eB]`. A single channel could not tell A from B.
- At the gate step c = 0, but e.g. eA > 0 and eB = 0, so cue identity is readable.
- **Averaging form**: the cue is a sustained input, so `(1 − λe)` keeps e independent of dt. Brief events (ȳ) use the jump form instead.
- No delay: e only has to bridge one step, so τe can be short. It is still needed, because c is already 0 at the outcome step.
- Not learned.

### Layer 2 of 4: trial-type layer

A fixed random expansion makes cue × outcome conjunctions linearly readable.

```
y(t) = φ(R·[eA, eB, r]ᵀ + b)·g(t)
R ∈ ℝ^(Ny × 3), b: random, fixed;   φ = ReLU
g(t) = off(t)
```

Why conjunctions: context is XOR(cue, outcome).

| | reward | no reward |
|---|---|---|
| cue A | A+ → context 1 | A− → context 2 |
| cue B | B+ → context 2 | B− → context 1 |

Diagonal cells share a context, so no weighted sum of cue and outcome separates them; conjunction units do.

- Gate `g(t) = off(t)` fires at cue offset = outcome. Computed from c(t), so no timing knowledge is needed. Fires on rewarded and omitted trials alike.
- **Variants:** four one-hot trial-type units instead of R (sanity check); surprise gate `g(|δ|)` instead of off(t).

### Layer 3 of 4: binding layer

Hebbian learning links trial types that follow each other within a context. The event trace ȳ bridges the ITI between trials.

```
ȳ(t) = λb·ȳ(t−1) + y(t),          τb ≈ 10–20 s   (jump form)
ΔWij = ηb·( yi(t)·ȳj(t−1) − yi(t)²·Wij ),   Wii = 0
z(t) = Wb·ȳ(t)                     (computed every step)
```

- Fixed point: `Wij = E[ȳj | unit i fires]`. Row i is the typical recent history before trial type i; zi measures how well the current history matches it.
- Expected Wb after context 1 (one-hot, 4×4; rows = current event i, columns = recent history j; diagonal 0):
  - **Strong:** A+ ↔ B− (same context).
  - **Faint, growing at each reversal:** A+ ↔ A−. Inhibition variants target this.
- **Feedforward:** z goes only to the RNN, never back to y.

### Layer 4 of 4: core RNN

Combines raw input with context evidence and learns by TD over a 1–2 trial window.

```
h(t) = f(Wh·h(t−1) + Wx·x(t) + Wz·z(t) + b)
V(t) = wᵀ·h(t)
δ(t) = r(t) + γ·V(t) − V(t−1)
```

- Nothing labels z as context. Wz is learned only because z helps predict reward.
- Long memory sits in ȳ and Wb, so the RNN only needs credit over the current trial.
- Gradient window: B1 (paper) W = 720 steps; Model 1 Ws ≈ 1–2 trials.
- **Variant:** e-prop instead of the short window (fully local).

---

## Model 2: the RNN binds traces of its own input-driven state (feedback loop)

Pipeline: `x(t) → h(t)`; input-driven part `h̃(t) → s(t)` (trace bank, log-spaced τ) `→ Wb → z(t) = Wb·s(t)`, fed back into the RNN. Surprise gate `g(|δ|)` on learning. Variant: raw x also traced.

- vs Model 1: no designed units. What gets bound comes from the RNN, and z changes what is traced next.
- `ηb ≪ ηRNN`. Same architecture as Model 3 with a Hebbian slow rule, giving a clean Hebbian-versus-gradient comparison.

### Component 1 of 3: core RNN with h̃ tap

Only the input-driven part h̃ is traced, so z cannot simply reinforce its own echo. z is added after the h̃ tap; h goes to V and the next step, h̃ goes to the traces.

Why exclude z: if s traced the full h, the loop z → h → s → Wb → z would let the Hebbian rule bind the binding layer's own output. Activity would reinforce itself instead of tracking the task.

**Open: define h̃** (with nonlinear f these differ; pick one and state it):
- (a) Second pass: `h̃ = f(Wh·h(t−1) + Wx·x(t) + b)` with z = 0.
- (b) Pre-activation: h̃ = pre-activation minus Wz·z, before f.

### Component 2 of 3: trace bank

Log-spaced time constants store h̃ at several timescales at once (e.g. τ = 6, 20, 60, 200, 600 s).

```
sk(t) = λk·sk(t−1) + (1 − λk)·u(t)
u = h̃   (variant: u = [h̃, x])
```

- Size: K timescales × Nh units.
- Coarse timing: a recent event shows in every trace, an old one only in the slow traces.
- **Open:** averaging form above (slow traces move very little per event) or jump form.
- Overnight: decay by `exp(−T/τ)`; all traces ≈ 0 at session start.

### Component 3 of 3: binding

Hebbian updates are slow and gated by surprise, so learning happens mainly after reversals.

```
ΔWb = ηb·g(|δ|)·( h̃(t)·s(t−1)ᵀ − Oja )
z(t) = Wb·s(t),   ηb ≪ ηRNN
```

- g: expected trials give little learning; surprises give a full update.
- **Oja term needs an explicit form**, e.g. row decay `g·h̃i²·Wij`.

| Known problem | Fix |
|---|---|
| TD discards A+ vs A− during the ITI | Surprise-weighted traces; raw x in traces; small predictive auxiliary loss |
| Moving target, self-reinforcement | Timescale separation; Oja normalization; trace only h̃ |
| Contexts are assemblies, not units | Decoding, representational similarity, fixed-point analysis |

---

## Model 3: slow diagonal units give gradient learning a blurred long memory, plus replay

Pipeline: `x(t) → h(t)` (fast RNN, short BPTT) `→ U·h → s(t)` (slow diagonal units) `→ Wz·s` back into h. Eligibility traces `εij(t)`; `∂L/∂s` from the short window; `ΔU = (∂L/∂s)·ε`. Replay buffer of (s snapshot, segment) pairs, replayed during breaks.

vs Model 2: same loop, but U is trained by the TD gradient through eligibility traces instead of a Hebbian rule.

### Component 1 of 4: fast RNN

Exact BPTT over a short window Ws (1–2 trials) trains the fast weights and supplies ∂L/∂s. The gradient is cut beyond the window.

- Trains Wh, Wx, Wz, w exactly within the window; nothing older gets direct credit.
- Produces `∂L/∂si(t)` at each in-window step, the error signal the slow units need.
- **Variant:** e-prop for the fast units, making every rule local.

### Component 2 of 4: slow units

Diagonal leaky integrators of a learned projection U·h, with log-spaced time constants (τ = 2, 6, 20, 60, 200, 600 s).

```
si(t) = λi·si(t−1) + (1 − λi)·Ui·h(t)
```

- Self-loop only (λi); no s–s connections.
- Timing precision falls in proportion to event age.
- Learnable: long-range co-occurrence. Learnable with Weber-like blur: long-range order. Not learnable: precise timing at long lags.
- **Variant:** learn τi too (cheap with diagonal dynamics).

### Component 3 of 4: credit for U

One eligibility trace per synapse is exact through each unit's own decay, and ignores the loop.

```
εij(t) = λi·εij(t−1) + (1 − λi)·hj(t)
ΔUij ∝ (∂L/∂si)(t) · εij(t)
```

- Kept (exact): s's own decay λ. Ignored: s → next h → later s, the loop that holds belief.
- ε has the same form as s, so it is local and forward-in-time.
- ∂L/∂s comes from the short window; ε reaches back over τi.
- Cost: one trace per U synapse (Ns × Nh).
- **Check:** cosine similarity with full BPTT on a small version, to measure the bias.

### Component 4 of 4: offline replay

Stored slow-state snapshots let short replayed segments carry long-range credit.

- Session 1 → overnight break (replay) → Session 2. Snapshots stored during sessions, denser near the reversal; replayed with the same learning rule.
- One buffer entry: s snapshot (minutes of history) + short segment (x over ≈ Ws steps), with noise added when replayed.
- Why it helps: the snapshot already compresses long history, so a short replay window still assigns long-range credit.
- **Staleness:** a snapshot was computed with an older U. Options: short-lived buffer, recompute s, or measure the bias.
- **Open:** does the fast RNN also learn offline? This decides between "sleep is necessary" and "sleep accelerates".
