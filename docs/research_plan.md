# Learning task structure without long-horizon credit assignment

*Three model families and a research plan*

Working document. Background: Lee, Hennig, Frelih, Gershman & Uchida, "Emergence of rapid value inference through meta-reinforcement learning" (bioRxiv, 10.64898/2025.11.30.691382).

> Markdown export of the project Google Doc. The Google Doc is the source of truth; re-export when it changes. For Model 1, `model_schematics.md` (revised spec) takes precedence over §4.1.

## 1. Motivation

In the paper, an RNN trained online with TD and truncated backpropagation through time (TBPTT) shifts from plasticity-based to dynamics-based value updating, and the TBPTT window length decides which regime appears. Long windows (W = 720 steps) are needed for meta-RL to emerge. Two problems follow:

- **Plausibility.** Exact credit assignment over minutes of stored history has no clear biological implementation; synaptic eligibility traces last about a second.
- **Training regime.** Mice see about one reversal per session, with overnight breaks, and reach expertise in about 12 sessions. A gradient window cannot span an overnight gap, and the number of reversals the RNN needs has not been compared with the mice.

Central claim to test: a small set of biologically motivated inductive biases (temporal contiguity, mixed selectivity, multiple memory timescales, separation of slow structure from fast state, offline replay) can replace long-horizon credit assignment for a broad class of tasks, with experience matched to the mice.

## 2. The three models at a glance

| | Model 1: fully layered | Model 2: Hebbian + RNN with feedback | Model 3: BPTT with blurred memory + offline |
|---|---|---|---|
| Role | Existence proof; most interpretable | Exploratory; emergent representation | Candidate main model; most general |
| What drives slow memory | Designed trial-type conjunctions | Input-driven part of the RNN state | Learned projection of the RNN state |
| Slow learning rule | Symmetric Hebbian + Oja | Asymmetric Hebbian, surprise-gated, + Oja | TD gradient via diagonal eligibility traces |
| Credit horizon | Short (1–2 trials) | Short | Long but blurred, plus replay |
| Main risk | Task rule built in | Unstable feedback loop | Weak credit signal; too slow to learn |

### Comments on the choices

- **Model 1** shows the principle works and gives clear predictions, but is open to the criticism that the task rule is built in. Its job is not to argue for generality.
- **Model 2** is general in what gets bound, but unsupervised Hebbian learning on its own output can be unstable. A careful characterization of when it fails is an acceptable outcome.
- **Model 3** is the most general and builds directly on the ValueRNN code. The open question is how many reversals it needs.
- **Models 2 and 3 share an architecture.** Hebbian binding on lagged traces is the unsupervised special case of the gradient rule through traces, so swapping only the slow learning rule gives a clean Hebbian-versus-gradient comparison.

## 3. Shared framework

### 3.1 Task stream

Timesteps of size dt with inputs x(t) = [c_A(t), c_B(t), r(t)] (cue, ITI, outcome). Sessions have two blocks and one reversal. Overnight breaks between sessions:

- Activity traces decay analytically by exp(−T/τ).
- The RNN state relaxes through a few hundred silent steps, or is reset to its ITI fixed point (report which).
- The TBPTT window is cut at the break; weights persist.
- Optional replay phase during the break (a separate experimental condition).

Minute-scale ITI-break probes (37.5–300 s) are simulated with real silent steps.

### 3.2 Core RNN (all models)

```
h(t) = f( W_h·h(t−1) + W_x·x(t) + W_z·z(t) + b )
V(t) = wᵀ·h(t),   δ(t) = r(t) + γ·V(t) − V(t−1)
```

Trained online with TD using a short exact BPTT window W_s of about one to two trials. Fully local variant: e-prop instead of the short window.

### 3.3 Slow memory (shared form)

```
s_k(t) = λ_k·s_k(t−1) + (1 − λ_k)·u_k(t),   λ_k = exp(−dt/τ_k)
```

Leaky integrator with λ_k as the exponential decay factor; τ_k log-spaced from seconds to tens of minutes. The models differ in what drives the traces (u) and how the traces are turned into z.

## 4. Model specifications

### 4.1 Model 1: trial-type layer + context binding + core RNN

**Trial-type layer (no learning).** Cue trace e(t) with τ_e ≈ 1–2 s, gated at outcome time and passed through a fixed random nonlinear expansion:

```
y(t) = φ( R·[e(t); r(t)] )·off(t),   off(t) = max(0, c(t−1) − c(t))
```

- Input trace: 3D vector [e_A, e_B, r(t)]
- R: fixed random expansion matrix
- φ: nonlinearity (ReLU)
- off(t): gates trial-type output only at the outcome step, assuming outcome comes right after cue ends (actual behavior setup)

**Context binding layer (slow Hebbian).** Activity trace ȳ with τ_b ≈ 10–20 s (one to two ITIs), lagged by one step:

```
ΔW_b = η_b·( y(t)·ȳ(t−1)ᵀ − Oja term ),   diag(W_b) = 0,   z(t) = W_b·ȳ(t)
```

- All-to-all connected (no self-connection, given diag(W_b) = 0)
- ȳ: slow trace of y, defined by the leaky integrator in 3.3
- Oja term: normalizes the weight vector to unit length, y·w²
- Feedforward: no recurrent feedback, one-pass feedforward; the binding layer provides evidence, while the core RNN holds and updates the belief.

**Variants:** fixed vs learned (anti-Hebbian) inhibition between A+ and A−; single τ_b vs multiple timescales; odor-offset gate vs a generic surprise gate.

### 4.2 Model 2: Hebbian binding on traces of the RNN state, with feedback

No designed units. The binding layer reads traces of the input-driven component h̃ of the RNN state (z's contribution removed):

```
ΔW_b = η_b·g(|δ(t)|)·( h̃(t)·s(t−1)ᵀ − Oja term ),   z(t) = W_b·s(t),   η_b ≪ η_RNN
```

Known problems and fixes:

- **Information discarded at the outcome:** TD gives no reason to keep A+ and A− distinct during the ITI. Fixes: surprise-weighted traces, raw inputs added to the traces, or a small predictive auxiliary loss.
- **Moving target and self-reinforcement:** strong timescale separation, Oja normalization, tracing only the input-driven part of h.
- **Interpretability:** contexts are assemblies in state space; analyze with decoding, representational similarity, and fixed points.

**Variants:** with and without raw inputs in the traces; with and without the surprise gate; with and without an auxiliary predictive loss.

### 4.3 Model 3: BPTT with blurred long-term memory and offline learning

Slow, diagonal units with learned input weights U and log-spaced time constants; the RNN receives s:

```
s_i(t) = λ_i·s_i(t−1) + (1 − λ_i)·U_i·h(t)
```

Credit assignment:

- **Fast units:** exact BPTT over the short window (or e-prop).
- **Slow units:** exact forward-mode credit through their own diagonal dynamics, one eligibility trace per synapse:

  ```
  ε_ij(t) = λ_i·ε_ij(t−1) + (1 − λ_i)·h_j(t),   ΔU_ij ∝ (∂L/∂s_i)·ε_ij(t)
  ```

  ∂L/∂s comes from the short window. This ignores the path from s back through h, an approximation in the spirit of e-prop; state it explicitly.

- **Offline:** a sparse buffer of (s snapshot, short trial segment) pairs, weighted toward reversals and perturbed with noise, replayed during breaks with the same rule. One snapshot carries minutes to hours of compressed history, so short replay windows still give long-range credit.

Generality: log-spaced traces keep coarse information about when events happened, with blur growing in proportion to age. Long-range co-occurrence should be learnable; long-range order with Weber-like loss of precision; precise long-lag timing not at all.

**Variants:** with and without replay; τ fixed vs learned (cheap with diagonal RTRL); with and without a Hebbian auxiliary term (links to Model 2).

## 5. Baselines

| | Description | Purpose |
|---|---|---|
| B1 | ValueRNN with long exact window (W = 720) | The paper's model; target to match |
| B2 | ValueRNN with short window W_s, no slow memory | Should fail; control for every model |
| B3 | Belief-state TD on the ideal observer's posterior | Upper bound |
| B4 | Linear readout of the traces, no RNN | Defines the linear speed–stability frontier |

## 6. Evaluation

### 6.1 Training trajectory (main result)

About 12 sessions, one reversal per session, overnight breaks, at least 20 seeds per model. Per session: reversal speed, discrimination at session start, and the session at which inference first appears, compared with the mouse data.

### 6.2 Benchmarks from the paper

- Reversal speed (mice: time constants of about 2–8 trials).
- ITI-break forgetting at 37.5–300 s.
- Weight freezing: blocks learning in the stable task but not the dynamic one.
- ITI perturbation of h: impairs only the dynamic task.
- Probe inference, including scaling with the number of opposite-cue trials.
- Structure-specific inference: correlated, anti-correlated, independent.

### 6.3 Mechanistic analyses

- Context axis, and whether drift along it predicts each model's forgetting.
- Fixed points vs slow points during the ITI.
- Position relative to the linear speed–stability frontier (B4).
- Reliance on slow memory: performance when z or s is ablated.

### 6.4 Generalization suite

Hyperparameters frozen after development on the anti-correlated task:

- Probabilistic reversal (e.g. 80/20).
- Three or more contexts.
- Partially overlapping contexts (e.g. A+ in both, only B differs).
- Order-defined contexts (AB vs BA).
- Variable-ISI hidden-state timing task.
- Reversal bandit with actions.

### 6.5 Fairness

Same hyperparameter-search budget and similar parameter counts for every model and baseline. Report every task tried, including failures.

## 7. Plan

### Phase 0: infrastructure and checking the motivation

- Task generator with sessions, breaks and long-ITI probes; ideal observer; metric pipeline.
- Reimplement ValueRNN and reproduce the paper's key figures.
- Run B1 under the realistic schedule (two blocks per session, overnight breaks); sweep W; record reversals to expertise.

**Go/no-go:** if B1 learns well under the realistic schedule, the motivation narrows to plausibility only. Still valid, but weaker; know this before investing further.

### Phase 1: Model 1

- Working model on the anti-correlated task.
- Ablations: no binding layer; separate features instead of conjunctions; linear readout only; feedforward core.
- Fixed vs learned inhibition (tests the built-in-task-knowledge concern).
- Long-ITI training simulation to sharpen the spacing prediction.

### Phase 2: Model 3

- With vs without replay; reversals to expertise compared with B1.
- Order-defined task, where it should beat Model 1.

Decides whether gradient learning through blurred memory is fast enough.

### Phase 3: Model 2

- Start from Model 3's architecture and swap in the Hebbian rule.
- Fixed time budget for stabilizing; if unstable, report when it does and doesn't stabilize.

### Phase 4: full comparison

All models and baselines on the generalization suite. Summarize each by where it succeeds and fails, and which failures match plausible animal limits.

### Phase 5: predictions back to mice

**Existing data:** session-start discrimination per day; reversal speed per session; whether improvement happens between or within sessions; changes in GLM history kernels across training.

**New experiments:** long-ITI training followed by short-ITI testing; probe inference on intermediate training days; sleep or ripple disruption; ITI-break forgetting at intermediate stages of training.

## 8. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Weak credit through blurred traces (Model 3) | Learning too slow to match ~12 sessions | Offline replay; Hebbian auxiliary term |
| Feedback loop does not stabilize (Model 2) | No usable Model 2 | Timescale separation, surprise gating; report failure conditions |
| Task rule built in (Model 1) | Generality criticism | Learned-inhibition variant; generalization suite |
| All models match on the paper's task | No discrimination between models | Build order-defined and timing tasks early |
| B1 works under realistic schedule | Motivation narrows | Decide at Phase 0 go/no-go |

## 9. Open design questions

- Should the core RNN learn only online with a short window, or also offline? This decides whether the claim is "sleep is necessary" or "sleep accelerates" meta-RL.
- Should slow-memory decay be time-based or event-based? The long-ITI training experiment distinguishes them.
- Is a fixed memory timescale enough, or must maintenance be learned? Compare ITI-break forgetting across training stages.
- Brain mapping (provisional): trial types in LEC or BLA; binding in ventral hippocampus (alternative: OFC); core RNN as a BLA–prefrontal loop; value and TD in ventral striatum and VTA dopamine.
