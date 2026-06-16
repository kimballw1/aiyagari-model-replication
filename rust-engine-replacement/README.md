# rust-engine-replacement-replication

This repository solves **and estimates** Rust's (1987) optimal bus-engine replacement model in Julia — the paper that founded modern structural econometrics by showing how to recover a forward-looking agent's preferences from observed choices. Harold Zurcher, maintenance superintendent of the Madison Metropolitan Bus Company, watches each bus's accumulated mileage and every month decides whether to **keep** the engine (and pay rising maintenance) or **replace** it (a big fixed cost that resets the engine to new). The code solves his dynamic program by value-function iteration, then runs the full inference pipeline on simulated Harold-Zurcher data: the **Nested Fixed-Point (NFXP)** maximum-likelihood estimator (Rust 1987), the two-step **Hotz–Miller (1993)** conditional-choice-probability estimator, standard errors, a Monte-Carlo sampling-distribution study, and a counterfactual policy experiment.

It is the dynamic-discrete-choice sibling of the McCall search model: where McCall has a reservation *wage*, Rust has a reservation *mileage*; both are forward-looking optimal-stopping problems, and both reduce to a single threshold. What Rust added — and what this repo is really about — is **estimation**: turning the solved model into a likelihood and recovering its deep parameters from data.

---

## The economics: what Rust is doing

### The one decision (again)

Each period Zurcher sees a bus at mileage state `x` and chooses `d ∈ {keep, replace}`:

- **Keep** (`d = 0`): pay maintenance cost `c(x) = θ₁·x`. The engine ages — next period's mileage is higher, so next period's maintenance is higher too.
- **Replace** (`d = 1`): pay a fixed replacement cost `RC`, and the engine is **good as new** (mileage resets to zero, then starts climbing again).

The tension is **pay-a-little-forever vs. pay-a-lot-once**: maintenance is cheap today but compounds as the engine ages; replacement is expensive today but buys a fresh, cheap-to-run engine. Because today's keep raises tomorrow's costs, the decision is genuinely dynamic — Zurcher must weigh the entire future path of maintenance against the one-time `RC`.

### The unobserved shocks — and why they're the whole point

If that were the entire model, every bus at the same mileage would make the identical choice, and the data would show a clean mileage cutoff. Real data don't: buses at *similar* mileage make *different* choices. Rust's modeling move was to add **i.i.d. Type-I Extreme-Value (logit) shocks** `ε(keep), ε(replace)` that Zurcher observes — weather, a complaining driver, parts on hand, gut feeling — but the econometrician does not. Choices become **probabilistic**, which (a) matches the data and (b), because of two magical properties of the EV distribution, keeps everything in closed form:

| Result | Formula | What it drives |
|---|---|---|
| **Expected maximum** (log-sum-exp) | `E[max_d(v_d + ε_d)] = log Σ_d e^{v_d} (+ γ)` | the **Bellman equation** |
| **Choice probability** (logit) | `P(d) = e^{v_d} / Σ_{d'} e^{v_{d'}}` | the **conditional choice probabilities (CCPs)** |

`analyze.jl` §2 verifies both by Monte Carlo (simulated vs. theory agree to ≈0.005). These are the same formulas as static discrete-choice econometrics — the only difference is that here `v_keep` and `v_replace` are forward-looking continuation values, not static payoffs.

### The cast of characters

| Symbol | Plain meaning | Baseline |
|---|---|---|
| `x` | **mileage state** — accumulated mileage, in `S` discrete bins | 0 … 49 |
| `d` | **decision** — keep (0) or replace (1) | — |
| `RC` | **replacement cost** — the fixed price of a new engine | 8.0 |
| `θ₁` | **maintenance slope** — how fast upkeep rises with mileage, `c(x)=θ₁·x` | 0.04 |
| `θ₃` | **mileage-transition** probabilities — `P(Δ = 0, 1, 2)` bins per period | [0.36, 0.48, 0.16] |
| `β` | **discount factor** — patience over future costs | 0.95 |
| `V(x)` | **value function** — expected discounted payoff at mileage `x` | solved |
| `P(replace\|x)` | **CCP** — replacement hazard at mileage `x` (the logit) | S-curve |

The default calibration is chosen so the replacement threshold sits *inside* the grid (`RC = 8`): see "A note on calibration" below.

### The headline object: the reservation mileage

Solving the Bellman equation gives `v_keep(x)` (falling in mileage — keeping a high-mileage bus means more costs ahead) and `v_replace(x)` (nearly flat — a new engine is a new engine regardless of where you started). They **cross** at a mileage `x*`, and the replacement hazard `P(replace | x) = 1/(1+e^{v_keep−v_replace})` is a smooth **S-curve** rising through 0.5 there:

> **`x* = 36`** at baseline — Zurcher is more likely than not to replace once a bus passes mileage bin 36. The implied **long-run replacement rate is 4.6%** per period, i.e. an engine lasts ≈ 22 periods.

This is the exact dynamic-discrete-choice analog of McCall's reservation wage: *keep below the threshold, replace above it*, with the EV shocks smoothing the hard cutoff into a probability. The comparative statics (§3) read off the cost trade-off directly:

| Change | Effect on `x*` | Why |
|---|---|---|
| **`RC ↑`** (engines pricier) | ↑ (wait longer) | a costlier new engine isn't worth it until maintenance is really biting |
| **`θ₁ ↑`** (upkeep steeper) | ↓ (replace sooner) | fast-rising maintenance makes a fresh engine pay off earlier |
| **`β ↑`** (more patient) | replace sooner | a forward-looking Zurcher pays now to avoid the future cost spiral |

### A note on calibration

The undergraduate lecture uses `RC = 20`, which (with `θ₁ = 0.04` and `S = 50`) makes replacement so cheap relative to lifetime maintenance that the hazard never reaches 0.5 on the grid — realistic for real bus engines, which almost never get replaced in a given month, but it hides the S-curve. This repo sets **`RC = 8`** so the reservation mileage `x* = 36` lands inside the grid and the full S-curve is visible. Rust's own estimates for his GMC fleet (groups 1–4, linear cost, mileage in thousands) are roughly **`RC ≈ 10`** and a maintenance slope an order of magnitude smaller, in thousands of 1985 dollars; the qualitative behavior is identical, only the units and the grid scale differ.

---

## Code structure

| File | Role |
|---|---|
| `params.jl` | `RustModel` — structural & numerical parameters; the linear cost function |
| `transitions.jl` | the keep/replace mileage transition matrices `F0, F1`; the controlled stationary distribution |
| `solve.jl` | the dynamic program: log-sum-exp Bellman operator, VFI, and the logit CCPs |
| `simulate.jl` | generates Harold-Zurcher panel data (the sawtooth mileage path) |
| `estimate.jl` | structural estimation: θ₃ by counting, NFXP MLE + standard errors, Hotz–Miller CCP estimator |
| `main.jl` | assembles the blocks and solves/prints the baseline |
| `analyze.jl` | diagnostics, EV Monte-Carlo checks, comparative statics, estimation, the sampling-distribution study, and the counterfactual |

Run order: `julia analyze.jl` (it `include`s `main.jl`, solves the baseline, then runs §1–§8 below and writes figures to `graphs/`). A full pass takes ~17 s. Requires `Optim`, `Distributions`, and `Plots`.

---

## Walkthrough of `analyze.jl`

### §1 — The model solution *(the central picture)*
`fig_value_function.png` shows `V(x)` falling with mileage and the choice-specific values crossing at the reservation mileage; `fig_ccp.png` is the replacement hazard S-curve through `x* = 36`. The stationary distribution implies a **4.6% replacement rate** and a ≈22-period engine life.

### §2 — The extreme-value machinery *(why it's all tractable)*
A Monte-Carlo check of the two EV results that make the model solvable: the expected maximum is log-sum-exp (errors ≈ 0.005) and the choice probability is logit (errors ≈ 0.002). `fig_ev_results.png` overlays simulation on the closed forms across the value gap `v₁ − v₀`. These are the load-bearing assumptions — everything downstream is built on them.

### §3 — Comparative statics *(the cost trade-off)*
Sweeping `RC` shifts the hazard curve **right** (costlier engines → wait longer → `x*`: 27 → 48); sweeping `θ₁` shifts it **left** (steeper upkeep → replace sooner). `fig_comparative_statics.png` shows both families of S-curves. This is the structural content: behavior responds to the deep cost parameters in an economically sensible, *predictable* way — the foundation for the counterfactual in §8.

### §4 — Simulation & the stationary mileage distribution *(model → data)*
Under the optimal policy a bus's mileage traces the famous **sawtooth**: a slow climb, then a sharp reset at replacement (`fig_simulation.png`, left). The long-run mileage histogram (right) piles up at **low** mileage — engines rarely survive to high mileage before replacement. The simulated replacement rate (4.5%) matches the stationary prediction (4.6%).

### §5 — Structural estimation: NFXP *(the headline method)*
From a 50,000-period panel, recover the parameters. Step 1: `θ₃` by counting mileage increments — identified *without* the model (true `[0.36,0.48,0.16]` → estimated `[0.362,0.481,0.158]`). Step 2: `(RC, θ₁)` by **nested fixed-point MLE** — an outer Nelder–Mead search where *every trial parameter re-solves the entire dynamic program*. Recovery is sharp, with Hessian-based standard errors:

| Parameter | True | NFXP estimate | Std. error |
|---|---:|---:|---:|
| `RC` | 8.0 | 8.28 | 0.156 |
| `θ₁` | 0.04 | 0.0418 | 0.0011 |

### §6 — Hotz–Miller (CCP) estimation *(the faster alternative, and its limits)*
The two-step CCP estimator replaces the inner fixed point with a **single linear solve**: given first-stage choice probabilities, invert the logit identity `V = v_keep − log P_keep` to get the value function in one shot — no VFI. It recovers `RC = 7.67, θ₁ = 0.0368`: close, but visibly less accurate than NFXP. `fig_estimation.png` shows why — and it is the most instructive figure in the repo. The **renewal structure** (replacing resets mileage) means engines are almost always replaced before reaching high mileage, so **only 34 of 50 states are ever visited**. The first-stage CCP is accurate where data are plentiful (low mileage), noisy in the thin transition region, and undefined above the recurrent set (it flatlines at the prior). NFXP sidesteps this by imposing the model everywhere; pure CCP methods cannot identify what never happens. This trade-off — NFXP's accuracy vs. CCP methods' speed — is the central methodological debate the literature (Aguirregabiria–Mira, Arcidiacono–Miller) grew out of.

### §7 — Monte-Carlo sampling distribution *(is the estimator any good?)*
Re-estimate by NFXP on 40 independent panels (`T = 8,000` each). The estimates center on the truth — **`RC`: mean 8.03 (true 8.0), `θ₁`: mean 0.0405 (true 0.04)** — confirming the MLE is essentially unbiased. The spread is an internal-consistency check on §5's analytic standard errors: the Monte-Carlo `sd(RC) = 0.40` matches the §5 standard error `0.156` rescaled for the shorter panel, `0.156 × √(50000/8000) ≈ 0.39`. `fig_monte_carlo.png` plots both sampling histograms with the truth marked.

### §8 — Counterfactual: a replacement subsidy *(the payoff of structure)*
The reason to bother estimating a *structural* model: counterfactuals. Lowering `RC` (a government subsidy on new engines) makes Zurcher replace sooner and more often — the long-run replacement rate climbs from 3.3% at `RC = 12` to **12.4% at `RC = 3`** (`fig_counterfactual.png`). A reduced-form hazard fit to the *original* data could never predict this: it would hold the old replacement pattern fixed, whereas Zurcher **re-optimizes** when the cost changes. Capturing that re-optimization — the Lucas critique made operational — is exactly what the structural parameters buy.

---

## Figures produced

| File | Section | Content |
|---|---|---|
| `fig_value_function.png` | §1 | `V(x)`, `v_keep(x)`, `v_replace(x)`, and the reservation mileage |
| `fig_ccp.png` | §1 | replacement hazard S-curve (the CCP) |
| `fig_ev_results.png` | §2 | Monte-Carlo checks of log-sum-exp and the logit formula |
| `fig_comparative_statics.png` | §3 | replacement hazard vs `RC` and vs `θ₁` |
| `fig_simulation.png` | §4 | sawtooth mileage path & stationary mileage distribution |
| `fig_estimation.png` | §6 | estimated vs true CCP — NFXP fit and Hotz–Miller first-stage |
| `fig_monte_carlo.png` | §7 | sampling distributions of `RĈ` and `θ̂₁` |
| `fig_counterfactual.png` | §8 | long-run replacement rate vs a replacement-cost subsidy |

---

## Research lineage

- **Rust, J. (1987).** "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher." *Econometrica* 55(5). The model, the EV-shock framework, and the Nested Fixed-Point MLE implemented here.
- **Rust, J. (1988).** "Maximum Likelihood Estimation of Discrete Control Processes." *SIAM J. Control* — the general theory behind NFXP.
- **Hotz, V. J. & Miller, R. A. (1993).** "Conditional Choice Probabilities and the Estimation of Dynamic Models." *REStud* 60(3). The CCP inversion behind the two-step estimator in §6.
- **Hotz, Miller, Sanders & Smith (1994).** The CCP forward-simulation estimator that generalized the inversion.
- **Aguirregabiria, V. & Mira, P. (2002, 2007).** Nested pseudo-likelihood — the iterated CCP estimator bridging Hotz–Miller and NFXP.
- **Arcidiacono, P. & Miller, R. (2011).** Finite dependence and CCP estimation with unobserved heterogeneity — the modern frontier the renewal structure in §6 points toward.
- **Ljungqvist, L. & Sargent, T. (2018).** *Recursive Macroeconomic Theory* — the dynamic-programming and discrete-choice background.
