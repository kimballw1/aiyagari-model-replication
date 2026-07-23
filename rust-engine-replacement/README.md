# Rust (1987): bus-engine replacement

This repo solves and estimates Rust's (1987) optimal bus-engine replacement model in Julia. It's the paper that started modern structural econometrics, by showing how to recover a forward-looking agent's preferences from observed choices. Harold Zurcher, maintenance superintendent of the Madison Metropolitan Bus Company, watches each bus's accumulated mileage and decides every month whether to keep the engine and pay rising maintenance, or replace it at a big fixed cost that resets the engine to new. The code solves his dynamic program by value-function iteration, then runs the inference pipeline on simulated data: the nested fixed-point (NFXP) maximum-likelihood estimator from the paper, standard errors, a Monte-Carlo sampling-distribution study, and a counterfactual policy experiment.

---

## The model

Each period Zurcher sees a bus at mileage state `x` and chooses `d ∈ {keep, replace}`:

- Keep (`d = 0`): pay maintenance cost `c(x) = θ₁·x`. The engine ages, so next period's mileage is higher and so is next period's maintenance.
- Replace (`d = 1`): pay a fixed replacement cost `RC`, and the engine is new (mileage resets to zero, then starts climbing again).

The tension is pay-a-little-forever against pay-a-lot-once. Maintenance is cheap today but compounds as the engine ages; replacement is expensive today but buys a fresh, cheap-to-run engine. Because today's keep raises tomorrow's costs, the decision is dynamic: Zurcher has to weigh the entire future path of maintenance against the one-time `RC`.

### The unobserved shocks

If that were the whole model, every bus at the same mileage would make the identical choice, and the data would show a clean mileage cutoff. Rust added i.i.d. Type-I Extreme-Value (logit) shocks `ε(keep), ε(replace)` that Zurcher observes but we don't: weather, a complaining driver, parts on hand, gut feeling. Choices become probabilistic, which matches the data, and two properties of the EV distribution keep everything in closed form:

| Result | Formula | What it drives |
|---|---|---|
| Expected maximum (log-sum-exp) | `E[max_d(v_d + ε_d)] = log Σ_d e^{v_d} (+ γ)` | the Bellman equation |
| Choice probability (logit) | `P(d) = e^{v_d} / Σ_{d'} e^{v_{d'}}` | the conditional choice probabilities (CCPs) |

### Variables

| Symbol | Meaning | Baseline |
|---|---|---|
| `x` | mileage state: accumulated mileage, in `S` discrete bins | 0 … 49 |
| `d` | decision: keep (0) or replace (1) | — |
| `RC` | replacement cost: the fixed price of a new engine | 8.0 |
| `θ₁` | maintenance slope: how fast upkeep rises with mileage, `c(x) = θ₁·x` | 0.04 |
| `θ₃` | mileage-transition probabilities, `P(Δ = 0, 1, 2)` bins per period | [0.36, 0.48, 0.16] |
| `β` | discount factor: patience over future costs | 0.95 |
| `V(x)` | value function: expected discounted payoff at mileage `x` | solved |
| `P(replace\|x)` | the CCP: replacement hazard at mileage `x` (the logit) | S-curve |

The default calibration puts the replacement threshold inside the grid (`RC = 8`), so the interesting region is actually visible.

### The reservation mileage

Solving the Bellman equation gives `v_keep(x)`, which falls with mileage (keeping a high-mileage bus means more costs ahead), and `v_replace(x)`, which is nearly flat (a new engine is a new engine regardless of where you started). They cross at a mileage `x*`, and the replacement hazard `P(replace | x) = 1/(1+e^{v_keep−v_replace})` is a smooth S-curve rising through 0.5 there. At the baseline `x* = 36`: Zurcher is more likely than not to replace once a bus passes mileage bin 36, the implied long-run replacement rate is 4.6% per period, and an engine lasts about 22 periods.

Below the threshold he keeps, above it he replaces, and the threshold moves the way you'd expect:

| Change | Effect on `x*` | Why |
|---|---|---|
| `RC ↑` (engines pricier) | ↑ (wait longer) | a costlier new engine isn't worth it until maintenance is really high |
| `θ₁ ↑` (upkeep steeper) | ↓ (replace sooner) | fast-rising maintenance makes a fresh engine pay off earlier |
| `β ↑` (more patient) | replace sooner | a forward-looking Zurcher pays now to avoid the future cost spiral |

---

## Files

| File | Role |
|---|---|
| `params.jl` | `RustModel`: structural and numerical parameters; the linear cost function |
| `transitions.jl` | the keep/replace mileage transition matrices `F0, F1`; the controlled stationary distribution |
| `solve.jl` | the dynamic program: log-sum-exp Bellman operator, VFI, and the logit CCPs |
| `simulate.jl` | generates Harold-Zurcher panel data (the sawtooth mileage path) |
| `estimate.jl` | structural estimation: θ₃ by counting, NFXP MLE plus standard errors |
| `main.jl` | assembles the blocks and solves/prints the baseline |
| `analyze.jl` | diagnostics, EV Monte-Carlo checks, comparative statics, estimation, the sampling-distribution study, and the counterfactual |

Run it with `julia analyze.jl` (it includes `main.jl`, solves the baseline, then runs §1–§7 and writes figures to `graphs/`). A full pass takes ~25 s. Requires `Optim`, `Distributions`, and `Plots`.

---

## Figures

| File | Section | Content |
|---|---|---|
| `fig_value_function.png` | §1 | `V(x)`, `v_keep(x)`, `v_replace(x)`, and the reservation mileage |
| `fig_ccp.png` | §1 | replacement hazard S-curve (the CCP) |
| `fig_ev_results.png` | §2 | Monte-Carlo checks of log-sum-exp and the logit formula |
| `fig_comparative_statics.png` | §3 | replacement hazard vs `RC` and vs `θ₁` |
| `fig_simulation.png` | §4 | sawtooth mileage path and stationary mileage distribution |
| `fig_estimation.png` | §5 | estimated vs true CCP: the NFXP fit |
| `fig_monte_carlo.png` | §6 | sampling distributions of `RĈ` and `θ̂₁` |
| `fig_counterfactual.png` | §7 | long-run replacement rate vs a replacement-cost subsidy |

---

## References

- Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher." *Econometrica* 55(5). The model, the EV-shock framework, and the nested fixed-point MLE implemented here.
- Rust, J. (1988). "Maximum Likelihood Estimation of Discrete Control Processes." *SIAM J. Control*. The general theory behind NFXP.
- Ljungqvist, L. & Sargent, T. (2018). *Recursive Macroeconomic Theory*. The dynamic-programming and discrete-choice background.
