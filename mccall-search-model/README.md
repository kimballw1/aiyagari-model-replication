# mccall-search-model-replication

I solve the **McCall (1970)** sequential job search model in Julia, from the original risk-neutral reservation wage problem through the modern extensions that turned it into the workhorse of labor economics: job separation (so unemployment has a *steady state*), risk aversion, persistent (serially correlated) wage offers, and a Monte-Carlo treatment of unemployment durations. An unemployed worker draws wage offers from a fixed distribution and decides each period whether to accept the job in hand or keep searching for something better. The whole model collapses to a single number (the **reservation wage**) and the code computes it two independent ways (a fast scalar fixed point and a full value-function iteration), then uses it to generate hazard rates, expected unemployment durations, the steady-state unemployment rate, and the cross-section of accepted wages.

The flow utility is CRRA (constant relative risk aversion), `u(x) = x^{1−γ}/(1−γ)` (with `u(x)=log x` at `γ=1`), and setting **`γ = 0` recovers McCall's original risk-neutral worker**, for whom utility is just income. The default calibration uses `γ = 2`, so the baseline already carries the risk-aversion margin that McCall's 1970 paper abstracted from — letting the same code show *both* the classic results and how risk aversion bends them.

---

## The economics: what McCall is doing

### The one decision

A worker is **unemployed** and gets a job offer this period — a wage `w` drawn from a distribution. She has exactly one choice:

- **Accept** and take the job and earn `w` (for a while, or forever).
- **Reject** and collect an unemployment benefit `c`, throw the offer back, and draw a fresh offer next period.

That's the entire model. The tension is between a **bird in the hand** (the offer `w` you can have right now) and the **option to wait** for something better (at the cost of a period spent unemployed earning only `c`). There are `2ⁿ` accept/reject paths after `n` periods.

### The headline result: the reservation wage

> The optimal policy is a **threshold rule**. There is a number `w̄` (the **reservation wage**) such that the worker **accepts any offer `w ≥ w̄` and rejects any offer below it**. Searching is valuable precisely because a rational worker throws back the low offers and waits for a good one.

Everything the model has to say is encoded in where `w̄` sits. Accepting too eagerly (low `w̄`) wastes the option to find a better match; holding out too long (high `w̄`) burns income while unemployed. The reservation wage is the exact balance point, and it satisfies a strikingly clean equation (derived below):

> **`u(w̄) = (1 − β)·U`**

where `U` is the lifetime value of being unemployed. In words: *the flow utility of the reservation wage equals the annuitized (per-period) value of continued search.* A worker accepts exactly when the job's per-period utility clears the bar that searching itself is worth. Everything — benefits, patience, risk, layoffs — moves `w̄` only through how it moves `U`.

### The cast of characters

| Symbol | Plain meaning | Baseline |
|---|---|---|
| `w` | a **wage offer** — one draw from the offer distribution | varies |
| `w̄` | the **reservation wage** — accept above it, reject below | solved → 45.29 |
| `c` | **value of unemployment** — UI benefit plus home production, per period | 25.0 |
| `β` | **patience** — how much next period is worth vs. this one | 0.96 |
| `γ` | **risk aversion** — curvature of utility; `γ=0` is risk-neutral McCall | 2.0 |
| `α` | **separation rate** — chance an employed worker is laid off each period | 0.05 |
| `U` | lifetime **value of being unemployed** (the continuation value) | −0.55 |
| `f` | **job-finding rate** (hazard) — probability a random offer is accepted, `P(w ≥ w̄)` | 0.315 |

The offer distribution itself is log-normal, `ln w ~ N(μ, σ²)`, with `μ = 3.6, σ = 0.45` — so the mean offer is `E[w] = exp(μ + σ²/2) ≈ 40.5` and offers are right-skewed, like real wage data.

### Why a *reservation wage* exists at all

Compare the two options as functions of the offer `w`:

- The value of **accepting**, `V_e(w)`, is **increasing** in `w` — a better wage is unambiguously better.
- The value of **rejecting** is `U` — it does **not depend** on the offer you're throwing back.

An increasing line crossing a flat line crosses exactly **once**. To the right of the crossing, accepting beats waiting; to the left, waiting wins. That single crossing point *is* `w̄`. This is why the `2ⁿ`-path problem has a one-number answer, and it is the picture in `fig_value_function.png`: a rising `V_e(w)` cutting the horizontal `U` at `w̄`.

### Deriving the reservation-wage equation

With separation rate `α`, an employed worker keeps her job with probability `1−α` and is thrown back into unemployment with probability `α`. The two Bellman equations are

```
V_e(w) = u(w) + β[(1−α)·V_e(w) + α·U]            (employed at wage w)
U      = u(c) + β·Σ_w pw·max{ V_e(w), U }         (unemployed, before drawing)
```

The first solves in closed form for the employment value, `V_e(w) = [u(w) + βαU]/(1 − β(1−α))`. Setting `V_e(w̄) = U` (the indifference point) and simplifying, the `α` terms cancel and you are left with `u(w̄) = (1 − β)·U` — **independent of the separation rate**. Separation changes `w̄` only by changing `U`, never the indifference condition itself. This is the equation the code inverts: solve the scalar fixed point for `U`, then `w̄ = u⁻¹((1−β)U)`.

### What the reservation wage *means* (and how it moves)

`w̄` is the **price of the worker's time** — the lowest wage at which she is willing to stop searching. Read the comparative statics in §3 through the single channel "does this make searching more attractive?":

| Change | Effect on `w̄` | Why |
|---|---|---|
| **`c ↑`** (more generous UI) | ↑ (31.6 → 55.5) | a better outside option makes waiting cheaper → hold out for more |
| **`β ↑`** (more patient) | ↑ (40.9 → 48.9) | the future payoff to waiting is discounted less → search longer |
| **`α ↑`** (riskier jobs) | ↓ (52.2 → 38.0) | if any job is soon lost anyway, there's less point being choosy |
| **`γ ↑`** (more risk-averse) | ↓ (56.3 → 33.5) | a risk-averse worker grabs the certainty of a job over the gamble of searching |

The job-finding rate `f = P(w ≥ w̄)` moves inversely to `w̄`, and (with separation) the steady-state unemployment rate is `u* = α/(α+f)`. The UI result — **more generous benefits raise the reservation wage, lengthen unemployment, and raise the unemployment rate** — is the single most policy-relevant prediction of the model, and it falls straight out of the table (`c: 10 → 40` drives `u*` from 7.4% to 22.0%).

### The equilibrium objects (baseline)

| Object | Value | Meaning |
|---|---:|---|
| `w̄` | 45.29 | reservation wage |
| `E[w]` | 40.50 | mean offer (worker accepts only the top ≈31%) |
| `f` | 0.315 | job-finding rate = `P(w ≥ w̄)` |
| `1/f` | 3.18 | expected unemployment duration (periods) |
| `u*` | 13.7% | steady-state unemployment rate `α/(α+f)` |
| `U` | −0.55 | lifetime value of unemployment (CRRA units, `γ=2`) |

---

## Code structure

| File | Role |
|---|---|
| `params.jl` | `McCallModel` — all economic & numerical parameters (with defaults) |
| `offers.jl` | discretizes the log-normal offer distribution (CDF-interval method) and the persistent AR(1) offer process (Tauchen) |
| `solve.jl` | the two solvers: scalar continuation-value fixed point (`solve_mccall`) and full value-function iteration (`solve_vfi`), plus CRRA utility and its inverse |
| `correlated.jl` | the persistent-offer extension: reservation policy when offers are serially correlated |
| `simulate.jl` | Monte-Carlo labor-market histories → duration distribution, accepted wages, time-average unemployment |
| `main.jl` | assembles the blocks and solves/prints the baseline |
| `analyze.jl` | diagnostics, comparative statics, the option-value experiment, persistent offers, and all figures |

Run order: `julia analyze.jl` (it `include`s `main.jl`, solves the baseline, then runs §1–§6 below and writes figures to `graphs/`). A full pass takes ~20 s. Requires `Distributions`, `QuantEcon`, and `Plots`.

### Two solvers, on purpose

`solve_mccall` iterates the **scalar** continuation value `U` to its fixed point — a contraction with modulus `β` that updates *one number* per step (the research-standard method). `solve_vfi` instead iterates the **entire** value function `V(w)` over the offer grid, exactly as the undergraduate lecture builds it up by backward induction. They converge to the same `U`, `w̄`, and policy; §1 verifies they agree to `1e-7`. Keeping both makes the speed-up from "solve for the threshold, not the whole function" concrete, and gives an independent check that neither is buggy.

---

## Walkthrough of `analyze.jl`

### §1 — Verification *(the sanity checks)*
Three things must hold before any economics is trustworthy: the two independent solvers agree (`|U_scalar − U_vfi| ≈ 2e-9`, `|w̄_scalar − w̄_vfi| ≈ 2e-7`), the closed-form reservation-wage identity `u(w̄) = (1−β)U` holds to machine zero, and the discretized offer distribution is a genuine probability measure (`Σ pw = 1`). The CDF-interval discretization in `offers.jl` additionally reproduces the true mean offer `E[w] = exp(μ+σ²/2)` to three digits — the pointwise-density alternative leaks ≈18% of the mean out through the truncated right tail, which would bias every downstream statistic.

### §2 — Value function & reservation wage *(the central picture)*
`fig_value_function.png` plots the increasing employment value `V_e(w)` against the flat unemployment value `U`; their single crossing is `w̄`. At the baseline `w̄ = 45.29` sits *above* the mean offer (40.5) because the worker is patient and well-insured (`c = 25`), so she accepts only the top ≈31% of offers — `f = 0.315`.

### §3 — Comparative statics *(the mechanism)*
Each of `c, β, α, γ` is swept one at a time; every row is a full re-solve. The direction of `w̄` is the thing to watch, and all four move as the "is search more attractive?" logic predicts (table above). `fig_comparative_statics.png` is the 2×2 panel. The `c`-sweep is the policy headline: **unemployment insurance raises the reservation wage and lengthens unemployment** — the moral-hazard cost of insurance, visible as a clean upward line.

### §4 — Hazard, duration & steady-state unemployment *(from policy to data)*
The reservation rule turns unemployment into a geometric waiting time with success probability `f`, so mean duration is `1/f` and the steady-state unemployment rate is `α/(α+f)`. A Monte-Carlo panel of 1,500 workers over 4,000 periods confirms both to within sampling error (simulated mean duration 3.18 vs. closed-form 3.18; simulated `u*` 13.7% vs. 13.7%). `fig_duration_hist.png` overlays the simulated spell-length distribution on the exact `geometric(f)`; `fig_wage_dist.png` shows the realized accepted-wage distribution as the offer distribution **truncated at `w̄`** — the model's prediction that observed wages are a *selected* sample, never the offers below the reservation wage.

### §5 — Mean-preserving spread: the option value of search *(why risk can help)*
Hold the mean offer fixed and raise its dispersion `σ`. For McCall's **risk-neutral** worker (`γ=0`) the value of search is convex in the wage — the `max` operator discards the downside — so more dispersion strictly **raises** both `U` and the reservation wage (`w̄`: 42.9 → 48.6 as `σ`: 0.2 → 0.6). This is the **option value of search**: a worker benefits from upside risk she can decline. Add risk aversion (`γ=2`) and a second force pushes back — a risk-averse worker dislikes dispersion — so `w̄` first rises then turns down. `fig_mps.png` plots both, isolating the option-value channel from the risk-aversion channel.

### §6 — Persistent (Markov) wage offers *(dropping i.i.d.)*
McCall (1970) assumes offers are i.i.d.; empirically, labor-market luck is persistent. Here the log offer follows an AR(1) (Tauchen-discretized), so the worker's *state is the current offer*. The reservation wage is no longer one number but a threshold whose location depends on persistence `ρ`: a higher `ρ` means a bad draw today predicts bad draws tomorrow, so there is less point holding out — `w̄` falls and the stationary job-finding rate rises (`fig_correlated.png`). This is the bridge to the modern correlated-offer search literature (QuantEcon's "Job Search II").

---

## Figures produced

| File | Section | Content |
|---|---|---|
| `fig_value_function.png` | §2 | `V_e(w)`, `U`, and the reservation wage `w̄` |
| `fig_comparative_statics.png` | §3 | `w̄` vs `c`, `β`, `α`, `γ` (2×2 panel) |
| `fig_duration_hist.png` | §4 | unemployment-duration distribution: simulated vs. `geometric(f)` |
| `fig_wage_dist.png` | §4 | offer distribution vs. accepted-wage distribution (truncation at `w̄`) |
| `fig_mps.png` | §5 | mean-preserving spread: risk-neutral vs. risk-averse `w̄` |
| `fig_correlated.png` | §6 | acceptance region under persistent offers, varying `ρ` |

---

## Research lineage

- **McCall, J. J. (1970).** "Economics of Information and Job Search." *QJE* 84(1). The original sequential-search model and the reservation-wage characterization — the i.i.d., risk-neutral, no-separation core recovered here at `γ = 0, α = 0`.
- **Mortensen, D. (1970, 1986).** Job search and the labor-market matching tradition that grew out of McCall; the separation margin (`α`) that gives unemployment a steady state.
- **Rothschild, M. & Stiglitz, J. (1970).** "Increasing Risk." The convexity argument behind the option-value-of-search result in §5.
- **Tauchen, G. (1986).** The finite-state discretization used for the persistent-offer process in §6.
- **Ljungqvist, L. & Sargent, T. (2018).** *Recursive Macroeconomic Theory*, ch. 6 — the modern textbook treatment (continuation-value formulation, separations, the reservation-wage fixed point) that this code follows.
- **Stachurski, J. & Sargent, T.** *QuantEcon* lectures "Job Search I–III" — the computational reference for the scalar fixed point, the separation model, and correlated offers.
