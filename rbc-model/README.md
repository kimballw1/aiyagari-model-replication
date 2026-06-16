# rbc-model-replication

This repository solves the **Real Business Cycle** model — Kydland & Prescott's (1982) "Time to Build and Aggregate Fluctuations," the paper that won the 2004 Nobel Prize and launched modern quantitative macroeconomics — in Julia. A representative household/planner in a perfectly competitive economy chooses consumption, labor, and capital accumulation in response to persistent **total-factor-productivity (TFP) shocks**. The radical claim of RBC theory is that business cycles need *no* market failures: they are the **optimal response** of an efficient economy to random shifts in technology. The code solves the model two ways — exact value-function iteration and a log-linear (Klein QZ) perturbation — calibrates it to the standard quarterly U.S. parameters, and confronts the simulated economy with the actual business-cycle moments it was *not* calibrated to match.

It is the macro capstone of the McCall → Rust → RBC progression: McCall and Rust are single-agent dynamic programs; here the same dynamic-programming and Markov-chain tools scale up to a whole economy. Beyond the undergraduate version (an inelastic-labor stochastic growth model solved by VFI), this repo restores the features that make RBC a serious quantitative theory: **endogenous labor supply**, **Hansen's (1985) indivisible-labor** refinement, the **log-linearization** that is the field's workhorse solver, **HP-filtered second moments vs. U.S. data**, and **impulse responses**.

---

## The economics: what Kydland–Prescott is doing

### The model in one breath

A single forward-looking planner maximizes expected lifetime utility

> `max E Σ βᵗ [ log cₜ − χ·nₜ^{1+1/ν}/(1+1/ν) ]`

subject to the economy's resource constraint

> `cₜ + kₜ₊₁ = zₜ·kₜ^α·nₜ^{1−α} + (1−δ)·kₜ`,  with  `log zₜ₊₁ = ρ·log zₜ + εₜ`.

Output is produced from capital `k` and labor `n` with a Cobb–Douglas technology scaled by productivity `z`. Each period the planner splits output between **consumption** (enjoyed now) and **investment** (capital for tomorrow). The *only* source of randomness is the TFP shock `z` — a persistent AR(1). The question KP asked: can this stripped-down, frictionless economy, hit only by technology shocks, reproduce the comovements we call the business cycle? The answer — surprisingly — is largely yes.

### The cast of characters

| Symbol | Plain meaning | Baseline (quarterly) |
|---|---|---|
| `z` | **TFP** — the technology shock driving everything | AR(1), mean 1 |
| `k` | **capital** — the economy's accumulated savings | `K/Y ≈ 2.6` (annual) |
| `c` | **consumption** | `C/Y ≈ 0.74` |
| `i` | **investment** = `k' − (1−δ)k` | `I/Y ≈ 0.26` |
| `n` | **labor** (hours worked) | `n* = 1/3` |
| `β` | **patience** — discount factor | 0.99 |
| `α` | **capital share** of output | 0.36 |
| `δ` | **depreciation** rate | 0.025 |
| `ρ, σ_ε` | TFP **persistence** and innovation size | 0.95, 0.007 |
| `ν` | **Frisch elasticity** of labor supply | 2 (or ∞ for Hansen) |

The calibration is the standard quarterly RBC one (King–Plosser–Rebelo / Hansen): `β` set so the annual real rate is ≈4%, `α` from capital's income share, `δ` from the investment rate, and `(ρ, σ_ε)` from the estimated Solow-residual process. `χ` is *not* free — it is pinned down so steady-state hours are `n* = 1/3`.

### The propagation mechanism: why a transitory shock makes a cycle

The heart of RBC is **capital accumulation as an internal propagation mechanism**. A positive TFP shock makes capital and labor more productive, so the planner produces more. What does it do with the extra output? It **invests** a lot of it — building up the capital stock. That higher capital stays productive for many periods, *after the TFP shock itself has decayed*. So a short-lived impulse to `z` becomes a long, hump-shaped swing in output: the economy's capital stock is a flywheel that smooths and stretches the shock. This is visible in `fig_irf.png` — TFP decays geometrically, but capital builds slowly and output's response outlives the shock.

A second margin does the amplifying: **consumption smoothing**. Because households dislike volatile consumption (concave utility), they let *investment* absorb most of the shock while keeping consumption smooth. That single optimization choice generates two of the central business-cycle facts at once.

### The headline results

**1. The RBC business-cycle facts.** Simulating the model, HP-filtering the log series (λ=1600, exactly as the data are treated), and comparing to postwar U.S. data:

| Variable | σ (model) | σ (US) | σ/σ_y (model) | σ/σ_y (US) | corr w/ y (model) | corr (US) |
|---|---:|---:|---:|---:|---:|---:|
| Output | 1.3% | 1.8% | 1.00 | 1.00 | 1.00 | 1.00 |
| Consumption | 0.4% | 1.4% | **0.31** | 0.74 | 0.89 | 0.88 |
| Investment | 4.1% | 5.3% | **3.11** | 2.93 | 0.99 | 0.80 |
| Hours | 0.6% | 1.8% | 0.49 | 0.99 | 0.98 | 0.88 |

The model nails the qualitative facts from a *single* shock: **consumption is smoother than output, investment is ≈3× more volatile, and everything is procyclical.** (US column: King & Rebelo 1999.)

**2. TFP shocks explain most of the cycle.** With endogenous (divisible) labor the model produces `σ_y = 1.3%` against the data's 1.8% — technology shocks alone account for ≈70% of output volatility, Prescott's famous and contested claim.

**3. Labor supply is the swing factor (Hansen 1985).** The baseline's weak point is hours: with inelastic labor, hours don't move at all (`σ_n/σ_y = 0`) and output volatility is only 0.9%. The single most important refinement to KP — **Hansen's (1985) indivisible labor** (employment lotteries, which make aggregate labor supply highly elastic) — fixes this:

| Labor specification | `σ_y` | `σ_n/σ_y` |
|---|---:|---:|
| Inelastic (`n ≡ 1`) | 0.9% | 0.00 |
| Divisible (Frisch `ν = 2`) | 1.3% | 0.49 |
| **Indivisible (Hansen)** | **1.8%** | **0.76** |

Hansen's indivisible labor lifts output volatility to **1.8% — essentially the U.S. number** — by letting hours respond strongly to productivity. This is why endogenous labor is non-negotiable in serious RBC work, and `fig_labor_comparison.png` shows the hours response growing from flat (inelastic) to large (Hansen).

---

## Code structure

| File | Role |
|---|---|
| `params.jl` | `RBCModel` — parameters, the labor-supply switch, the quarterly calibration |
| `steadystate.jl` | deterministic steady state and the implied calibration of `χ` |
| `rouwenhorst.jl` | discretizes the AR(1) TFP process into a Markov chain |
| `linearize.jl` | **the workhorse**: log-linearizes the equilibrium and solves it with Klein's (2000) QZ method; impulse responses |
| `vfi.jl` | exact value-function iteration with interpolation (the nonlinear benchmark) |
| `moments.jl` | simulation, the Hodrick–Prescott filter, and the business-cycle scorecard |
| `main.jl` | assembles the blocks; prints the calibration and headline moments |
| `analyze.jl` | calibration, policies, linear-vs-exact validation, the scorecard, IRFs, the role of labor, comparative statics |

Run order: `julia analyze.jl` (it `include`s `main.jl`, then runs §1–§7 and writes figures to `graphs/`). A full pass takes ~50 s — the exact VFI benchmark dominates; everything built on the log-linear solution is near-instant. Requires `Interpolations`, `Optim`, and `Plots`.

### Two solvers, on purpose

`solve_linear` (Klein QZ) is the **first-order perturbation** solution — the standard fast RBC method behind Dynare-style toolkits. It log-linearizes the equilibrium conditions around the steady state and solves the resulting saddle-path-stable rational-expectations system via the generalized Schur decomposition. It is instant and handles endogenous labor for free, so it drives the moments, IRFs, and comparative statics. `solve_vfi` is the **exact, nonlinear** value-function iteration of the undergraduate lecture (inelastic labor). §3 confirms the two savings policies coincide near the steady state, validating the linearization (capital persistence: VFI 0.978 vs. linear 0.965; the small gap is VFI grid/interpolation noise).

---

## Walkthrough of `analyze.jl`

### §1 — Calibration & steady state *(matching the long run first)*
Parameters are pinned to long-run averages — the capital share, the real interest rate, the investment rate, the Solow-residual process — *not* to the cyclical moments we then ask the model to explain. The steady-state great ratios come out at `K/Y ≈ 2.6` (annual), `C/Y ≈ 0.74`, `I/Y ≈ 0.26`, all in the empirical range. This calibration-not-estimation discipline is a methodological signature of the RBC program.

### §2 — Policy functions *(exact, by VFI)*
`fig_policy_functions.png` shows the savings policy `k'(k,z)` crossing the 45° line at each productivity-specific steady state, and the consumption policy rising in both capital and TFP; `fig_value_function.png` shows the concave value function. Investment responds far more than consumption — the first sign of the volatility ranking to come.

### §3 — Log-linear vs. exact *(validating the perturbation)*
The log-linear savings policy is a first-order Taylor expansion about the steady state; `fig_linear_vs_vfi.png` overlays it on the exact VFI policy. They are indistinguishable near `k_ss` and fan apart only in the tails — confirming both solvers are correct and that the (fast) linearization is trustworthy for business-cycle analysis, where the economy stays near the steady state.

### §4 — The business-cycle scorecard *(the moment of truth)*
Simulate, HP-filter, and confront the model with U.S. data (table above). Given one shock, the model reproduces the relative volatilities and comovements of the macro aggregates — moments it never targeted. `fig_simulation.png` shows a sample path with smooth consumption and volatile investment tracking output.

### §5 — Impulse responses *(the dynamic signature)*
A one-standard-deviation TFP innovation (`fig_irf.png`): output jumps on impact, **investment jumps several times more** (it is the adjustment margin), **consumption rises smoothly** (smoothing), hours rise with productivity, and **capital builds slowly** — propagating the shock long after TFP itself has decayed. This is the internal propagation mechanism made visible.

### §6 — The role of labor *(inelastic → divisible → Hansen)*
The progression that turned RBC from suggestive to quantitative. Inelastic labor produces almost no hours fluctuation; divisible labor adds some; **Hansen's (1985) indivisible labor** makes aggregate hours highly elastic and brings the model's output and hours volatility close to the data (table above). `fig_labor_comparison.png` plots the hours impulse response under each specification.

### §7 — Comparative statics *(persistence and propagation)*
Sweeping TFP persistence `ρ` shows two things at once (`fig_comparative_statics.png`). The *impulse responses* grow larger and longer-lived as `ρ → 1` — a more persistent shock keeps productivity elevated for longer, so output's response is bigger and more drawn out. But the *HP-filtered* output volatility is hump-shaped in `ρ` (≈1.3–1.4% for `ρ ≤ 0.95`, then falling at `ρ = 0.99`): a near-random-walk shock pushes most of its variance to low frequencies, which the HP filter — by construction — strips out as "trend." This gap between the raw propagation and the *filtered* moment is itself an important lesson about how the RBC scorecard is constructed. Note also that capital's own persistence (`P[1,1] ≈ 0.954`) is *independent of `ρ`*: it is the stable eigenvalue of the capital-accumulation dynamics, set by `α, β, δ` alone, while `ρ` governs only how TFP loads onto capital.

---

## Figures produced

| File | Section | Content |
|---|---|---|
| `fig_policy_functions.png` | §2 | exact savings & consumption policies `k'(k,z)`, `c(k,z)` |
| `fig_value_function.png` | §2 | value function `V(k,z)` |
| `fig_linear_vs_vfi.png` | §3 | log-linear vs. exact savings policy (validation) |
| `fig_simulation.png` | §4 | simulated output, consumption, investment |
| `fig_irf.png` | §5 | impulse responses of `y, c, i, n, k, z` to a TFP shock |
| `fig_labor_comparison.png` | §6 | hours impulse response by labor specification |
| `fig_comparative_statics.png` | §7 | output volatility and IRF vs. TFP persistence `ρ` |

---

## Research lineage

- **Kydland, F. & Prescott, E. (1982).** "Time to Build and Aggregate Fluctuations." *Econometrica* 50(6). The founding RBC paper (2004 Nobel). This repo implements the core stochastic-growth mechanism; KP's literal *time-to-build* investment lag and labor-leisure choice are the features the undergraduate version drops and this one partly restores (endogenous labor).
- **Hansen, G. (1985).** "Indivisible Labor and the Business Cycle." *JME*. The employment-lottery model that makes aggregate labor supply highly elastic — the key refinement (§6) that brings hours and output volatility to the data.
- **King, R., Plosser, C. & Rebelo, S. (1988).** "Production, Growth and Business Cycles." *JME*. The canonical balanced-growth RBC and the preferences/calibration used here.
- **Prescott, E. (1986).** "Theory Ahead of Business Cycle Measurement." The "TFP explains ~70% of fluctuations" claim and the HP-filter methodology.
- **Hodrick, R. & Prescott, E. (1997).** The HP filter (`moments.jl`).
- **Klein, P. (2000).** "Using the generalized Schur form to solve a multivariate linear rational expectations model." *JEDC*. The QZ solution method in `linearize.jl`.
- **King, R. & Rebelo, S. (1999).** "Resuscitating Real Business Cycles." *Handbook of Macroeconomics*. The U.S. moments in the §4 scorecard.
- **Rouwenhorst, K. (1995).** The Markov discretization of the TFP process.
