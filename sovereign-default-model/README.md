# sovereign-default-model-replication

This repository solves the **Eaton–Gersovitz (1981) / Arellano (2008)** model of sovereign default in Julia — the workhorse theory of why governments borrow from abroad, when they choose to default, and how lenders price that risk into bond spreads. A small open economy's government receives a stochastic endowment and borrows by issuing one-period bonds to risk-neutral international lenders. Each period, after seeing its income, it chooses whether to **repay** or **default**; default erases the debt but triggers exclusion from credit markets and an output loss. The key object is the **endogenous bond price schedule** `q(b', y)`: lenders, anticipating default, charge more (a lower price) for larger loans and for loans made in bad times, so the interest rate the government faces is determined inside the model. The code solves the joint fixed point of value functions and prices, then confronts the simulated economy with the business-cycle facts of emerging-market debt crises.

It is the open-economy capstone of the McCall → Rust → RBC → sovereign-default progression. Like Rust, it is a dynamic discrete choice (repay vs. default) smoothed by a taste shock; like the RBC model, it is driven by a persistent income process and judged by how well it matches data moments. What is new is the **price feedback**: the government's default policy determines the bond price, and the bond price determines how much it wants to borrow — two fixed points solved together.

---

## The economics: what Arellano is doing

### The government's problem

A benevolent government maximizes the representative household's expected utility, `E Σ βᵗ u(cₜ)`, facing a stochastic endowment `y` (an AR(1) in logs). It can borrow or save through one-period bonds `b'` (with `b' < 0` meaning debt). Each period it makes two nested choices:

1. **Repay or default?** If it repays, it honors `b` and can issue new debt `b'` at price `q(b', y)`. If it defaults, it keeps the resources it would have repaid but suffers two costs.
2. **If repaying, how much to borrow?** Choose `b'` to trade off consumption today against the price of debt and the risk it creates for tomorrow.

The two costs of default are what make debt sustainable at all:

- **Exclusion** from credit markets — the country regains access only with probability `λ` each period (here `λ = 0.282`, ≈ 3.5 quarters of autarky).
- **An output loss** while in default. Following Arellano, the loss is **asymmetric**: output is capped at `ŷ·E[y]`, so default is *costlier in good times* (when output is high) and nearly free in bad times. This asymmetry is the engine of **countercyclical default**.

### The cast of characters

| Symbol | Plain meaning | Baseline |
|---|---|---|
| `y` | **endowment** (income) — a persistent AR(1) shock | mean 1 |
| `b` | **bond position** — `b < 0` is debt owed, `b > 0` is savings | varies |
| `q(b',y)` | **bond price** — what lenders pay per unit of promised repayment | ≤ 1/(1+r) |
| `β` | **patience** — discount factor | 0.96 |
| `σ` | **risk aversion** (CRRA) | 2.0 |
| `r` | **risk-free rate** lenders can earn elsewhere | 1.7%/qtr |
| `λ` | **re-entry probability** after default | 0.282 |
| `ŷ` | **output-cost ceiling** in default (× mean income) | 0.969 |
| `ρ, σ_ε` | income **persistence** and innovation size | 0.945, 0.025 |

The calibration is Arellano's quarterly Argentina one; `β` is set to ≈4% annual default (Arellano's published value is 0.953, chosen the same way — to match the default frequency).

### The three equilibrium relationships

**1. The default decision is a debt threshold that tightens in bad times.** The government defaults when the value of repaying, `Vᴿ(b,y)`, falls below the value of defaulting, `Vᴰ(y)`. Since `Vᴿ` rises with bond holdings (less debt is better) while `Vᴰ` does not depend on `b`, there is a **maximum sustainable debt** for each income level — and it collapses when income is low:

| Income `y/E[y]` | Max sustainable debt `−b` |
|---|---:|
| 0.79 (low) | ≈ 0 (default at almost any debt) |
| 1.00 (mean) | 0.097 |
| 1.27 (high) | 0.45 (the grid ceiling) |

A government that draws a bad income shock while carrying debt is pushed over its threshold and defaults — so **default happens in recessions**, exactly as in the data.

**2. The bond price schedule is an endogenous credit limit.** Risk-neutral lenders break even, so

> `q(b', y) = (1 − E[default next period]) / (1 + r)`.

The price **falls as the government borrows more** (more debt → more likely to default → lenders demand a discount) and **rises with income**. Past a point, lending more actually *reduces* the revenue `q·b'` raised — the schedule is a soft, endogenous debt limit, with no hard borrowing constraint imposed anywhere.

**3. Spreads are the mirror image of the price.** The interest-rate spread over the risk-free rate, `(1/q)⁴ − (1+r)⁴` annualized, rises steeply with debt and in bad states. Because default clusters in recessions, **spreads are countercyclical** — they spike precisely when income is low and the government is most stretched.

### The headline results (the Arellano scorecard)

Simulating the model and comparing to Argentine data (Arellano 2008):

| Moment | Model | Argentina |
|---|---:|---:|
| Default rate (annual) | 4.0% | ≈ 3.0% |
| Mean debt / output | 3.6% | ≈ 6.0% |
| Mean spread (annual) | 9.2% | ≈ 3.6% |
| σ(c)/σ(y) | **1.02** | ≈ 1.10 |
| corr(spread, y) | **−0.53** | ≈ −0.7 |
| corr(trade balance, y) | **−0.10** | ≈ −0.25 |

The model reproduces the **defining qualitative facts** of emerging-market business cycles, all of which a frictionless model gets *wrong*:

- **Consumption is more volatile than output** (`σ_c/σ_y > 1`). In a normal economy, access to borrowing *smooths* consumption below output volatility. Here, default risk does the opposite: when income falls, spreads rise and credit dries up exactly when the country wants to borrow, so it is forced to cut consumption by *more* than output fell.
- **Countercyclical interest-rate spreads** — borrowing costs spike in recessions.
- **Countercyclical trade balance** — the country runs surpluses (repays/saves) in bad times and deficits (borrows) in good times.

The `fig_default_episode.png` event study makes the mechanism vivid: in the run-up to a default, debt is high and spreads climb; at default, income has dropped sharply and consumption collapses; afterward the country is excluded and re-enters deleveraged.

### An honest limitation (and where the literature went)

The model matches the default rate and every qualitative fact, but it **overstates spreads (9% vs. 3.6%) and understates debt (3.6% vs. 6%)**. This is not a bug — it is the well-known structural limitation of the **one-period-bond** model: to deter default at realistic debt levels, short-term debt needs counterfactually high and volatile spreads. Fixing it is exactly what the modern literature did, by adding **long-maturity debt** (Hatchondo–Martinez 2009; Chatterjee–Eyigungor 2012), which lets governments sustain more debt at lower spreads because rolling over long bonds dilutes the cost of a marginal default. The comparative statics in §6 show the same tension cleanly: a *longer* exclusion penalty (lower `λ`) lets the country carry **9.2% debt at a 2.8% spread** — much closer to the data — because costlier default supports more borrowing.

---

## Code structure

| File | Role |
|---|---|
| `params.jl` | `SovDefaultModel` — parameters, grids, and the output-cost specification |
| `income.jl` | Rouwenhorst discretization of the endowment, CRRA utility, the default output cost |
| `solve.jl` | the joint fixed point of value functions `(Vᴿ, Vᴰ)` and the bond price `q`, with taste-shock smoothing |
| `simulate.jl` | simulates debt, spreads, and default episodes (hard-max default rule) |
| `moments.jl` | the Arellano business-cycle scorecard |
| `main.jl` | assembles the blocks and solves/prints the baseline |
| `analyze.jl` | value functions, the price schedule, the default event study, the scorecard, and comparative statics |

Run order: `julia analyze.jl` (it `include`s `main.jl`, solves the baseline, then runs §1–§6 and writes figures to `graphs/`). A full pass takes ~60 s. Requires `Distributions`, `Interpolations`, and `Plots`.

### A note on the two fixed points and the taste shock

Unlike the RBC model (one fixed point, on `V`), the sovereign-default model iterates **two coupled fixed points**: given the price `q`, the government's values `(Vᴿ, Vᴰ)` are a contraction; given those values, lenders re-price the bonds — and the new prices change borrowing, which changes default, which changes prices. The iteration is stabilized by **damping** the price update and by a vanishing **Type-I-EV taste shock** (scale `τ = 0.001`, the same log-sum-exp device as the Rust model) that smooths the binary repay/default choice into a logistic default probability, keeping `q(b', y)` continuous near the default boundary. The taste shock is a *numerical* smoother only: the simulation applies the true **hard-max** default rule (default iff `Vᴰ > Vᴿ`), so it does not inject spurious defaults.

---

## Walkthrough of `analyze.jl`

### §1 — Value functions & the default decision *(when does a country default?)*
`fig_value_functions.png` plots the repayment value `Vᴿ(b,y)` against the flat default value `Vᴰ(y)`; their crossing is the default threshold. `fig_default_set.png` maps the default region in `(b, y)` space — a wedge in the bottom-left (high debt, low income). The threshold table above is the model's core prediction: **default tolerance collapses in recessions.**

### §2 — The bond price schedule & spreads *(how lenders price risk)*
`fig_bond_price.png` shows `q(b', y)` falling with borrowing and rising with income; `fig_spreads.png` translates it into the spread schedule (steeply rising) and the smooth default probability. The price schedule *is* the endogenous debt limit — no hard constraint is imposed.

### §3 — Borrowing policy *(how much to borrow)*
`fig_borrowing_policy.png` plots `b'(b, y)`: below the 45° line the government accumulates debt, above it deleverages. Higher income supports more borrowing because the price schedule is looser there.

### §4 — A default episode *(the anatomy of a crisis)*
An event study averaging the dynamics in a 12-quarter window around default onsets (`fig_default_episode.png`): spreads climb and debt peaks beforehand, income drops sharply at the default, consumption collapses (the output cost), and the country re-enters deleveraged. This is the model's account of a sovereign-debt crisis.

### §5 — The business-cycle scorecard *(model vs. Argentina)*
The full moment comparison (table above). The signature successes — `σ_c/σ_y > 1`, countercyclical spreads, countercyclical trade balance — are facts a frictionless small-open-economy model gets *backwards*, and they fall out of default risk and credit exclusion. The spread/debt gap is discussed under "An honest limitation" above.

### §6 — Comparative statics *(what disciplines default)*
Two clean experiments (`fig_comparative_statics.png`). **Patience** (`β`): a more patient government values future market access more and defaults far less (default falls from 10% at `β=0.94` to 3.6% at `β=0.96`). **Exclusion length** (`λ`): a *shorter* exclusion (higher `λ`) makes default cheaper, so the country defaults more and can borrow less — and conversely a longer exclusion supports much more debt at lower spreads, illuminating the debt-vs-spread tension at the heart of the model.

---

## Figures produced

| File | Section | Content |
|---|---|---|
| `fig_value_functions.png` | §1 | repayment value `Vᴿ(b,y)` vs. default value `Vᴰ(y)` |
| `fig_default_set.png` | §1 | the default region in `(b, y)` space |
| `fig_bond_price.png` | §2 | the endogenous bond price schedule `q(b',y)` |
| `fig_spreads.png` | §2 | the spread schedule and default probability |
| `fig_borrowing_policy.png` | §3 | borrowing policy `b'(b,y)` |
| `fig_default_episode.png` | §4 | event study of income, consumption, debt, and spreads around defaults |
| `fig_comparative_statics.png` | §6 | default rate vs. patience `β` and exclusion length `λ` |

---

## Research lineage

- **Eaton, J. & Gersovitz, M. (1981).** "Debt with Potential Repudiation." *REStud* 48(2). The founding model of sovereign borrowing with the option to default and the exclusion penalty.
- **Arellano, C. (2008).** "Default Risk and Income Fluctuations in Emerging Economies." *AER* 98(3). The quantitative model implemented here: the asymmetric output cost, the endogenous price schedule, and the countercyclical-spread results.
- **Aguiar, M. & Gopinath, G. (2006).** "Defaultable Debt, Interest Rates and the Current Account." *JIE*. The companion quantitative default model emphasizing trend shocks.
- **Hatchondo, J. C. & Martinez, L. (2009).** Long-duration bonds — the first fix for the debt/spread tension noted above.
- **Chatterjee, S. & Eyigungor, B. (2012).** "Maturity, Indebtedness, and Default Risk." *AER*. Long-term debt with a continuous shock for computability; the modern benchmark.
- **Hatchondo, Martinez & Sapriza (2010).** The taste-shock / smoothing and computational techniques used in `solve.jl`.
- **Tauchen/Rouwenhorst** — discretization of the endowment process.
