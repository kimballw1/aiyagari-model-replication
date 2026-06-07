# aiyagari-model-replication

This repository solves the **Aiyagari (1994)** heterogeneous-agent model in Julia. Households face uninsurable idiosyncratic income shocks and a borrowing constraint, and can self-insure only by accumulating a single risk-free asset (capital). A representative firm with Cobb-Douglas technology sets the interest rate and wage competitively. The code finds the stationary general equilibrium (the interest rate at which households' precautionary savings exactly equal the capital the firm demands) via value function iteration for the household problem, forward iteration of the policy-induced (asset, income) Markov transition for the stationary wealth distribution, and a root-find over the interest rate to clear the capital market.

The borrowing constraint can be either an ad-hoc fixed bound (`borrow_limit = :adhoc`, the default, e.g. `a_min = 0`) or the Aiyagari natural borrowing limit `−w·e_min/r` (`borrow_limit = :natural`), which depends on equilibrium prices. The inequality analysis assumes non-negative wealth, so `:adhoc` is the default; `:natural` lets a large share of households borrow, which is economically valid but makes Lorenz/Gini ill-defined. *(The planned continuous / linear-interpolation variant will use the Young (2010) lottery method for the distribution; the discrete variant here does not need it, since savings land exactly on grid points.)*

---

## The economics: what Aiyagari is doing

### The cast of characters

The model has two players connected by one market:

- **Households** earn a wage by working and **save** part of it. Those savings are lent out and earn interest. Each household's income bounces around randomly over time (some years good, some bad), and crucially **they can't buy insurance against it** — no one sells a policy that pays out in their bad years.
- **Firms** rent capital (the households' pooled savings) and hire labor to produce output. They pay interest to use the capital and a wage to use the labor.
- **The asset** is the one thing households can save in — think of it as physical capital, or a bank deposit that earns interest. There's only one, and it's risk-free.

A handful of symbols show up constantly. Here's the whole glossary:

| Symbol | Plain meaning | Baseline |
|---|---|---|
| `r` | the **interest rate** — the yearly return a household earns on its savings (and what firms pay to rent capital) | solved for → 2.52% |
| `w` | the **wage** — pay per unit of work | 1.28 |
| `K` | **capital** — the economy's total stock of savings (= everyone's wealth added up) | 7.59 |
| `β` (beta) | **patience** — how much a household values next year vs. this year (0.96 = fairly patient) | 0.96 |
| `δ` (delta) | **depreciation** — fraction of capital that wears out each year | 0.08 |
| `a` | one household's **assets** (its wealth) | varies |
| `e` | a household's current **income state** (how good its luck is right now) | varies |

### The benchmark: a world with perfect insurance

Imagine first that households *could* fully insure — every bad year is covered, so income is effectively smooth and certain. Economists call this the **representative-agent** world (everyone ends up behaving identically). It has a famous, simple answer for the interest rate:

> **`r = 1/β − 1`** &nbsp;→&nbsp; with `β = 0.96`, that's **≈ 4.17%**.

The intuition: `1/β − 1` is just a measure of how impatient people are. In a calm, certain world the interest rate only has to compensate households for waiting — nothing more. Call this the **RA rate**; it's our reference point.

### Aiyagari's twist: take the insurance away

Now make the world realistic. Income is **random and uninsurable**, and households **can't borrow without limit** (there's a floor on how far into debt they can go). A household that gets unlucky can't call an insurer — its *only* defense is to have built a savings cushion ahead of time. So households **save extra, as a precaution**, beyond what they'd save in the calm benchmark. This **precautionary saving** motive is the heart of the model.

Follow the chain of consequences:

1. **Everyone over-saves for safety.** Worried about a bad income draw, each household holds a bigger buffer of assets than it would if income were certain. Add this up across everyone and the economy's total savings `K` is large.
2. **More savings means a lower return.** Firms only need so much capital. When households collectively pile up savings, capital becomes abundant — and abundant things are cheap — so the interest rate `r` they can earn **falls**. (Mechanically: firms rent capital until its payoff `r + δ` equals what one more machine produces; more capital -> each extra machine produces less -> lower `r`.)
3. **Equilibrium is where the two sides meet.** The interest rate settles at the level where the savings households *want* to hold exactly equals the capital firms *want* to rent. That balancing `r` is what the code solves for.

### The headline result

Because the precautionary cushion creates *extra* savings, the interest rate is pushed **below** the calm-world benchmark:

> **`r* = 2.52%`** &nbsp;vs.&nbsp; **`r_RA = 4.17%`** &nbsp;→&nbsp; a gap of **1.65 percentage points.**

That gap has a name — the **precautionary wedge** — and it *is* the point of the model: it's the price tag uninsurable risk puts on the economy. More risk -> more precautionary saving -> a wider wedge. Everything `analyze.jl` does is, ultimately, measuring where this wedge comes from and what the same risk does to the gap between rich and poor households.

### The equilibrium objects (baseline)

| Object | Value | Meaning |
|---|---:|---|
| `r*` | 2.52% | equilibrium net return on capital |
| `r_RA = 1/β−1` | 4.17% | complete-markets / RA rate |
| precautionary wedge | 1.65 pp | how far incomplete markets push `r` down |
| `K*` | 7.59 | aggregate capital = mean household wealth |
| `w*` | 1.28 | wage = marginal product of labor |
| wealth Gini | 0.52 | cross-sectional wealth inequality |

---

## Code structure

| File | Role |
|---|---|
| `params.jl` | `AiyagariParams` — all economic & numerical parameters (with defaults) |
| `rouwenhorst.jl` | discretizes the AR(1) income process into a finite Markov chain |
| `grids.jl` | builds the curved asset grid (dense near the borrowing constraint) |
| `vfi.jl` | household problem: value function iteration → savings & consumption policies |
| `distributions.jl` | stationary wealth distribution from the policy-induced Markov transition |
| `main.jl` | assembles the blocks and root-finds `r*` for general equilibrium |
| `analyze.jl` | post-solution diagnostics, inequality, comparative statics, plots |

Run order: `julia analyze.jl` (it `include`s `main.jl`, which solves the baseline, then runs §1–§7 below). A full pass takes ~50 s.

---

## Walkthrough of `analyze.jl`

### §1 — Baseline verification *(the sanity checks)*
Before interpreting anything, confirm the solution is internally consistent:
- **Market clearing:** `|K_supply − K_demand| ≈ 2e-4` — household savings really do equal firm demand at `r*`.
- **Valid distribution:** `Σ μ[i,j] = 1.0` — the stationary measure is a genuine probability distribution.
- **The wedge:** reports `r*`, `r_RA`, and their gap. This is the one-number summary of incomplete markets.

*Why it matters:* a general-equilibrium fixed point is easy to get subtly wrong. If the market doesn't clear or the measure doesn't sum to 1, every downstream statistic is meaningless.

### §2 — Income process (Markov chain) *(is the input risk right?)*
The wealth distribution is only as good as the income process driving it. Checks on the discretized AR(1):
- **Eigenvalues of `P`** = `[1.0, 0.9, 0.81, 0.729, …]` — these are exactly `ρ^k`, confirming Rouwenhorst reproduces the persistence `ρ = 0.9`.
- **Spectral gap** `1 − |λ₂| = 0.1` and **mixing time ≤ 56 steps** — how fast a household's income "forgets" its past. Slow mixing (persistent shocks) is what makes income risk hard to self-insure.
- **Stationary income**: `σ_e = 0.53`, income Gini `0.25`. Note income inequality (0.25) is much *lower* than the wealth inequality it generates (0.52) — incomplete markets **amplify** inequality.

### §3 — Wealth distribution & inequality *(the headline output)*
The cross-section of wealth in steady state:
- Mean 7.59, **median 5.15** (mean > median -> right-skewed), std 7.72, coefficient of variation ≈ 1.0.
- **Gini 0.52**, **top-10% share 34%**, **top-1% share 5%**.

`fig_lorenz.png` plots the **Lorenz curve** — cumulative wealth share vs. cumulative population share. The gap below the 45° "perfect equality" line *is* the inequality; the Gini is twice that area. `fig_wealth_dist.png` shows the wealth density: a mass of constrained households near `a = 0` and a long right tail.

*Key takeaway:* Aiyagari generates real wealth inequality **from identical households** — they differ only in their luck-of-the-draw income histories. (It still *understates* the top tail vs. US data ≈ 35% top-1%; fixing that needs richer ingredients — heterogeneous returns, etc.)

### §4 — Policy functions *(household behavior)*
`fig_policy_savings.png` plots next-period assets `a'(a, e)`; `fig_policy_consumption.png` plots `c(a, e)`:
- Where the savings curve crosses the 45° line is the household's **target wealth** for that income state — it saves toward it and dissaves above it.
- Higher income `e` shifts both curves up.
- Near the borrowing constraint the savings policy flattens onto the floor (constrained households consume their entire cash-on-hand). The **concavity of `c(a,e)`** is the precautionary motive made visible: marginal propensity to consume is high when poor, low when rich.

### §5 — Euler-equation residuals *(numerical accuracy)*
At an interior optimum the household's Euler equation `u'(c) = β(1+r)·E[u'(c')]` holds exactly. The residual `|LHS/RHS − 1|` measures how far the computed policy is from the true optimum:
- **Weighted-average residual ≈ 0.016** (averaged over states households actually visit), max ≈ 0.22 near the constraint kink.
- The larger errors sit exactly where the discrete grid struggles — at the borrowing-constraint kink. *This is the motivation for the planned linear-interpolation / EGM variant*, which removes the grid-snapping error.

### §6 — Comparative statics
Each parameter is swept one at a time; every row is a full re-solve. This is where the economic mechanism becomes visible. **Direction of `r*` is the thing to watch** — it reveals the strength of the precautionary motive.

| Sweep | Effect on `r*` | Economic logic |
|---|---|---|
| **β ↑** (more patient) | ↓ (6.3% → 0.8%) | patient households save more → more capital → lower return |
| **σ ↑** (more income risk) | ↓ (4.0% → −0.1%) | bigger shocks → stronger precaution → more saving |
| **ρ ↑** (more persistent) | ↓ (3.9% → 1.5%) | persistent shocks are harder to self-insure → more precaution |
| **γ ↑** (more risk averse) | ↓ (3.4% → 0.0%) | more curvature → stronger precautionary motive |
| **α ↑** (capital share) | ↑ (1.8% → 3.0%) | capital more productive → higher marginal product → higher `r` |
| **δ ↑** (depreciation) | ↓ `K*` (14.1 → 4.8) | faster wear ⇒ more investment just to stand still |
| **a_min looser** (0 → −2) | ↑ (2.5% → 2.8%) | being able to borrow weakens precaution → less saving → higher `r` |
| **n_e** (income states) | converges by 7 | confirms the Rouwenhorst approximation has stabilized |

Two limits worth internalizing:
- **As risk vanishes** (σ→0.05, γ→1, ρ low), the wedge collapses toward 0 (σ=0.05 ⇒ wedge **0.16 pp**) and `r* → r_RA`. The model continuously nests the representative agent — *no risk, no precaution, no wedge*.
- **The borrowing-constraint sweep** (`fig_borrowing_constraint.png`) is the cleanest test of the incomplete-markets channel: a tighter constraint → more precautionary saving → **lower `r*`, higher `K*`, more inequality**, all monotone.

`fig_uncertainty_tradeoff.png` isolates the σ channel: more income uncertainty raises *both* aggregate capital `K*` and the wealth Gini — risk simultaneously drives saving and inequality.

### §7 — Ergodicity *(does the steady state actually exist?)*
The stationary distribution `μ*` is only meaningful if the economy converges to it from *any* starting point. We forward-iterate the population from three extreme initial conditions (everyone poor / everyone rich / everyone at the midpoint) and track `‖μ_t − μ*‖∞`:

| t | from all-poor | from all-rich |
|---:|---:|---:|
| 50 | 0.014 | 0.016 |
| 150 | 0.0006 | 0.012 |
| 300 | 5e-5 | 0.0004 |

All paths decay geometrically to the same `μ*` (`fig_ergodicity.png`, log scale). This is the **ergodic theorem** in action: the cross-sectional distribution at a point in time equals the time-average of any single household's history — which is *why* a stationary cross-section is a legitimate object to study.

---

## Figures produced

| File | Section | Content |
|---|---|---|
| `fig_lorenz.png` | §3 | Lorenz curve + Gini |
| `fig_wealth_dist.png` | §3 | stationary wealth density |
| `fig_policy_savings.png` | §4 | savings policy `a'(a,e)` |
| `fig_policy_consumption.png` | §4 | consumption policy `c(a,e)` |
| `fig_comparative_statics.png` | §6 | `r*` vs β, σ, ρ, γ |
| `fig_uncertainty_tradeoff.png` | §6 | `K*` and Gini vs σ |
| `fig_comparative_statics2.png` | §6 | `r*` vs α, δ, a_min, n_e |
| `fig_borrowing_constraint.png` | §6 | `r*`, `K*`, Gini vs borrowing limit |
| `fig_ergodicity.png` | §7 | convergence to `μ*` from 3 starts |
