
# Walkthrough of `analyze.jl`

### §1 — Baseline verification *(the sanity checks)*
Before interpreting anything, confirm the solution is internally consistent:
- **Market clearing:** `|K_supply − K_demand| ≈ 2e-4` — household savings really do equal firm demand at `r*`.
- **Valid distribution:** `Σ μ[i,j] = 1.0` — the stationary measure is a genuine probability distribution.
- **The wedge:** reports `r*`, `r_RA`, and their gap. This is the one-number summary of incomplete markets.

### §2 — Income process (Markov chain) *(is the input risk right?)*
The wealth distribution is only as good as the income process driving it. Checks on the discretized AR(1):
- **Eigenvalues of `P`** = `[1.0, 0.9, 0.81, 0.729, …]` — these are exactly `ρ^k`, confirming Rouwenhorst reproduces the persistence `ρ = 0.9`.
- **Spectral gap** `1 − |λ₂| = 0.1` and **mixing time ≤ 56 steps** — how fast a household's income "forgets" its past. Slow mixing (persistent shocks) is what makes income risk hard to self-insure.
- **Stationary income**: `σ_e = 0.53`, income Gini `0.25`. Note income inequality (0.25) is much *lower* than the wealth inequality it generates (0.52) — incomplete markets **amplify** inequality.

### §3 — Wealth distribution & inequality *(the headline output)*
The cross-section of wealth in steady state:
- Mean 7.59, **median 5.15** (mean > median -> right-skewed), std 7.72, coefficient of variation ≈ 1.0.
- **Gini 0.52**, **top-10% share 34%**, **top-1% share 5%**.

`fig_lorenz.png` plots the **Lorenz curve** — cumulative wealth share vs. cumulative population share. The gap below the 45° "perfect equality" line *is* the inequality; the Gini is twice that area. (Both are produced only when wealth is non-negative; under the natural default they are skipped — see the Gini note below and "Reading `μ`".) `fig_wealth_dist.png` shows the wealth density, whose shape depends on the borrowing limit: under the **ad-hoc** limit, a spike of constrained households at `a = 0` plus a long right tail; under the **natural** default, a smooth hump that spreads into negative wealth with no spike (see "Reading `μ`" for why).

*Key takeaway:* Aiyagari generates real wealth inequality **from identical households** — they differ only in their luck-of-the-draw income histories. (It still *understates* the top tail vs. US data ≈ 35% top-1%; fixing that needs richer ingredients — heterogeneous returns, etc.)

#### Reading `μ` — why households sit where they do

`μ[i,j]` is the **fraction of the population** simultaneously holding asset level `a_i` and income state `e_j` in the long-run steady state; summing over income gives the marginal wealth density `μ_a[i]` in `fig_wealth_dist.png`. The key point of interpretation: `μ` is a **stationary distribution**, *not* a snapshot of "everyone parked at their optimal asset." Every household follows the *same* optimal savings policy `g_idx`, but its **position** is the accumulated result of its own random income history. Each income state has a **target wealth** (where `a'(a, e_j)` crosses the 45° line in §4), and ongoing income shocks keep knocking households off those targets — so `μ` is the long-run *scatter* of households fluctuating around them. **The policy is optimal at every state; the location is luck.**

Two regions are worth reading explicitly, because the borrowing limit reshapes them completely:

- **The spike at the constraint (ad-hoc, `a_min = 0`).** A household pushed to `a = 0` by a run of bad income draws would like to borrow to smooth consumption but cannot, so it is pinned on the floor consuming only its current cash-on-hand — *hand-to-mouth*. Many low-`(a, e)` states all map to this one point, so probability piles up there as a **spike**. These households are genuinely **constrained**: the limit binds and their Euler equation holds as a strict inequality.

- **The left tail (natural limit, `a_min = −w·e_min/r ≈ −12`).** Here households *can* borrow, so the distribution spreads into negative wealth — these are households carrying debt to ride out a low-income spell. The density is **small and tapers to zero before the limit**, with essentially no mass exactly at it, for two reasons: (1) sitting at the limit would require consuming ≈ 0 forever to stay solvent, and `u'(c) → ∞` as `c → 0`, so households **precautionarily keep a buffer away from it**; (2) reaching it takes a *sustained* run of bad luck, which is rare. The few households at the far left have **both very low assets and a low current income** — income is persistent, so deep debt and a bad income state travel together. They are doing the best they can from a bad spot (the policy is optimal), but **the state itself is an unlucky outcome, not a chosen optimum**. Crucially, unlike the ad-hoc case almost none of them are actually *constrained*: only the thin slice right at the edge has the limit binding; the rest are unconstrained borrowers choosing their debt freely.

So the same diagnostic plot tells opposite stories under the two limits: ad-hoc produces a **spike of constrained, hand-to-mouth households** at the floor, while the natural limit produces a **smooth tail of mostly-unconstrained borrowers** that fades out before the floor is reached.

#### Why households borrow — and why they sometimes don't

**What "borrowing" is, mathematically.** Each period a household chooses next-period assets `a'`, and consumption is the residual in the budget constraint:

> `c = (1+r)·a + w·e − a'  =  (cash-on-hand) − a'`, &nbsp; subject to `a' ≥ a_min`.

So `a'` is the single dial between consuming now and carrying resources forward:

- `a' > 0` — the household **saves**: it consumes *less* than its cash-on-hand and carries a positive buffer into next period.
- `a' = 0` — it holds **no assets**; consumption equals its full cash-on-hand `(1+r)a + w·e`. Under the ad-hoc limit this is the hand-to-mouth floor.
- `a' < 0` — it **borrows**: since `−a' = +|a'|`, consumption `c = (cash-on-hand) + |a'|` is *more than its entire current resources*. It pulls `|a'|` of future income into today and starts next period in the hole at `(1+r)·a'` (principal plus interest). Borrowing is therefore stronger than "running out of savings" — it is a negative asset *level*, available only when `a_min < 0` (the natural limit). In the code this is simply `g_idx[i,j]` pointing to a grid index where `a_grid[·] < 0`.

**Two motives in tension.** Whether a household picks `a' < 0` is a contest between two forces:

1. **Consumption smoothing — pushes *toward* borrowing.** When income is *temporarily* low and expected to recover, the household would like to borrow now and repay later, so consumption doesn't track every dip in income.
2. **Precaution — pushes *away from* borrowing.** In an uninsured world, debt is dangerous: a household already borrowing that draws *another* bad shock is shoved against the constraint and forced to cut consumption hard. Fear of future bad luck makes households want a positive buffer instead of debt.

Negative wealth appears in `μ` only when smoothing wins, and the parameters decide which motive dominates:

- **More risk** (`σ`, `ρ`) or **more risk aversion** (`γ`) strengthens precaution → households build buffers and refuse to borrow. Push risk high enough and *no one* holds negative wealth: the natural limit goes unused and the economy behaves like the ad-hoc `a_min = 0` case.
- **More patience** (`β`) → save rather than pull consumption forward → fewer borrowers.
- **More persistent shocks** (`ρ`) → a low income today predicts low income for a while, so there is no imminent recovery to borrow against → less borrowing. (Higher `ρ` and `σ` also lower the worst-case income `e_min`, which tightens the natural limit `−w·e_min/r` itself — less *room* to borrow — though that is partly offset in equilibrium, since more risk lowers `r*`, which loosens it.)

So when the negative-wealth mass appears or vanishes as you move a parameter, read it as the smoothing/precaution balance tipping: borrowers exist when risk is mild, shocks are transitory, and households are impatient; they disappear when risk, persistence, or patience make a precautionary buffer the better choice.

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
The stationary distribution `μ*` is only meaningful if the economy converges to it from *any* starting point. We forward-iterate the population from three extreme initial conditions (everyone poor / everyone rich / everyone at the midpoint) and track the **total-variation distance** `d_TV(μ_t, μ*) = ½·Σ|μ_t − μ*|` over `T = 1200` steps (`fig_ergodicity.png`, log scale). All three collapse onto the same `μ*` and **meet at the same floor**, confirming ergodicity.

This is the **ergodic theorem** in action: the cross-sectional distribution at a point in time equals the time-average of any single household's history — which is *why* a stationary cross-section is a legitimate object to study.

Three details in that plot are worth understanding, because they are easy to misread:

- **Why total variation, not the sup-norm.** TV is the standard metric for Markov mixing (it is what a "mixing time" refers to) and is provably non-increasing along the chain. The sup-norm `‖·‖∞` tracks only the single largest cell, which can cross `μ*` from above to below near convergence and momentarily collapse to zero — a spurious downward spike on a log plot. The earlier sup-norm version had exactly that artifact.
- **Why the three lines sit at different heights before meeting.** Convergence is geometric, `d_t ≈ C·|λ₂|^t`, but the constant `C` depends on how far the starting point is from `μ*`. "Everyone rich" starts in the extreme top wealth corner — farthest from where `μ*` puts its mass — so it has the largest `C` and lags; "midpoint" starts nearest the bulk and converges first. They are roughly parallel lines at different heights, not coincident lines, until they all bottom out at the numerical floor.
- **Why convergence is slow (and what sets the rate).** The measured decay rate is `|λ₂| ≈ 0.98`, far slower than the *income* chain's second eigenvalue `ρ = 0.9` (whose mixing time is ~56 steps, §2). The bottleneck is **wealth accumulation, not income mixing**: it physically takes many periods for a household to save its way from a corner of the asset grid out to the stationary spread. So the economy's relaxation time is governed by the slow asset dimension. (`T = 1200` is chosen long enough that even this slow mode reaches the floor, so the lines visibly meet. The reference `μ*` is solved to a tight tolerance — see `solve_distribution` — otherwise an imperfect reference would leave the trajectories passing *through* it and producing a dip.)

---

## Figures produced

| File | Section | Content |
|---|---|---|
| `fig_lorenz.png` | §3 | Lorenz curve + Gini *(skipped when wealth can be negative, i.e. under the natural limit)* |
| `fig_wealth_dist.png` | §3 | stationary wealth density |
| `fig_policy_savings.png` | §4 | savings policy `a'(a,e)` |
| `fig_policy_consumption.png` | §4 | consumption policy `c(a,e)` |
| `fig_comparative_statics.png` | §6 | `r*` vs β, σ, ρ, γ |
| `fig_uncertainty_tradeoff.png` | §6 | `K*` and Gini vs σ |
| `fig_comparative_statics2.png` | §6 | `r*` vs α, δ, a_min, n_e |
| `fig_borrowing_constraint.png` | §6 | `r*`, `K*`, Gini vs borrowing limit |
| `fig_ergodicity.png` | §7 | TV-distance convergence to `μ*` from 3 starts |
