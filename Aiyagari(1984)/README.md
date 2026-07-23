# Aiyagari (1994): incomplete markets and precautionary saving

This repository solves the Aiyagari (1994) heterogeneous-agent model in Julia. Households face uninsurable idiosyncratic income shocks and a borrowing constraint, and can self-insure only by accumulating a single risk-free asset (capital). A representative firm with Cobb-Douglas technology sets the interest rate and wage competitively. The code finds the stationary general equilibrium, the interest rate at which households' precautionary savings exactly equal the capital the firm demands, via value function iteration for the household problem, forward iteration of the policy-induced (asset, income) Markov transition for the stationary wealth distribution, and a root-find over the interest rate to clear the capital market.

The borrowing constraint can be either the Aiyagari natural borrowing limit `−w·e_min/r` (`borrow_limit = :natural`, the default), which depends on equilibrium prices and lets a household borrow up to what it could repay out of the lowest possible income stream, or an ad-hoc fixed bound (`borrow_limit = :adhoc`, e.g. `a_min = 0`) that forbids borrowing entirely. The natural limit is the more standard Aiyagari object, but it lets a large share of households hold negative wealth, which makes the Lorenz curve and Gini ill-defined. So the worked baseline numbers in this README use the ad-hoc limit (`a_min = 0`) as a clean inequality reference.

---

## The model

The model has two players connected by one market:

- Households earn a wage by working and save part of it. Those savings are lent out and earn interest. Each household's income bounces around randomly over time (some years good, some bad), and crucially they can't buy insurance against it: no one sells a policy that pays out in their bad years.
- Firms rent capital (the households' pooled savings) and hire labor to produce output. They pay interest to use the capital and a wage to use the labor.
- The asset is the one thing households can save in. Think of it as physical capital, or a bank deposit that earns interest. There's only one, and it's risk-free.

A handful of symbols show up constantly:

| Symbol | Meaning | Baseline |
|---|---|---|
| `r` | the interest rate: the yearly return a household earns on its savings (and what firms pay to rent capital) | 2.52% (solved) |
| `w` | the wage: pay per unit of work | 1.28 |
| `K` | capital: the economy's total stock of savings (everyone's wealth added up) | 7.59 |
| `β` (beta) | patience: how much a household values next year vs. this year (0.96 = fairly patient) | 0.96 |
| `δ` (delta) | depreciation: fraction of capital that wears out each year | 0.08 |
| `a` | one household's assets (its wealth) | varies |
| `e` | a household's current income state (how good its luck is right now) | varies |

### The benchmark: a world with perfect insurance

Imagine first that households could fully insure, so every bad year is covered and income is effectively smooth and certain. This is the representative-agent world (everyone ends up behaving identically), and there the interest rate is pinned by patience alone:

> `r = 1/β − 1`, which at `β = 0.96` is ≈ 4.17%.

The intuition: `1/β − 1` measures how impatient people are. In a calm, certain world the interest rate only has to compensate households for waiting. Call this the RA rate; it's the reference point.

### Taking the insurance away

Now make the world realistic. Income is random and uninsurable, and households can't borrow without limit (there's a floor on how far into debt they can go). A household that gets unlucky can't call an insurer; its only defense is a savings cushion built ahead of time. So households save extra as a precaution, beyond what they'd save in the calm benchmark. This precautionary saving motive is the heart of the model, and the chain of consequences runs:

1. Everyone over-saves for safety. Worried about a bad income draw, each household holds a bigger buffer of assets than it would if income were certain. Add this up across everyone and the economy's total savings `K` is large.
2. More savings means a lower return. Firms only need so much capital. When households collectively pile up savings, capital becomes abundant, and abundant things are cheap, so the interest rate `r` they can earn falls. (Mechanically: firms rent capital until its payoff `r + δ` equals what one more machine produces; more capital means each extra machine produces less, which means lower `r`.)
3. Equilibrium is where the two sides meet. The interest rate settles at the level where the savings households want to hold exactly equal the capital firms want to rent. That balancing `r` is what the code solves for.

### The main result

Because the precautionary cushion creates extra savings, the interest rate is pushed below the calm-world benchmark:

> `r* = 2.52%` vs. `r_RA = 4.17%`, a gap of 1.65 percentage points.

That gap is called the precautionary wedge, and it's the model's central object: the price uninsurable risk puts on the economy. More risk means more precautionary saving and a wider wedge. Everything `analyze.jl` does is, one way or another, measuring where this wedge comes from and what the same risk does to the gap between rich and poor households.

### Equilibrium objects (baseline)

| Object | Value | Meaning |
|---|---:|---|
| `r*` | 2.52% | equilibrium net return on capital |
| `r_RA = 1/β−1` | 4.17% | complete-markets / RA rate |
| precautionary wedge | 1.65 pp | how far incomplete markets push `r` down |
| `K*` | 7.59 | aggregate capital = mean household wealth |
| `w*` | 1.28 | wage = marginal product of labor |
| wealth Gini | 0.52 | cross-sectional wealth inequality |

### What the interest rate really is (and why it can go negative)

`r` is the net real return on capital: the marginal product of one more unit of capital, minus depreciation, `r = MPK − δ`. Equivalently, it is the price at which a household trades consumption today for consumption next period by saving. In the complete-markets benchmark it equals pure impatience, `1/β − 1`; incomplete markets push it down by the precautionary wedge.

Nothing pins `r` above zero. If the precautionary motive is strong enough (high `σ`, `ρ`, `γ`, or a tight constraint), households pile up so much capital that its marginal product falls below depreciation, so `r = MPK − δ < 0`. Saving then literally loses value over time, yet households still do it, because in this economy capital is the only store of value: there is no other way to carry resources into a rainy day. The rate is bounded below by `−δ`. As `r → −δ`, firms' capital demand `K = L·((r+δ)/α)^{1/(α−1)}` explodes, so the market always clears at some `r > −δ` (in the code this is the `r_low = −δ + 1e-4` floor of the root-find). You can watch `r*` cross into negative territory by pushing `σ` up in §6; at `σ = 0.4`, `r* ≈ −0.1%`.

---

## Files

| File | Role |
|---|---|
| `params.jl` | `AiyagariParams`: all economic and numerical parameters (with defaults) |
| `rouwenhorst.jl` | discretizes the AR(1) income process into a finite Markov chain |
| `grids.jl` | builds the curved asset grid (dense near the borrowing constraint) |
| `vfi.jl` | household problem: value function iteration for the savings and consumption policies |
| `distributions.jl` | stationary wealth distribution from the policy-induced Markov transition |
| `main.jl` | assembles the blocks and root-finds `r*` for general equilibrium |
| `analyze.jl` | post-solution diagnostics, inequality, comparative statics, plots |

Run it with `julia analyze.jl` (it includes `main.jl`, which solves the baseline, then runs §1–§7 below). A full pass takes ~50 s.

---
