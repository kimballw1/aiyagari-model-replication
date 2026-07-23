# Kydland–Prescott (1982): real business cycles

In this repo I solve the Real Business Cycle model of Kydland and Prescott (1982) in Julia. The paper won the 2004 Nobel Prize and launched modern quantitative macroeconomics. A representative household/planner in a perfectly competitive economy chooses consumption and capital accumulation in response to persistent total-factor-productivity (TFP) shocks. The claim of RBC theory is that business cycles need no market failures: they are the optimal response of an efficient economy to random shifts in technology.

I solve the model by value function iteration, with inelastic labor (everyone works a fixed amount every period). An earlier version of this repo also had a log-linearization solved with Klein's QZ method, which is the standard fast way to solve RBC models and the practical way to add an hours choice. I pulled it out: I don't understand the method well enough yet to defend every line of it, and I'd rather this repo only contain things I do. Once I've properly learned the linearization it comes back, and endogenous labor with it.

---

## The planner's problem

A single forward-looking planner maximizes expected lifetime utility

> `max E Σ βᵗ log cₜ`

subject to the economy's resource constraint

> `cₜ + kₜ₊₁ = zₜ·kₜ^α + (1−δ)·kₜ`,  with  `log zₜ₊₁ = ρ·log zₜ + εₜ`.

Output is produced from capital `k` (and a fixed unit of labor) with a Cobb–Douglas technology scaled by productivity `z`. Each period the planner splits output between consumption (enjoyed now) and investment (capital for tomorrow). The only source of randomness is the TFP shock `z`, a persistent AR(1). The question Kydland and Prescott asked: can this stripped-down, frictionless economy, hit only by technology shocks, reproduce the comovements we call the business cycle? The answer, surprisingly, is largely yes.

### Variables

| Symbol | Meaning | Baseline (quarterly) |
|---|---|---|
| `z` | TFP, the technology shock driving everything | AR(1), mean 1 |
| `k` | capital, the economy's accumulated savings | `K/Y ≈ 2.6` (annual) |
| `c` | consumption | `C/Y ≈ 0.74` |
| `i` | investment = `k' − (1−δ)k` | `I/Y ≈ 0.26` |
| `β` | discount factor | 0.99 |
| `α` | capital share of output | 0.36 |
| `δ` | depreciation rate | 0.025 |
| `ρ, σ_ε` | TFP persistence and innovation size | 0.95, 0.007 |

The calibration is the standard quarterly RBC one (King–Plosser–Rebelo / Hansen): `β` set so the annual real rate is ≈4%, `α` from capital's income share, `δ` from the investment rate, and `(ρ, σ_ε)` from the estimated Solow-residual process.

### Why a transitory shock makes a cycle

The heart of RBC is capital accumulation as an internal propagation mechanism. A positive TFP shock makes capital and labor more productive, so the planner produces more. What does it do with the extra output? It invests a lot of it, building up the capital stock. That higher capital stays productive for many periods, after the TFP shock itself has decayed. So a short-lived impulse to `z` becomes a long, hump-shaped swing in output: the capital stock is a flywheel that smooths and stretches the shock. You can see this in `fig_irf.png`, where TFP decays geometrically but capital builds slowly, peaking around quarter 28, and output's response outlives the shock.

A second margin does the amplifying: consumption smoothing. Because households dislike volatile consumption (concave utility), they let investment absorb most of the shock while keeping consumption smooth. In the impulse responses, a 0.7% TFP shock moves investment 1.9% on impact but consumption only 0.3%. That single optimization choice generates two of the central business-cycle facts at once.

### Results

**1. The business-cycle facts.** Simulating the model, HP-filtering the log series (λ=1600, exactly as the data are treated), and comparing to postwar U.S. data:

| Variable | σ (model) | σ (US) | σ/σ_y (model) | σ/σ_y (US) | corr w/ y (model) | corr (US) |
|---|---:|---:|---:|---:|---:|---:|
| Output | 0.95% | 1.81% | 1.00 | 1.00 | 1.00 | 1.00 |
| Consumption | 0.37% | 1.35% | 0.39 | 0.74 | 0.92 | 0.88 |
| Investment | 2.75% | 5.30% | 2.90 | 2.93 | 0.99 | 0.80 |

The model gets the qualitative facts right from a single shock: consumption is smoother than output, investment is about 3× more volatile, and everything is procyclical. (US column: King & Rebelo 1999.)

**2. What's missing: hours.** With inelastic labor, output volatility is 0.95% against the data's 1.81%, so TFP shocks alone explain only about half the cycle here. The gap is hours: in the data hours are nearly as volatile as output, and in this version they cannot move at all. This is exactly the margin the literature fixed. Endogenous labor supply gets σ_y to about 1.3%, and Hansen's (1985) indivisible labor (employment lotteries, which make aggregate labor supply highly elastic) gets it to essentially the U.S. number. Adding an hours choice to the VFI is possible but slow and fiddly; the linearized model handles it almost for free, which is why that extension left with the Klein solver and will return with it.

**3. Persistence matters for the shock, not the economy.** Sweeping the TFP persistence `ρ` from 0.80 to 0.99 raises output volatility from 0.86% to 0.97%, and the output impulse response changes character: at `ρ = 0.99` it is hump-shaped, still rising five years after the shock. Meanwhile the savings policy's slope at the steady state, the economy's internal capital persistence, stays at ≈0.98 across the whole sweep. The economy's propagation is set by `α, β, δ`; `ρ` only governs how long the impulse keeps feeding it.

---

## Files

| File | Role |
|---|---|
| `params.jl` | `RBCModel`: parameters and the quarterly calibration |
| `steadystate.jl` | deterministic steady state |
| `rouwenhorst.jl` | discretizes the AR(1) TFP process into a Markov chain |
| `vfi.jl` | value-function iteration, plus impulse responses from the VFI policy |
| `moments.jl` | simulation, the Hodrick–Prescott filter, and the business-cycle scorecard |
| `main.jl` | assembles the blocks; prints the calibration and headline moments |
| `analyze.jl` | calibration, policies, the scorecard, IRFs, and comparative statics |

Run `julia analyze.jl`; it includes `main.jl`, works through §1–§5, and writes the figures to `graphs/`. A full pass takes 2–3 minutes: everything runs off the VFI solution, and the §5 sweep re-solves the model once per `ρ` at ~30 s a solve. Requires `Interpolations`, `Optim`, and `Plots`.

## Figures

| File | Section | Content |
|---|---|---|
| `fig_policy_functions.png` | §2 | savings and consumption policies `k'(k,z)`, `c(k,z)` |
| `fig_value_function.png` | §2 | value function `V(k,z)` |
| `fig_simulation.png` | §3 | simulated output, consumption, investment |
| `fig_irf.png` | §4 | impulse responses of `y, c, i, k, z` to a TFP shock |
| `fig_comparative_statics.png` | §5 | output volatility and IRF vs. TFP persistence `ρ` |

---

## References

- Kydland, F. & Prescott, E. (1982). "Time to Build and Aggregate Fluctuations." *Econometrica* 50(6). The founding RBC paper. This repo implements the core stochastic-growth mechanism; KP's time-to-build investment lag and labor-leisure choice are not implemented here.
- Hansen, G. (1985). "Indivisible Labor and the Business Cycle." *JME*. The employment-lottery refinement that fixes the hours margin — the main thing this version leaves on the table.
- King, R., Plosser, C. & Rebelo, S. (1988). "Production, Growth and Business Cycles." *JME*. The canonical balanced-growth RBC and the calibration used here.
- Prescott, E. (1986). "Theory Ahead of Business Cycle Measurement." The "TFP explains most of the cycle" claim and the HP-filter methodology.
- Hodrick, R. & Prescott, E. (1997). The HP filter (`moments.jl`).
- King, R. & Rebelo, S. (1999). "Resuscitating Real Business Cycles." *Handbook of Macroeconomics*. The U.S. moments in the §3 scorecard.
- Rouwenhorst, K. (1995). The Markov discretization of the TFP process.
