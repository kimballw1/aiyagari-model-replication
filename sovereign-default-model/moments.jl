# =============================================================================
# moments.jl — business-cycle statistics (the Arellano scorecard)
# =============================================================================

using Statistics

"""
    default_statistics(m, sim) -> NamedTuple

Compute the moments Arellano (2008) targets, on the market-access subsample
(periods where the government is borrowing, not in default or exclusion):

- `def_rate`   — annualized default rate (%), `100·(1−(1−p_q)⁴)` with `p_q` the
  quarterly default frequency among access periods.
- `mean_debt`  — average debt-to-output ratio `E[−b/y]` (%).
- `mean_spread`, `std_spread` — mean and volatility of the annualized spread (%).
- `σ_c`, `σ_y` — std. dev. of HP-detrended log consumption and output (%).
- `rel_c`      — `σ_c/σ_y` (sovereign-default models deliver > 1: default and
  exclusion make consumption *more* volatile than output).
- `corr_cy`    — corr(consumption, output) (procyclical).
- `corr_sy`    — corr(spread, output): **negative** ⇒ countercyclical spreads.
- `corr_tby`   — corr(trade balance / output, output): **negative** ⇒
  countercyclical net exports.
"""
function default_statistics(m::SovDefaultModel, sim)
    access = (sim.default .== 0) .& (sim.excluded .== 0)   # borrowing periods
    p_q = sum(sim.default) / max(sum((sim.excluded .== 0)), 1)
    def_rate = 100 * (1 - (1 - p_q)^4)

    yA = sim.y[access]; cA = sim.c[access]; spA = sim.spread[access]
    tbA = sim.tb[access]; bA = sim.b[access]
    ly = log.(yA) .- mean(log.(yA))
    lc = log.(cA) .- mean(log.(cA))
    σy = std(ly); σc = std(lc)

    return (; def_rate,
            mean_debt   = 100 * mean(-bA ./ yA),
            mean_spread = mean(spA),
            med_spread  = median(spA),
            std_spread  = std(spA),
            σ_y = 100σy, σ_c = 100σc, rel_c = σc / σy,
            corr_cy  = cor(ly, lc),
            corr_sy  = cor(spA, ly),
            corr_tby = cor(tbA, ly))
end
