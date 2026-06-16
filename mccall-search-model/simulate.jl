# =============================================================================
# simulate.jl — Monte-Carlo labor-market histories
# =============================================================================
#
# The reservation-wage policy turns the model into a simple two-state Markov
# process over (unemployed, employed). Simulating it lets us read off objects the
# closed-form solution only predicts in expectation: the *distribution* of
# unemployment-spell lengths (geometric with the job-finding rate `f`), the
# realized distribution of accepted wages (the offer distribution truncated at
# `w̄`), and the time-average unemployment rate (which should match the
# steady-state `α/(α+f)`).

using Statistics
using Random

# Draw an index from the pmf `cdf` (a precomputed cumulative sum of probabilities)
# by inverse-CDF sampling — base-Julia replacement for StatsBase.sample.
@inline draw_idx(cdf, u) = searchsortedfirst(cdf, u)

"""
    simulate_panel(p, sol; T = 5_000, N = 2_000, seed = 1) -> NamedTuple

Simulate `N` workers for `T` periods under the reservation-wage policy in `sol`
(from `solve_mccall`). Workers start unemployed; an unemployed worker draws an
offer each period and accepts iff `w ≥ w_res`; an employed worker is separated
with probability `α` and returns to unemployment.

# Returns
- `u_rate`        — time-average unemployment rate across the panel.
- `mean_dur`      — average completed unemployment-spell length.
- `durations`     — vector of completed spell lengths (for a histogram).
- `accepted_wages`— vector of wages at acceptance (truncated offer dist.).
- `mean_wage`     — average accepted wage.
"""
function simulate_panel(p::McCallModel, sol; T::Int = 5_000, N::Int = 2_000, seed::Int = 1)
    (; α) = p
    w, pw, w_res = sol.w, sol.pw, sol.w_res
    rng = MersenneTwister(seed)                         # reproducible stream
    cdf = cumsum(pw)

    n_unemp_periods = 0
    durations       = Int[]
    accepted_wages  = Float64[]

    for n in 1:N
        employed = false
        spell    = 0
        for t in 1:T
            if employed
                if rand(rng) < α                         # separated
                    employed = false
                    spell    = 0
                end
            else
                n_unemp_periods += 1
                spell += 1
                wd = w[draw_idx(cdf, rand(rng))]         # draw an offer
                if wd >= w_res                           # accept
                    employed = true
                    push!(durations, spell)
                    push!(accepted_wages, wd)
                end
            end
        end
    end

    u_rate   = n_unemp_periods / (N * T)
    mean_dur = isempty(durations) ? Inf : mean(durations)
    mean_w   = isempty(accepted_wages) ? NaN : mean(accepted_wages)
    return (; u_rate, mean_dur, durations, accepted_wages, mean_wage = mean_w)
end
