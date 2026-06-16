# =============================================================================
# simulate.jl — simulate debt, spreads, and default episodes
# =============================================================================

using Random
using Distributions: Categorical

"""
    simulate_sovereign(m, sol, T; seed = 42, burn = 500) -> NamedTuple

Simulate the equilibrium for `T` periods. Each period the government draws its
endowment, and — if it has market access — defaults with the model's logistic
probability; otherwise it borrows according to the policy `bp_itp` at the price
`q_itp`. In default it consumes the cost-adjusted endowment and regains access
with probability `λ`, re-entering at zero debt.

# Returns per-period series `(; b, y, c, spread, default, excluded, tb)` where
`spread` is the annualized bond spread (NaN while excluded) and `tb` is the trade
balance / output ratio.
"""
function simulate_sovereign(m::SovDefaultModel, sol, T::Int; seed = 42, burn = 500)
    rng = MersenneTwister(seed)
    (; r, λ, τ, N_y, P_y, y_grid, b_grid) = m
    h = default_income(m)
    Tt = T + burn

    b = 0.0
    iy = (N_y + 1) ÷ 2
    excluded = false
    B = zeros(Tt); Y = zeros(Tt); C = zeros(Tt)
    SP = fill(NaN, Tt); DEF = zeros(Int, Tt); EX = zeros(Int, Tt); TB = zeros(Tt)

    for t in 1:Tt
        y = y_grid[iy]
        B[t] = b; Y[t] = y; EX[t] = excluded ? 1 : 0
        if excluded
            C[t] = h[iy]
            excluded = rand(rng) > λ                 # try to regain access
            excluded || (b = 0.0)                    # re-enter at zero debt
        else
            # Default decision: hard max (the τ→0 model). The taste shock in the
            # solver is only a numerical smoother; the economy's actual default
            # rule is "default iff it is strictly better", which avoids the
            # spurious extra defaults a stochastic logistic rule would inject.
            if sol.VD[iy] > sol.VR_itp[iy](b)        # default this period
                C[t] = h[iy]; DEF[t] = 1; excluded = true
            else                                     # repay and re-borrow
                bp   = clamp(sol.bp_itp[iy](b), b_grid[1], b_grid[end])
                qv   = sol.q_itp[iy](bp)
                C[t] = y + b - qv * bp
                SP[t] = 100 * ((1 / max(qv, 1e-8))^4 - (1 + r)^4)   # annualized %
                TB[t] = (y - C[t]) / y
                b = bp
            end
        end
        iy = rand(rng, Categorical(P_y[iy, :]))
    end

    keep = burn+1:Tt
    return (; b = B[keep], y = Y[keep], c = C[keep], spread = SP[keep],
            default = DEF[keep], excluded = EX[keep], tb = TB[keep])
end
