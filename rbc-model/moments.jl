# =============================================================================
# moments.jl — simulation, HP filter, and business-cycle statistics
# =============================================================================
#
# The RBC "scorecard": simulate the solved model, Hodrick–Prescott filter the log
# series exactly as macroeconomists treat the data, and report the second moments
# that Kydland & Prescott put the model up against — relative volatilities and
# comovement with output.

using LinearAlgebra
using SparseArrays
using Statistics
using Random

"""
    hp_filter(y, λ) -> (trend, cycle)

Hodrick–Prescott filter. Splits `y` into a smooth trend and a cyclical component
`cycle = y − trend` by solving `(I + λ·D'D)·trend = y`, where `D` is the
second-difference operator. Use `λ = 1600` for quarterly data (the RBC standard).
"""
function hp_filter(y::AbstractVector, λ::Real)
    T = length(y)
    # Second-difference operator as a sparse matrix; (I + λ·D'D) is pentadiagonal,
    # so the factorization is O(T) rather than the O(T³) of a dense solve.
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:T-2
        append!(rows, (i, i, i)); append!(cols, (i, i+1, i+2)); append!(vals, (1.0, -2.0, 1.0))
    end
    D = sparse(rows, cols, vals, T - 2, T)
    trend = (sparse(1.0I, T, T) + λ * (D' * D)) \ collect(y)
    return trend, y .- trend
end

"""
    simulate_vfi(sol, m::RBCModel, T; seed = 1, burn = 200) -> NamedTuple

Simulate the solved model: draw the TFP Markov chain from `P_z`, roll capital
forward with the savings policy, and back out output, consumption, and
investment from the budget constraint. Returns the log series (after a
burn-in) for output, consumption, investment, capital, and TFP.
"""
function simulate_vfi(sol, m::RBCModel, T::Int; seed = 1, burn = 200)
    rng = MersenneTwister(seed)
    (; α, δ) = m
    (; kp_itp, k_grid, z_grid, P_z) = sol
    N_z = length(z_grid)
    Pcum = cumsum(P_z, dims = 2)

    Tt = T + burn
    y = zeros(Tt); c = zeros(Tt); i = zeros(Tt); k = zeros(Tt); z = zeros(Tt)
    iz = (N_z + 1) ÷ 2                       # start at mean TFP
    kt = steady_state(m).k                   # and at the steady-state capital
    for t in 1:Tt
        zt = z_grid[iz]
        kp = clamp(kp_itp[iz](kt), k_grid[1], k_grid[end])
        yt = zt * kt^α
        it = kp - (1 - δ) * kt
        y[t] = log(yt); c[t] = log(yt - it); i[t] = log(it)
        k[t] = log(kt); z[t] = log(zt)
        kt = kp
        iz = min(searchsortedfirst(view(Pcum, iz, :), rand(rng)), N_z)
    end
    keep = burn+1:Tt
    return (; y = y[keep], c = c[keep], i = i[keep], k = k[keep], z = z[keep])
end

"""
    business_cycle_stats(sol, m; T = 10_000, λ = 1600, seed = 1) -> NamedTuple

The business-cycle scorecard. Simulate, HP-filter each log series, and report
for output, consumption, and investment:
- `σ`         — percent standard deviation of the cyclical component,
- `σ_rel`     — volatility relative to output,
- `corr_y`    — contemporaneous correlation with output,
- `autocorr`  — first-order autocorrelation (persistence).

(Hours are constant by construction here — inelastic labor — so there is no
hours row.)
"""
function business_cycle_stats(sol, m::RBCModel; T = 10_000, λ = 1600, seed = 1)
    sim = simulate_vfi(sol, m, T; seed)
    cyc(x) = hp_filter(x, λ)[2]
    yc, cc, ic = cyc(sim.y), cyc(sim.c), cyc(sim.i)
    σy = std(yc)
    ac(x) = cor(x[1:end-1], x[2:end])
    rows = [
        (:output,      100σy,        1.0,            1.0,          ac(yc)),
        (:consumption, 100std(cc),   std(cc)/σy,     cor(yc, cc),  ac(cc)),
        (:investment,  100std(ic),   std(ic)/σy,     cor(yc, ic),  ac(ic)),
    ]
    return (; rows, σy = 100σy)
end
