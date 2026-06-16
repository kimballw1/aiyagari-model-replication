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
    simulate_linear(lin::LinearRBC, m::RBCModel, T; seed = 1, burn = 200) -> NamedTuple

Simulate the log-linear state-space: draw TFP innovations `ε ~ N(0, σ_ε²)`,
propagate `[k̂, ẑ]` and map to `[ŷ, ĉ, î, n̂]` via the loadings `C`. Returns the
log-deviation series (after a burn-in) for output, consumption, investment,
hours, capital, and TFP.
"""
function simulate_linear(lin::LinearRBC, m::RBCModel, T::Int; seed = 1, burn = 200)
    rng = MersenneTwister(seed)
    s = zeros(2)
    Tt = T + burn
    out = (y = zeros(Tt), c = zeros(Tt), i = zeros(Tt), n = zeros(Tt), k = zeros(Tt), z = zeros(Tt))
    for t in 1:Tt
        v = lin.C * s
        out.y[t], out.c[t], out.i[t], out.n[t] = v
        out.k[t], out.z[t] = s
        ε = m.σ_ε * randn(rng)
        s = lin.P * s + [0.0, ε]
    end
    keep = burn+1:Tt
    return (; y = out.y[keep], c = out.c[keep], i = out.i[keep],
            n = out.n[keep], k = out.k[keep], z = out.z[keep])
end

"""
    business_cycle_stats(lin, m; T = 10_000, λ = 1600, seed = 1) -> NamedTuple

The business-cycle scorecard. Simulate, HP-filter each log series, and report for
output, consumption, investment, and hours:
- `σ`         — percent standard deviation of the cyclical component,
- `σ_rel`     — volatility relative to output,
- `corr_y`    — contemporaneous correlation with output,
- `autocorr`  — first-order autocorrelation (persistence).
"""
function business_cycle_stats(lin::LinearRBC, m::RBCModel; T = 10_000, λ = 1600, seed = 1)
    sim = simulate_linear(lin, m, T; seed)
    # Series are already log-deviations; HP-filter to mimic the empirical treatment.
    cyc(x) = hp_filter(x, λ)[2]
    yc, cc, ic, nc = cyc(sim.y), cyc(sim.c), cyc(sim.i), cyc(sim.n)
    σy = std(yc)
    ac(x) = cor(x[1:end-1], x[2:end])
    rows = [
        (:output,      100σy,        1.0,            1.0,          ac(yc)),
        (:consumption, 100std(cc),   std(cc)/σy,     cor(yc, cc),  ac(cc)),
        (:investment,  100std(ic),   std(ic)/σy,     cor(yc, ic),  ac(ic)),
        (:hours,       100std(nc),   std(nc)/σy,     cor(yc, nc),  ac(nc)),
    ]
    return (; rows, σy = 100σy)
end
