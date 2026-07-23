# =============================================================================
# offers.jl — wage-offer distribution
# =============================================================================

using Distributions
using QuantEcon: tauchen

"""
    offer_grid(p::McCallModel) -> (w, pw)

Discretize the i.i.d. log-normal wage-offer distribution `ln w ~ N(μ, σ²)` onto
`n_w` points.

# Returns
- `w::Vector{Float64}`  — wage levels (length `n_w`), increasing.
- `pw::Vector{Float64}` — offer probabilities, summing to 1.
"""
function offer_grid(p::McCallModel)
    (; μ, σ, n_w, m) = p
    logw = range(μ - m*σ, μ + m*σ, length = n_w)   # even grid in logs
    w    = exp.(logw)                              # wage levels (skewed)
    Φ    = z -> cdf(Normal(μ, σ), z)               # CDF of log wage
    mids = (logw[1:end-1] .+ logw[2:end]) ./ 2     # cell boundaries (in logs)

    pw = Vector{Float64}(undef, n_w)
    pw[1]   = Φ(mids[1])                           # left endpoint absorbs the lower tail
    pw[end] = 1 - Φ(mids[end])                     # right endpoint absorbs the upper tail
    for i in 2:n_w-1
        pw[i] = Φ(mids[i]) - Φ(mids[i-1])          # interior cell mass
    end
    return collect(w), pw
end

"""
    persistent_offers(p::McCallModel) -> (w, P, π)

Discretize a *persistent* offer process in which the log wage follows an AR(1),
`ln w' = (1−ρ)·μ + ρ·ln w + ε`, `ε ~ N(0, σ²(1−ρ²))` (so the stationary log-wage
distribution still has mean `μ` and variance `σ²`, matching `offer_grid`). This
is the input to the correlated-offer extension in `correlated.jl`, where a good
draw today signals good draws tomorrow.

Uses the Tauchen (1986) method.

# Returns
- `w::Vector{Float64}`  — wage levels at the AR(1) nodes (length `n_w`).
- `P::Matrix{Float64}`  — `n_w × n_w` Markov transition matrix over offers.
- `π::Vector{Float64}`  — stationary offer distribution.
"""
function persistent_offers(p::McCallModel)
    (; μ, σ, ρ, n_w, m) = p
    σ_ε = σ * sqrt(1 - ρ^2)                          # innovation std ⇒ stationary var σ²
    mc  = tauchen(n_w, ρ, σ_ε, (1 - ρ) * μ, m)      # AR(1) with mean μ
    logw = mc.state_values
    w    = exp.(logw)
    P    = mc.p
    π    = stationary_distributions(mc)[1]
    return collect(w), Matrix(P), π
end
