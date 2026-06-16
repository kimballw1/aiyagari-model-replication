# =============================================================================
# income.jl — endowment process and model setup
# =============================================================================

"""
    rouwenhorst(N, ρ, σ_ε) -> (z_grid, P)

Discretize `log yₜ = ρ·log yₜ₋₁ + εₜ`, `εₜ ~ N(0, σ_ε²)` into an `N`-state
Markov chain (Rouwenhorst 1995). Accurate for the highly persistent endowment
processes (`ρ ≈ 0.95`) used in sovereign-default calibrations.
"""
function rouwenhorst(N::Int, ρ::Float64, σ_ε::Float64)
    σ_z = σ_ε / sqrt(1 - ρ^2)
    p   = (1 + ρ) / 2
    P   = [p 1-p; 1-p p]
    for n in 3:N
        z = zeros(n - 1)
        P = p     .* [P z; z' 0] + (1 - p) .* [z P; 0 z'] +
            (1 - p) .* [z' 0; P z] + p     .* [0 z'; z P]
        P[2:end-1, :] ./= 2
    end
    z_grid = collect(LinRange(-sqrt(N - 1) * σ_z, sqrt(N - 1) * σ_z, N))
    return z_grid, P
end

"""
    setup!(m::SovDefaultModel) -> m

Fill the endowment grid (in levels), its transition matrix, and the sparse/fine
bond grids from the model primitives. Mutates `m`.
"""
function setup!(m::SovDefaultModel)
    (; ρ, σ_ε, N_y, N_b, N_b_fine, b_min, b_max) = m
    log_y, P = rouwenhorst(N_y, ρ, σ_ε)
    m.y_grid = exp.(log_y)
    m.P_y    = P
    m.b_grid      = collect(LinRange(b_min, b_max, N_b))
    m.b_grid_fine = collect(LinRange(b_min, b_max, N_b_fine))
    return m
end

# CRRA utility; large penalty for infeasible (non-positive) consumption.
function u(c, σ)
    c <= 0 && return -1e14
    σ == 1.0 && return log(c)
    return c^(1 - σ) / (1 - σ)
end

"""
    default_income(m) -> Vector

Effective income while in default (output cost), one entry per endowment state:
- `:arellano`     — `min(y, ŷ·E[y])`: a ceiling that bites only in good states,
  so default is costlier in booms — the source of *countercyclical* default.
- `:proportional` — `(1−φ)·y`: a flat output loss.
"""
function default_income(m::SovDefaultModel)
    Ey = sum(stationary(m.P_y) .* m.y_grid)        # mean endowment
    if m.cost == :proportional
        return (1 - m.φ) .* m.y_grid
    else
        return min.(m.y_grid, m.ŷ * Ey)            # Arellano asymmetric ceiling
    end
end

# Stationary distribution of a Markov matrix (dominant left eigenvector).
function stationary(P)
    n = size(P, 1)
    π = fill(1.0 / n, n)
    for _ in 1:100_000
        πn = P' * π
        maximum(abs.(πn .- π)) < 1e-14 && return πn ./ sum(πn)
        π = πn
    end
    return π ./ sum(π)
end
