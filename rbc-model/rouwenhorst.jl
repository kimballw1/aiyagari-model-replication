# =============================================================================
# rouwenhorst.jl — discretize the AR(1) TFP process
# =============================================================================

"""
    rouwenhorst(N, ρ, σ_ε) -> (z_grid, P)

Discretize `log zₜ = ρ·log zₜ₋₁ + εₜ`, `εₜ ~ N(0, σ_ε²)`, into an `N`-state
Markov chain by the Rouwenhorst method.

# Returns
- `z_grid::Vector{Float64}` — log-TFP nodes (exponentiate for levels).
- `P::Matrix{Float64}`      — `N × N` transition matrix.
"""
function rouwenhorst(N::Int, ρ::Float64, σ_ε::Float64)
    σ_z = σ_ε / sqrt(1 - ρ^2)                 # unconditional std of log z
    p   = (1 + ρ) / 2
    P   = [p 1-p; 1-p p]
    for n in 3:N
        z = zeros(n - 1)
        P = p     .* [P z; z' 0] + (1 - p) .* [z P; 0 z'] +
            (1 - p) .* [z' 0; P z] + p     .* [0 z'; z P]
        P[2:end-1, :] ./= 2                    # renormalize interior rows
    end
    z_grid = collect(LinRange(-sqrt(N - 1) * σ_z, sqrt(N - 1) * σ_z, N))
    return z_grid, P
end
