# =============================================================================
# transitions.jl — mileage transition matrices
# =============================================================================

"""
    build_transition(θ₃, S) -> (F0, F1)

Construct the two `S × S` mileage transition matrices.

- `F0` (**keep**): from state `i`, mileage advances to `min(i + Δ, S)` with
  `Δ ∈ {0,1,2}` drawn from `θ₃`. A banded matrix — mileage can only rise.
- `F1` (**replace**): the engine resets to new, so the next state is
  `min(1 + Δ, S)` regardless of the current state. Every row of `F1` is
  identical, which is the model's *renewal* structure: replacing erases the past.

At the top bin `S`, mileage is absorbed (capped).
"""
function build_transition(θ₃::Vector{Float64}, S::Int)
    K  = length(θ₃)
    F0 = zeros(S, S)
    F1 = zeros(S, S)
    for i in 1:S
        for k in 1:K
            j_keep    = min(i + k - 1, S)       # keep: current mileage + (k−1)
            F0[i, j_keep] += θ₃[k]
            j_replace = min(k, S)               # replace: reset to 1, then + (k−1)
            F1[i, j_replace] += θ₃[k]
        end
    end
    return F0, F1
end

"""
    stationary_mileage(P_rep, θ₃, S) -> π

Stationary distribution of mileage states under the replacement policy with
choice probabilities `P_rep` (the long-run histogram of `fig_mileage_dist`).

The controlled chain mixes the keep- and replace-transitions state by state:
row `i` is `(1−P_rep[i])·F0[i,:] + P_rep[i]·F1[i,:]`. The stationary `π` is the
dominant left eigenvector, found by iterating `π ← πᵀT` to convergence.
"""
function stationary_mileage(P_rep::Vector{Float64}, θ₃::Vector{Float64}, S::Int)
    F0, F1 = build_transition(θ₃, S)
    T = (1 .- P_rep) .* F0 .+ P_rep .* F1           # controlled transition matrix
    π = fill(1.0 / S, S)
    for _ in 1:100_000
        π_new = T' * π
        maximum(abs.(π_new .- π)) < 1e-14 && (π = π_new; break)
        π = π_new
    end
    return π ./ sum(π)
end
