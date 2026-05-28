# =============================================================================
# rouwenhorst.jl
# =============================================================================

"""
rouwenhorst(p::AiyagariParams)

Discretize an AR(1) process using the Rouwenhorst (1995) method.
The process is given by: y_t = ρ * y_{t-1} + e_t, where e_t ~ N(0, σ^2).

# Arguments
- `p::AiyagariParams`: Struct containing model parameters (requires `n_e`, `ρ`, and `σ`).

# Returns
- `z::Vector{Float64}`: State space grid of length `n_e`.
- `P::Matrix{Float64}`: `n_e` x `n_e` Markov transition probability matrix.
"""
function rouwenhorst(p::AiyagariParams)
    if p.n_e == 1
        return [0.0], ones(1, 1)
    end

    prob = (1.0 + p.ρ) / 2.0

    P = [prob 1.0-prob; 1.0-prob prob]

    for i in 3:p.n_e
        P_new = zeros(i, i)
        P_new[1:i-1, 1:i-1] += prob .* P
        P_new[1:i-1, 2:i] += (1.0 - prob) .* P
        P_new[2:i, 1:i-1] += (1.0 - prob) .* P
        P_new[2:i, 2:i] += prob .* P
        
        # Normalize the interior rows so probabilities sum to 1
        P_new[2:i-1, :] ./= 2.0
        P = P_new
    end

    # Calculate the bounds of the state space grid
    sigma_y = p.σ / sqrt(1.0 - p.ρ^2)
    psi = sqrt(p.n_e - 1) * sigma_y
    
    # Generate linearly spaced grid (unconditional mean is 0.0)
    e_grid = collect(range(-psi, psi, length=p.n_e))
    
    return e_grid, P
end
