# =============================================================================
# steadystate.jl — deterministic steady state
# =============================================================================

"""
    steady_state(m::RBCModel) -> NamedTuple

Compute the deterministic (z = 1) steady state.

The Euler equation at the steady state pins the gross return on capital,
`R = α·y/k + 1−δ = 1/β`, hence `y/k = (1/β − 1 + δ)/α` and (with n ≡ 1)
`k = (α/(1/β−1+δ))^{1/(1−α)}`.

# Returns `(; k, y, c, i, R, r, ky, cy, iy)` — levels, the net rental rate
`r = R − 1`, and the great ratios `K/Y, C/Y, I/Y`.
"""
function steady_state(m::RBCModel)
    (; α, β, δ) = m
    R = 1 / β                       # gross return on capital in steady state
    yk = (R - 1 + δ) / α            # output–capital ratio  y/k = (1/β−1+δ)/α
    k = (α / (R - 1 + δ))^(1 / (1 - α))
    y = yk * k
    i = δ * k
    c = y - i
    return (; k, y, c, i, R, r = R - 1, ky = k / y, cy = c / y, iy = i / y)
end
