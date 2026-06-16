# =============================================================================
# steadystate.jl — deterministic steady state
# =============================================================================

"""
    steady_state(m::RBCModel) -> NamedTuple

Compute the deterministic (z = 1) steady state and the implied calibration of
the labor-disutility weight `χ`.

The Euler equation at the steady state gives the gross return
`R = α·y/k + 1−δ = 1/β`, hence `y/k = (1/β − 1 + δ)/α`. With endogenous labor we
target steady-state hours `n* = n_target` and back out `χ` from the static
labor first-order condition `χ·c·n^{1/ν} = (1−α)·y/n`. With inelastic labor
`n* = 1`.

# Returns `(; k, y, c, i, n, R, r, χ, ky, cy, iy)` — levels, the net rental rate
`r = R − 1`, and the great ratios `K/Y, C/Y, I/Y`.
"""
function steady_state(m::RBCModel)
    (; α, β, δ) = m
    R = 1 / β                       # gross return on capital in steady state
    yk = (R - 1 + δ) / α            # output–capital ratio  y/k = (1/β−1+δ)/α
    n  = m.labor == :inelastic ? 1.0 : m.n_target

    # y = k^α n^{1−α} ⇒ y/k = (n/k)^{1−α} ⇒ k = n·(α/(1/β−1+δ))^{1/(1−α)}
    k = n * (α / (R - 1 + δ))^(1 / (1 - α))
    y = yk * k
    i = δ * k
    c = y - i

    # Calibrate χ from the static labor FOC (irrelevant under inelastic labor).
    if m.labor == :inelastic
        χ = 0.0
    else
        invν = inv_frisch(m)
        χ = (1 - α) * y / (n^(1 + invν) * c)
    end

    r = R - 1
    return (; k, y, c, i, n, R, r, χ, ky = k / y, cy = c / y, iy = i / y)
end
