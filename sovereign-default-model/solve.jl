# =============================================================================
# solve.jl — the nested fixed point: value functions + the bond price schedule
# =============================================================================
#
# Two fixed points are solved jointly. Given the bond price schedule q, the
# government's repay/default values (V^R, V^D) are a contraction; given those
# values, competitive lenders re-price the bonds. Unlike a plain consumption-
# smoothing problem, the price q(b', y) is endogenous — it embeds next period's
# default probability — and feeds back into how much the government wants to
# borrow. We iterate both to convergence with damping on q.
#
# A vanishingly small Type-I-EV "taste shock" (scale τ) smooths the binary
# repay/default choice into a logistic default probability — the same log-sum-exp
# device as the Rust model — which keeps q(b',y) continuous and the iteration
# stable near the default boundary.

using LinearAlgebra
using Interpolations

# Numerically stable smooth maximum (log-sum-exp) of the repay/default values.
function smooth_max(vR, vD, τ)
    vmax = max(vR, vD)
    return vmax + τ * log(exp((vR - vmax) / τ) + exp((vD - vmax) / τ))
end

# Logistic default probability P(default) = 1/(1+exp((V^R−V^D)/τ)).
default_prob(vR, vD, τ) = 1.0 / (1.0 + exp((vR - vD) / τ))

"""
    solve_sovereign(m; tol = 1e-7, maxiter = 2000, damp = 0.5, verbose = false) -> NamedTuple

Solve for the recursive equilibrium `(V, V^R, V^D, q, policy, default_prob)` by
iterating:

1. **Government** — given `q`, update the value of repayment
   `V^R(b,y) = max_{b'} u(y + b − q(b',y)·b') + β·E V(b',y')`,
   the value of default
   `V^D(y) = u(h(y)) + β·E[λ·V(0,y') + (1−λ)·V^D(y')]`,
   and `V = smooth_max(V^R, V^D)`.
2. **Lenders** — re-price bonds from next period's default probability,
   `q(b',y) = (1 − E δ(b',y')) / (1 + r)`, damped to prevent oscillation.

The repayment search runs over a *fine* bond grid (interpolating `V` and `q`),
while values are stored on a *sparse* grid — separating optimization accuracy
from storage. Continuation values are precomputed once per iteration for speed.
"""
function solve_sovereign(m::SovDefaultModel; tol = 1e-7, maxiter = 2000, damp = 0.5, verbose = false)
    (; β, σ, r, λ, τ, N_y, N_b, N_b_fine, P_y, y_grid, b_grid, b_grid_fine) = m
    h = default_income(m)

    VR = zeros(N_b, N_y)
    VD = zeros(N_y)
    V  = zeros(N_b, N_y)
    q  = fill(1 / (1 + r), N_b, N_y)
    bp = zeros(N_b, N_y)

    for iter in 1:maxiter
        # ---- precompute continuation values on the fine grid -----------------
        V_itp = [linear_interpolation(b_grid, V[:, iy], extrapolation_bc = Line()) for iy in 1:N_y]
        q_itp = [linear_interpolation(b_grid, q[:, iy], extrapolation_bc = Line()) for iy in 1:N_y]
        Vmat_fine = reduce(vcat, [V_itp[iy](b_grid_fine)' for iy in 1:N_y])   # N_y × N_b_fine
        EV_fine   = P_y * Vmat_fine                                            # N_y × N_b_fine
        q_fine    = reduce(vcat, [q_itp[iy](b_grid_fine)' for iy in 1:N_y])    # N_y × N_b_fine
        V0        = [V_itp[iy](0.0) for iy in 1:N_y]                           # value at zero debt

        # ---- value of default ------------------------------------------------
        VD_new = similar(VD)
        for iy in 1:N_y
            EV_def = sum(P_y[iy, iy2] * (λ * V0[iy2] + (1 - λ) * VD[iy2]) for iy2 in 1:N_y)
            VD_new[iy] = u(h[iy], σ) + β * EV_def
        end

        # ---- value of repayment (vectorized over (b, b')) --------------------
        VR_new = similar(VR)
        for iy in 1:N_y
            y = y_grid[iy]
            for ib in 1:N_b
                b = b_grid[ib]
                best_v = -Inf; best_bp = b_grid_fine[1]
                @inbounds for ibp in 1:N_b_fine
                    c = y + b - q_fine[iy, ibp] * b_grid_fine[ibp]
                    c <= 0 && continue
                    val = u(c, σ) + β * EV_fine[iy, ibp]
                    if val > best_v
                        best_v = val; best_bp = b_grid_fine[ibp]
                    end
                end
                VR_new[ib, iy] = best_v
                bp[ib, iy]     = best_bp
            end
        end

        V_new = [smooth_max(VR_new[ib, iy], VD_new[iy], τ) for ib in 1:N_b, iy in 1:N_y]

        # ---- lenders re-price bonds -----------------------------------------
        q_new = similar(q)
        for iy in 1:N_y, ibp in 1:N_b
            δ = sum(P_y[iy, iy2] * default_prob(VR_new[ibp, iy2], VD_new[iy2], τ) for iy2 in 1:N_y)
            q_new[ibp, iy] = (1 - δ) / (1 + r)
        end
        q_new = damp .* q_new .+ (1 - damp) .* q

        err = max(maximum(abs.(V_new - V)), maximum(abs.(q_new - q)))
        VR, VD, V, q = VR_new, VD_new, V_new, q_new
        verbose && iter % 50 == 0 && println("  iter $iter | err = $(round(err, sigdigits = 3))")
        if err < tol
            verbose && println("  converged in $iter iterations")
            break
        end
    end

    def_prob = [default_prob(VR[ib, iy], VD[iy], τ) for ib in 1:N_b, iy in 1:N_y]
    VR_itp = [linear_interpolation(b_grid, VR[:, iy], extrapolation_bc = Line()) for iy in 1:N_y]
    bp_itp = [linear_interpolation(b_grid, bp[:, iy], extrapolation_bc = Line()) for iy in 1:N_y]
    q_itp  = [linear_interpolation(b_grid, q[:, iy],  extrapolation_bc = Line()) for iy in 1:N_y]
    return (; V, VR, VD, q, bp, def_prob, VR_itp, bp_itp, q_itp)
end
