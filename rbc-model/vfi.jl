# =============================================================================
# vfi.jl — exact nonlinear solution by value-function iteration (benchmark)
# =============================================================================
#
# The "ground-truth" solver: value-function iteration with linear interpolation
# of the continuation value, for the inelastic-labor stochastic growth model
# (n ≡ 1). It is far slower than the log-linear solution in `linearize.jl`, but
# it is *exact* (no local approximation), so analyze.jl §3 uses it to validate
# the linearization — the two policy functions should coincide near the steady
# state and fan apart only far from it.

using Interpolations
using Optim

"""
    solve_vfi(m::RBCModel; tol = 1e-6, maxiter = 1000) -> NamedTuple

Iterate the Bellman operator
`V(k,z) = max_{k'} log(z·kᵃ + (1−δ)k − k') + β·Σ_z' P(z,z')·V(k',z')`
on an `N_k × N_z` grid, evaluating the continuation value off-grid by linear
interpolation and choosing `k'` with Brent's method. Returns the value function
and interpolated savings/consumption policies.

# Returns `(; V, kp, c, kp_itp, c_itp, k_grid, z_grid, P_z)`.
"""
function solve_vfi(m::RBCModel; tol = 1e-6, maxiter = 1000)
    (; α, δ, β, N_k, N_z) = m
    z_log, P_z = rouwenhorst(N_z, m.ρ, m.σ_ε)
    z_grid = exp.(z_log)
    k_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))   # inelastic-labor steady state
    k_grid = collect(LinRange(0.3k_ss, 1.8k_ss, N_k))

    V  = zeros(N_k, N_z)
    kp = zeros(N_k, N_z)
    for iter in 1:maxiter
        V_itp = [linear_interpolation(k_grid, V[:, jz], extrapolation_bc = Line()) for jz in 1:N_z]
        V_new = similar(V)
        for iz in 1:N_z
            z = z_grid[iz]
            for ik in 1:N_k
                k = k_grid[ik]
                budget = z * k^α + (1 - δ) * k
                obj(kp_) = begin
                    c = budget - kp_
                    c <= 0 && return 1e10
                    EV = 0.0
                    @inbounds for jz in 1:N_z
                        EV += P_z[iz, jz] * V_itp[jz](kp_)
                    end
                    return -(log(c) + β * EV)
                end
                k_hi = min(k_grid[end], budget - 1e-8)
                res = optimize(obj, k_grid[1], k_hi, Brent())
                V_new[ik, iz] = -Optim.minimum(res)
                kp[ik, iz]    = Optim.minimizer(res)
            end
        end
        dist = maximum(abs.(V_new - V))
        V = V_new
        dist < tol && break
    end

    c = [z_grid[iz] * k_grid[ik]^α + (1 - δ) * k_grid[ik] - kp[ik, iz] for ik in 1:N_k, iz in 1:N_z]
    kp_itp = [linear_interpolation(k_grid, kp[:, iz], extrapolation_bc = Line()) for iz in 1:N_z]
    c_itp  = [linear_interpolation(k_grid, c[:, iz],  extrapolation_bc = Line()) for iz in 1:N_z]
    return (; V, kp, c, kp_itp, c_itp, k_grid, z_grid, P_z)
end
