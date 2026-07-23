# =============================================================================
# vfi.jl — value-function iteration
# =============================================================================
#
# The solver: value-function iteration with linear interpolation of the
# continuation value, choosing k' by Brent's method at every (k, z) point.
# Exact (no local approximation around the steady state) — just slow, because
# every grid point is a one-dimensional maximization every iteration.

using Interpolations
using Optim

"""
    solve_vfi(m::RBCModel; tol = 1e-6, maxiter = 2000) -> NamedTuple

Iterate the Bellman operator
`V(k,z) = max_{k'} log(z·kᵃ + (1−δ)k − k') + β·Σ_z' P(z,z')·V(k',z')`
on an `N_k × N_z` grid, evaluating the continuation value off-grid by linear
interpolation and choosing `k'` with Brent's method. Returns the value function
and interpolated savings/consumption policies.

# Returns `(; V, kp, c, kp_itp, c_itp, k_grid, z_grid, P_z)`.
"""
function solve_vfi(m::RBCModel; tol = 1e-6, maxiter = 2000)
    (; α, δ, β, N_k, N_z) = m
    z_log, P_z = rouwenhorst(N_z, m.ρ, m.σ_ε)
    z_grid = exp.(z_log)
    k_ss = steady_state(m).k
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

"""
    irf_vfi(sol, m::RBCModel; T = 40) -> NamedTuple

Impulse response to a one-standard-deviation TFP innovation, computed from the
VFI policy. The economy starts at the policy's own resting point (z held at its
mean), gets hit with `log z₀ = σ_ε`, and z then decays deterministically at
rate `ρ`. Off-node z values evaluate the savings policy by linear interpolation
across the (evenly spaced) log-z grid. Responses are % deviations from the
resting point.
"""
function irf_vfi(sol, m::RBCModel; T = 40)
    (; α, δ, ρ, σ_ε) = m
    z_log = log.(sol.z_grid)

    # savings policy at an arbitrary (k, log z): interpolate across the z nodes
    function kp_at(k, lz)
        j = clamp(searchsortedlast(z_log, lz), 1, length(z_log) - 1)
        w = (lz - z_log[j]) / (z_log[j+1] - z_log[j])
        return (1 - w) * sol.kp_itp[j](k) + w * sol.kp_itp[j+1](k)
    end

    # resting point of the policy at z = 1 (grid version of the steady state)
    k0 = steady_state(m).k
    for _ in 1:5000
        k1 = kp_at(k0, 0.0)
        abs(k1 - k0) < 1e-10 && (k0 = k1; break)
        k0 = k1
    end
    y0 = k0^α
    i0 = δ * k0                       # at the resting point k' = k, so i = δk
    c0 = y0 - i0

    y = zeros(T); c = zeros(T); i = zeros(T); kk = zeros(T); zz = zeros(T)
    k = k0
    for t in 1:T
        lz = ρ^(t - 1) * σ_ε
        z  = exp(lz)
        kp = kp_at(k, lz)
        yt = z * k^α
        it = kp - (1 - δ) * k
        y[t]  = 100 * (yt / y0 - 1)
        c[t]  = 100 * ((yt - it) / c0 - 1)
        i[t]  = 100 * (it / i0 - 1)
        kk[t] = 100 * (k / k0 - 1)
        zz[t] = 100 * lz
        k = kp
    end
    return (; y, c, i, k = kk, z = zz)
end
