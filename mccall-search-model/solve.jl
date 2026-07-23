# =============================================================================
# solve.jl — value-function and reservation-wage solvers
# =============================================================================

using LinearAlgebra

# CRRA flow utility. γ = 0 ⇒ u(x) = x (risk-neutral, McCall 1970);
# γ = 1 ⇒ log; otherwise the standard power form. Returns −Inf for x ≤ 0.
function flow_u(x, γ)
    x > 0 || return -Inf
    γ == 0 && return x
    γ == 1 && return log(x)
    return x^(1 - γ) / (1 - γ)
end

# Inverse of `flow_u` in its argument: given v = u(x), return x. Used to turn the
# reservation *utility* (1−β)·U into a reservation *wage*.
function flow_u_inv(v, γ)
    γ == 0 && return v
    γ == 1 && return exp(v)
    return ((1 - γ) * v)^(1 / (1 - γ))
end

"""
    solve_mccall(p::McCallModel)

Solve the McCall model with job separation by iterating the continuation
value of unemployment `U` to its fixed point, rather than iterating the
whole value function over the wage grid (see `solve_vfi`).

The two Bellman objects are

    V_e(w) = u(w) + β[(1−α)·V_e(w) + α·U]          (employed at wage w)
    U      = u(c) + β·Σ_w pw·max{ V_e(w), U }       (unemployed, pre-offer)

Given a guess for the scalar `U`, the employment value has the closed form

    V_e(w) = [u(w) + β·α·U] / (1 − β(1−α)),

so each iteration is one cheap sweep over the offer grid. The worker accepts iff
`V_e(w) ≥ U`; because `V_e` is increasing in `w` this is a reservation-wage
rule, and setting `V_e(w̄) = U` gives the clean closed form

    u(w̄) = (1 − β)·U                  (independent of the separation rate α).

# Returns
- `U`        — value of unemployment (the fixed point).
- `V_e`      — employment value at each grid wage (length `n_w`).
- `accept`   — Bool policy, `true` where the worker accepts.
- `w_res`    — reservation wage `w̄ = u⁻¹((1−β)U)`.
- `f`        — job-finding rate (hazard) = Σ pw over accepted offers.
- `u_rate`   — steady-state unemployment rate `α/(α+f)`.
- `w, pw`    — the offer grid and its probabilities (for downstream use).
"""
function solve_mccall(p::McCallModel)
    (; β, γ, c, α, tol, max_iter) = p
    w, pw = offer_grid(p)

    denom = 1 - β * (1 - α)
    U = flow_u(c, γ) / (1 - β)          # initial guess: never-accept value
    local V_e
    for iter in 1:max_iter
        V_e   = (flow_u.(w, γ) .+ β * α * U) ./ denom
        U_new = flow_u(c, γ) + β * dot(pw, max.(V_e, U))
        if abs(U_new - U) < tol
            U = U_new
            break
        end
        U = U_new
    end

    V_e    = (flow_u.(w, γ) .+ β * α * U) ./ denom
    accept = V_e .>= U
    w_res  = flow_u_inv((1 - β) * U, γ)
    f      = dot(pw, accept)             # probability a random offer is accepted
    u_rate = α / (α + f)                 # steady-state unemployment (α=0 ⇒ 0)

    return (; U, V_e, accept, w_res, f, u_rate, w, pw)
end

"""
    solve_vfi(p::McCallModel) -> NamedTuple

Slower alternative to `solve_mccall`, kept as a cross-check (§1 of analyze.jl):
iterate the full value function `V(w) = max{ V_e(w), U }` over the entire offer
grid until it converges.
"""
function solve_vfi(p::McCallModel)
    (; β, γ, c, α, tol, max_iter) = p
    w, pw = offer_grid(p)
    denom = 1 - β * (1 - α)

    V = flow_u.(w, γ) ./ (1 - β)        # initial guess: accept everything
    U = flow_u(c, γ) / (1 - β)
    for iter in 1:max_iter
        EV    = dot(pw, V)
        U_new = flow_u(c, γ) + β * EV
        V_e   = (flow_u.(w, γ) .+ β * α * U_new) ./ denom
        V_new = max.(V_e, U_new)
        err   = max(norm(V_new - V, Inf), abs(U_new - U))
        V, U  = V_new, U_new
        err < tol && break
    end

    V_e    = (flow_u.(w, γ) .+ β * α * U) ./ denom
    accept = V_e .>= U
    w_res  = flow_u_inv((1 - β) * U, γ)
    return (; U, V, V_e, accept, w_res, w, pw)
end
