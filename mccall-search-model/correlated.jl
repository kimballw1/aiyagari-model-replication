# =============================================================================
# correlated.jl — McCall search with persistent (Markov) wage offers
# =============================================================================
#
# McCall (1970) assumes offers are i.i.d.: today's draw says nothing about
# tomorrow's. Empirically, labor-market luck is persistent — a worker who can
# command a high offer this period tends to command one next period too. Here the
# log offer follows an AR(1) (discretized in `offers.jl::persistent_offers`), so
# the worker's *state* is the current offer itself.
#
# With employment absorbing (set α elsewhere for the separation case), the value
# of being unemployed and holding offer w_i solves
#
#     V_u(i) = max{ u(w_i)/(1−β),  u(c) + β·Σ_j P[i,j]·V_u(j) }
#                    └ accept ┘     └────── reject, redraw from P[i,·] ──────┘
#

using LinearAlgebra

"""
    solve_correlated(p::McCallModel) -> NamedTuple

Solve the persistent-offer McCall model by iterating `V_u` over the offer grid.

# Returns
- `V_u`     — value of unemployment at each offer state.
- `accept`  — Bool acceptance policy over the offer grid.
- `w_res`   — reservation wage (lowest accepted offer level).
- `w, P, π` — the persistent offer grid, transition matrix, and stationary dist.
- `f`       — stationary job-finding rate `Σ_i π_i·accept_i`.
"""
function solve_correlated(p::McCallModel)
    (; β, γ, c, tol, max_iter) = p
    w, P, π = persistent_offers(p)

    V_e = flow_u.(w, γ) ./ (1 - β)           # absorbing employment value
    V_u = copy(V_e)
    for iter in 1:max_iter
        reject = flow_u(c, γ) .+ β .* (P * V_u)
        V_new  = max.(V_e, reject)
        err    = norm(V_new - V_u, Inf)
        V_u    = V_new
        err < tol && break
    end

    reject = flow_u(c, γ) .+ β .* (P * V_u)
    accept = V_e .>= reject
    idx    = findfirst(accept)
    w_res  = idx === nothing ? Inf : w[idx]
    f      = dot(π, accept)
    return (; V_u, accept, w_res, w, P, π, f)
end
