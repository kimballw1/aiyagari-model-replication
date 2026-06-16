# =============================================================================
# solve.jl — dynamic program: value-function iteration + choice probabilities
# =============================================================================

using LinearAlgebra

const γ_euler = Base.MathConstants.eulergamma   # ≈ 0.5772, Euler–Mascheroni

"""
    bellman_update(m, V, F0, F1) -> V_new

One application of Rust's Bellman operator. With Type-I EV shocks the expected
maximum is the **log-sum-exp** of the choice-specific values, so

    v_keep    = −c   + β·F0·V
    v_replace = −RC  + β·F1·V
    V(x)      = log( exp(v_keep) + exp(v_replace) )

(The additive Euler constant `γ` is a level shift that cancels out of all choice
probabilities, so we drop it from `V` — and keep the Hotz–Miller inversion in
`estimate.jl` consistent with that convention.) The `vmax` subtraction is the
standard log-sum-exp guard against overflow.
"""
function bellman_update(m::RustModel, V, F0, F1)
    (; β, RC) = m
    c = cost(m)
    v_keep    = -c    .+ β .* (F0 * V)
    v_replace = -RC   .+ β .* (F1 * V)
    vmax = max.(v_keep, v_replace)
    return vmax .+ log.(exp.(v_keep .- vmax) .+ exp.(v_replace .- vmax))
end

"""
    solve_bellman(m, F0, F1; tol = 1e-12, max_iter = 10_000) -> V

Value-function iteration to the fixed point. The operator is a contraction with
modulus `β`, so convergence is geometric (≈ `log(tol)/log(β)` steps).
"""
function solve_bellman(m::RustModel, F0, F1; tol = 1e-12, max_iter = 10_000)
    V = zeros(m.S)
    for _ in 1:max_iter
        V_new = bellman_update(m, V, F0, F1)
        err   = norm(V_new - V, Inf)
        V     = V_new
        err < tol && break
    end
    return V
end

"""
    solve_rust(m::RustModel) -> NamedTuple

Solve the model end to end: build transitions, iterate the Bellman equation, and
return the value function, the choice-specific values, and the **conditional
choice probability** of replacement at each mileage (the logit formula),

    P(replace | x) = 1 / (1 + exp(v_keep(x) − v_replace(x))).

# Returns `(; V, P, v_keep, v_replace, F0, F1)`.
"""
function solve_rust(m::RustModel)
    (; β, RC, θ₃, S) = m
    c = cost(m)
    F0, F1 = build_transition(θ₃, S)
    V = solve_bellman(m, F0, F1)
    v_keep    = -c  .+ β .* (F0 * V)
    v_replace = -RC .+ β .* (F1 * V)
    P_rep = 1 ./ (1 .+ exp.(v_keep .- v_replace))
    return (; V, P = P_rep, v_keep, v_replace, F0, F1)
end
