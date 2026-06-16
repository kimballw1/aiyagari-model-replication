# =============================================================================
# estimate.jl — structural estimation: NFXP and Hotz–Miller (CCP)
# =============================================================================
#
# Rust's insight: the structural parameters (RC, θ₁) can be *recovered* from
# observed (mileage, decision) data. The likelihood of the choices is built from
# the model's conditional choice probabilities, which require solving the dynamic
# program. Two estimators are implemented:
#
#   • NFXP  (Rust 1987)        — the **N**ested **F**ixed-**P**oint MLE: an outer
#     optimizer searches over (RC, θ₁); every trial value re-solves the full DP.
#   • Hotz–Miller (1993)       — a two-step CCP estimator that replaces the inner
#     fixed point with a single linear solve, given a first-stage estimate of the
#     choice probabilities. Much faster, slightly less efficient.
#
# Both take the mileage transition θ₃ as pre-estimated (step 1 below), since it
# is identified from mileage increments alone — independent of costs.

using LinearAlgebra
using Optim

# -----------------------------------------------------------------------------
# Step 1 — transition probabilities θ₃ (pure counting, no model needed)
# -----------------------------------------------------------------------------
"""
    estimate_theta3(x, d, S) -> θ̂₃

Estimate the mileage-increment probabilities `P(Δ = 0, 1, 2)` by counting
observed increments in the panel, skipping boundary states where mileage is
censored by the cap. Identified without solving the model — this is what makes
Rust's two-step split possible.
"""
function estimate_theta3(x::Vector{Int}, d::Vector{Int}, S::Int)
    counts = zeros(3)
    for t in 1:length(x)-1
        if d[t] == 0 && x[t] <= S - 2                 # keep, away from the cap
            Δ = x[t+1] - x[t]
            0 <= Δ <= 2 && (counts[Δ+1] += 1)
        elseif d[t] == 1                              # replace ⇒ next = 1 + Δ
            Δ = x[t+1] - 1
            0 <= Δ <= 2 && (counts[Δ+1] += 1)
        end
    end
    return counts ./ sum(counts)
end

# -----------------------------------------------------------------------------
# Choice log-likelihood (shared by both estimators given a CCP vector)
# -----------------------------------------------------------------------------
"""
    choice_loglik(P_rep, x, d) -> Float64

Log-likelihood of the observed replacement decisions given a replacement-CCP
vector `P_rep` indexed by mileage state. Probabilities are floored at `1e-15` so
a perfectly-separated state cannot send the likelihood to `−∞`.
"""
function choice_loglik(P_rep::Vector{Float64}, x::Vector{Int}, d::Vector{Int})
    ll = 0.0
    @inbounds for t in eachindex(d)
        p = clamp(P_rep[x[t]], 1e-15, 1 - 1e-15)
        ll += d[t] == 1 ? log(p) : log(1 - p)
    end
    return ll
end

# -----------------------------------------------------------------------------
# Step 2a — NFXP (nested fixed-point MLE)
# -----------------------------------------------------------------------------
"""
    nfxp_negloglik([RC, θ₁], x, d, β, θ₃, S) -> Float64

Negative choice log-likelihood at candidate `(RC, θ₁)`: build the trial model,
**solve the entire DP** (the nested fixed point), read off the model CCPs, and
score the data. Returns a large penalty for infeasible negative parameters.
"""
function nfxp_negloglik(params, x, d, β, θ₃, S)
    RC, θ₁ = params
    (RC < 0 || θ₁ < 0) && return 1e10
    sol = solve_rust(RustModel(; β, S, RC, θ₁, θ₃))
    return -choice_loglik(sol.P, x, d)
end

"""
    estimate_nfxp(x, d, β, θ₃, S; x0 = [15.0, 0.03]) -> NamedTuple

Maximize the choice likelihood over `(RC, θ₁)` by NFXP (Nelder–Mead outer loop,
full VFI inner loop). Standard errors come from the numerical Hessian of the
negative log-likelihood at the optimum (the observed information matrix).

# Returns `(; RC, θ₁, se_RC, se_θ₁, ll, niter)`.
"""
function estimate_nfxp(x, d, β, θ₃, S; x0 = [15.0, 0.03])
    obj = p -> nfxp_negloglik(p, x, d, β, θ₃, S)
    res = optimize(obj, x0, NelderMead(), Optim.Options(iterations = 5_000))
    RC, θ₁ = Optim.minimizer(res)
    H  = num_hessian(obj, [RC, θ₁])
    se = sqrt.(abs.(diag(inv(H))))
    return (; RC, θ₁, se_RC = se[1], se_θ₁ = se[2], ll = -Optim.minimum(res),
            niter = Optim.iterations(res))
end

# -----------------------------------------------------------------------------
# Step 2b — Hotz–Miller (two-step CCP estimator)
# -----------------------------------------------------------------------------
"""
    fit_ccp_frequency(x, d, S; prior = 0.5, strength = 1.0) -> (P_rep, visits)

First-stage **nonparametric** conditional choice probabilities — the standard
Hotz–Miller first stage. For each mileage state, `P(replace | x)` is the observed
replacement frequency, Laplace-smoothed toward `prior` with weight `strength`:

    P_rep[s] = (replacements at s + strength·prior) / (visits to s + strength).

A parametric logit-in-mileage is *not* used on purpose: the true CCP saturates at
high mileage (the value differences flatten), which a logistic in `x` cannot
represent — fitting one biases the recovered value function and badly inflates
`RC`. The frequency estimator instead lets the data speak state by state.

The renewal structure has a sharp consequence visible in `visits`: engines are
almost always replaced *before* reaching the highest mileage bins, so those
states are essentially never observed and their CCPs are identified only by the
model's extrapolation, not by data — the Achilles' heel of pure CCP estimators
here, and why NFXP remains more accurate (see §6 of `analyze.jl`).

# Returns the smoothed CCP over all `S` states and the per-state visit counts.
"""
function fit_ccp_frequency(x::Vector{Int}, d::Vector{Int}, S::Int;
                           prior = 0.5, strength = 1.0)
    visits = zeros(Int, S)
    nrep   = zeros(Int, S)
    for t in eachindex(x)
        visits[x[t]] += 1
        nrep[x[t]]   += d[t]
    end
    P_rep = @. (nrep + strength * prior) / (visits + strength)
    return P_rep, visits
end

"""
    hm_negloglik([RC, θ₁], x, d, β, θ₃, S, P_keep) -> Float64

Hotz–Miller pseudo-likelihood. Given the first-stage keep-CCPs `P_keep`, recover
the value function **without iterating** via the logit inversion identity
`V = v_keep − log P_keep` together with `v_keep = −c + β·F0·V`, i.e.

    (I − β·F0)·V = −c − log P_keep      ⇒      V = (I − β F0)⁻¹(−c − log P_keep).

Then form the model CCPs from `v_keep, v_replace = −RC + β F1 V` and score the
data. Each evaluation is one linear solve — no nested fixed point.
"""
function hm_negloglik(params, x, d, β, θ₃, S, P_keep)
    RC, θ₁ = params
    (RC < 0 || θ₁ < 0) && return 1e10
    x̄ = Float64.(0:S-1)
    c  = θ₁ .* x̄
    F0, F1 = build_transition(θ₃, S)
    V  = (I - β .* F0) \ (-c .- log.(P_keep))          # logit inversion (no VFI)
    v_keep    = -c  .+ β .* (F0 * V)
    v_replace = -RC .+ β .* (F1 * V)
    P_rep = 1 ./ (1 .+ exp.(v_keep .- v_replace))
    return -choice_loglik(P_rep, x, d)
end

"""
    estimate_hotz_miller(x, d, β, θ₃, S; x0 = [15.0, 0.03]) -> NamedTuple

Two-step Hotz–Miller estimator: fit smoothed CCPs, then minimize the
pseudo-likelihood over `(RC, θ₁)` with the DP-free value inversion above.

# Returns `(; RC, θ₁, P_rep_hat)` — estimates plus the first-stage CCP.
"""
function estimate_hotz_miller(x, d, β, θ₃, S; x0 = [15.0, 0.03])
    P_rep_hat, _ = fit_ccp_frequency(x, d, S)
    P_keep = clamp.(1 .- P_rep_hat, 1e-6, 1 - 1e-6)
    obj = p -> hm_negloglik(p, x, d, β, θ₃, S, P_keep)
    res = optimize(obj, x0, NelderMead(), Optim.Options(iterations = 5_000))
    RC, θ₁ = Optim.minimizer(res)
    return (; RC, θ₁, P_rep_hat)
end

# -----------------------------------------------------------------------------
# Numerical Hessian (central differences) — for MLE standard errors
# -----------------------------------------------------------------------------
"""
    num_hessian(f, θ; h = 1e-4) -> Matrix

Central-difference Hessian of scalar `f` at `θ`. Used to turn the curvature of
the negative log-likelihood into the observed-information standard errors.
"""
function num_hessian(f, θ; h = 1e-4)
    n = length(θ)
    H = zeros(n, n)
    for i in 1:n, j in 1:n
        ei = zeros(n); ei[i] = h
        ej = zeros(n); ej[j] = h
        H[i, j] = (f(θ + ei + ej) - f(θ + ei - ej) - f(θ - ei + ej) + f(θ - ei - ej)) / (4h^2)
    end
    return (H + H') / 2                                 # symmetrize
end
