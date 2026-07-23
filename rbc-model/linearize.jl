# =============================================================================
# linearize.jl — first-order (log-linear) solution via Klein's (2000) QZ method
# =============================================================================
#
# The main RBC solver. Log-linearize the equilibrium conditions around the
# deterministic steady state and solve the resulting linear rational-
# expectations system with the generalized Schur (QZ) decomposition. This is the
# standard fast method behind Dynare-style toolkits, and unlike value-function
# iteration it handles endogenous labor with no extra cost and delivers clean
# impulse responses and second moments.
#
# Variables are log-deviations from steady state: k̂, ẑ (states) and ĉ (the jump),
# with hours n̂, output ŷ, investment î recovered from the static relations. The
# linearized equilibrium (separable log-c / power-disutility preferences) is:
#
#   labor FOC:   n̂ = (ẑ + α·k̂ − ĉ) / D,         D = α + 1/ν  (D→∞ if inelastic)
#   production:  ŷ = G·(α·k̂ + ẑ) − H·ĉ,          G = 1+(1−α)/D,  H = (1−α)/D
#   resource:    (c/y)·ĉ + (k/y)·k̂' = ŷ + (1−δ)(k/y)·k̂
#   Euler:       ĉ_t = E_t[ ĉ_{t+1} − φ·(ŷ_{t+1} − k̂_{t+1}) ],   φ = 1 − β(1−δ)
#   TFP:         ẑ_{t+1} = ρ·ẑ_t + ε_{t+1}
#
# Stacked as A·E_t[x_{t+1}] = B·x_t with x = [k̂, ẑ, ĉ] and two predetermined
# variables (k̂, ẑ).

using LinearAlgebra

"""
    klein(A, B, n_pred) -> (P, F, eigs)

Solve `A·E_t[x_{t+1}] = B·x_t` for a saddle-path-stable rational-expectations
equilibrium with `n_pred` predetermined variables (Klein 2000). Partition
`x = [x₁; x₂]` into predetermined `x₁` and jump `x₂`.

# Returns
- `P` — law of motion of the predetermined block, `x₁_{t+1} = P·x₁_t`.
- `F` — jump policy, `x₂_t = F·x₁_t`.
- `eigs` — the dynamic multipliers `|T_ii/S_ii|` (sorted), for the BK check.

Errors if the number of stable multipliers (`|μ| < 1`) differs from `n_pred`
(Blanchard–Kahn violated ⇒ no unique stable solution).
"""
function klein(A::Matrix, B::Matrix, n_pred::Int)
    sf = schur(complex(A), complex(B))                 # generalized Schur of (A,B)
    μ  = abs.(diag(sf.T) ./ diag(sf.S))                # dynamic multipliers |T/S|
    stable = μ .< 1.0
    ns = count(stable)
    ns == n_pred || error("Blanchard–Kahn fails: $ns stable modes vs $n_pred predetermined.")
    ordschur!(sf, stable)                              # move stable modes to top-left
    S, T, Z = sf.S, sf.T, sf.Z
    np = n_pred
    Z11 = Z[1:np, 1:np];   Z21 = Z[np+1:end, 1:np]
    S11 = S[1:np, 1:np];   T11 = T[1:np, 1:np]
    F = real(Z21 / Z11)                                # jump = F · predetermined
    P = real(Z11 * (S11 \ T11) / Z11)                  # x₁' = P · x₁
    return P, F, sort(μ)
end

"""
    LinearRBC

Container for a solved log-linear RBC model: the steady state `ss`, the state
law of motion `P` (for `[k̂, ẑ]`), and a loadings matrix `C` mapping the state
`[k̂, ẑ]` to every variable of interest, with row order
`[ŷ, ĉ, î, n̂, k̂', ẑ]`. All entries are elasticities (log-dev per log-dev).
"""
struct LinearRBC
    ss::NamedTuple
    P::Matrix{Float64}        # [k̂';ẑ'] = P·[k̂;ẑ]
    C::Matrix{Float64}        # [ŷ,ĉ,î,n̂] = C·[k̂;ẑ]
    γ::Vector{Float64}        # ĉ = γ·[k̂;ẑ]   (the jump policy, for reference)
    vars::Vector{Symbol}
end

"""
    solve_linear(m::RBCModel) -> LinearRBC

Build the linearized coefficient matrices, solve with `klein`, and assemble the
state-space (state law of motion `P` and the variable loadings `C`).
"""
function solve_linear(m::RBCModel)
    (; α, β, δ, ρ) = m
    ss = steady_state(m)
    ky, cy = ss.ky, ss.cy
    φ = 1 - β * (1 - δ)

    D = m.labor == :inelastic ? Inf : α + inv_frisch(m)
    G = isinf(D) ? 1.0 : 1 + (1 - α) / D
    H = isinf(D) ? 0.0 : (1 - α) / D

    # A·E[x_{t+1}] = B·x_t,  x = [k̂, ẑ, ĉ]
    A = [ 1.0           0.0   0.0;
          0.0           1.0   0.0;
         -φ*(G*α - 1)  -φ*G   (1 + φ*H) ]
    B = [ (G*α/ky + (1 - δ))   G/ky   -((H + cy)/ky);
          0.0                  ρ       0.0;
          0.0                  0.0     1.0 ]

    P_full, F, _ = klein(A, B, 2)        # x₁=[k̂,ẑ] predetermined, ĉ jump
    γ = vec(F)                            # ĉ = γ_k·k̂ + γ_z·ẑ

    # Variable loadings on the state [k̂, ẑ].
    Cmat = zeros(4, 2)
    # ŷ = G(αk̂+ẑ) − Hĉ
    Cmat[1, :] = [G*α, G] .- H .* γ
    # ĉ
    Cmat[2, :] = γ
    # n̂ = (ẑ + αk̂ − ĉ)/D   (zero under inelastic)
    Cmat[4, :] = isinf(D) ? [0.0, 0.0] : ([α, 1.0] .- γ) ./ D
    # î = (k̂' − (1−δ)k̂)/δ,   k̂' = P_full[1,:]·[k̂,ẑ]
    kp = P_full[1, :]
    Cmat[3, :] = (kp .- [(1 - δ), 0.0]) ./ δ

    return LinearRBC(ss, P_full, Cmat, γ, [:y, :c, :i, :n])
end

"""
    irf(lin::LinearRBC, σ_shock; T = 40) -> NamedTuple

Impulse responses to a one-standard-deviation TFP innovation: start from steady
state, hit `ẑ` with `σ_shock`, and propagate the linear state law of motion.
Returns percent log-deviation paths for output, consumption, investment, hours,
capital, and TFP.
"""
function irf(lin::LinearRBC, σ_shock; T::Int = 40)
    s = [0.0, σ_shock]                    # k̂₀ = 0, ẑ₀ = one-std shock
    Y = zeros(T); C = zeros(T); I = zeros(T); N = zeros(T); K = zeros(T); Z = zeros(T)
    for t in 1:T
        v = lin.C * s                     # [ŷ, ĉ, î, n̂]
        Y[t], C[t], I[t], N[t] = v
        K[t], Z[t] = s
        s = lin.P * s                     # advance the state
    end
    pct = x -> 100 .* x
    return (; y = pct(Y), c = pct(C), i = pct(I), n = pct(N), k = pct(K), z = pct(Z))
end
