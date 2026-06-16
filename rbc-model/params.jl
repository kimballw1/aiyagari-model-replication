# =============================================================================
# params.jl
# =============================================================================

"""
RBCModel

Parameters of the stochastic neoclassical growth model — the core of Kydland &
Prescott's (1982) real-business-cycle theory. A representative household/planner
chooses consumption, labor, and capital accumulation in response to a persistent
total-factor-productivity (TFP) shock `z`:

    max E Σ βᵗ [ log cₜ − χ·nₜ^{1+1/ν}/(1+1/ν) ]
    s.t. cₜ + kₜ₊₁ = zₜ·kₜ^α·nₜ^{1−α} + (1−δ)·kₜ
         log zₜ₊₁ = ρ·log zₜ + εₜ₊₁,   εₜ₊₁ ~ N(0, σ_ε²)

Labor supply is governed by `labor`:
- `:inelastic`   — `n ≡ 1` (the undergraduate-lecture / textbook growth model).
- `:divisible`   — interior labor with Frisch elasticity `ν` (King–Plosser–Rebelo).
- `:indivisible` — Hansen (1985): linear labor disutility (`ν → ∞`), which makes
  the *aggregate* labor-supply elasticity effectively infinite and is the key
  refinement that lets the model match hours volatility.

Defaults are the standard **quarterly** RBC calibration (Hansen 1985 / King–
Plosser–Rebelo): `β = 0.99` (≈4%/yr real rate), `α = 0.36`, `δ = 0.025`,
`ρ = 0.95`, `σ_ε = 0.007`, steady-state hours `n* = 1/3`. `χ` is *not* a free
parameter — it is pinned down by the `n* = 1/3` target in `steadystate.jl`.
"""
@kwdef struct RBCModel
    β::Float64   = 0.99          # discount factor (quarterly)
    α::Float64   = 0.36          # capital share
    δ::Float64   = 0.025         # depreciation rate (quarterly)
    ρ::Float64   = 0.95          # TFP persistence
    σ_ε::Float64 = 0.007         # TFP innovation std. dev.
    ν::Float64   = 2.0           # Frisch labor-supply elasticity (:divisible)
    n_target::Float64 = 1/3      # steady-state hours target (pins down χ)
    labor::Symbol = :divisible   # :inelastic | :divisible | :indivisible

    # numerical (used by the VFI benchmark)
    N_z::Int = 7                 # productivity states (Rouwenhorst)
    N_k::Int = 200               # capital grid points
    m_z::Float64 = 3.0           # not used by Rouwenhorst (kept for reference)
end

# Effective inverse Frisch elasticity 1/ν: 0 for indivisible (Hansen, linear
# disutility); ∞-Frisch ⇒ 1/ν = 0. Inelastic labor is handled separately (n≡1).
inv_frisch(m::RBCModel) = m.labor == :indivisible ? 0.0 : 1.0 / m.ν
