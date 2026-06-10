
# =============================================================================
# params.jl  —  Two-asset (liquid + illiquid) continuous-choice Aiyagari / KMV
# =============================================================================

"""
TwoAssetParams

All economic and numerical parameters for the two-asset continuous-choice model.

Two assets:
  b — LIQUID asset.   Freely adjustable each period. Return r_b. Can go negative
                      (unsecured borrowing) down to `b_min`.
  a — ILLIQUID asset. Adjusting it costs X(d,a). Return r_a > r_b. The wedge
                      r_a − r_b is the illiquidity premium, sustained in
                      equilibrium by the adjustment friction. a ≥ 0.

The illiquid asset is physical capital (priced by the firm's MPK). The liquid
asset is in fixed net supply `B_supply` (a government bond / zero-net-supply
when B_supply = 0); r_b adjusts to clear the liquid market.
"""

@kwdef struct TwoAssetParams

# ---- ECONOMIC PARAMETERS -------------------------------------------------
    β::Float64  = 0.96     # discount factor
    γ::Float64  = 2.0      # CRRA risk aversion (note: discrete code calls this `y`)
    α::Float64  = 0.36     # capital share, Y = K^α L^(1-α)
    δ::Float64  = 0.08     # depreciation of illiquid capital
    ρ::Float64  = 0.9      # income persistence (AR(1) in logs)
    σ::Float64  = 0.2      # std of stationary log income

    # ---- ASSET RETURNS / SUPPLY ---------------------------------------------
    # r_a (illiquid return) is pinned by the firm: r_a = α(K/L)^(α-1) − δ.
    # r_b (liquid return) is solved for to clear the liquid market.
    B_supply::Float64 = 0.0   # net supply of the liquid asset (0 = zero-net-supply bond)

    # ---- ADJUSTMENT COST  χ(d, a) -------------------------------------------
    # d = a' − (1+r_a)·a  is the net deposit into / withdrawal from illiquid.
    # KMV convex form (suggested):  χ(d,a) = χ0·|d| + χ1·|d / (a + χ_kink)|^χ2 · (a + χ_kink)
    χ0::Float64    = 0.0      # linear (fixed-rate) component
    χ1::Float64    = 2.0      # convex component scale
    χ2::Float64    = 1.5      # convex component curvature ( >1 )
    χ_kink::Float64 = 0.5     # avoids divide-by-zero / softens cost near a = 0

    # ---- BORROWING LIMITS ----------------------------------------------------
    b_min::Float64 = 0.0      # liquid borrowing limit ( <0 allows unsecured debt )
    # illiquid a_min is 0 by construction

    # ---- GRID BOUNDS ---------------------------------------------------------
    b_max::Float64 = 50.0
    a_max::Float64 = 150.0    # illiquid grid top (must be slack in equilibrium)

    # ---- NUMERICAL PARAMETERS ------------------------------------------------
    n_b::Int = 60             # liquid grid points
    n_a::Int = 60             # illiquid grid points
    n_e::Int = 7              # income states
    E_b::Int = 3              # liquid grid curvature (more points near b_min)
    E_a::Int = 3              # illiquid grid curvature (more points near 0)

    method::Symbol = :vfi     # :vfi  → household_vfi.jl ;  :egm → household_egm.jl

    tol_vfi::Float64  = 1e-6
    tol_dist::Float64 = 1e-6
    tol_eq::Float64   = 1e-4
    max_iter_vfi::Int  = 2000
    max_iter_dist::Int = 5000
    howard_every::Int = 20    # (VFI path) Howard policy-improvement cadence
    howard_steps::Int = 30
end
