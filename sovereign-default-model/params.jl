# =============================================================================
# params.jl
# =============================================================================

"""
SovDefaultModel

Parameters of the Eaton–Gersovitz (1981) / Arellano (2008) sovereign-default
model. A benevolent government borrows from risk-neutral international lenders by
issuing one-period discount bonds, and each period — after observing a stochastic
endowment `y` — chooses whether to **repay** its debt or **default**. Default
wipes out the debt but triggers two costs: temporary exclusion from credit
markets (re-entry probability `λ` per period) and an output loss while excluded.

Lenders price the bonds competitively, so the bond price `q(b', y)` already
embeds the equilibrium probability that the government defaults next period —
borrowing more, or borrowing in a bad state, raises the implied interest rate.
This endogenous price schedule is what makes the model a sovereign-debt theory
rather than a plain consumption-smoothing problem.

Defaults follow Arellano's (2008) quarterly Argentina calibration. `β = 0.953`
is deliberately *low* — a patient government would never borrow enough to default.

Output cost of default (`cost`):
- `:arellano`  — asymmetric ceiling `h(y) = min(y, ŷ·E[y])`, so default is
  *more* costly in good times (the mechanism that makes default countercyclical).
- `:proportional` — a flat fraction `φ` of output is lost: `h(y) = (1−φ)·y`.
"""
@kwdef mutable struct SovDefaultModel
    β::Float64   = 0.96       # discount factor (calibrated to ≈4% default; Arellano's published value is 0.953)
    σ::Float64   = 2.0        # CRRA risk aversion
    r::Float64   = 0.017      # risk-free (lenders') interest rate, quarterly
    ρ::Float64   = 0.945      # endowment persistence (Argentine GDP)
    σ_ε::Float64 = 0.025      # endowment innovation std dev
    λ::Float64   = 0.282      # probability of regaining market access
    ŷ::Float64   = 0.969      # default output ceiling as a fraction of mean income
    φ::Float64   = 0.02       # proportional output loss (cost = :proportional)
    τ::Float64   = 0.001      # taste-shock scale (smooths the default boundary)
    cost::Symbol = :arellano  # :arellano | :proportional

    N_y::Int = 21             # endowment states
    N_b::Int = 251            # bond grid (value-function storage)
    N_b_fine::Int = 251       # fine bond grid (b' optimization search)
    b_min::Float64 = -0.45    # most debt the government can owe (b < 0 is debt)
    b_max::Float64 = 0.45     # most the government can save (b > 0 is assets)

    y_grid::Vector{Float64}      = Float64[]   # filled by setup!
    P_y::Matrix{Float64}         = zeros(0, 0) # filled by setup!
    b_grid::Vector{Float64}      = Float64[]   # sparse grid
    b_grid_fine::Vector{Float64} = Float64[]   # fine grid
end
