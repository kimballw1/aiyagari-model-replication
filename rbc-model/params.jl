# =============================================================================
# params.jl
# =============================================================================

"""
RBCModel

Parameters. Defaults are the standard quarterly RBC calibration: `β = 0.99` (≈4%/yr real
rate), `α = 0.36` (capital's income share), `δ = 0.025` (≈10%/yr depreciation),
`(ρ, σ_ε) = (0.95, 0.007)` from the estimated Solow-residual process.
"""
@kwdef struct RBCModel
    β::Float64   = 0.99          # discount factor (quarterly)
    α::Float64   = 0.36          # capital share
    δ::Float64   = 0.025         # depreciation rate (quarterly)
    ρ::Float64   = 0.95          # TFP persistence
    σ_ε::Float64 = 0.007         # TFP innovation std. dev.

    # numerical
    N_z::Int = 7                 # productivity states (Rouwenhorst)
    N_k::Int = 200               # capital grid points
end
