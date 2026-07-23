# =============================================================================
# params.jl
# =============================================================================

"""
RustModel

All structural and numerical parameters of Rust's (1987) optimal bus-engine
replacement model.
"""
@kwdef struct RustModel
    β::Float64  = 0.95                          # discount factor
    S::Int      = 50                            # number of mileage bins
    RC::Float64 = 8.0                           # engine replacement cost (interior threshold)
    θ₁::Float64 = 0.04                          # per-bin maintenance-cost slope
    θ₃::Vector{Float64} = [0.36, 0.48, 0.16]    # P(Δ mileage = 0, 1, 2)
    x̄::Vector{Float64}  = Float64.(0:S-1)       # mileage levels at each state
end

# Maintenance cost vector c(x) = θ₁·x at every mileage state.
cost(m::RustModel) = m.θ₁ .* m.x̄
