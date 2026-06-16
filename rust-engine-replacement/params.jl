# =============================================================================
# params.jl
# =============================================================================

"""
RustModel

All structural and numerical parameters of Rust's (1987) optimal bus-engine
replacement model. Harold Zurcher, superintendent of the Madison Metropolitan
Bus Company, observes each bus's accumulated mileage `x` (discretized into `S`
bins) and each period chooses `d = 0` (keep and pay maintenance) or `d = 1`
(replace the engine, resetting mileage to zero).

The flow payoffs are

    u(x, d=0) = −c(x)  = −θ₁·x        (maintenance rises linearly with mileage)
    u(x, d=1) = −RC                   (a fixed replacement cost)

plus i.i.d. Type-I Extreme-Value (logit) shocks `ε(d)` that Zurcher sees but the
econometrician does not. Mileage advances by `Δ ∈ {0,1,2}` bins per period with
probabilities `θ₃`.

Defaults follow the undergraduate lecture's clean calibration; Rust's own
estimates (groups 1–4, linear cost) are roughly `RC ≈ 10` and `θ₁` an order of
magnitude smaller, in thousands of 1985 dollars — see the README.
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
