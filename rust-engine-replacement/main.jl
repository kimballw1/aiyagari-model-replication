# =============================================================================
# main.jl — assemble the blocks and solve the baseline
# =============================================================================

using LinearAlgebra

include("params.jl")
include("transitions.jl")
include("solve.jl")
include("simulate.jl")
include("estimate.jl")

"""
    run_baseline(m = RustModel()) -> NamedTuple

Solve the baseline replacement model and print the headline objects: the
reservation mileage (where replacement becomes more likely than not) and the
long-run replacement rate implied by the stationary mileage distribution.
"""
function run_baseline(m::RustModel = RustModel())
    sol = solve_rust(m)
    res_mile = m.x̄[findfirst(sol.P .>= 0.5)]
    π = stationary_mileage(sol.P, m.θ₃, m.S)
    repl_rate = dot(π, sol.P)
    println("\n=== Rust (1987) Engine Replacement — Baseline ===")
    println("  replacement cost   RC      = ", m.RC)
    println("  maintenance slope  θ₁      = ", m.θ₁)
    println("  reservation mileage x*      = ", round(res_mile, digits = 1),
            "  (P(replace) crosses 0.5)")
    println("  long-run replacement rate   = ", round(repl_rate, digits = 4),
            "  ⇒ replace ≈ every ", round(1 / repl_rate, digits = 0), " periods")
    return sol
end

if abspath(PROGRAM_FILE) == @__FILE__
    m   = RustModel()
    sol = run_baseline(m)
end
