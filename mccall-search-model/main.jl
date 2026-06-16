# =============================================================================
# main.jl — assemble the blocks and solve the baseline
# =============================================================================

using LinearAlgebra
using QuantEcon: tauchen, stationary_distributions, MarkovChain

include("params.jl")
include("offers.jl")
include("solve.jl")
include("correlated.jl")
include("simulate.jl")

"""
    run_baseline(p = McCallModel()) -> NamedTuple

Solve the baseline McCall model (with separation) and print the headline
objects: the reservation wage, the job-finding rate, expected unemployment
duration, and the steady-state unemployment rate.
"""
function run_baseline(p::McCallModel = McCallModel())
    sol = solve_mccall(p)
    println("\n=== McCall Search Model — Baseline Equilibrium ===")
    println("  reservation wage  w̄        = ", round(sol.w_res, digits = 3))
    println("  job-finding rate  f        = ", round(sol.f, digits = 4))
    println("  expected duration 1/f      = ", round(1 / sol.f, digits = 2), " periods")
    println("  separation rate   α        = ", round(p.α, digits = 4))
    println("  steady-state unemployment  = ", round(100 * sol.u_rate, digits = 2), "%")
    println("  value of unemployment  U   = ", round(sol.U, digits = 3))
    return sol
end

# Run when executed directly (not when included by analyze.jl).
if abspath(PROGRAM_FILE) == @__FILE__
    p   = McCallModel()
    sol = run_baseline(p)
end
