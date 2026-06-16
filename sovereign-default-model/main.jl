# =============================================================================
# main.jl — assemble the blocks and solve the baseline
# =============================================================================

using LinearAlgebra

include("params.jl")
include("income.jl")
include("solve.jl")
include("simulate.jl")
include("moments.jl")

"""
    run_baseline(m = setup!(SovDefaultModel())) -> (m, sol)

Solve the baseline Arellano (2008) model and print the headline business-cycle
statistics from a long simulation.
"""
function run_baseline(m::SovDefaultModel = setup!(SovDefaultModel()))
    sol = solve_sovereign(m; verbose = true)
    sim = simulate_sovereign(m, sol, 20_000)
    st  = default_statistics(m, sim)
    println("\n=== Sovereign Default (Arellano 2008) — Baseline ===")
    println("  annualized default rate    = ", round(st.def_rate, digits = 2), "%")
    println("  mean debt / output         = ", round(st.mean_debt, digits = 2), "%")
    println("  mean spread (annualized)   = ", round(st.mean_spread, digits = 2), "%")
    println("  std spread                 = ", round(st.std_spread, digits = 2), "%")
    println("  σ(c)/σ(y)                  = ", round(st.rel_c, digits = 2), "  (>1: consumption amplified)")
    println("  corr(spread, y)            = ", round(st.corr_sy, digits = 2), "  (<0: countercyclical)")
    println("  corr(trade bal, y)         = ", round(st.corr_tby, digits = 2), "  (<0: countercyclical)")
    return m, sol
end

if abspath(PROGRAM_FILE) == @__FILE__
    m, sol = run_baseline()
end
