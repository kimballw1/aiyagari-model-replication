# =============================================================================
# main.jl — assemble the blocks and solve the baseline
# =============================================================================

using LinearAlgebra

include("params.jl")
include("steadystate.jl")
include("rouwenhorst.jl")
include("linearize.jl")
include("vfi.jl")
include("moments.jl")

"""
    run_baseline(m = RBCModel()) -> NamedTuple

Solve the baseline RBC model by log-linearization, print the calibration, the
steady-state great ratios, and the headline business-cycle moments.
"""
function run_baseline(m::RBCModel = RBCModel())
    lin = solve_linear(m)
    ss  = lin.ss
    println("\n=== RBC Model (Kydland–Prescott 1982) — Baseline ===")
    println("  labor supply: ", m.labor, "   (β=$(m.β), α=$(m.α), δ=$(m.δ), ρ=$(m.ρ), σ_ε=$(m.σ_ε))")
    println("  steady state:  K/Y = ", round(ss.ky, digits = 2),
            "   C/Y = ", round(ss.cy, digits = 2),
            "   I/Y = ", round(ss.iy, digits = 2),
            "   n* = ", round(ss.n, digits = 3))
    bc = business_cycle_stats(lin, m)
    println("  σ(output)           = ", round(bc.σy, digits = 2), "%")
    for r in bc.rows
        println("    ", rpad(string(r[1]), 13), " σ=", rpad(round(r[2], digits = 2), 6),
                " σ/σy=", rpad(round(r[3], digits = 2), 6), " corr_y=", round(r[4], digits = 2))
    end
    return lin
end

if abspath(PROGRAM_FILE) == @__FILE__
    lin = run_baseline(RBCModel())
end
