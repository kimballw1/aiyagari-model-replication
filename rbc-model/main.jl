# =============================================================================
# main.jl — assemble the blocks and solve the baseline
# =============================================================================

using LinearAlgebra

include("params.jl")
include("steadystate.jl")
include("rouwenhorst.jl")
include("vfi.jl")
include("moments.jl")

"""
    run_baseline(m = RBCModel()) -> NamedTuple

Solve the baseline RBC model by value-function iteration, print the
calibration, the steady-state great ratios, and the headline business-cycle
moments. Returns the VFI solution.
"""
function run_baseline(m::RBCModel = RBCModel())
    ss = steady_state(m)
    println("\n=== RBC Model (Kydland–Prescott 1982) — Baseline ===")
    println("  β=$(m.β), α=$(m.α), δ=$(m.δ), ρ=$(m.ρ), σ_ε=$(m.σ_ε)   (inelastic labor, n ≡ 1)")
    println("  steady state:  K/Y = ", round(ss.ky / 4, digits = 2), " (annual)",
            "   C/Y = ", round(ss.cy, digits = 2),
            "   I/Y = ", round(ss.iy, digits = 2))
    t = @elapsed sol = solve_vfi(m)
    println("  VFI solved in ", round(t, digits = 1), " s")
    bc = business_cycle_stats(sol, m)
    println("  σ(output)           = ", round(bc.σy, digits = 2), "%")
    for r in bc.rows
        println("    ", rpad(string(r[1]), 13), " σ=", rpad(round(r[2], digits = 2), 6),
                " σ/σy=", rpad(round(r[3], digits = 2), 6), " corr_y=", round(r[4], digits = 2))
    end
    return sol
end

if abspath(PROGRAM_FILE) == @__FILE__
    sol = run_baseline(RBCModel())
end
