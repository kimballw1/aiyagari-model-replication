# =============================================================================
# analyze.jl — RBC model: calibration, policies, the business-cycle scorecard,
#              impulse responses, and comparative statics
# =============================================================================
#
# Run with:  julia analyze.jl   (includes main.jl, then works through §1–§5 and
# writes figures to graphs/). Everything runs off the VFI solution, so the
# comparative-statics sweep in §5 re-solves the model once per ρ — a full pass
# takes ~2–3 minutes.

include("main.jl")

using Statistics
using Plots

const FIG = joinpath(@__DIR__, "graphs")
isdir(FIG) || mkdir(FIG)

m  = RBCModel()                  # baseline: quarterly, inelastic labor
ss = steady_state(m)

println("\n", "═"^64)
println("  RBC ANALYSIS (Kydland–Prescott 1982) — quarterly calibration")
println("═"^64)


# =============================================================================
# §1  CALIBRATION & STEADY STATE
# =============================================================================
# Parameters are pinned to long-run averages; then we ask whether the model
# reproduces business-cycle moments it was NOT calibrated to match.

println("\n", "─"^64); println("  §1  CALIBRATION & STEADY STATE"); println("─"^64)
println("  ", rpad("β (discount)", 22), m.β, "   → annual real rate ≈ ", round(100*((1/m.β)^4 - 1), digits = 1), "%")
println("  ", rpad("α (capital share)", 22), m.α)
println("  ", rpad("δ (depreciation)", 22), m.δ, "   (", round(100*m.δ, digits = 1), "%/qtr)")
println("  ", rpad("ρ, σ_ε (TFP AR(1))", 22), m.ρ, ", ", m.σ_ε)
println()
println("  steady-state great ratios (annualized for K/Y):")
println("  ", rpad("K/Y", 8), round(ss.ky/4, digits = 2), "   (data ≈ 2.5–3.0)")
println("  ", rpad("C/Y", 8), round(ss.cy, digits = 2), "   (data ≈ 0.6–0.7)")
println("  ", rpad("I/Y", 8), round(ss.iy, digits = 2), "   (data ≈ 0.25–0.30)")


# =============================================================================
# §2  POLICY FUNCTIONS  (value-function iteration)
# =============================================================================
# Solve the model. The savings policy crosses the 45° line at the (z-specific)
# steady state; consumption and investment rise with both capital and
# productivity, and investment responds far more than consumption.

println("\n", "─"^64); println("  §2  POLICY FUNCTIONS (VFI)"); println("─"^64)
t_vfi = @elapsed sol = solve_vfi(m)
kg, zg = sol.k_grid, sol.z_grid
println("  VFI solved in ", round(t_vfi, digits = 1), " s   (k_ss = ", round(ss.k, digits = 2), ")")

idx = [1, (m.N_z+1)÷2, m.N_z]
plt_pol = plot(layout = (1, 2), size = (900, 360))
for iz in idx
    plot!(plt_pol, kg, sol.kp[:, iz], lw = 2, label = "z = $(round(zg[iz], digits = 3))",
          xlabel = "capital k", ylabel = "k'", title = "Savings Policy k'(k,z)", subplot = 1, legend = :topleft)
end
plot!(plt_pol, kg, kg, ls = :dash, color = :black, label = "45°", subplot = 1)
vline!(plt_pol, [ss.k], ls = :dot, color = :gray, label = "", subplot = 1)
for iz in idx
    plot!(plt_pol, kg, sol.c[:, iz], lw = 2, label = "z = $(round(zg[iz], digits = 3))",
          xlabel = "capital k", ylabel = "c", title = "Consumption Policy c(k,z)", subplot = 2, legend = :topleft)
end
savefig(plt_pol, joinpath(FIG, "fig_policy_functions.png"))

plt_v = plot(xlabel = "capital k", ylabel = "V(k,z)", title = "Value Function", legend = :bottomright)
for iz in idx
    plot!(plt_v, kg, sol.V[:, iz], lw = 2, label = "z = $(round(zg[iz], digits = 3))")
end
vline!(plt_v, [ss.k], ls = :dash, color = :gray, label = "k_ss")
savefig(plt_v, joinpath(FIG, "fig_value_function.png"))
println("  → saved: fig_policy_functions.png, fig_value_function.png")


# =============================================================================
# §3  THE BUSINESS-CYCLE SCORECARD
# =============================================================================
# Simulate, HP-filter (λ=1600), and confront the model with US data. The model
# is given ONE shock (TFP) and asked to reproduce the relative volatilities and
# comovements of the macro aggregates — moments it was never calibrated to.
# Hours are constant by construction (inelastic labor), so the hours margin —
# σ_n/σ_y ≈ 1 in the data — is the model's known missing piece.

println("\n", "─"^64); println("  §3  BUSINESS-CYCLE SCORECARD"); println("─"^64)

# US data, postwar quarterly, HP-filtered (King & Rebelo 1999).
us = Dict(:output => (1.81, 1.00, 1.00), :consumption => (1.35, 0.74, 0.88),
          :investment => (5.30, 2.93, 0.80))
bc = business_cycle_stats(sol, m)
println("  ", rpad("variable", 13), rpad("σ model", 9), rpad("σ US", 8),
        rpad("σ/σy mod", 10), rpad("σ/σy US", 9), rpad("corr mod", 10), "corr US")
for r in bc.rows
    u = us[r[1]]
    println("  ", rpad(string(r[1]), 13), rpad(round(r[2], digits = 2), 9), rpad(u[1], 8),
            rpad(round(r[3], digits = 2), 10), rpad(u[2], 9), rpad(round(r[4], digits = 2), 10), u[3])
end
println("  → σ_y = ", round(bc.σy, digits = 2), "% vs US 1.81%  (TFP shocks alone, with no labor margin,",
        " explain ≈ ", round(Int, 100*bc.σy/1.81), "%)")

# Sample simulated path for the figure (HP-filtered, like the scorecard)
sim = simulate_vfi(sol, m, 400; seed = 3)
cyc(x) = hp_filter(x, 1600)[2]
yc, cc, ic = cyc(sim.y), cyc(sim.c), cyc(sim.i)
plt_sim = plot(1:160, 100 .* yc[1:160], lw = 2, color = :black, label = "output",
               xlabel = "quarter", ylabel = "% deviation from trend", title = "Simulated RBC Economy", legend = :topright)
plot!(plt_sim, 1:160, 100 .* cc[1:160], lw = 2, color = :seagreen, label = "consumption")
plot!(plt_sim, 1:160, 100 .* ic[1:160], lw = 2, color = :crimson, label = "investment")
savefig(plt_sim, joinpath(FIG, "fig_simulation.png"))
println("  → saved: fig_simulation.png")


# =============================================================================
# §4  IMPULSE RESPONSES TO A TFP SHOCK
# =============================================================================
# The dynamic signature of the model: a 1-std TFP innovation raises output on
# impact; investment jumps far more than output (it is the adjustment margin) and
# consumption rises smoothly; capital builds up slowly and propagates the shock
# long after TFP itself has decayed.

println("\n", "─"^64); println("  §4  IMPULSE RESPONSES"); println("─"^64)
ir = irf_vfi(sol, m; T = 40)
println("  peak responses to a 1-std (", round(100*m.σ_ε, digits = 1), "%) TFP shock:")
println("    output ", round(maximum(ir.y), digits = 2), "%   investment ", round(maximum(ir.i), digits = 2),
        "%   consumption ", round(maximum(ir.c), digits = 2), "%   capital ", round(maximum(ir.k), digits = 3), "%")

plt_irf = plot(layout = (2, 3), size = (960, 540), legend = false)
for (j, (lab, col)) in enumerate([(:y, :black), (:c, :seagreen), (:i, :crimson),
                                  (:k, :navy), (:z, :darkorange)])
    plot!(plt_irf, 0:39, getfield(ir, lab), lw = 2, color = col, subplot = j,
          title = uppercase(string(lab)), xlabel = "quarter", ylabel = "%")
    hline!(plt_irf, [0], ls = :dot, color = :gray, subplot = j)
end
plot!(plt_irf, [NaN], subplot = 6, framestyle = :none)
savefig(plt_irf, joinpath(FIG, "fig_irf.png"))
println("  → saved: fig_irf.png")


# =============================================================================
# §5  COMPARATIVE STATICS — TFP PERSISTENCE
# =============================================================================
# Persistence ρ governs propagation: more persistent shocks generate larger,
# longer-lived output responses. Each ρ is a full VFI re-solve (~30 s each).
# The savings-policy slope dk'/dk at the steady state is the model's internal
# capital persistence — set by α, β, δ, essentially independent of ρ.

println("\n", "─"^64); println("  §5  COMPARATIVE STATICS"); println("─"^64)
ρ_vals = [0.80, 0.90, 0.95, 0.99]
println("  ", rpad("ρ", 8), rpad("σ_y(%)", 10), "dk'/dk at k_ss")
h = 0.01 * ss.k
irs_ρ = Dict{Float64,Any}()
σy_by_ρ = Float64[]
for ρ in ρ_vals
    mr = RBCModel(ρ = ρ)
    sr = ρ == m.ρ ? sol : solve_vfi(mr)
    bcr = business_cycle_stats(sr, mr)
    irs_ρ[ρ] = irf_vfi(sr, mr; T = 40)
    push!(σy_by_ρ, bcr.σy)
    iz_mid = (mr.N_z + 1) ÷ 2
    slope = (sr.kp_itp[iz_mid](ss.k + h) - sr.kp_itp[iz_mid](ss.k - h)) / (2h)
    println("  ", rpad(ρ, 8), rpad(round(bcr.σy, digits = 2), 10), round(slope, digits = 4))
end

plt_cs = plot(layout = (1, 2), size = (840, 340))
plot!(plt_cs, ρ_vals, σy_by_ρ, lw = 2, marker = :circle, color = :navy, legend = false,
      xlabel = "TFP persistence ρ", ylabel = "σ(output) %", title = "Output Volatility vs ρ", subplot = 1)
for ρ in ρ_vals
    plot!(plt_cs, 0:39, irs_ρ[ρ].y, lw = 2, label = "ρ = $ρ",
          xlabel = "quarter", ylabel = "output %", title = "Output IRF vs ρ", subplot = 2, legend = :topright)
end
savefig(plt_cs, joinpath(FIG, "fig_comparative_statics.png"))
println("  → saved: fig_comparative_statics.png")


# =============================================================================
# SUMMARY
# =============================================================================
println("\n", "═"^64); println("  ALL FIGURES SAVED → graphs/"); println("═"^64)
for (f, d) in [
    ("fig_policy_functions.png",   "VFI savings & consumption policies"),
    ("fig_value_function.png",     "value function V(k,z)"),
    ("fig_simulation.png",         "simulated output/consumption/investment"),
    ("fig_irf.png",                "impulse responses to a TFP shock"),
    ("fig_comparative_statics.png","output volatility & IRF vs ρ"),
]
    println("  ", rpad(f, 30), " ", d)
end
println("═"^64)
