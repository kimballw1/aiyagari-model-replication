# =============================================================================
# analyze.jl — RBC model: calibration, policies, the business-cycle scorecard,
#              impulse responses, the role of labor, and comparative statics
# =============================================================================
#
# Run with:  julia analyze.jl   (includes main.jl, then works through §1–§7 and
# writes figures to graphs/). A full pass is ~40 s; the VFI benchmark in §2–§3
# dominates the time (the log-linear solution everywhere else is instant).

include("main.jl")

using Statistics
using Plots

const FIG = joinpath(@__DIR__, "graphs")
isdir(FIG) || mkdir(FIG)

m   = RBCModel()                 # baseline: quarterly, divisible labor
lin = solve_linear(m)
ss  = lin.ss

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
println("  ", rpad("n*", 8), round(ss.n, digits = 3), "   (target 1/3)")


# =============================================================================
# §2  POLICY FUNCTIONS  (exact, by value-function iteration)
# =============================================================================
# Solve the inelastic-labor growth model exactly. The savings policy crosses the
# 45° line at the (z-specific) steady state; consumption and investment rise with
# both capital and productivity, and investment responds far more than consumption.

println("\n", "─"^64); println("  §2  POLICY FUNCTIONS (VFI)"); println("─"^64)
mi  = RBCModel(labor = :inelastic)
t_vfi = @elapsed solv = solve_vfi(mi)
k_ss_i = (mi.α*mi.β/(1 - mi.β*(1-mi.δ)))^(1/(1-mi.α))
kg, zg = solv.k_grid, solv.z_grid
println("  VFI solved in ", round(t_vfi, digits = 1), " s   (k_ss = ", round(k_ss_i, digits = 2), ")")

idx = [1, (mi.N_z+1)÷2, mi.N_z]
plt_pol = plot(layout = (1, 2), size = (900, 360))
for iz in idx
    plot!(plt_pol, kg, solv.kp[:, iz], lw = 2, label = "z = $(round(zg[iz], digits = 3))",
          xlabel = "capital k", ylabel = "k'", title = "Savings Policy k'(k,z)", subplot = 1, legend = :topleft)
end
plot!(plt_pol, kg, kg, ls = :dash, color = :black, label = "45°", subplot = 1)
vline!(plt_pol, [k_ss_i], ls = :dot, color = :gray, label = "", subplot = 1)
for iz in idx
    plot!(plt_pol, kg, solv.c[:, iz], lw = 2, label = "z = $(round(zg[iz], digits = 3))",
          xlabel = "capital k", ylabel = "c", title = "Consumption Policy c(k,z)", subplot = 2, legend = :topleft)
end
savefig(plt_pol, joinpath(FIG, "fig_policy_functions.png"))

plt_v = plot(xlabel = "capital k", ylabel = "V(k,z)", title = "Value Function", legend = :bottomright)
for iz in idx
    plot!(plt_v, kg, solv.V[:, iz], lw = 2, label = "z = $(round(zg[iz], digits = 3))")
end
vline!(plt_v, [k_ss_i], ls = :dash, color = :gray, label = "k_ss")
savefig(plt_v, joinpath(FIG, "fig_value_function.png"))
println("  → saved: fig_policy_functions.png, fig_value_function.png")


# =============================================================================
# §3  LOG-LINEAR vs EXACT  (validating the perturbation solution)
# =============================================================================
# The log-linear policy is a first-order Taylor expansion about the steady state.
# It should be indistinguishable from the exact VFI policy near k_ss and drift
# away only in the tails — confirming both solvers are correct.

println("\n", "─"^64); println("  §3  LOG-LINEAR vs EXACT VFI"); println("─"^64)
lin_i = solve_linear(mi)
iz_mid = (mi.N_z+1)÷2
kp_vfi = solv.kp[:, iz_mid]
# linear savings policy in levels: k' = k_ss·exp(k̂'),  k̂' = P[1,1]·k̂  (z at mean ⇒ ẑ=0)
k̂  = log.(kg ./ k_ss_i)
kp_lin = k_ss_i .* exp.(lin_i.P[1, 1] .* k̂)
h = 0.01k_ss_i
slope_vfi = (linear_interpolation(kg, kp_vfi)(k_ss_i + h) - linear_interpolation(kg, kp_vfi)(k_ss_i - h))/(2h)
println("  ", rpad("d k'/d k at k_ss", 22), "VFI = ", round(slope_vfi, digits = 4),
        "   linear = ", round(lin_i.P[1,1], digits = 4))
println("  ", rpad("max |VFI − linear|/k_ss", 24), round(maximum(abs.(kp_vfi .- kp_lin))/k_ss_i, digits = 4))

plt_lv = plot(kg, kp_vfi, lw = 2.5, color = :black, label = "VFI (exact)",
              xlabel = "capital k", ylabel = "k'", title = "Savings Policy: Exact vs Log-Linear", legend = :topleft)
plot!(plt_lv, kg, kp_lin, lw = 2, ls = :dash, color = :crimson, label = "log-linear")
plot!(plt_lv, kg, kg, lw = 1, ls = :dot, color = :gray, label = "45°")
vline!(plt_lv, [k_ss_i], lw = 1, ls = :dot, color = :gray, label = "")
savefig(plt_lv, joinpath(FIG, "fig_linear_vs_vfi.png"))
println("  → saved: fig_linear_vs_vfi.png")


# =============================================================================
# §4  THE BUSINESS-CYCLE SCORECARD
# =============================================================================
# Simulate, HP-filter (λ=1600), and confront the model with US data. The model
# is given ONE shock (TFP) and asked to reproduce the relative volatilities and
# comovements of the macro aggregates — moments it was never calibrated to.

println("\n", "─"^64); println("  §4  BUSINESS-CYCLE SCORECARD"); println("─"^64)

# US data, postwar quarterly, HP-filtered (King & Rebelo 1999).
us = Dict(:output => (1.81, 1.00, 1.00), :consumption => (1.35, 0.74, 0.88),
          :investment => (5.30, 2.93, 0.80), :hours => (1.79, 0.99, 0.88))
bc = business_cycle_stats(lin, m)
println("  ", rpad("variable", 13), rpad("σ model", 9), rpad("σ US", 8),
        rpad("σ/σy mod", 10), rpad("σ/σy US", 9), rpad("corr mod", 10), "corr US")
for r in bc.rows
    u = us[r[1]]
    println("  ", rpad(string(r[1]), 13), rpad(round(r[2], digits = 2), 9), rpad(u[1], 8),
            rpad(round(r[3], digits = 2), 10), rpad(u[2], 9), rpad(round(r[4], digits = 2), 10), u[3])
end
println("  → the model delivers σ_y = ", round(bc.σy, digits = 2),
        "% vs US 1.81% (TFP shocks alone explain ≈ ", round(100*bc.σy/1.81, digits = 0), "%)")

# Sample simulated path for the figure
sim = simulate_linear(lin, m, 200; seed = 3, burn = 100)
plt_sim = plot(1:160, 100 .* sim.y[1:160], lw = 2, color = :black, label = "output",
               xlabel = "quarter", ylabel = "% deviation", title = "Simulated RBC Economy", legend = :topright)
plot!(plt_sim, 1:160, 100 .* sim.c[1:160], lw = 2, color = :seagreen, label = "consumption")
plot!(plt_sim, 1:160, 100 .* sim.i[1:160], lw = 2, color = :crimson, label = "investment")
savefig(plt_sim, joinpath(FIG, "fig_simulation.png"))
println("  → saved: fig_simulation.png")


# =============================================================================
# §5  IMPULSE RESPONSES TO A TFP SHOCK
# =============================================================================
# The dynamic signature of the model: a 1-std TFP innovation raises output on
# impact; investment jumps far more than output (it is the adjustment margin) and
# consumption rises smoothly; capital builds up slowly and propagates the shock
# long after TFP itself has decayed.

println("\n", "─"^64); println("  §5  IMPULSE RESPONSES"); println("─"^64)
ir = irf(lin, m.σ_ε; T = 40)
println("  peak responses to a 1-std (", round(100*m.σ_ε, digits = 1), "%) TFP shock:")
println("    output ", round(maximum(ir.y), digits = 2), "%   investment ", round(maximum(ir.i), digits = 2),
        "%   consumption ", round(maximum(ir.c), digits = 2), "%   hours ", round(maximum(ir.n), digits = 2), "%")

plt_irf = plot(layout = (2, 3), size = (960, 540), legend = false)
for (k, (lab, col)) in enumerate([(:y, :black), (:c, :seagreen), (:i, :crimson),
                                  (:n, :purple), (:k, :navy), (:z, :darkorange)])
    plot!(plt_irf, 0:39, getfield(ir, lab), lw = 2, color = col, subplot = k,
          title = uppercase(string(lab)), xlabel = "quarter", ylabel = "%")
    hline!(plt_irf, [0], ls = :dot, color = :gray, subplot = k)
end
savefig(plt_irf, joinpath(FIG, "fig_irf.png"))
println("  → saved: fig_irf.png")


# =============================================================================
# §6  THE ROLE OF LABOR — INELASTIC vs DIVISIBLE vs HANSEN
# =============================================================================
# Kydland–Prescott's competitive economy needs a volatile labor input to match
# the data. Inelastic labor produces almost no hours fluctuation; divisible
# labor adds some; Hansen's (1985) INDIVISIBLE labor — lotteries over employment
# — makes aggregate hours highly elastic and is the standard fix that brings the
# model's hours volatility close to the data.

println("\n", "─"^64); println("  §6  THE ROLE OF LABOR"); println("─"^64)
labs = [:inelastic, :divisible, :indivisible]
println("  ", rpad("labor spec", 14), rpad("σ_y(%)", 9), rpad("σ_n/σ_y", 9), rpad("σ_i/σ_y", 9), "corr(n,y)")
irs = Dict{Symbol,Any}()
for lab in labs
    ml = RBCModel(labor = lab); Ll = solve_linear(ml); bcl = business_cycle_stats(Ll, ml)
    irs[lab] = irf(Ll, ml.σ_ε; T = 40)
    n = bcl.rows[4]; i = bcl.rows[3]; y = bcl.rows[1]
    println("  ", rpad(string(lab), 14), rpad(round(y[2], digits = 2), 9),
            rpad(round(n[3], digits = 2), 9), rpad(round(i[3], digits = 2), 9), round(n[4], digits = 2))
end

plt_lab = plot(xlabel = "quarter", ylabel = "hours response %", title = "Hours Response to TFP Shock by Labor Spec", legend = :topright)
for (lab, col) in [(:inelastic, :gray), (:divisible, :seagreen), (:indivisible, :crimson)]
    plot!(plt_lab, 0:39, irs[lab].n, lw = 2, color = col, label = string(lab))
end
savefig(plt_lab, joinpath(FIG, "fig_labor_comparison.png"))
println("  → saved: fig_labor_comparison.png")


# =============================================================================
# §7  COMPARATIVE STATICS — TFP PERSISTENCE & VOLATILITY
# =============================================================================
# Persistence ρ governs propagation: more persistent shocks generate larger,
# longer-lived output fluctuations and a more sluggish capital response. The
# innovation size σ_ε scales volatility roughly one-for-one.

println("\n", "─"^64); println("  §7  COMPARATIVE STATICS"); println("─"^64)
ρ_vals = [0.80, 0.90, 0.95, 0.99]
println("  ", rpad("ρ", 8), rpad("σ_y(%)", 10), "capital persistence P[1,1]")
σy_by_ρ = Float64[]
for ρ in ρ_vals
    mr = RBCModel(ρ = ρ); Lr = solve_linear(mr); bcr = business_cycle_stats(Lr, mr)
    push!(σy_by_ρ, bcr.σy)
    println("  ", rpad(ρ, 8), rpad(round(bcr.σy, digits = 2), 10), round(Lr.P[1,1], digits = 4))
end

plt_cs = plot(layout = (1, 2), size = (840, 340))
plot!(plt_cs, ρ_vals, σy_by_ρ, lw = 2, marker = :circle, color = :navy, legend = false,
      xlabel = "TFP persistence ρ", ylabel = "σ(output) %", title = "Output Volatility vs ρ", subplot = 1)
for ρ in [0.5, 0.9, 0.99]
    mr = RBCModel(ρ = ρ); Lr = solve_linear(mr); ir_ρ = irf(Lr, mr.σ_ε; T = 40)
    plot!(plt_cs, 0:39, ir_ρ.y, lw = 2, label = "ρ = $ρ",
          xlabel = "quarter", ylabel = "output % ", title = "Output IRF vs ρ", subplot = 2, legend = :topright)
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
    ("fig_linear_vs_vfi.png",      "log-linear vs exact savings policy"),
    ("fig_simulation.png",         "simulated output/consumption/investment"),
    ("fig_irf.png",                "impulse responses to a TFP shock"),
    ("fig_labor_comparison.png",   "hours response by labor specification"),
    ("fig_comparative_statics.png","output volatility & IRF vs ρ"),
]
    println("  ", rpad(f, 30), " ", d)
end
println("═"^64)
