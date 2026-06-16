# =============================================================================
# analyze.jl — Rust (1987): diagnostics, estimation, counterfactuals, figures
# =============================================================================
#
# Run with:  julia analyze.jl   (includes main.jl, solves the baseline, then
# works through §1–§8 and writes figures to graphs/). A full pass is ~90 s
# (the Monte-Carlo estimation study in §7 dominates the time).

include("main.jl")

using Statistics
using Random
using Optim
using Plots

const FIG = joinpath(@__DIR__, "graphs")
isdir(FIG) || mkdir(FIG)

m   = RustModel()
sol = solve_rust(m)

println("\n", "═"^64)
println("  RUST (1987) ANALYSIS — RC = ", m.RC, ", θ₁ = ", m.θ₁)
println("═"^64)


# =============================================================================
# §1  THE MODEL SOLUTION
# =============================================================================
# The value function falls with mileage (more costs ahead); the choice-specific
# values cross at the reservation mileage; the replacement CCP is a smooth
# S-curve — the dynamic-logit analog of McCall's reservation wage.

println("\n", "─"^64); println("  §1  MODEL SOLUTION"); println("─"^64)
x̄ = m.x̄
res_mile = x̄[findfirst(sol.P .>= 0.5)]
π = stationary_mileage(sol.P, m.θ₃, m.S)
repl_rate = dot(π, sol.P)
println("  ", rpad("reservation mileage x* (P=0.5)", 34), " ", round(res_mile, digits = 1))
println("  ", rpad("long-run replacement rate", 34), " ", round(repl_rate, digits = 4))
println("  ", rpad("expected engine life (periods)", 34), " ", round(1 / repl_rate, digits = 1))

plt_v = plot(x̄, sol.V, lw = 2, color = :darkblue, label = "V(x)",
             xlabel = "mileage state x", ylabel = "value",
             title = "Value Function & Choice-Specific Values", legend = :bottomleft)
plot!(plt_v, x̄, sol.v_keep, lw = 2, ls = :dash, color = :seagreen, label = "v_keep(x)")
plot!(plt_v, x̄, sol.v_replace, lw = 2, ls = :dash, color = :crimson, label = "v_replace")
vline!(plt_v, [res_mile], lw = 1, ls = :dot, color = :gray, label = "x* = $(Int(res_mile))")
savefig(plt_v, joinpath(FIG, "fig_value_function.png"))

plt_ccp = plot(x̄, sol.P, lw = 2.5, color = :purple, marker = :circle, ms = 2,
               xlabel = "mileage state x", ylabel = "P(replace | x)",
               title = "Replacement Hazard (conditional choice probability)", legend = false)
hline!(plt_ccp, [0.5], ls = :dot, color = :gray)
vline!(plt_ccp, [res_mile], ls = :dot, color = :gray)
savefig(plt_ccp, joinpath(FIG, "fig_ccp.png"))
println("  → saved: fig_value_function.png, fig_ccp.png")


# =============================================================================
# §2  THE EXTREME-VALUE MACHINERY
# =============================================================================
# Everything tractable in the model rests on two facts about i.i.d. Type-I EV
# shocks: the expected max is log-sum-exp (drives the Bellman equation) and the
# choice probability is logit (drives the CCPs). Verify both by Monte Carlo.

println("\n", "─"^64); println("  §2  EXTREME-VALUE RESULTS (Monte Carlo)"); println("─"^64)

ev() = -log.(-log.(rand(200_000)))                 # Type-I EV draws via inverse CDF
Δgrid = range(-4, 4, length = 41)
emax_sim = Float64[]; emax_th = Float64[]
p_sim = Float64[];   p_th = Float64[]
for Δ in Δgrid
    ε0, ε1 = ev(), ev()
    push!(emax_sim, mean(max.(0 .+ ε0, Δ .+ ε1)))
    push!(emax_th,  log(exp(0) + exp(Δ)) + γ_euler)
    push!(p_sim, mean((Δ .+ ε1) .> (0 .+ ε0)))
    push!(p_th,  exp(Δ) / (1 + exp(Δ)))
end
println("  ", rpad("max |E[max] sim − theory|", 30), " ", round(maximum(abs.(emax_sim .- emax_th)), sigdigits = 2))
println("  ", rpad("max |P(d=1) sim − logit|", 30), " ", round(maximum(abs.(p_sim .- p_th)), sigdigits = 2))

plt_ev = plot(layout = (1, 2), size = (820, 340))
plot!(plt_ev, Δgrid, emax_th, lw = 2, color = :navy, label = "log-sum-exp + γ",
      xlabel = "v₁ − v₀", ylabel = "E[max]", title = "Result 1: Expected Maximum", subplot = 1)
scatter!(plt_ev, Δgrid, emax_sim, ms = 2.5, color = :orange, label = "Monte Carlo", subplot = 1)
plot!(plt_ev, Δgrid, p_th, lw = 2, color = :navy, label = "logit formula",
      xlabel = "v₁ − v₀", ylabel = "P(d=1)", title = "Result 2: Choice Probability", subplot = 2)
scatter!(plt_ev, Δgrid, p_sim, ms = 2.5, color = :orange, label = "Monte Carlo", subplot = 2)
savefig(plt_ev, joinpath(FIG, "fig_ev_results.png"))
println("  → saved: fig_ev_results.png")


# =============================================================================
# §3  COMPARATIVE STATICS
# =============================================================================
# A higher replacement cost RC pushes the hazard curve RIGHT (wait longer); a
# steeper maintenance slope θ₁ pushes it LEFT (replace sooner). Patience β raises
# the willingness to pay now to avoid future maintenance.

println("\n", "─"^64); println("  §3  COMPARATIVE STATICS"); println("─"^64)

res_at(; kw...) = (s = solve_rust(RustModel(; kw...)); i = findfirst(s.P .>= 0.5);
                   i === nothing ? NaN : x̄[i])

RC_vals = [4.0, 6.0, 8.0, 12.0, 16.0]
θ₁_vals = [0.02, 0.03, 0.04, 0.06, 0.08]
β_vals  = [0.90, 0.93, 0.95, 0.97, 0.99]

println("  reservation mileage x*:")
for RC in RC_vals; println("    RC = ", rpad(RC, 5), "  x* = ", res_at(RC = RC)); end
for θ in θ₁_vals;  println("    θ₁ = ", rpad(θ, 5), "  x* = ", res_at(θ₁ = θ)); end

plt_cs = plot(layout = (1, 2), size = (840, 340), legend = :topleft)
for RC in RC_vals
    s = solve_rust(RustModel(RC = RC))
    plot!(plt_cs, x̄, s.P, lw = 2, label = "RC = $(Int(RC))",
          xlabel = "mileage x", ylabel = "P(replace)", title = "Hazard vs Replacement Cost", subplot = 1)
end
for θ in θ₁_vals
    s = solve_rust(RustModel(θ₁ = θ))
    plot!(plt_cs, x̄, s.P, lw = 2, label = "θ₁ = $θ",
          xlabel = "mileage x", ylabel = "P(replace)", title = "Hazard vs Maintenance Slope", subplot = 2)
end
savefig(plt_cs, joinpath(FIG, "fig_comparative_statics.png"))
println("  → saved: fig_comparative_statics.png")


# =============================================================================
# §4  SIMULATION & THE STATIONARY MILEAGE DISTRIBUTION
# =============================================================================
# Under the optimal policy the mileage path is a sawtooth: a slow climb, then a
# sharp reset at replacement. The long-run histogram concentrates at low mileage
# because engines rarely survive to high mileage before being replaced.

println("\n", "─"^64); println("  §4  SIMULATION"); println("─"^64)
x_sim, d_sim = simulate_rust(sol.P, m.θ₃, m.S, 20_000; seed = 7)
println("  ", rpad("simulated replacement rate", 30), " ", round(mean(d_sim), digits = 4))
println("  ", rpad("stationary replacement rate", 30), " ", round(repl_rate, digits = 4))
println("  ", rpad("max mileage state reached", 30), " ", maximum(x_sim), " / ", m.S)

plt_sim = plot(layout = (1, 2), size = (880, 330))
plot!(plt_sim, x̄[x_sim[1:300]], lw = 1, color = :steelblue, legend = false,
      xlabel = "period", ylabel = "mileage", title = "Sawtooth Mileage Path (first 300)", subplot = 1)
histogram!(plt_sim, x̄[x_sim], bins = 0:2:m.S, normalize = true, color = :steelblue, alpha = 0.7,
           legend = false, xlabel = "mileage", ylabel = "share", title = "Stationary Mileage Distribution", subplot = 2)
savefig(plt_sim, joinpath(FIG, "fig_simulation.png"))
println("  → saved: fig_simulation.png")


# =============================================================================
# §5  STRUCTURAL ESTIMATION — NFXP
# =============================================================================
# Rust's nested fixed-point MLE. Generate a long panel from the true model, then
# recover (θ₃, RC, θ₁): θ₃ by counting increments (step 1), (RC, θ₁) by MLE with
# a full DP solve at every trial (step 2). Standard errors from the Hessian.

println("\n", "─"^64); println("  §5  NFXP ESTIMATION"); println("─"^64)
T_est = 50_000
x_e, d_e = simulate_rust(sol.P, m.θ₃, m.S, T_est; seed = 1)
θ₃_hat = estimate_theta3(x_e, d_e, m.S)
nfxp = estimate_nfxp(x_e, d_e, m.β, θ₃_hat, m.S)

println("  ", rpad("θ₃ true", 14), round.(m.θ₃, digits = 3))
println("  ", rpad("θ₃ est", 14), round.(θ₃_hat, digits = 3))
println("  ", rpad("Param", 8), rpad("true", 10), rpad("NFXP est", 14), "std error")
println("  ", rpad("RC", 8), rpad(m.RC, 10), rpad(round(nfxp.RC, digits = 3), 14), round(nfxp.se_RC, digits = 3))
println("  ", rpad("θ₁", 8), rpad(m.θ₁, 10), rpad(round(nfxp.θ₁, digits = 4), 14), round(nfxp.se_θ₁, digits = 4))


# =============================================================================
# §6  HOTZ–MILLER (CCP) ESTIMATION
# =============================================================================
# The two-step CCP estimator replaces the inner fixed point with a single linear
# solve (the logit value inversion), given first-stage CCPs. Faster, but less
# accurate here: the renewal structure means high-mileage states are essentially
# never visited, so their CCPs are not data-identified.

println("\n", "─"^64); println("  §6  HOTZ–MILLER (CCP) ESTIMATION"); println("─"^64)
hm = estimate_hotz_miller(x_e, d_e, m.β, θ₃_hat, m.S)
_, visits = fit_ccp_frequency(x_e, d_e, m.S)
println("  ", rpad("states ever visited", 30), " ", count(>(0), visits), " / ", m.S,
        "   (max = ", maximum(x_e), ")")
println("  ", rpad("Param", 8), rpad("true", 10), rpad("NFXP", 12), "Hotz–Miller")
println("  ", rpad("RC", 8), rpad(m.RC, 10), rpad(round(nfxp.RC, digits = 3), 12), round(hm.RC, digits = 3))
println("  ", rpad("θ₁", 8), rpad(m.θ₁, 10), rpad(round(nfxp.θ₁, digits = 4), 12), round(hm.θ₁, digits = 4))

sol_nfxp = solve_rust(RustModel(RC = nfxp.RC, θ₁ = nfxp.θ₁, θ₃ = θ₃_hat))
plt_est = plot(x̄, sol.P, lw = 3, color = :black, label = "true CCP",
               xlabel = "mileage x", ylabel = "P(replace)", title = "Estimated vs True CCP", legend = :topleft)
plot!(plt_est, x̄, sol_nfxp.P, lw = 2, ls = :dash, color = :crimson, label = "NFXP fit")
plot!(plt_est, x̄, hm.P_rep_hat, lw = 2, ls = :dot, color = :seagreen, label = "Hotz–Miller 1st-stage CCP")
savefig(plt_est, joinpath(FIG, "fig_estimation.png"))
println("  → saved: fig_estimation.png")


# =============================================================================
# §7  MONTE-CARLO SAMPLING DISTRIBUTION
# =============================================================================
# Re-estimate on many independent panels to trace the finite-sample sampling
# distribution of the NFXP estimates. The histograms should center on the truth,
# and their spread should match the analytic standard errors from §5.

println("\n", "─"^64); println("  §7  MONTE-CARLO SAMPLING DISTRIBUTION"); println("─"^64)
R, T_mc = 40, 8_000
RC_draws = Float64[]; θ₁_draws = Float64[]
for r in 1:R
    xr, dr = simulate_rust(sol.P, m.θ₃, m.S, T_mc; seed = 1000 + r)
    θ3r = estimate_theta3(xr, dr, m.S)
    obj = p -> nfxp_negloglik(p, xr, dr, m.β, θ3r, m.S)
    res = optimize(obj, [10.0, 0.03], NelderMead(), Optim.Options(iterations = 5_000))
    push!(RC_draws, Optim.minimizer(res)[1])
    push!(θ₁_draws, Optim.minimizer(res)[2])
end
println("  ", rpad("RC : mean (true 8.0)", 30), " ", round(mean(RC_draws), digits = 3),
        "   sd ", round(std(RC_draws), digits = 3))
println("  ", rpad("θ₁ : mean (true 0.04)", 30), " ", round(mean(θ₁_draws), digits = 4),
        "   sd ", round(std(θ₁_draws), digits = 4))

plt_mc = plot(layout = (1, 2), size = (840, 330))
histogram!(plt_mc, RC_draws, bins = 12, color = :steelblue, alpha = 0.75, legend = false,
           xlabel = "RC estimate", ylabel = "count", title = "Sampling Dist of RĈ", subplot = 1)
vline!(plt_mc, [m.RC], lw = 2, color = :crimson, subplot = 1)
histogram!(plt_mc, θ₁_draws, bins = 12, color = :seagreen, alpha = 0.75, legend = false,
           xlabel = "θ₁ estimate", ylabel = "count", title = "Sampling Dist of θ̂₁", subplot = 2)
vline!(plt_mc, [m.θ₁], lw = 2, color = :crimson, subplot = 2)
savefig(plt_mc, joinpath(FIG, "fig_monte_carlo.png"))
println("  → saved: fig_monte_carlo.png")


# =============================================================================
# §8  COUNTERFACTUAL — A REPLACEMENT SUBSIDY
# =============================================================================
# The payoff to structural estimation: counterfactuals. Lowering the replacement
# cost RC (e.g. a government subsidy on new engines) makes Zurcher replace sooner
# and more often. A reduced-form hazard fit to the old data could not predict
# this — agents re-optimize, which only the structural model captures.

println("\n", "─"^64); println("  §8  COUNTERFACTUAL — REPLACEMENT SUBSIDY"); println("─"^64)
RC_cf = range(3.0, 12.0, length = 19)
rates = Float64[]; lives = Float64[]
for RC in RC_cf
    s = solve_rust(RustModel(RC = RC))
    πr = stationary_mileage(s.P, m.θ₃, m.S)
    rr = dot(πr, s.P)
    push!(rates, rr); push!(lives, 1 / rr)
end
println("  ", rpad("RC", 8), rpad("repl. rate", 14), "engine life")
for (RC, rr, lf) in zip(RC_cf, rates, lives)
    (RC in (3.0, 6.0, 8.0, 12.0)) && println("  ", rpad(RC, 8), rpad(round(rr, digits = 4), 14), round(lf, digits = 1))
end

plt_cf = plot(RC_cf, 100 .* rates, lw = 2, marker = :circle, color = :darkorange, legend = false,
              xlabel = "replacement cost RC (counterfactual)", ylabel = "long-run replacement rate (%)",
              title = "Counterfactual: Subsidizing Engine Replacement")
vline!(plt_cf, [m.RC], ls = :dot, color = :gray)
savefig(plt_cf, joinpath(FIG, "fig_counterfactual.png"))
println("  → saved: fig_counterfactual.png")


# =============================================================================
# SUMMARY
# =============================================================================
println("\n", "═"^64); println("  ALL FIGURES SAVED → graphs/"); println("═"^64)
for (f, d) in [
    ("fig_value_function.png",     "V(x) and choice-specific values"),
    ("fig_ccp.png",                "replacement hazard (CCP S-curve)"),
    ("fig_ev_results.png",         "log-sum-exp & logit Monte-Carlo checks"),
    ("fig_comparative_statics.png","hazard vs RC and θ₁"),
    ("fig_simulation.png",         "sawtooth path & stationary mileage dist"),
    ("fig_estimation.png",         "estimated vs true CCP (NFXP, Hotz–Miller)"),
    ("fig_monte_carlo.png",        "sampling distribution of RĈ, θ̂₁"),
    ("fig_counterfactual.png",     "replacement rate vs a cost subsidy"),
]
    println("  ", rpad(f, 30), " ", d)
end
println("═"^64)
