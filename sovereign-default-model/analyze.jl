# =============================================================================
# analyze.jl — Sovereign default: value functions, the price schedule, default
#              episodes, the Arellano scorecard, and comparative statics
# =============================================================================
#
# Run with:  julia analyze.jl   (includes main.jl, solves the baseline, then
# works through §1–§6 and writes figures to graphs/). A full pass is ~60 s; the
# comparative statics in §6 re-solve the model several times.

include("main.jl")

using Statistics
using Plots

const FIG = joinpath(@__DIR__, "graphs")
isdir(FIG) || mkdir(FIG)

m   = setup!(SovDefaultModel())
sol = solve_sovereign(m; verbose = false)
Ey  = sum(stationary(m.P_y) .* m.y_grid)

println("\n", "═"^64)
println("  SOVEREIGN DEFAULT ANALYSIS (Arellano 2008) — quarterly Argentina")
println("═"^64)

iy_lo, iy_mid, iy_hi = 4, (m.N_y + 1) ÷ 2, m.N_y - 3
ycols = [iy_lo, iy_mid, iy_hi]
ylab(iy) = "y = $(round(m.y_grid[iy], digits = 3))"


# =============================================================================
# §1  VALUE FUNCTIONS & THE DEFAULT DECISION
# =============================================================================
# The government defaults when the value of repaying its debt falls below the
# value of default. Because V^R rises with bond holdings b (less debt is better)
# while V^D is flat in b, there is a debt threshold for each income level — and
# the threshold is tighter (default comes sooner) when income is low.

println("\n", "─"^64); println("  §1  VALUE FUNCTIONS & DEFAULT DECISION"); println("─"^64)
# default threshold: the most debt still repaid is the smallest b (largest debt)
# at which repayment beats default. V_R rises with b, so this is the first
# crossing of V_R above V_D as we move from heavy debt toward zero.
thr = [m.b_grid[something(findfirst(i -> sol.VR[i, iy] >= sol.VD[iy], 1:m.N_b), m.N_b)] for iy in 1:m.N_y]
println("  max sustainable debt (−b) by income:")
for iy in ycols
    println("    ", rpad(ylab(iy), 14), "→ default if debt exceeds ", round(-thr[iy], digits = 3))
end

plt_v = plot(xlabel = "bond holdings b  (b < 0 is debt)", ylabel = "value",
             title = "Repayment vs Default Value", legend = :bottomright)
for iy in ycols
    plot!(plt_v, m.b_grid, sol.VR[:, iy], lw = 2, label = "V_R, " * ylab(iy))
    hline!(plt_v, [sol.VD[iy]], lw = 1.5, ls = :dash, label = "V_D, " * ylab(iy))
end
savefig(plt_v, joinpath(FIG, "fig_value_functions.png"))

# default set heatmap over (b, y)
defset = [sol.VD[iy] > sol.VR[ib, iy] ? 1.0 : 0.0 for iy in 1:m.N_y, ib in 1:m.N_b]
plt_ds = heatmap(m.b_grid, m.y_grid ./ Ey, defset, color = :viridis,
                 xlabel = "bond holdings b", ylabel = "income y / E[y]",
                 title = "Default Set (yellow = default)", colorbar = false)
savefig(plt_ds, joinpath(FIG, "fig_default_set.png"))
println("  → saved: fig_value_functions.png, fig_default_set.png")


# =============================================================================
# §2  THE BOND PRICE SCHEDULE & SPREADS
# =============================================================================
# Lenders price default risk: q(b',y) falls as the government borrows more, and
# is higher when income is high (default less likely). The schedule acts as an
# endogenous debt limit — borrow enough and the price collapses, so the revenue
# from extra borrowing q·b' actually falls.

println("\n", "─"^64); println("  §2  BOND PRICE SCHEDULE & SPREADS"); println("─"^64)
qrf = 1 / (1 + m.r)
plt_q = plot(xlabel = "new debt b'", ylabel = "bond price q(b', y)",
             title = "Bond Price Schedule", legend = :topleft)
for iy in ycols
    plot!(plt_q, m.b_grid, sol.q[:, iy], lw = 2, label = ylab(iy))
end
hline!(plt_q, [qrf], ls = :dash, color = :gray, label = "risk-free 1/(1+r)")
savefig(plt_q, joinpath(FIG, "fig_bond_price.png"))

plt_sp = plot(layout = (2, 1), size = (720, 560))
for iy in ycols
    spread = clamp.(100 .* ((1 ./ max.(sol.q[:, iy], 1e-8)).^4 .- (1 + m.r)^4), 0, 100)
    plot!(plt_sp, m.b_grid, spread, lw = 2, label = ylab(iy), subplot = 1,
          xlabel = "new debt b'", ylabel = "spread (ann. %)", title = "Sovereign Spread", legend = :topright)
end
for iy in ycols
    plot!(plt_sp, m.b_grid, sol.def_prob[:, iy], lw = 2, label = ylab(iy), subplot = 2,
          xlabel = "bond holdings b", ylabel = "P(default)", title = "Default Probability", legend = :topleft)
end
savefig(plt_sp, joinpath(FIG, "fig_spreads.png"))
println("  → saved: fig_bond_price.png, fig_spreads.png")


# =============================================================================
# §3  BORROWING POLICY
# =============================================================================
# Below the 45° line the government accumulates debt; above it, it deleverages or
# saves. Higher income supports more borrowing (the price schedule is looser).

println("\n", "─"^64); println("  §3  BORROWING POLICY"); println("─"^64)
plt_bp = plot(xlabel = "current bond holdings b", ylabel = "next-period b'",
              title = "Borrowing Policy b'(b, y)", legend = :topleft)
for iy in ycols
    plot!(plt_bp, m.b_grid, sol.bp[:, iy], lw = 2, label = ylab(iy))
end
plot!(plt_bp, m.b_grid, m.b_grid, ls = :dash, color = :black, label = "45°")
savefig(plt_bp, joinpath(FIG, "fig_borrowing_policy.png"))
println("  → saved: fig_borrowing_policy.png")


# =============================================================================
# §4  A DEFAULT EPISODE  (event study)
# =============================================================================
# Average the dynamics in a window around default events. Spreads climb and debt
# peaks in the run-up; at default, consumption drops (the output cost) and the
# country is excluded; afterwards it re-enters deleveraged.

println("\n", "─"^64); println("  §4  DEFAULT EPISODE (event study)"); println("─"^64)
sim = simulate_sovereign(m, sol, 60_000; seed = 11)
W = 12
events = [t for t in W+1:length(sim.default)-W if sim.default[t] == 1 &&
          all(sim.default[t-W:t-1] .== 0)]                # isolated default onsets
println("  isolated default events found: ", length(events))
avg(f) = [mean(f[e+k] for e in events) for k in -W:W]
y_e  = avg(sim.y); c_e = avg(sim.c); b_e = avg(sim.b)
sp_e = [mean(filter(!isnan, [sim.spread[e+k] for e in events])) for k in -W:W]

plt_ev = plot(layout = (2, 2), size = (880, 600), legend = false)
plot!(plt_ev, -W:W, 100 .* (y_e ./ Ey .- 1), lw = 2, color = :darkgreen, subplot = 1,
      title = "Income (% from mean)", xlabel = "quarters around default")
plot!(plt_ev, -W:W, 100 .* (c_e ./ Ey .- 1), lw = 2, color = :purple, subplot = 2,
      title = "Consumption (% from mean)", xlabel = "quarters around default")
plot!(plt_ev, -W:W, -b_e, lw = 2, color = :steelblue, subplot = 3,
      title = "Debt (−b)", xlabel = "quarters around default")
plot!(plt_ev, -W:W, sp_e, lw = 2, color = :crimson, subplot = 4,
      title = "Spread (ann. %)", xlabel = "quarters around default")
vline!(plt_ev, [0], ls = :dot, color = :gray, subplot = 1); vline!(plt_ev, [0], ls = :dot, color = :gray, subplot = 2)
vline!(plt_ev, [0], ls = :dot, color = :gray, subplot = 3); vline!(plt_ev, [0], ls = :dot, color = :gray, subplot = 4)
savefig(plt_ev, joinpath(FIG, "fig_default_episode.png"))
println("  → saved: fig_default_episode.png")


# =============================================================================
# §5  THE BUSINESS-CYCLE SCORECARD vs ARGENTINA
# =============================================================================
# Confront the model with the targeted and untargeted moments of Argentine data
# (Arellano 2008, Table 4). The signature successes: countercyclical spreads and
# trade balance, and consumption MORE volatile than output — the opposite of a
# frictionless economy, generated by default risk and exclusion.

println("\n", "─"^64); println("  §5  BUSINESS-CYCLE SCORECARD vs ARGENTINA"); println("─"^64)
st = default_statistics(m, sim)
function row(name, mod, dat)
    println("  ", rpad(name, 24), rpad(round(mod, digits = 2), 10), dat)
end
println("  ", rpad("moment", 24), rpad("model", 10), "Argentina (Arellano 2008)")
row("default rate (ann. %)",   st.def_rate,    "≈ 3.0")
row("mean debt / output (%)",  st.mean_debt,   "≈ 6.0")
row("mean spread (ann. %)",    st.mean_spread, "≈ 3.6")
row("std spread (%)",          st.std_spread,  "≈ 6.4")
row("σ(c)/σ(y)",               st.rel_c,       "≈ 1.1")
row("corr(spread, y)",         st.corr_sy,     "≈ −0.7")
row("corr(trade bal, y)",      st.corr_tby,    "≈ −0.25")


# =============================================================================
# §6  COMPARATIVE STATICS
# =============================================================================
# Patience and the output cost discipline default. A more patient government
# (higher β) values future market access more and defaults less; a larger output
# cost makes default more painful and also reduces it.

println("\n", "─"^64); println("  §6  COMPARATIVE STATICS"); println("─"^64)

function quick_stats(; kwargs...)
    mm = setup!(SovDefaultModel(; N_b = 151, N_b_fine = 151, kwargs...))
    ss = solve_sovereign(mm; verbose = false)
    sm = simulate_sovereign(mm, ss, 20_000; seed = 5)
    default_statistics(mm, sm)
end

println("  β  (patience):")
β_vals = [0.94, 0.95, 0.953, 0.96]
β_def = Float64[]; β_sp = Float64[]
for β in β_vals
    s = quick_stats(β = β); push!(β_def, s.def_rate); push!(β_sp, s.mean_spread)
    println("    β = ", rpad(β, 7), "def = ", rpad(round(s.def_rate, digits = 2), 7), "%  spread = ", round(s.mean_spread, digits = 2), "%")
end

println("  λ  (re-entry prob — higher λ ⇒ shorter exclusion ⇒ default cheaper ⇒ more default):")
λ_vals = [0.10, 0.20, 0.282, 0.40]
λ_def = Float64[]; λ_debt = Float64[]
for λ in λ_vals
    s = quick_stats(λ = λ); push!(λ_def, s.def_rate); push!(λ_debt, s.mean_debt)
    println("    λ = ", rpad(λ, 7), "def = ", rpad(round(s.def_rate, digits = 2), 7),
            "%  debt/y = ", round(s.mean_debt, digits = 2), "%")
end

plt_cs = plot(layout = (1, 2), size = (840, 340))
plot!(plt_cs, β_vals, β_def, lw = 2, marker = :circle, color = :navy, legend = false,
      xlabel = "discount factor β", ylabel = "default rate (ann. %)", title = "Default vs Patience", subplot = 1)
plot!(plt_cs, λ_vals, λ_def, lw = 2, marker = :diamond, color = :crimson, legend = false,
      xlabel = "re-entry probability λ", ylabel = "default rate (ann. %)", title = "Default vs Exclusion Length", subplot = 2)
savefig(plt_cs, joinpath(FIG, "fig_comparative_statics.png"))
println("  → saved: fig_comparative_statics.png")


# =============================================================================
# SUMMARY
# =============================================================================
println("\n", "═"^64); println("  ALL FIGURES SAVED → graphs/"); println("═"^64)
for (f, d) in [
    ("fig_value_functions.png",   "repayment vs default value functions"),
    ("fig_default_set.png",       "default region in (b, y) space"),
    ("fig_bond_price.png",        "endogenous bond price schedule q(b',y)"),
    ("fig_spreads.png",           "sovereign spreads & default probability"),
    ("fig_borrowing_policy.png",  "borrowing policy b'(b,y)"),
    ("fig_default_episode.png",   "event study around default events"),
    ("fig_comparative_statics.png","default rate vs β and output cost"),
]
    println("  ", rpad(f, 30), " ", d)
end
println("═"^64)
