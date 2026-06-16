# =============================================================================
# analyze.jl — McCall Search Model: diagnostics, comparative statics, figures
# =============================================================================
#
# Run with:  julia analyze.jl   (it includes main.jl, solves the baseline, then
# works through §1–§6 below, saving figures into graphs/). A full pass is ~20 s.

include("main.jl")

using Statistics
using Plots

const FIG = joinpath(@__DIR__, "graphs")
isdir(FIG) || mkdir(FIG)

p   = McCallModel()
sol = solve_mccall(p)

println("\n", "═"^64)
println("  McCALL SEARCH ANALYSIS — w̄ = ", round(sol.w_res, digits = 2),
        "   f = ", round(sol.f, digits = 3))
println("═"^64)

# Convenience: headline stats for a parameter override, used by the sweeps.
function stats(; kwargs...)
    pp = McCallModel(; kwargs...)
    s  = solve_mccall(pp)
    return (; w_res = s.w_res, f = s.f, U = s.U, u_rate = s.u_rate)
end


# =============================================================================
# §1  VERIFICATION
# =============================================================================
# Two independent solvers must agree, and the reservation-wage closed form
# u(w̄) = (1−β)·U must hold to machine precision.

println("\n", "─"^64); println("  §1  VERIFICATION"); println("─"^64)

sv = solve_vfi(p)
res_closed = flow_u(sol.w_res, p.γ)            # u(w̄)
res_target = (1 - p.β) * sol.U                 # (1−β)·U
println("  ", rpad("|U_scalar − U_vfi|", 30), " ", round(abs(sol.U - sv.U), sigdigits = 3))
println("  ", rpad("|w̄_scalar − w̄_vfi|", 30), " ", round(abs(sol.w_res - sv.w_res), sigdigits = 3))
println("  ", rpad("|u(w̄) − (1−β)U|", 30), " ", round(abs(res_closed - res_target), sigdigits = 3))
println("  ", rpad("Σ pw", 30), " ", round(sum(sol.pw), digits = 10))


# =============================================================================
# §2  VALUE FUNCTION & RESERVATION WAGE
# =============================================================================
# V_e(w) is increasing in w and crosses the flat unemployment value U exactly at
# the reservation wage w̄ — the wage at which the worker is indifferent.

println("\n", "─"^64); println("  §2  VALUE FUNCTION & RESERVATION WAGE"); println("─"^64)
println("  ", rpad("reservation wage w̄", 30), " ", round(sol.w_res, digits = 3))
println("  ", rpad("mean offer  E[w]", 30), " ", round(dot(sol.pw, sol.w), digits = 3))
println("  ", rpad("P(accept random offer)", 30), " ", round(sol.f, digits = 4))

plt_v = plot(sol.w, sol.V_e, lw = 2, color = :darkblue, label = "V_e(w)  (employed)",
             xlabel = "wage offer w", ylabel = "value",
             title = "Value Function & Reservation Wage", legend = :topleft)
hline!(plt_v, [sol.U], lw = 2, ls = :dash, color = :darkorange, label = "U  (unemployed)")
vline!(plt_v, [sol.w_res], lw = 1.5, ls = :dot, color = :gray,
       label = "w̄ = $(round(sol.w_res, digits = 1))")
savefig(plt_v, joinpath(FIG, "fig_value_function.png"))
println("  → saved: fig_value_function.png")


# =============================================================================
# §3  COMPARATIVE STATICS
# =============================================================================
# Vary one parameter at a time. The reservation wage w̄ rises whenever search
# becomes more attractive (better outside option, more patience, lower layoff
# risk); the job-finding rate f moves inversely.

println("\n", "─"^64); println("  §3  COMPARATIVE STATICS"); println("─"^64)

function sweep(header, sym, vals)
    println("\n  ", header)
    println("  ", rpad("value", 8), " ", lpad("w̄", 9), " ", lpad("f", 9), " ",
            lpad("1/f", 9), " ", lpad("u*(%)", 9))
    println("  ", "─"^48)
    out = NamedTuple[]
    for v in vals
        s = stats(; (sym => v,)...)
        push!(out, s)
        println("  ", rpad(round(Float64(v), sigdigits = 3), 8), " ",
                lpad(round(s.w_res, digits = 3), 9), " ",
                lpad(round(s.f, digits = 4), 9), " ",
                lpad(round(1 / s.f, digits = 2), 9), " ",
                lpad(round(100 * s.u_rate, digits = 2), 9))
    end
    return out
end

c_vals = [10.0, 20.0, 25.0, 30.0, 40.0]
β_vals = [0.90, 0.94, 0.96, 0.98, 0.99]
α_vals = [0.0, 0.02, 0.05, 0.10, 0.20]
γ_vals = [0.0, 1.0, 2.0, 4.0, 6.0]

c_st = sweep("c  (UI / outside option) — higher c ⇒ pickier ⇒ higher w̄, lower f", :c, c_vals)
β_st = sweep("β  (patience) — more patient ⇒ wait for better offers ⇒ higher w̄", :β, β_vals)
α_st = sweep("α  (layoff rate) — riskier jobs ⇒ less picky ⇒ lower w̄", :α, α_vals)
γ_st = sweep("γ  (risk aversion) — more risk-averse ⇒ grab a job sooner ⇒ lower w̄", :γ, γ_vals)

wres(st) = [s.w_res for s in st]
opts = (ylabel = "reservation wage w̄", marker = :circle, lw = 2, legend = false)
plt_cs = plot(layout = (2, 2), size = (880, 660))
plot!(plt_cs, c_vals, wres(c_st); xlabel = "c (UI)",            title = "w̄ vs c", subplot = 1, opts...)
plot!(plt_cs, β_vals, wres(β_st); xlabel = "β (patience)",      title = "w̄ vs β", subplot = 2, opts...)
plot!(plt_cs, α_vals, wres(α_st); xlabel = "α (layoff rate)",   title = "w̄ vs α", subplot = 3, opts...)
plot!(plt_cs, γ_vals, wres(γ_st); xlabel = "γ (risk aversion)", title = "w̄ vs γ", subplot = 4, opts...)
savefig(plt_cs, joinpath(FIG, "fig_comparative_statics.png"))
println("\n  → saved: fig_comparative_statics.png")


# =============================================================================
# §4  HAZARD, DURATION & STEADY-STATE UNEMPLOYMENT
# =============================================================================
# The reservation-wage rule makes unemployment a geometric waiting time with
# success probability f, so the closed-form mean duration is 1/f and the
# steady-state unemployment rate is α/(α+f). A Monte-Carlo panel confirms both.

println("\n", "─"^64); println("  §4  HAZARD, DURATION & STEADY-STATE UNEMPLOYMENT"); println("─"^64)

sim = simulate_panel(p, sol; T = 4_000, N = 1_500, seed = 1)
println("  ", rpad("job-finding rate f", 30), " ", round(sol.f, digits = 4))
println("  ", rpad("mean duration (closed form 1/f)", 32), " ", round(1 / sol.f, digits = 3))
println("  ", rpad("mean duration (simulated)", 32), " ", round(sim.mean_dur, digits = 3))
println("  ", rpad("u* (closed form α/(α+f))", 32), " ", round(100 * sol.u_rate, digits = 3), "%")
println("  ", rpad("u* (simulated)", 32), " ", round(100 * sim.u_rate, digits = 3), "%")
println("  ", rpad("mean accepted wage", 32), " ", round(sim.mean_wage, digits = 3))

# Simulated duration distribution vs. the geometric prediction
maxd  = quantile(sim.durations, 0.99)
edges = 1:ceil(Int, maxd)
counts = [count(==(d), sim.durations) for d in edges] ./ length(sim.durations)
geom   = [(1 - sol.f)^(d - 1) * sol.f for d in edges]
plt_dur = bar(edges, counts, alpha = 0.6, color = :steelblue, label = "simulated",
              xlabel = "unemployment-spell length", ylabel = "share of spells",
              title = "Unemployment Duration: simulated vs. geometric(f)")
plot!(plt_dur, edges, geom, lw = 2, color = :crimson, marker = :circle, label = "geometric(f)")
savefig(plt_dur, joinpath(FIG, "fig_duration_hist.png"))

# Offer distribution vs. realized accepted-wage distribution (truncation at w̄)
plt_w = plot(sol.w, sol.pw ./ maximum(sol.pw), lw = 2, color = :gray, fill = (0, 0.15, :gray),
             label = "offer dist (all w)", xlabel = "wage", ylabel = "density (scaled)",
             title = "Offers vs. Accepted Wages")
ahist = [count(wv -> sol.w[i] <= wv < get(sol.w, i + 1, Inf), sim.accepted_wages)
         for i in eachindex(sol.w)]
plot!(plt_w, sol.w, ahist ./ maximum(ahist), lw = 2, color = :darkgreen,
      label = "accepted wages (w ≥ w̄)")
vline!(plt_w, [sol.w_res], ls = :dot, color = :black, label = "w̄")
savefig(plt_w, joinpath(FIG, "fig_wage_dist.png"))
println("  → saved: fig_duration_hist.png, fig_wage_dist.png")


# =============================================================================
# §5  MEAN-PRESERVING SPREAD — THE OPTION VALUE OF SEARCH
# =============================================================================
# Hold the *mean* offer fixed and raise its dispersion. For McCall's original
# RISK-NEUTRAL worker (γ = 0) the value of search is convex in the wage (the max
# operator throws away the downside), so a mean-preserving spread RAISES both the
# value of search and the reservation wage — the pure option value of waiting.
# Add risk aversion (γ > 0) and a second force pushes the other way: a risk-averse
# worker dislikes dispersion, so at high σ w̄ can turn back down. We report both.

println("\n", "─"^64); println("  §5  MEAN-PRESERVING SPREAD — OPTION VALUE"); println("─"^64)

# E[w] = exp(μ + σ²/2). To raise σ at fixed mean, lower μ by σ²/2 accordingly.
mean_logwage = p.μ + p.σ^2 / 2
# Kept to σ ≤ 0.6: beyond that the log-normal's right tail runs past the ±mσ grid
# and the *discretized* mean drifts below target, contaminating the experiment.
σ_vals = [0.20, 0.30, 0.40, 0.50, 0.60]

function mps_sweep(γv)
    out = NamedTuple[]
    for σv in σ_vals
        μv = mean_logwage - σv^2 / 2             # keep E[w] constant
        s  = stats(; μ = μv, σ = σv, γ = γv, m = 6.0)   # wider grid for fat tails
        push!(out, (; σ = σv, Ew = exp(μv + σv^2 / 2), s.w_res, s.U, s.f))
    end
    return out
end

mps_rn = mps_sweep(0.0)    # risk-neutral: clean monotone option value
mps_ra = mps_sweep(p.γ)    # risk-averse baseline: option value vs. risk distaste

println("  risk-neutral worker (γ = 0): option value ⇒ w̄ rises monotonically")
println("  ", rpad("σ", 8), " ", lpad("E[w]", 9), " ", lpad("w̄", 9), " ", lpad("U", 11))
println("  ", "─"^40)
for m in mps_rn
    println("  ", rpad(m.σ, 8), " ", lpad(round(m.Ew, digits = 2), 9), " ",
            lpad(round(m.w_res, digits = 2), 9), " ", lpad(round(m.U, digits = 2), 11))
end
println("\n  risk-averse worker (γ = $(p.γ)): w̄ turns down once risk distaste dominates")
for m in mps_ra
    println("  ", rpad(m.σ, 8), " ", lpad(round(m.Ew, digits = 2), 9), " ",
            lpad(round(m.w_res, digits = 2), 9), " ", lpad(round(m.U, digits = 3), 11))
end

plt_mps = plot([m.σ for m in mps_rn], [m.w_res for m in mps_rn], lw = 2, marker = :circle,
               color = :purple, label = "risk-neutral (γ=0)",
               xlabel = "offer dispersion σ (mean held fixed)",
               ylabel = "reservation wage w̄", legend = :topleft,
               title = "Mean-Preserving Spread & the Option Value of Search")
plot!(plt_mps, [m.σ for m in mps_ra], [m.w_res for m in mps_ra], lw = 2, marker = :diamond,
      color = :darkorange, label = "risk-averse (γ=$(p.γ))")
savefig(plt_mps, joinpath(FIG, "fig_mps.png"))
println("  → saved: fig_mps.png")


# =============================================================================
# §6  PERSISTENT (MARKOV) WAGE OFFERS
# =============================================================================
# Drop the i.i.d. assumption: the log offer follows an AR(1). The reservation
# wage is no longer a single number — it is a threshold whose location depends on
# how persistent offers are. Higher persistence ρ means a bad draw today predicts
# bad draws tomorrow, weakening the incentive to hold out.

println("\n", "─"^64); println("  §6  PERSISTENT (MARKOV) WAGE OFFERS"); println("─"^64)

ρ_vals = [0.0, 0.5, 0.9, 0.95]
plt_cor = plot(xlabel = "current offer w", ylabel = "accept (1) / reject (0)",
               title = "Acceptance Region under Persistent Offers", legend = :right)
for ρv in ρ_vals
    sc = solve_correlated(McCallModel(ρ = ρv))
    plot!(plt_cor, sc.w, Float64.(sc.accept), lw = 2,
          label = "ρ = $ρv  (w̄ = $(round(sc.w_res, digits = 1)))")
    println("  ", rpad("ρ = $ρv", 12), "  w̄ = ", rpad(round(sc.w_res, digits = 2), 8),
            "  stationary f = ", round(sc.f, digits = 4))
end
savefig(plt_cor, joinpath(FIG, "fig_correlated.png"))
println("  → saved: fig_correlated.png")


# =============================================================================
# SUMMARY
# =============================================================================
println("\n", "═"^64); println("  ALL FIGURES SAVED → graphs/"); println("═"^64)
for (f, d) in [
    ("fig_value_function.png",     "V_e(w), U, and the reservation wage"),
    ("fig_comparative_statics.png","w̄ vs c, β, α, γ"),
    ("fig_duration_hist.png",      "unemployment duration: simulated vs geometric"),
    ("fig_wage_dist.png",          "offer dist vs accepted-wage dist"),
    ("fig_mps.png",                "mean-preserving spread raises w̄"),
    ("fig_correlated.png",         "acceptance region under persistent offers"),
]
    println("  ", rpad(f, 30), " ", d)
end
println("═"^64)
