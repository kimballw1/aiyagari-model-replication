# =============================================================================
# analyze.jl — Aiyagari Model Analysis
# =============================================================================

include("main.jl")

using Statistics
using Plots

# figures go to graphs/ next to this file, same as the other model folders
const FIG = joinpath(@__DIR__, "graphs")
isdir(FIG) || mkdir(FIG)

# -----------------------------------------------------------------------------
# Inequality helpers: Lorenz curve and Gini coefficent 
# -----------------------------------------------------------------------------

"""
    lorenz_pts(vals, wts) -> (F, L)

Compute the Lorenz curve of a discrete distribution where outcome `vals[i]`
(e.g. a wealth level) carries probability mass `wts[i]`.

Points are sorted by value, so `(F[k], L[k])` reads as "the poorest `F[k]`
fraction of the population holds `L[k]` of total wealth". Both series are
prefixed with the origin `(0, 0)` so the curve can be plotted directly.

# Arguments
- `vals`: outcome levels (need not be sorted; may be negative).
- `wts`: nonnegative masses; normalized to sum to 1.

# Returns
- `F`: cumulative population share, length `length(vals) + 1`, ending at 1.
- `L`: cumulative share of the total `∑ vals·wts`, ending at 1.
"""
function lorenz_pts(vals, wts)
    ord = sortperm(collect(vals)) #returns a vector of indicies that represent the sorted order
    v   = collect(vals)[ord] #
    w   = collect(wts)[ord] ./ sum(wts)     # weights normalized to sum to 1
    F = [0.0; cumsum(w)]                     # cumulative population share
    L = [0.0; cumsum(v .* w) ./ dot(v, w)]   # cumulative wealth share
    return F, L
end

"""
    gini(vals, wts) -> Float64

Gini coefficient of the discrete distribution `(vals, wts)`, computed as
`1 − 2·A`, where `A` is the area under the Lorenz curve

Returns a value in `[0, 1]`: `0` is perfect equality (Lorenz curve on the 45°
line) and values approaching `1` mean wealth concentrated in a vanishing share
of the population.

Returns `NaN` if any value is negative. The Lorenz/Gini construction assumes
non-negative outcomes; under the natural borrowing limit a large share of
households hold negative wealth, which sends the Lorenz curve below zero and
makes the coefficient leave `[0, 1]` (it can exceed 1). The statistic is not
meaningful there, so we decline to report it rather than print a misleading number.
"""
function gini(vals, wts)
    any(<(0), vals) && return NaN
    F, L = lorenz_pts(vals, wts)
    area = sum((F[k] - F[k-1]) * (L[k] + L[k-1]) / 2 for k in 2:lastindex(F))
    return 1.0 - 2.0 * area
end

# -----------------------------------------------------------------------------
# Shared objects from the baseline run
# -----------------------------------------------------------------------------
(; n_a, n_e, β, y) = p
z_grid, P_mat = rouwenhorst(p) # income grid + transition matrix
π_inc = stationary_distributions(MarkovChain(P_mat))[1] # stationary income probabilities

println("\n", "═"^64)
println("  AIYAGARI ANALYSIS — r* = ", round(result.r * 100, digits=3), "%   K* = ", round(result.K, digits=3))
println("═"^64)


# =============================================================================
# §1  BASELINE VERIFICATION
# =============================================================================
# Confirm the equilibrium satisfies the two key conditions:
#   (1) Capital market clears: aggregate household savings = firm capital demand
#   (2) The stationary distribution is a valid probability measure (sums to 1)
# Also reports the precautionary wedge: how far r* sits below the
# representative-agent rate 1/β−1 due to incomplete markets.

println("\n", "─"^64)
println("  §1  BASELINE VERIFICATION")
println("─"^64)

# Recompute K_supply from policy + distribution to verify market clearing
K_s = sum(result.a_grid[result.g_idx[i, j]] * result.μ[i, j] for i in 1:n_a, j in 1:n_e)

println("  ", rpad("|K_supply − K_demand|", 26), " ", round(abs(K_s - result.K), sigdigits=3))
println("  ", rpad("Σ μ[i,j]", 26), " ", round(sum(result.μ), digits=8))
println("  ", rpad("r*  (Aiyagari)", 26), " ", round(result.r, digits=4))
println("  ", rpad("r_RA = 1/β−1", 26), " ", round(1/β - 1, digits=4))
println("  ", rpad("Precautionary wedge", 26), " ", round((1/β - 1 - result.r) * 100, digits=3), " pp")


# =============================================================================
# §2  INCOME PROCESS — MARKOV CHAIN ANALYSIS
# =============================================================================
# We look at:
#   Eigenvalues of P  |λ₂| controls the speed of mixing
#   Spectral gap      1 − |λ₂|: how quickly the chain forgets its past
#   TV(total variation)-mixing time    upper bound on steps to reach ε-close to stationarity

println("\n", "─"^64)
println("  §2  INCOME PROCESS — MARKOV CHAIN")
println("─"^64)

# Eigenvalues sorted by |λ| descending. λ₁ = 1 always (stochastic matrix);
# λ₂ governs the mixing rate.
λ_sort = sort(eigvals(P_mat), by=abs, rev=true)
sgap = 1.0 - abs(λ_sort[2])

# TV-mixing bound
t_mix = ceil(Int, log(n_e / 0.02) / (-log(abs(λ_sort[2])))) 

println("  ", rpad("Eigenvalues of P", 26), " ", round.(real.(λ_sort), digits=5))
println("  ", rpad("Spectral gap (1 − |λ₂|)", 26), " ", round(sgap, digits=5))
println("  ", rpad("TV-mixing time (ε=0.01)", 26), " ≤ $t_mix steps")

# Stationary income summary
var_e = dot(π_inc, (result.e_grid .- dot(π_inc, result.e_grid)).^2)
println()
println("  ", rpad("Stationary income std σ_e", 26), " ", round(sqrt(var_e), digits=4))
println("  ", rpad("Income Gini G_e", 26), " ", round(gini(result.e_grid, π_inc), digits=4))

# =============================================================================
# §3  STATIONARY WEALTH DISTRIBUTION & INEQUALITY
# =============================================================================
# Characterize the cross-sectional wealth distribution.
#   Mean / median / std   basic moments of the marginal wealth distribution
#   Gini coefficient      0 = perfect equality, 1 = perfect inequality
#   Top shares            wealth held by the top 10% and top 1%
#   Lorenz curve          visual representation of inequality

println("\n", "─"^64)
println("  §3  WEALTH DISTRIBUTION & INEQUALITY")
println("─"^64)

# Marginal asset distribution: sum out the income dimension
μ_a   = vec(sum(result.μ, dims=2))
a_vec = result.a_grid
cdf_a = cumsum(μ_a)

mean_a = dot(a_vec, μ_a)            # mean wealth = E[a] = total wealth
med_a  = a_vec[findfirst(cdf_a .>= 0.5)]
std_a  = sqrt(dot(μ_a, (a_vec .- mean_a).^2))
G_w    = gini(repeat(a_vec, n_e), vec(result.μ))   # Gini over full (a,e) joint dist

# Top shares: fraction of total wealth held by households above p90 / p99
p90  = findfirst(cdf_a .>= 0.90)
p99  = findfirst(cdf_a .>= 0.99)
sh10 = dot(a_vec[p90:end], μ_a[p90:end]) / mean_a
sh1  = dot(a_vec[p99:end], μ_a[p99:end]) / mean_a

println("  ", rpad("Mean wealth", 26), " ", round(mean_a, digits=4))
println("  ", rpad("Median wealth", 26), " ", round(med_a, digits=4))
println("  ", rpad("Std wealth", 26), " ", round(std_a, digits=4))
println("  ", rpad("Coeff. of variation", 26), " ", round(std_a / mean_a, digits=4))
println("  ", rpad("Gini (wealth)", 26), " ", isnan(G_w) ? "n/a — negative wealth" : round(G_w, digits=4))
println("  ", rpad("Top-10% share", 26), " ", round(sh10, digits=4))
println("  ", rpad("Top-1%  share", 26), " ", round(sh1, digits=4))

# Plot: Lorenz curve — gap below the 45° line shows inequality. Skipped when
# households hold negative wealth (natural limit): the curve dips below zero and
# the Gini is undefined (gini() returns NaN), so the plot would be misleading.
if isnan(G_w)
    println("  Lorenz/Gini skipped — households hold negative wealth (natural borrowing limit).")
else
    F_L, L_L = lorenz_pts(a_vec, μ_a)
    plt_lorenz = plot(F_L, L_L, linewidth=2, color=:darkblue,
                      label="Lorenz  (Gini = $(round(G_w, digits=3)))",
                      xlabel="Cumulative population share",
                      ylabel="Cumulative wealth share",
                      title="Lorenz Curve — Wealth Distribution")
    plot!(plt_lorenz, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="Perfect equality")
    savefig(plt_lorenz, joinpath(FIG, "fig_lorenz.png"))
end

# Plot: wealth density as a UNIFORM-BIN histogram.
# The asset grid is curved (a_min + (a_max−a_min)·u^E), so the first points are
# packed into a tiny interval near the constraint. Plotting raw mass μ_a at each
# grid node therefore smears the constrained pile across many near-zero-width bars
# and over-weights the sparse tail, hiding the true right-skewed shape. Rebinning
# the mass onto evenly spaced wealth bins recovers it. The display range is capped
# at the 99.5th wealth percentile so the long thin tail doesn't flatten the bulk.
disp_max = a_vec[findfirst(cdf_a .>= 0.995)]
nbins    = 60
edges    = range(a_vec[1], disp_max, length = nbins + 1)
centers  = (edges[1:end-1] .+ edges[2:end]) ./ 2
binwidth = step(edges)
hist     = zeros(nbins)
for i in 1:n_a
    b = clamp(floor(Int, (a_vec[i] - edges[1]) / binwidth) + 1, 1, nbins)
    a_vec[i] <= disp_max && (hist[b] += μ_a[i])   # let the thin tail beyond the cap fade out
end
plt_wdist = bar(centers, hist, color=:steelblue, alpha=0.8, legend=false,
                xlabel="Wealth a", ylabel="Share of households",
                title="Stationary Wealth Distribution")
savefig(plt_wdist, joinpath(FIG, "fig_wealth_dist.png"))
println("  → saved: fig_lorenz.png, fig_wealth_dist.png")


# =============================================================================
# §4  POLICY FUNCTIONS
# =============================================================================
# Plot savings a'(a, e) and consumption c(a, e) across assets for each income
# state. a'(a,e) crossing the 45° line marks the household's asset target;
# higher income e shifts both curves up; c(a,e) is concave in a.

println("\n", "─"^64)
println("  §4  POLICY FUNCTIONS")
println("─"^64)

clrs   = palette(:viridis, n_e)
e_lbls = ["e = $(round(result.e_grid[j], digits=2))" for j in 1:n_e]

# Savings policy: next-period assets a' vs current assets a
plt_sav = plot(xlabel="Current assets a", ylabel="Next-period assets a'",
               title="Savings Policy  a'(a, e)", legend=:topleft)
for j in 1:n_e
    plot!(plt_sav, a_vec, a_vec[result.g_idx[:, j]], label=e_lbls[j], color=clrs[j])
end
plot!(plt_sav, a_vec, a_vec, linestyle=:dash, color=:black, label="45°", linewidth=0.8)
savefig(plt_sav, joinpath(FIG, "fig_policy_savings.png"))

# Consumption policy: consumption c vs current assets a
plt_cons = plot(xlabel="Current assets a", ylabel="Consumption c",
                title="Consumption Policy  c(a, e)", legend=:bottomright)
for j in 1:n_e
    plot!(plt_cons, a_vec, result.c[:, j], label=e_lbls[j], color=clrs[j])
end
savefig(plt_cons, joinpath(FIG, "fig_policy_consumption.png"))
println("  → saved: fig_policy_savings.png, fig_policy_consumption.png")


# =============================================================================
# §5  EULER EQUATION RESIDUALS
# =============================================================================
# Check how closely the optimal policy satisfies the household Euler equation:
#   u'(c(a,e)) = β(1+r) · Σ_{e'} P[e,e'] · u'(c(a'(a,e), e'))
# The relative residual |LHS/RHS − 1| measures distance from the true optimum
# and should be close to tol_vfi. The equation holds with equality only away
# from the borrowing constraint (g_idx = 1) and grid ceiling (g_idx = n_a),
# so those states are skipped.

println("\n", "─"^64)
println("  §5  EULER EQUATION RESIDUALS")
println("─"^64)

u_prime(c) = c > 0 ? c^(-y) : Inf   # CRRA marginal utility u'(c) = c^{-γ}

ee = zeros(n_a, n_e)
for i in 1:n_a, j in 1:n_e
    k = result.g_idx[i, j]
    (k == 1 || k == n_a) && continue   # skip constraint-binding states

    # Expected marginal utility next period: Σ_{j'} P[j,j'] · u'(c(k, j'))
    Eu_next = sum(P_mat[j, j2] * u_prime(result.c[k, j2]) for j2 in 1:n_e)
    ee[i, j] = abs(u_prime(result.c[i, j]) / (β * (1 + result.r) * Eu_next) - 1.0)
end

# Weight residuals by the stationary distribution over interior states, so we
# report accuracy at states households actually visit in equilibrium.
interior = (result.g_idx .!= 1) .& (result.g_idx .!= n_a)
μ_int    = result.μ .* interior
μ_int  ./= sum(μ_int)

mean_ee = sum(ee[i, j] * μ_int[i, j] for i in 1:n_a, j in 1:n_e)
max_ee  = maximum(ee[interior])

println("  ", rpad("Weighted-avg residual", 26), " ", round(mean_ee, sigdigits=3))
println("  ", rpad("Max residual", 26), " ", round(max_ee, sigdigits=3))


# =============================================================================
# §6  COMPARATIVE STATICS
# =============================================================================
# Vary one parameter at a time, holding all others at baseline:
#   β     discount factor:  higher β = more patient = more saving = lower r*
#   σ     income std dev:    higher σ = more uncertainty = more precaution = lower r*
#   ρ     persistence:       higher ρ = more permanent shocks = lower r*
#   γ (y) risk aversion:     higher γ = stronger precautionary motive = lower r*
#   α     capital share:     changes the production side
#   δ     depreciation:      higher δ = more investment needed = lower K*
#   a_min borrowing limit:   tighter = more precaution = lower r*
#   n_e   income states:     tests Rouwenhorst approximation convergence
# Each row runs a full solve (VFI + distribution + equilibrium); takes minutes.

println("\n", "─"^64)
println("  §6  COMPARATIVE STATICS")
println("─"^64)

"""
    eq_stats(; kwargs...) -> NamedTuple

Solve the model for a stationary equilibrium with the baseline parameters
overridden by `kwargs`, and return its headline statistics. Each call is a full
solve (VFI → stationary distribution → market-clearing root find).

`kwargs` are forwarded to [`AiyagariParams`](@ref), e.g. `eq_stats(β=0.94)` or
`eq_stats(σ=0.2, ρ=0.9)`.

# Returns
A `NamedTuple` with fields:
- `r`: equilibrium interest rate.
- `K`: aggregate capital (= mean wealth).
- `w`: equilibrium wage.
- `G`: wealth Gini over the joint `(a, e)` distribution.
- `wedge`: precautionary wedge `(1/β − 1) − r`, the gap below the
  representative-agent rate.
"""
function eq_stats(; kwargs...)
    pp  = AiyagariParams(; kwargs...)
    res = solve_aiyagari(pp)
    G     = gini(repeat(res.a_grid, pp.n_e), vec(res.μ))  # Gini over joint dist
    wedge = (1/pp.β - 1) - res.r                          # precautionary wedge
    return (r=res.r, K=res.K, w=res.w, G=G, wedge=wedge)
end

"""
    sweep(header, sym, vals) -> Vector{NamedTuple}

Comparative-statics sweep: re-solve the equilibrium for each value in `vals`,
varying the single parameter named `sym`, and print an aligned results table
under `header`.

# Arguments
- `header`: title line printed above the table.
- `sym`: the [`AiyagariParams`](@ref) field to vary, as a `Symbol` (e.g. `:β`).
- `vals`: the values to sweep over; one table row and one solve per value.

# Returns
The vector of [`eq_stats`](@ref) results, one per value in `vals`, for reuse in
the comparative-statics plots.
"""
function sweep(header, sym, vals)
    println("\n  ", header)
    println("  ", rpad("Param", 8), " ", lpad("r*(%)", 8), " ", lpad("K*", 9), " ",
            lpad("w*", 9), " ", lpad("Gini", 8), " ", lpad("Wedge(pp)", 10))
    println("  ", "─"^62)
    stats = [eq_stats(; (sym => v,)...) for v in vals]
    for (v, s) in zip(vals, stats)
        println("  ", rpad(round(Float64(v), sigdigits=3), 8), " ", lpad(round(s.r * 100, digits=3), 8), " ",
                lpad(round(s.K, digits=4), 9), " ", lpad(round(s.w, digits=4), 9), " ",
                lpad(round(s.G, digits=4), 8), " ", lpad(round(s.wedge * 100, digits=3), 10))
    end
    return stats
end

rpct(stats) = [s.r * 100 for s in stats]   # equilibrium r* in percent, for plotting

# ── 6a–6d. Household preference parameters ───────────────────────────────────
β_vals = [0.92, 0.94, 0.96, 0.98]
σ_vals = [0.05, 0.10, 0.20, 0.30, 0.40]
ρ_vals = [0.50, 0.70, 0.85, 0.90, 0.95]
γ_vals = [1.001, 1.5, 2.0, 3.0, 5.0]

β_st = sweep("β  (discount factor) — higher β = more patient = lower r*", :β, β_vals)
σ_st = sweep("σ  (income std dev) — higher σ = more uncertainty = lower r*", :σ, σ_vals)
ρ_st = sweep("ρ  (income persistence) — higher ρ = more permanent shocks = lower r*", :ρ, ρ_vals)
γ_st = sweep("γ  (risk aversion) — higher γ = stronger precautionary motive = lower r*", :y, γ_vals)

# Shared styling for the line plots below
line_opts = (ylabel="r* (%)", marker=:circle, lw=2, legend=false)

# Panel 1: r* vs the four household preference parameters
plt_cs1 = plot(layout=(2, 2), size=(860, 640))
plot!(plt_cs1, β_vals, rpct(β_st); xlabel="β", title="r* vs β", subplot=1, line_opts...)
plot!(plt_cs1, σ_vals, rpct(σ_st); xlabel="σ", title="r* vs σ", subplot=2, line_opts...)
plot!(plt_cs1, ρ_vals, rpct(ρ_st); xlabel="ρ", title="r* vs ρ", subplot=3, line_opts...)
plot!(plt_cs1, γ_vals, rpct(γ_st); xlabel="γ", title="r* vs γ", subplot=4, line_opts...)
savefig(plt_cs1, joinpath(FIG, "fig_comparative_statics.png"))
println("  → saved: fig_comparative_statics.png")

# K* and Gini vs σ: more uncertainty raises both capital and inequality
plt_kg = plot(
    plot(σ_vals, [s.K for s in σ_st], color=:navy, marker=:circle, lw=2,
         xlabel="σ (income std)", ylabel="K*", title="Capital vs Uncertainty", legend=false),
    plot(σ_vals, [s.G for s in σ_st], color=:darkorange, marker=:diamond, lw=2,
         xlabel="σ (income std)", ylabel="Gini", title="Inequality vs Uncertainty", legend=false),
    layout=(1, 2), size=(760, 340)
)
savefig(plt_kg, joinpath(FIG, "fig_uncertainty_tradeoff.png"))
println("  → saved: fig_uncertainty_tradeoff.png")

# ── 6e–6h. Production, constraint, and grid parameters ───────────────────────
# a_min < 0 lets households borrow up to |a_min|; relaxing it weakens the
# precautionary motive (the most direct test of the incomplete-markets channel).
# n_e tests whether the discrete Rouwenhorst approximation has converged.
α_vals    = [0.25, 0.30, 0.33, 0.36, 0.40, 0.45]
δ_vals    = [0.04, 0.06, 0.08, 0.10, 0.12]
amin_vals = [-2.0, -1.0, -0.5, 0.0]
ne_vals   = [3, 5, 7, 9, 11]

α_st    = sweep("α  (capital share) — changes production function shape", :α, α_vals)
δ_st    = sweep("δ  (depreciation) — higher δ = more investment needed = lower K*", :δ, δ_vals)
amin_st = sweep("a_min (borrow limit) — looser = less precaution = higher r*, lower K*", :a_min, amin_vals)
ne_st   = sweep("n_e  (income states) — convergence of Rouwenhorst approximation", :n_e, ne_vals)

# Panel 2: r* vs the four production/constraint/grid parameters
plt_cs2 = plot(layout=(2, 2), size=(860, 640))
plot!(plt_cs2, α_vals, rpct(α_st);          xlabel="α (capital share)",  title="r* vs α", subplot=1, line_opts...)
plot!(plt_cs2, δ_vals, rpct(δ_st);          xlabel="δ (depreciation)",   title="r* vs δ", subplot=2, line_opts...)
plot!(plt_cs2, amin_vals, rpct(amin_st);    xlabel="a_min (borrow limit)", title="r* vs Borrowing Limit", subplot=3, line_opts...)
plot!(plt_cs2, Float64.(ne_vals), rpct(ne_st); xlabel="n_e (income states)", title="r* vs n_e  (convergence)", subplot=4, line_opts...)
savefig(plt_cs2, joinpath(FIG, "fig_comparative_statics2.png"))
println("  → saved: fig_comparative_statics2.png")

# Borrowing constraint, the key incomplete-markets result:
# tighter constraint → more precautionary saving → lower r*, higher K*, more inequality
amin_opts = (xlabel="a_min", marker=:circle, lw=2, legend=false)
plt_amin = plot(layout=(1, 3), size=(960, 320))
plot!(plt_amin, amin_vals, rpct(amin_st);          ylabel="r* (%)", title="r* vs Borrowing Limit", color=:steelblue, subplot=1, amin_opts...)
plot!(plt_amin, amin_vals, [s.K for s in amin_st]; ylabel="K*",     title="K* vs Borrowing Limit", color=:darkgreen, subplot=2, amin_opts...)
plot!(plt_amin, amin_vals, [s.G for s in amin_st]; ylabel="Gini",   title="Gini vs Borrowing Limit", color=:darkorange, subplot=3, amin_opts...)
savefig(plt_amin, joinpath(FIG, "fig_borrowing_constraint.png"))
println("  → saved: fig_borrowing_constraint.png")


# =============================================================================
# §7  ERGODICITY — CONVERGENCE TO THE STATIONARY DISTRIBUTION
# =============================================================================
# Verify the joint (a, e) chain is ergodic: from ANY initial distribution.

println("\n", "─"^64)
println("  §7  ERGODICITY — CONVERGENCE TO STATIONARY DISTRIBUTION")
println("─"^64)

"""
    point_mass(i, j) -> Matrix{Float64}

A degenerate `n_a × n_e` distribution placing all mass on the single state
`(a_grid[i], e_grid[j])`. Used as an extreme initial condition for the
ergodicity test.
"""
point_mass(i, j) = (m = zeros(n_a, n_e); m[i, j] = 1.0; m)

"""
    fwd_iter_dist(μ0, g_idx, P, μ_star, T) -> Vector{Float64}

Push the population distribution forward `T` periods under the household policy
and income process, tracking how fast it approaches the stationary distribution.

Each step applies the joint `(a, e)` transition: assets move deterministically
via the savings policy `g_idx`, while income mixes via the Markov matrix `P`.

# Arguments
- `μ0`: initial `n_a × n_e` distribution (its `size` fixes the grid dimensions).
- `g_idx`: savings policy as next-asset grid indices, `g_idx[i, j]`.
- `P`: income transition matrix, `P[j, j']`.
- `μ_star`: the stationary distribution to measure convergence against.
- `T`: number of forward iterations.

# Returns
A length-`T` vector whose `t`-th entry is the total-variation distance
`½·Σ|μ_t − μ_star|`, which should decay geometrically if the chain is ergodic.

We use total variation, not the sup-norm, because TV is the standard metric for
Markov mixing (it is what the reported mixing time refers to) and because it is
provably non-increasing along the chain. The sup-norm tracks only the single
largest cell, which can cross `μ_star` from above to below near convergence and
momentarily collapse to zero, producing a spurious downward spike on a log plot.
"""
function fwd_iter_dist(μ0, g_idx, P, μ_star, T)
    na, ne = size(μ0)
    μ_t = copy(μ0)
    d   = zeros(T)
    for t in 1:T
        # One Markov step: μ_new[k, j'] = Σ_{i,j} P[j,j'] · μ_t[i,j] · 1(g(i,j) = k)
        μ_new = zeros(na, ne)
        for i in 1:na, j in 1:ne
            k = g_idx[i, j]
            for j2 in 1:ne
                μ_new[k, j2] += P[j, j2] * μ_t[i, j]
            end
        end
        d[t] = 0.5 * sum(abs.(μ_new .- μ_star))   # total-variation distance to stationary dist
        μ_t  = μ_new
    end
    return d
end

# T_erg is long enough that even the slow asset-accumulation mode (|λ₂| ≈ 0.98,
# much slower than the income chain's ρ) drives every start down to the numerical
# floor, so the three curves visibly flatten and meet at the same μ*.
T_erg = 1200
iter(μ0) = fwd_iter_dist(μ0, result.g_idx, P_mat, result.μ, T_erg)
d_poor = iter(point_mass(1, 1))            # start: all poorest (a_min, e_min)
d_rich = iter(point_mass(n_a, n_e))        # start: all richest (a_max, e_max)
d_mid  = iter(point_mass(n_a÷2, n_e÷2+1))  # start: all at grid midpoint

# All three lines converging to the same μ* confirms ergodicity
plt_erg = plot(1:T_erg, log10.(d_poor .+ 1e-16),
               label="Start: all poor  (a_min, e_min)",
               xlabel="Iterations t", ylabel="log10 TV(μ_t, μ*)",
               title="Convergence to Stationary Distribution  (Ergodic theorem)",
               linewidth=2, color=:crimson)
plot!(plt_erg, 1:T_erg, log10.(d_rich .+ 1e-16),
      label="Start: all rich  (a_max, e_max)", linewidth=2, color=:navy)
plot!(plt_erg, 1:T_erg, log10.(d_mid .+ 1e-16),
      label="Start: midpoint", linewidth=2, color=:darkgreen, linestyle=:dash)
savefig(plt_erg, joinpath(FIG, "fig_ergodicity.png"))

println("\n  Total-variation distance TV(μ_t, μ*):")
println("  ", rpad("t", 8), " ", lpad("poor start", 12), " ", lpad("rich start", 12))
for t in (50, 150, T_erg)
    println("  ", rpad(t, 8), " ", lpad(round(d_poor[t], sigdigits=2), 12), " ", lpad(round(d_rich[t], sigdigits=2), 12))
end
println("  → saved: fig_ergodicity.png")


# =============================================================================
# SUMMARY
# =============================================================================
println("\n", "═"^64)
println("  ALL FIGURES SAVED")
println("═"^64)
for (file, desc) in [
    ("fig_lorenz.png",               "Lorenz curve and Gini coefficient"),
    ("fig_wealth_dist.png",          "Stationary wealth density"),
    ("fig_policy_savings.png",       "Savings policy a'(a,e)"),
    ("fig_policy_consumption.png",   "Consumption policy c(a,e)"),
    ("fig_comparative_statics.png",  "r* vs β, σ, ρ, γ  (2×2 panel)"),
    ("fig_uncertainty_tradeoff.png", "K* and Gini vs σ  (side-by-side)"),
    ("fig_comparative_statics2.png", "r* vs α, δ, a_min, n_e  (2×2 panel)"),
    ("fig_borrowing_constraint.png", "r*, K*, Gini vs a_min  (3-panel)"),
    ("fig_ergodicity.png",           "Distribution convergence from 3 starts"),
]
    println("  ", rpad(file, 30), " ", desc)
end
println("═"^64)
