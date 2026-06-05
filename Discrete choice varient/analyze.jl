

# analyze.jl - Aiyagari (1994) Model Analysis


"""
analyze.jl

Sections:
    §1  Baseline verification   - market clearing, distribution validity
    §2  Income process          - Markov chain eigenvalues, mixing time, ACF
    §3  Wealth distribution     - stationary distribution, Lorenz curve, Gini
    §4  Policy functions        - savings and consumption rules a'(a,e), c(a,e)
    §5  Euler equation check    - numerical accuracy of the VFI solution
    §6  Comparative statics     - how equilibrium shifts with β, σ, ρ, γ, α, δ, a_min, n_e
    §7  Ergodicity              - distribution convergence from extreme starting points

Run: include("analyze.jl") in a fresh Julia REPL from this directory.
Requires Plots: ] add Plots
"""

include("main.jl")  # includes all components, runs baseline → `result`, `p`

using Statistics
using Plots


# Helper functions
"""
gini(vals, wts)

Goal: Compute the Gini coefficient from a discrete (value, weight) distribution.
      Gini = 1 means perfect inequality, Gini = 0 means perfect equality.

Output: Gini coefficient ∈ [0, 1]
"""
function gini(vals, wts)
    ord = sortperm(collect(vals))
    v   = collect(vals)[ord]
    w   = collect(wts)[ord]
    w ./= sum(w)                         # normalize weights to sum to 1
    tw  = dot(v, w)
    tw ≤ 0 && return 0.0
    L = cumsum(v .* w) ./ tw             # Lorenz ordinates: cumulative wealth share
    F = cumsum(w)                        # cumulative population share
    # area under Lorenz curve via trapezoids, then Gini = 1 - 2*area
    area = (F[1] * L[1]) / 2 +
           sum((F[k] - F[k-1]) * (L[k] + L[k-1]) / 2 for k in 2:lastindex(F))
    return 1.0 - 2.0 * area
end

"""
lorenz_pts(vals, wts)

Goal: Compute Lorenz curve data points from a discrete distribution.
      Returns cumulative population share (F) and cumulative wealth share (L),
      both starting at (0, 0).

Output: (F, L) - two vectors of length length(vals)+1
"""
function lorenz_pts(vals, wts)
    ord = sortperm(collect(vals))
    v   = collect(vals)[ord]
    w   = collect(wts)[ord]
    w ./= sum(w)
    tw  = dot(v, w)
    F   = [0.0; cumsum(w)]               # cumulative population share
    L   = [0.0; cumsum(v .* w) ./ tw]   # cumulative wealth share
    return F, L
end

# Pull parameters and shared objects from the baseline run
(; n_a, n_e, β, y) = p
z_grid, P_mat = rouwenhorst(p)                                  # income grid and transition matrix
π_inc = stationary_distributions(MarkovChain(P_mat))[1]        # stationary income probabilities

println("=" ^ 62)
println("  Aiyagari Analysis — r* = $(round(result.r*100, digits=3))%  K* = $(round(result.K, digits=3))")
println("=" ^ 62)


# §1  BASELINE VERIFICATION


"""
§1 Baseline Verification

Goal: Confirm that the equilibrium solution satisfies the two key conditions:
      (1) Capital market clears: aggregate household savings = firm capital demand
      (2) The stationary distribution is a valid probability measure (sums to 1)

Also reports the precautionary savings wedge: how much lower r* is compared to
the representative-agent rate 1/β−1 due to incomplete markets.
"""

println("\n")
println("# §1  BASELINE VERIFICATION")
println("")

# Recompute K_supply from the policy and distribution to verify market clearing
K_s = sum(result.a_grid[result.g_idx[i,j]] * result.μ[i,j]
          for i in 1:n_a, j in 1:n_e)

println("  |K_supply − K_demand|  = $(round(abs(K_s - result.K), sigdigits=3))")
println("  Σ μ[i,j]               = $(round(sum(result.μ), digits=8))")
println("  r*  (Aiyagari)         = $(round(result.r, digits=4))")
println("  r_RA = 1/β−1           = $(round(1/β - 1, digits=4))")
println("  Precautionary wedge    = $(round((1/β - 1 - result.r)*100, digits=3)) pp")


# §2  INCOME PROCESS — MARKOV CHAIN ANALYSIS


"""
§2 Markov Chain Analysis — Income Process

Goal: Analyze the stochastic income process z_t = ρ*z_{t-1} + ε_t, which is
      approximated by a finite Markov chain using the Rouwenhorst (1995) method.

Key quantities:
    Eigenvalues of P   - the second eigenvalue |λ₂| controls the speed of mixing
    Spectral gap       - 1 - |λ₂| determines how quickly the chain forgets its past
    TV-mixing time     - upper bound on how many steps to reach ε-close to stationarity
    Autocorrelation    - should match the AR(1) theory: ACF(h) = ρ^h
"""

println("\n")
println("# §2  INCOME PROCESS — MARKOV CHAIN ANALYSIS")
println("")

# Eigenvalues: sorted by absolute value descending
# λ₁ = 1 always (stochastic matrix), λ₂ governs mixing rate
λ_all  = eigvals(P_mat)
λ_sort = sort(λ_all, by=abs, rev=true)
sgap   = 1.0 - abs(λ_sort[2])

println("  Eigenvalues of P: $(round.(real.(λ_sort), digits=5))")
println("  Spectral gap (1 − |λ₂|) = $(round(sgap, digits=5))")

# TV-mixing time bound: ||μ_t − π||_TV ≤ (n_e/2) · |λ₂|^t < ε
# Solving for t: t > log(n_e / 2ε) / log(1 / |λ₂|)
t_mix = ceil(Int, log(n_e / 0.02) / (-log(abs(λ_sort[2]))))
println("  TV-mixing time bound (ε=0.01) ≤ $t_mix steps")

# Autocorrelation of log income: compare Markov numerics to AR(1) theory ρ^h
# Cov(z_0, z_h) = π' .* z_grid * P^h * z_grid − mean_z²
mean_z  = dot(π_inc, z_grid)
var_z   = dot(π_inc, (z_grid .- mean_z).^2)
max_lag = 10
acf_mc  = zeros(max_lag + 1)
Ph = Matrix{Float64}(I, n_e, n_e)   # P^0 = identity
for h in 0:max_lag
    acf_mc[h+1] = (dot(π_inc .* z_grid, Ph * z_grid) - mean_z^2) / var_z
    Ph = Ph * P_mat
end

println("\n  Log-income ACF — Markov chain numerics vs AR(1) theory ρ^h:")
println("  Lag    Markov      ρ^h (theory)")
for h in 0:max_lag
    println("   $h      $(round(acf_mc[h+1], digits=5))     $(round(p.ρ^h, digits=5))")
end

# Stationary income distribution summary
var_e = dot(π_inc, (result.e_grid .- dot(π_inc, result.e_grid)).^2)
println("\n  Stationary income std  σ_e = $(round(sqrt(var_e), digits=4))")
println("  Income Gini            G_e = $(round(gini(result.e_grid, π_inc), digits=4))")

# Plot: ACF bar chart with theoretical AR(1) overlay
plt_acf = bar(0:max_lag, acf_mc, color=:steelblue, alpha=0.85,
              xlabel="Lag h", ylabel="Autocorrelation",
              title="Log-Income ACF  (ρ = $(p.ρ))", legend=:topright, label="Markov")
plot!(plt_acf, 0:max_lag, p.ρ .^ (0:max_lag),
      linestyle=:dash, linewidth=2, color=:crimson, label="ρ^h (theory)")
savefig(plt_acf, "fig_acf.png")
println("  → fig_acf.png saved")


# §3  STATIONARY WEALTH DISTRIBUTION & INEQUALITY


"""
§3 Stationary Wealth Distribution & Inequality

Goal: Characterize the cross-sectional distribution of wealth in the stationary
      equilibrium. The distribution μ is the unique invariant measure of the
      joint (a, e) Markov chain induced by the optimal policy g_idx.

Key quantities:
    Mean, median, std  - basic moments of the marginal wealth distribution
    Gini coefficient   - 0 = perfect equality, 1 = perfect inequality
    Top shares         - fraction of total wealth held by the top 10% and top 1%
    Lorenz curve       - visual representation of wealth inequality
"""

println("\n")
println("# §3  WEALTH DISTRIBUTION & INEQUALITY")
println("")

# Marginal distribution over assets: sum out the income dimension
μ_a   = vec(sum(result.μ, dims=2))
a_vec = result.a_grid
cdf_a = cumsum(μ_a)
tw    = dot(a_vec, μ_a)   # total (mean) wealth = E[a]

mean_a = tw
med_a  = a_vec[findfirst(cdf_a .>= 0.5)]
std_a  = sqrt(dot(μ_a, (a_vec .- mean_a).^2))
G_w    = gini(repeat(a_vec, n_e), vec(result.μ))   # Gini over full (a,e) joint distribution

# Top-share computation: fraction of total wealth held by households above p90/p99
p90  = findfirst(cdf_a .>= 0.90)
p99  = findfirst(cdf_a .>= 0.99)
sh10 = dot(a_vec[p90:end], μ_a[p90:end]) / tw
sh1  = dot(a_vec[p99:end], μ_a[p99:end]) / tw

println("  Mean wealth          = $(round(mean_a, digits=4))")
println("  Median wealth        = $(round(med_a, digits=4))")
println("  Std wealth           = $(round(std_a, digits=4))")
println("  Coeff. of variation  = $(round(std_a / mean_a, digits=4))")
println("  Gini (wealth)        = $(round(G_w, digits=4))")
println("  Top-10% share        = $(round(sh10, digits=4))")
println("  Top-1%  share        = $(round(sh1, digits=4))")

# Plot: Lorenz curve — deviations below the 45° equality line show inequality
F_L, L_L = lorenz_pts(a_vec, μ_a)
plt_lorenz = plot(F_L, L_L, linewidth=2, color=:darkblue,
                  label="Lorenz  (Gini = $(round(G_w, digits=3)))",
                  xlabel="Cumulative population share",
                  ylabel="Cumulative wealth share",
                  title="Lorenz Curve — Wealth Distribution")
plot!(plt_lorenz, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="Perfect equality")
savefig(plt_lorenz, "fig_lorenz.png")

# Plot: wealth density — mass at each grid point
nz = μ_a .> 1e-10   # drop near-zero mass points for readability
plt_wdist = bar(a_vec[nz], μ_a[nz], color=:steelblue, alpha=0.7, legend=false,
                xlabel="Wealth a", ylabel="Mass",
                title="Stationary Wealth Distribution")
savefig(plt_wdist, "fig_wealth_dist.png")
println("  → fig_lorenz.png, fig_wealth_dist.png saved")


# §4  POLICY FUNCTIONS


"""
§4 Policy Functions

Goal: Plot the savings policy a'(a, e) and consumption policy c(a, e) across
      asset levels for each income state.

Key features to observe:
    - a'(a, e) > a  means the household is accumulating wealth
    - a'(a, e) < a  means the household is decumulating (running down assets)
    - The crossing point with the 45° line is the household's asset target
    - Higher income e shifts both curves up (richer households save more)
    - All c(a, e) curves are concave — diminishing marginal utility of consumption
"""

println("\n")
println("# §4  POLICY FUNCTIONS")
println("")

clrs    = palette(:viridis, n_e)
e_lbls  = ["e = $(round(result.e_grid[j], digits=2))" for j in 1:n_e]

# Savings policy: next-period assets a' as a function of current assets a
plt_sav = plot(xlabel="Current assets a", ylabel="Next-period assets a'",
               title="Savings Policy  a'(a, e)", legend=:topleft)
for j in 1:n_e
    plot!(plt_sav, a_vec, a_vec[result.g_idx[:, j]], label=e_lbls[j], color=clrs[j])
end
plot!(plt_sav, a_vec, a_vec, linestyle=:dash, color=:black, label="45°", linewidth=0.8)
savefig(plt_sav, "fig_policy_savings.png")

# Consumption policy: consumption c as a function of current assets a
plt_cons = plot(xlabel="Current assets a", ylabel="Consumption c",
                title="Consumption Policy  c(a, e)", legend=:bottomright)
for j in 1:n_e
    plot!(plt_cons, a_vec, result.c[:, j], label=e_lbls[j], color=clrs[j])
end
savefig(plt_cons, "fig_policy_consumption.png")

println("  → fig_policy_savings.png, fig_policy_consumption.png saved")


# §5  EULER EQUATION RESIDUALS


"""
§5 Euler Equation Residuals

Goal: Verify the numerical accuracy of the VFI solution by checking how closely
      the optimal policy satisfies the household Euler equation:

          u'(c(a,e)) = β(1+r) · Σ_{e'} P[e,e'] · u'(c(a'(a,e), e'))

      The relative residual |LHS/RHS − 1| measures how far the numerical
      solution is from the true optimum. Should be close to tol_vfi.

Note: Euler equation holds with equality only away from the borrowing constraint
      (g_idx = 1) and the grid ceiling (g_idx = n_a). We skip those states.
"""

println("\n")
println("# §5  EULER EQUATION RESIDUALS")
println("")

u_prime(c) = c > 0 ? c^(-y) : Inf   # CRRA marginal utility: u'(c) = c^{-γ}

ee = zeros(n_a, n_e)
for i in 1:n_a, j in 1:n_e
    k = result.g_idx[i, j]
    (k == 1 || k == n_a) && continue   # skip constraint-binding states

    # Expected marginal utility next period: Σ_{j'} P[j,j'] · u'(c(k, j'))
    Eu_next = sum(P_mat[j, j2] * u_prime(result.c[k, j2]) for j2 in 1:n_e)

    # Relative residual: how far LHS / (β(1+r) · E[u'(c')]) is from 1
    ee[i, j] = abs(u_prime(result.c[i, j]) / (β * (1 + result.r) * Eu_next) - 1.0)
end

# Weight residuals by stationary distribution so we report accuracy
# at states households actually visit in equilibrium
interior  = (result.g_idx .!= 1) .& (result.g_idx .!= n_a)
μ_int     = result.μ .* interior
μ_int   ./= sum(μ_int)   # renormalize to get conditional distribution

mean_ee = sum(ee[i,j] * μ_int[i,j] for i in 1:n_a, j in 1:n_e)
max_ee  = maximum(ee[interior])

println("  Weighted-avg relative Euler residual = $(round(mean_ee, sigdigits=3))")
println("  Max relative Euler residual          = $(round(max_ee, sigdigits=3))")


# §6  COMPARATIVE STATICS


"""
§6 Comparative Statics

Goal: Show how the stationary equilibrium changes when we vary one parameter
      at a time, holding all others at baseline values.

Parameters tested:
    β     - discount factor: higher β = more patient = more saving = lower r*
    σ     - income std dev:  higher σ = more uncertainty = more precaution = lower r*
    ρ     - persistence:     higher ρ = more permanent shocks = more precaution = lower r*
    γ (y) - risk aversion:   higher γ = stronger precautionary motive = lower r*
    α     - capital share:   changes the production side of the model
    δ     - depreciation:    higher δ = more investment needed
    a_min - borrowing limit: less negative = tighter constraint = more precaution = lower r*
    n_e   - income states:   tests convergence of the Rouwenhorst Markov approximation

Each row runs a full model solve (VFI + distribution + equilibrium).
This section may take several minutes to complete.
"""

println("\n")
println("# §6  COMPARATIVE STATICS")
println("")

# Helper: run the full model with modified parameters and return key statistics
function eq_stats(; kwargs...)
    pp  = AiyagariParams(; kwargs...)
    res = solve_aiyagari(pp)

    # Gini from the full joint distribution (includes income-wealth correlation)
    G = gini(repeat(res.a_grid, pp.n_e), vec(res.μ))

    # Precautionary savings wedge: how much lower r* is vs complete markets
    wedge = (1/pp.β - 1) - res.r

    return (r=res.r, K=res.K, w=res.w, G=G, wedge=wedge)
end

# Helper: print a formatted results table (no Printf — use round() + string interpolation)
function show_table(header, vals, stats)
    println("\n  $header")
    println("  Param     r*(%)    K*        w*        Gini    Wedge(pp)")
    for (v, s) in zip(vals, stats)
        v_s = lpad(round(Float64(v), digits=3), 6)
        r_s = lpad(round(s.r * 100, digits=3), 7)
        K_s = lpad(round(s.K, digits=4), 8)
        w_s = lpad(round(s.w, digits=4), 8)
        G_s = lpad(round(s.G, digits=4), 6)
        wd_s = lpad(round(s.wedge * 100, digits=3), 7)
        println("  $v_s   $r_s   $K_s   $w_s   $G_s   $wd_s")
    end
end

# ── 6a. Discount factor β ────────────────────────────────────────────────────
# Higher β = more patient households = more saving = larger capital stock = lower r*
β_vals = [0.92, 0.94, 0.96, 0.98]
β_st   = [eq_stats(β=v) for v in β_vals]
show_table("β  (discount factor) — higher β = more patient = lower r*", β_vals, β_st)

# ── 6b. Income volatility σ ──────────────────────────────────────────────────
# Higher σ = more income uncertainty = stronger precautionary motive = lower r*
σ_vals = [0.05, 0.10, 0.20, 0.30, 0.40]
σ_st   = [eq_stats(σ=v) for v in σ_vals]
show_table("σ  (income std dev) — higher σ = more uncertainty = lower r*", σ_vals, σ_st)

# ── 6c. Income persistence ρ ─────────────────────────────────────────────────
# Higher ρ = shocks are more permanent = harder to self-insure = lower r*
ρ_vals = [0.50, 0.70, 0.85, 0.90, 0.95]
ρ_st   = [eq_stats(ρ=v) for v in ρ_vals]
show_table("ρ  (income persistence) — higher ρ = more permanent shocks = lower r*", ρ_vals, ρ_st)

# ── 6d. Risk aversion γ ──────────────────────────────────────────────────────
# Higher γ = stronger precautionary motive = more saving = lower r*
γ_vals = [1.001, 1.5, 2.0, 3.0, 5.0]
γ_st   = [eq_stats(y=v) for v in γ_vals]
show_table("γ  (risk aversion) — higher γ = stronger precautionary motive = lower r*", γ_vals, γ_st)

# Panel 1: r* vs the four household preference parameters
plt_cs1 = plot(layout=(2,2), size=(860, 640))
plot!(plt_cs1, β_vals, [s.r*100 for s in β_st],
      xlabel="β", ylabel="r* (%)", title="r* vs β",  marker=:circle, lw=2, legend=false, subplot=1)
plot!(plt_cs1, σ_vals, [s.r*100 for s in σ_st],
      xlabel="σ", ylabel="r* (%)", title="r* vs σ",  marker=:circle, lw=2, legend=false, subplot=2)
plot!(plt_cs1, ρ_vals, [s.r*100 for s in ρ_st],
      xlabel="ρ", ylabel="r* (%)", title="r* vs ρ",  marker=:circle, lw=2, legend=false, subplot=3)
plot!(plt_cs1, γ_vals, [s.r*100 for s in γ_st],
      xlabel="γ", ylabel="r* (%)", title="r* vs γ",  marker=:circle, lw=2, legend=false, subplot=4)
savefig(plt_cs1, "fig_comparative_statics.png")
println("  → fig_comparative_statics.png saved")

# Plot K* and Gini vs σ — shows that more uncertainty raises both capital and inequality
plt_kg = plot(
    plot(σ_vals, [s.K for s in σ_st], color=:navy, marker=:circle, lw=2,
         xlabel="σ (income std)", ylabel="K*", title="Capital vs Uncertainty", legend=false),
    plot(σ_vals, [s.G for s in σ_st], color=:darkorange, marker=:diamond, lw=2,
         xlabel="σ (income std)", ylabel="Gini", title="Inequality vs Uncertainty", legend=false),
    layout=(1,2), size=(760, 340)
)
savefig(plt_kg, "fig_uncertainty_tradeoff.png")
println("  → fig_uncertainty_tradeoff.png saved")

# ── 6e. Capital share α ──────────────────────────────────────────────────────
# α sets the production function curvature: Y = K^α L^{1-α}
# Higher α = capital more productive = firms want more capital = affects r* and wages
α_vals = [0.25, 0.30, 0.33, 0.36, 0.40, 0.45]
α_st   = [eq_stats(α=v) for v in α_vals]
show_table("α  (capital share) — changes production function shape", α_vals, α_st)

# ── 6f. Depreciation rate δ ──────────────────────────────────────────────────
# Higher δ = capital wears out faster = firms need more gross investment = lower K*
δ_vals = [0.04, 0.06, 0.08, 0.10, 0.12]
δ_st   = [eq_stats(δ=v) for v in δ_vals]
show_table("δ  (depreciation) — higher δ = more investment needed = lower K*", δ_vals, δ_st)

# ── 6g. Borrowing limit a_min ────────────────────────────────────────────────
# a_min < 0 means households can borrow up to |a_min| units.
# Relaxing the constraint reduces the precautionary savings motive:
# households can borrow when income is low instead of pre-saving.
# This is the most direct test of the incomplete-markets mechanism.
amin_vals = [-2.0, -1.0, -0.5, 0.0]
amin_st   = [eq_stats(a_min=v) for v in amin_vals]
show_table("a_min (borrow limit) — less tight = less precaution = higher r*, lower K*",
           amin_vals, amin_st)

# ── 6h. Number of income states n_e ──────────────────────────────────────────
# Tests whether the discrete Rouwenhorst Markov approximation has converged.
# The equilibrium r*, K* should stabilize as n_e increases.
# If results change a lot between n_e=7 and n_e=11, the baseline grid is too coarse.
ne_vals = [3, 5, 7, 9, 11]
ne_st   = [eq_stats(n_e=v) for v in ne_vals]
show_table("n_e  (income states) — convergence of Rouwenhorst Markov approximation",
           ne_vals, ne_st)

# Panel 2: r* vs the four production/constraint/grid parameters
plt_cs2 = plot(layout=(2,2), size=(860, 640))
plot!(plt_cs2, α_vals, [s.r*100 for s in α_st],
      xlabel="α (capital share)", ylabel="r* (%)", title="r* vs α",
      marker=:circle, lw=2, legend=false, subplot=1)
plot!(plt_cs2, δ_vals, [s.r*100 for s in δ_st],
      xlabel="δ (depreciation)", ylabel="r* (%)", title="r* vs δ",
      marker=:circle, lw=2, legend=false, subplot=2)
plot!(plt_cs2, amin_vals, [s.r*100 for s in amin_st],
      xlabel="a_min (borrow limit)", ylabel="r* (%)", title="r* vs Borrowing Limit",
      marker=:circle, lw=2, legend=false, subplot=3)
plot!(plt_cs2, Float64.(ne_vals), [s.r*100 for s in ne_st],
      xlabel="n_e (income states)", ylabel="r* (%)", title="r* vs n_e  (convergence)",
      marker=:circle, lw=2, legend=false, subplot=4)
savefig(plt_cs2, "fig_comparative_statics2.png")
println("  → fig_comparative_statics2.png saved")

# Borrowing constraint: show r*, K*, and Gini together
# This triple-panel is the key incomplete-markets result:
# tighter constraint → more precautionary saving → lower r*, higher K*, more inequality
plt_amin = plot(layout=(1,3), size=(960, 320))
plot!(plt_amin, amin_vals, [s.r*100 for s in amin_st],
      xlabel="a_min", ylabel="r* (%)", title="r* vs Borrowing Limit",
      marker=:circle, lw=2, color=:steelblue, legend=false, subplot=1)
plot!(plt_amin, amin_vals, [s.K for s in amin_st],
      xlabel="a_min", ylabel="K*", title="K* vs Borrowing Limit",
      marker=:circle, lw=2, color=:darkgreen, legend=false, subplot=2)
plot!(plt_amin, amin_vals, [s.G for s in amin_st],
      xlabel="a_min", ylabel="Gini", title="Gini vs Borrowing Limit",
      marker=:circle, lw=2, color=:darkorange, legend=false, subplot=3)
savefig(plt_amin, "fig_borrowing_constraint.png")
println("  → fig_borrowing_constraint.png saved")


# §7  ERGODICITY — CONVERGENCE TO THE STATIONARY DISTRIBUTION


"""
§7 Ergodicity — Distribution Convergence

Goal: Verify that the joint (a, e) Markov chain is ergodic: starting from ANY
      initial distribution, the distribution μ_t converges to the unique
      stationary distribution μ* as t → ∞.

This is the Markov chain Law of Large Numbers: cross-sectional averages in the
model equal time-series averages for any individual household path.

We test by forward-iterating μ_t from three very different starting points:
    - All households at (a_min, e_min): the poorest possible state
    - All households at (a_max, e_max): the richest possible state
    - All households at the midpoint of the grid

If all three converge to the same μ*, the chain is ergodic.
The sup-norm distance ||μ_t − μ*||∞ should decay geometrically.
"""

println("\n")
println("# §7  ERGODICITY — CONVERGENCE TO THE STATIONARY DISTRIBUTION")
println("")

# Forward-iterate the distribution T times starting from μ_init
# Returns the sup-norm distance ||μ_t - μ*||∞ at each step t
function fwd_iter_dist(μ_init, g_idx_eq, P_loc, na, ne, μ_star, T)
    μ_t = copy(μ_init)
    d   = zeros(T)
    for t in 1:T
        # Apply one step of the Markov transition:
        # μ_new[k, j'] = Σ_{i,j} P[j,j'] · μ_t[i,j] · 1(g(i,j) = k)
        μ_new = zeros(na, ne)
        for i in 1:na, j in 1:ne
            k = g_idx_eq[i, j]
            for j2 in 1:ne
                μ_new[k, j2] += P_loc[j, j2] * μ_t[i, j]
            end
        end
        d[t] = maximum(abs.(μ_new .- μ_star))   # sup-norm distance to stationary dist
        μ_t  = μ_new
    end
    return d
end

T_erg  = 300
μ_poor = zeros(n_a, n_e);  μ_poor[1,       1]       = 1.0
μ_rich = zeros(n_a, n_e);  μ_rich[n_a,     n_e]     = 1.0
μ_mid  = zeros(n_a, n_e);  μ_mid[n_a÷2,    n_e÷2+1] = 1.0

d_poor = fwd_iter_dist(μ_poor, result.g_idx, P_mat, n_a, n_e, result.μ, T_erg)
d_rich = fwd_iter_dist(μ_rich, result.g_idx, P_mat, n_a, n_e, result.μ, T_erg)
d_mid  = fwd_iter_dist(μ_mid,  result.g_idx, P_mat, n_a, n_e, result.μ, T_erg)

# All three lines should converge to the same μ* — confirms ergodicity
plt_erg = plot(1:T_erg, log10.(d_poor .+ 1e-16),
               label="Start: all poor  (a_min, e_min)",
               xlabel="Iterations t", ylabel="log10 ||μ_t − μ*||∞",
               title="Convergence to Stationary Distribution  (Ergodic theorem)",
               linewidth=2, color=:crimson)
plot!(plt_erg, 1:T_erg, log10.(d_rich .+ 1e-16),
      label="Start: all rich  (a_max, e_max)", linewidth=2, color=:navy)
plot!(plt_erg, 1:T_erg, log10.(d_mid  .+ 1e-16),
      label="Start: midpoint", linewidth=2, color=:darkgreen, linestyle=:dash)
savefig(plt_erg, "fig_ergodicity.png")

println("  ||μ_t − μ*||∞ at t=50:   poor=$(round(d_poor[50], sigdigits=2))   rich=$(round(d_rich[50], sigdigits=2))")
println("  ||μ_t − μ*||∞ at t=150:  poor=$(round(d_poor[150], sigdigits=2))   rich=$(round(d_rich[150], sigdigits=2))")
println("  ||μ_t − μ*||∞ at t=300:  poor=$(round(d_poor[T_erg], sigdigits=2))   rich=$(round(d_rich[T_erg], sigdigits=2))")
println("  → fig_ergodicity.png saved")


# Summary


println("\n" * "=" ^ 62)
println("  All figures saved:")
println("    fig_acf.png                  ACF of log income vs AR(1) theory")
println("    fig_lorenz.png               Lorenz curve and Gini coefficient")
println("    fig_wealth_dist.png          Stationary wealth density")
println("    fig_policy_savings.png       Savings policy a'(a,e)")
println("    fig_policy_consumption.png   Consumption policy c(a,e)")
println("    fig_comparative_statics.png  r* vs β, σ, ρ, γ  (2x2 panel)")
println("    fig_uncertainty_tradeoff.png K* and Gini vs σ  (side-by-side)")
println("    fig_comparative_statics2.png r* vs α, δ, a_min, n_e  (2x2 panel)")
println("    fig_borrowing_constraint.png r*, K*, Gini vs a_min  (3-panel)")
println("    fig_ergodicity.png           Distribution convergence from 3 starts")
println("=" ^ 62)
