
# =============================================================================
# main.jl
# =============================================================================

include("params.jl")
include("grids.jl")
include("rouwenhorst.jl")
include("vfi.jl")
include("distributions.jl")

using LinearAlgebra
using QuantEcon
using NonlinearSolve

"""
borrowing_limit(p, w, r, e_min)

Lower bound on next-period assets.

- `:adhoc`   — returns the fixed `p.a_min` (e.g. the classic a_min = 0).
- `:natural` — returns the Aiyagari (1994) natural borrowing limit −φ, where
  φ = w·e_min / r is the present value of an unending stream of the *lowest*
  labor income, i.e. the most a household could borrow and still repay with
  certainty. `nbl_buffer` sits inside φ so consumption stays strictly
  positive at the constraint (keeps the value function finite rather than −Inf
  at the corner), and cap total borrowing at `b_max` for robustness when the
  root-finder probes r ≤ 0, where φ is not well defined.
"""
function borrowing_limit(p::AiyagariParams, w, r, e_min)
    p.borrow_limit == :natural || return p.a_min
    φ = r > 0 ? w * e_min / r : Inf          # PV of an unending stream of the lowest income
    return -min(p.b_max, (1 - p.nbl_buffer) * φ)
end

"""
solve_aiyagari

Find the stationary general equilibrium of the model by
solving for the interest rate r* where the capital market clears:
    K_supply(r) == K_demand(r)

Returns a named tuple with equilibrium prices, aggregates, and household objects.

Economic interpretation of output:
    r (interest rate): the equilibrium net return on capital.

    w (wage):          the competitive wage equal to the marginal product of labor.
                       Higher K* (from more precautionary saving) means a higher K/L ratio
                       and therefore a higher wage, households benefit from the capital
                       accumulation driven by income risk.

    K (capital):       the aggregate capital stock in the stationary equilibrium.
                       Equal to average household wealth: K = Σ_{i,j} a_grid[i] * μ[i,j].
                       Higher than the complete-markets level because precautionary saving
                       pushes households to hold more assets than they otherwise would.

    L (labor):         aggregate effective labor supply, fixed at the stationary mean of
                       the income endowment process: L = Σ_j π_j * e_j. Households supply
                       labor inelastically so L does not respond to prices.

    V (n_a × n_e):     value function. V[i,j] is the maximum lifetime utility of a household
                       currently holding a_grid[i] units of assets and earning income e_grid[j].

    g_idx (n_a × n_e): savings policy function, stored as grid indices.

    c (n_a × n_e):     consumption policy function. c[i,j] is optimal consumption at state
                       (a_grid[i], e_grid[j]).

    μ (n_a × n_e):     stationary distribution. μ[i,j] is the fraction of households
                       simultaneously holding assets a_grid[i] and earning income e_grid[j]
                       in the long-run steady state.

    a_grid (n_a,):     the non-uniform asset grid, curved to put more points near the
                       borrowing constraint where the policy function changes most rapidly.

    e_grid (n_e,):     the discrete income states, in levels (not logs). 
"""        
function solve_aiyagari(p::AiyagariParams)
    (; α, δ, β, n_a, n_e, tol_eq) = p

    # Build grids. The income grid is fixed, but the asset grid's lower bound is
    # the borrowing limit, which under the :natural variation depends on equilibrium
    # prices (r, w) — so the asset grid is rebuilt per price inside K_supply.
    z_grid, P = rouwenhorst(p)
    e_grid = exp.(z_grid) #convert from AR(1) from logs back to levels
    e_min  = minimum(e_grid)

    # Aggregate labor supply: stationary mean of labor endowments
    π_stat = stationary_distributions(MarkovChain(P))[1]
    L = dot(π_stat, e_grid)  #Labor caculated from the fraction of households in each income state j in steadt state 

    # Capital demand: invert MPK = r + δ for K
    K_demand(r) = L * ((r + δ) / α)^(1 / (α - 1))

     # Firm pricing: given K (capital), return (r, w)
    function firm_prices(K)
        r = α * (K/L)^(α - 1) - δ    # MPK - depreciation
        w = (1 - α) * (K/L)^α        # MPL
        return r, w
    end

    # Household block: aggregate savings given (r, w)
    function K_supply(r, w)
        a_grid = CreateAssetGrid(p, borrowing_limit(p, w, r, e_min))
        _, g_idx, _ = solve_vfi(a_grid, e_grid, P, r, w, p)
        μ = solve_distribution(g_idx, P, p)
        return sum(a_grid[g_idx[i, j]] * μ[i, j] for i in 1:n_a, j in 1:n_e) #asset level * the density in that asset income pair for all pairs
    end

    # Root solve over r: find r* such that K_supply(r*) == K_demand(r*).
    r_low = -δ + 1e-4         # K_demand → ∞ as r → −δ, so excess < 0 at this floor
    r_cap = 1 / β - 1 - 1e-6  # hard ceiling: VFI needs β(1+r) < 1 to contract; at r = 1/β−1 saving is unbounded

    function excess(r, _)
        Kd = K_demand(r)
        _, w = firm_prices(Kd)
        Ks = K_supply(r, w)
        return Ks - Kd
    end

    # Start from a conservative ceiling (keeps the endpoint VFI fast), then widen it toward the
    # RA rate only if it doesn't yet bracket r*. When the precautionary motive is weak (low σ, ρ,
    # or γ — or the loose :natural limit) the economy is near complete markets and r* sits just
    # below 1/β−1, above the conservative ceiling. K_supply explodes as r → 1/β−1, so once r_high
    # is close enough excess(r_high) > 0 and [r_low, r_high] brackets r*.
    r_high = 1 / β - 1 - (p.borrow_limit == :natural ? 0.001 : 0.01)
    while excess(r_high, nothing) < 0 && r_high < r_cap
        r_high = min(r_cap, (r_high + (1 / β - 1)) / 2)   # halve the remaining gap to the RA rate
    end

    prob = IntervalNonlinearProblem(excess, (r_low, r_high))
    sol  = NonlinearSolve.solve(prob)
    println("Solver retcode: ", sol.retcode)
    r_eq = sol.u
    K_eq = K_demand(r_eq)
    _, w_eq = firm_prices(K_eq)

    # solve at equilibrium prices (rebuild the grid at the equilibrium borrowing limit)
    a_grid = CreateAssetGrid(p, borrowing_limit(p, w_eq, r_eq, e_min))
    V, g_idx, c = solve_vfi(a_grid, e_grid, P, r_eq, w_eq, p)
    μ = solve_distribution(g_idx, P, p)

    println("\n=== Equilibrium Results ===")
    println("  r* = $r_eq")
    println("  w* = $w_eq")
    println("  K* = $K_eq")
    println("  L  = $L")

    return (r=r_eq, w=w_eq, K=K_eq, L=L,
            V=V, g_idx=g_idx, c=c, μ=μ,
            a_grid=a_grid, e_grid=e_grid)
end

# Run
p = AiyagariParams()
result = solve_aiyagari(p)