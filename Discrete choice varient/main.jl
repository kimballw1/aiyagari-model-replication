
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
solve_aiyagari

Find the stationary general equilibrium of the model by
solving for the interest rate r* where the capital market clears:
    K_supply(r) == K_demand(r)

Returns a named tuple with equilibrium prices, aggregates, and household objects.
"""
function solve_aiyagari(p::AiyagariParams)
    (; α, δ, β, n_a, n_e, tol_eq) = p

    # Build grids
    a_grid = CreateAssetGrid(p)
    z_grid, P = rouwenhorst(p)
    e_grid = exp.(z_grid) #convert from AR(1) from logs back to levels

    # Aggregate labor supply: stationary mean of labor endowments
    π_stat = stationary_distributions(MarkovChain(P))[1]
    L = dot(π_stat, e_grid)  #Labor caculated from the fraction of households in each income state j in steadt state 

    # Firm pricing: given K (capital), return (r, w)
    function firm_prices(K)
        r = α * (K/L)^(α - 1) - δ    # MPK - depreciation
        w = (1 - α) * (K/L)^α        # MPL
        return r, w
    end

    # Capital demand: invert MPK = r + δ for K
    K_demand(r) = L * ((r + δ) / α)^(1 / (α - 1))

    # Household block: aggregate savings given (r, w)
    function K_supply(r, w)
        _, g_idx, _ = solve_vfi(a_grid, e_grid, P, r, w, p)
        μ = solve_distribution(g_idx, P, p)
        return sum(a_grid[g_idx[i, j]] * μ[i, j] for i in 1:n_a, j in 1:n_e) #asset level * the density in that asset income pair for all pairs
    end

    # Root solve over r: finding r* such that K_supply(r*) == K_demand(r*)
    r_low  = -δ + 1e-4 #K_demand would go to infinity since when r = -δ the equation is 0/int
    r_high =  1 / β - 1 - 1e-4 #if above, the return on saving exceeds the households impatience so they would save forever
    # from euler equation in equillibrium β(1+r) = 1, so when we are above 1, we would save forever

    function excess(r, _)
        Kd = K_demand(r)
        _, w = firm_prices(Kd)
        Ks = K_supply(r, w)
        return Ks - Kd
    end

    prob = IntervalNonlinearProblem(excess, (r_low, r_high))
    sol  = NonlinearSolve.solve(prob)
    r_eq = sol.u
    K_eq = K_demand(r_eq)
    _, w_eq = firm_prices(K_eq)

    # solve at equilibrium prices
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
