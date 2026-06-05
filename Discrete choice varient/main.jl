
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

    # Build grids
    a_grid = CreateAssetGrid(p)
    z_grid, P = rouwenhorst(p)
    e_grid = exp.(z_grid) #convert from AR(1) from logs back to levels

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
        _, g_idx, _ = solve_vfi(a_grid, e_grid, P, r, w, p)
        μ = solve_distribution(g_idx, P, p)
        return sum(a_grid[g_idx[i, j]] * μ[i, j] for i in 1:n_a, j in 1:n_e) #asset level * the density in that asset income pair for all pairs
    end

    # Root solve over r: finding r* such that K_supply(r*) == K_demand(r*)
    r_low  = -δ + 1e-4 #K_demand would go to infinity since when r = -δ the equation is 0/int
    r_high =  1 / β - 1 - 0.01 # stay well below 1/β-1: VFI needs β(1+r) << 1 to contract fast enough
    # from euler equation in equillibrium β(1+r) = 1, so when we are above 1, we would save forever

    function excess(r, _)
        Kd = K_demand(r)
        _, w = firm_prices(Kd)
        Ks = K_supply(r, w)
        return Ks - Kd
    end

    prob = IntervalNonlinearProblem(excess, (r_low, r_high))
    sol  = NonlinearSolve.solve(prob)
    println("Solver retcode: ", sol.retcode)
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