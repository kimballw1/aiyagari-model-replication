
# =============================================================================
# params.jl
# =============================================================================

"""
AiyagariParams

A struct containing all economic and numerical parameters for the Aiyagari (1994) model.
Default values are provided, but can be overridden
"""
@kwdef struct AiyagariParams
    
    #ECONOMIC PARAMTERS 
    β::Float64 = 0.96 #discount factor: how much future consumption is valued vs today
    y::Float64 = 2.0  #risk aversion: higher = more precautionary saving
    α::Float64 = 0.36 #Capital share of income (output) (Y = K^α * L^(1-α))
    δ::Float64 = 0.08 #deprecition rate: 8% of cpaital wears out per year 
    ρ::Float64 = 0.9 #income persistence: how long incomme shocks last
    σ::Float64 = 0.2 #std of stationary log income
    a_min::Float64 = 0.0 #ad-hoc borrowing limit (used when borrow_limit = :adhoc)
    borrow_limit::Symbol = :natural #:adhoc = fixed a_min (clean inequality analysis); :natural = Aiyagari limit −w·e_min/r
    b_max::Float64 = 30.0 #cap on borrowing for the natural limit (robust when solver probes r ≤ 0)
    nbl_buffer::Float64 = 1e-2 #borrow this fraction inside the natural limit so c > 0 at the constraint

    #NUMERICAL PARAMETERS (for discretization)
    a_max::Float64 = 100.0 #top of asset grid: must be high enough so no one pules up here
    n_a::Int = 1000 #asset grid points: more = accruate but slower
    n_e::Int = 20 #income states approximating the AR(1)
    m::Int = 3 #Tauchen grid width in std devs (+/- covers ~99.7%)
    E::Int = 4 #exponent for asset grid spacing: higher = more points near borrowing limit
    tol_vfi::Float64 = 1e-6 #tolerance for convergence in value function iteration
    tol_dist::Float64 = 1e-6 #per-step tolerance for the distribution iteration INSIDE the equilibrium root-find (only the mean of μ is used there, so loose is fine and fast)
    tol_dist_final::Float64 = 1e-10 #tighter tolerance for the FINAL reported μ; the loose 1e-6 stops early on the slow-mixing chain and biases the distribution/Gini (see solve_distribution)
    tol_eq::Float64 = 1e-4 #equillibrium biscection tolerance: how close supply and demand must be to stop iterating
    max_iter_vfi::Int = 2000 #max iterations for value function iteration
    max_iter_dist::Int = 5000 #max iterations for distribution iteration
    max_iter_dist_final::Int = 50000 #max iterations for the final tight distribution solve
    howard_every::Int = 20 #how often to perform Howard policy iteration
    howard_steps::Int = 30 #Howard improvement iterations
end 
