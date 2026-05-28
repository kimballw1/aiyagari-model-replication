
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
    a_min::Float64 = 0.0 #borrowing limit
    
    #NUMERICAL PARAMETERS (for discretization)
    a_max::Float64 = 80.0 #top of asset grid: must be high enough so no one pules up here
    n_a::Int = 500 #asset grid points: more = accruate but slower
    n_e::Int = 7 #income states approximating the AR(1)
    m::Int = 3 #Tauchen grid width in std devs (+/- covers ~99.7%)
    E::Int = 3 #exponent for asset grid spacing: higher = more points near borrowing limit
    tol_vfi::Float64 = 1e-6 #tolerance for convergence in value function iteration
    tol_dist::Float64 = 1e-6 #tolerance for convergence in distribution iteration
    tol_eq::Float64 = 1e-4 #equillibrium biscection tolerance: how close supply and demand must be to stop iterating
    max_iter_vfi::Int = 2000 #max iterations for value function iteration
    max_iter_dist::Int = 5000 #max iterations for distribution iteration
    howard_every::Int = 20 #how often to perform Howard policy iteration
    howard_steps::Int = 30 #Howard improvement iterations
end 
