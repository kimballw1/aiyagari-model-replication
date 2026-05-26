
# =============================================================================
# params.jl — constuct paramters for Aiyagari model
# =============================================================================

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
end 
  