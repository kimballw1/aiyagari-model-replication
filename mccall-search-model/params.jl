# =============================================================================
# params.jl
# =============================================================================

"""
McCallModel

All economic and numerical parameters of the McCall (1970) sequential-search
model and its modern extensions. Default values are provided but can be
overridden, e.g. `McCallModel(c = 25.0, α = 0.10)`.

The worker is risk averse with CRRA flow utility `u(x) = x^{1−γ}/(1−γ)` (and
`u(x) = log x` at `γ = 1`); set `γ = 0` to recover McCall's original
risk-neutral worker, for whom utility is income itself.

Wage offers are drawn i.i.d. from a discretized log-normal distribution
(`ln w ~ N(μ, σ²)`), the standard empirical wage-offer distribution. The
persistent-offer extension (`correlated.jl`) instead lets the *log* offer follow
an AR(1) with autocorrelation `ρ`.
"""
@kwdef struct McCallModel
    # preferences
    β::Float64 = 0.96     # discount factor (annual ⇒ patient worker)
    γ::Float64 = 2.0      # CRRA risk aversion; γ = 0 ⇒ risk-neutral (McCall 1970)

    # labor market
    c::Float64 = 25.0     # flow value of unemployment (UI benefit + home production)
    α::Float64 = 0.05     # job-separation (layoff) rate; α = 0 ⇒ permanent jobs

    # wage-offer distribution: ln w ~ N(μ, σ²)
    μ::Float64 = 3.6      # mean of log wage  (e^3.6 ≈ 36.6)
    σ::Float64 = 0.45     # std. dev. of log wage (offer dispersion)
    ρ::Float64 = 0.90     # offer autocorrelation (persistent-offer variant only)

    # numerical parameters
    n_w::Int = 200        # number of grid points discretizing the offer dist.
    m::Float64 = 4.0      # grid half-width in std. devs. (covers ±4σ ≈ 99.99%)
    tol::Float64 = 1e-8   # convergence tolerance for the fixed-point iterations
    max_iter::Int = 5_000 # iteration cap for the value/reservation-wage solvers
end
