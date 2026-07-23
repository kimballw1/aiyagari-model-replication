# =============================================================================
# simulate.jl — generate Harold Zurcher-style panel data
# =============================================================================

using Random
using Distributions: Categorical

"""
    simulate_rust(P_rep, θ₃, S, T; seed = 42) -> (x, d)

Simulate a single bus for `T` periods under the replacement policy `P_rep`.
Starting from a new engine (state 1), each period: draw the replacement decision
`d_t` as a Bernoulli with success probability `P_rep[x_t]`, then advance mileage
by `Δ ∈ {0,1,2}` drawn from `θ₃`, from the reset state if it replaced and from
the current state otherwise. The mileage path is the famous sawtooth: a slow
climb punctuated by sharp drops at replacement.

# Returns
- `x::Vector{Int}` — mileage-state index each period.
- `d::Vector{Int}` — replacement decisions (`0` keep, `1` replace).
"""
function simulate_rust(P_rep::Vector{Float64}, θ₃::Vector{Float64}, S::Int, T::Int; seed = 42)
    rng = MersenneTwister(seed)
    x = zeros(Int, T)
    d = zeros(Int, T)
    x[1] = 1
    incr = Categorical(θ₃)
    for t in 1:T
        d[t] = rand(rng) < P_rep[x[t]] ? 1 : 0
        if t < T
            Δ = rand(rng, incr) - 1
            x[t+1] = d[t] == 1 ? min(1 + Δ, S) : min(x[t] + Δ, S)
        end
    end
    return x, d
end
