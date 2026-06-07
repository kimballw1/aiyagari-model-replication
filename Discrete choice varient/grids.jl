
# =============================================================================
# grids.jl
# =============================================================================

"""
CreateAssetGrid(p::AiyagariParams)

Build a curved (non-uniform) asset grid.
The savings policy g(a,e) curves sharply near the borrowing constraint and is
nearly straight far from it. We use an exponential grid to cluster points 
densely near `a_min` and sparsely near `a_max`.

# Arguments
- `p::AiyagariParams`: Struct containing model parameters (requires `a_max`, `n_a`, and `E`).
- `a_min::Real`: lower bound of the grid; defaults to `p.a_min`, but can be overridden with a
  price-dependent natural borrowing limit (see `borrowing_limit`).

# Returns
- `a_grid::Vector{Float64}`: 1-D array of non-uniformly spaced asset grid points.
"""

function CreateAssetGrid(p::AiyagariParams, a_min::Real = p.a_min)
    u = LinRange(0.0, 1.0, p.n_a) #Asset grid
    a_grid = a_min .+ (p.a_max - a_min) .* u.^p.E
    return a_grid #1-D matrix of asset grid points, non-uniformly spaced
end