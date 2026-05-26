
# =============================================================================
# grids.jl — Build the curved (non-uniform) asset grid
# =============================================================================
# WHY CURVED:
# The savings policy g(a,e) curves sharply near the borrowing constraint and is
# nearly straight far from it. Poor households are sensitive to small wealth
# changes (one bad shock from the constraint); rich households barely care. When
# approximating a function, you need points where it BENDS — so we want points
# dense near a_min and sparse near a_max. A uniform grid does the opposite and
# wastes points where they're not needed.
#
# HOW:
#   a_grid = a_min + (a_max - a_min) * u^E,  where u is uniform [0,1], E > 1
# Raising small numbers to a power makes them much smaller, so early points get
# crushed toward a_min (dense) while points near u=1 spread out (sparse).
# Higher E = more bunching at the constraint.
#
# a_grid[1] = a_min exactly (since 0^E = 0) — this is the borrowing constraint.
# OUTPUT: 1-D array, denser near a_min, sparser toward a_max. (not really a economic meaning)
# =============================================================================

function CreateAssetGrid(p::AiyagariParams)
    u = LinRange(0.0, 1.0, p.n_a) #Asset grid
    a_grid = p.a_min .+ (p.a_max - p.a_min) .* u.^p.E
    return a_grid #1-D matrix of asset grid points, non-uniformly spaced
end