# =============================================================================
# grids.jl  —  curved grids for the liquid and illiquid assets
# =============================================================================
#
# Same idea as the single-asset curved grid: cluster points where the policy
# bends most.  Liquid policy bends near b_min; illiquid policy bends near 0.
# Both grids are FIXED here (returns/prices don't move the grid bounds), unlike
# the natural-borrowing-limit case in the discrete variant.
# =============================================================================

