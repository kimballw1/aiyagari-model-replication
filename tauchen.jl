
# =============================================================================
# tauchen.jl — Discretize an AR(1) into a finite-state Markov chain
# =============================================================================
#
# PURPOSE
#   Next period's log income is normally distributed, centered at the
#   conditional mean ρ·e_i (where you're headed given today's state i).
#   We chop the income axis into n_e bins of equal width d. The probability
#   of transitioning from i to j is just the area of that bell curve sitting
#   over bin j.
#
# THE FORMULA
#   Each bin has number is centered in the middle of the bin, so its edges are at ±d/2.
#   For each bin we standardize its edges into z-scores and read off the
#   normal CDF. Every term inside has the same shape:
#
#       (bin edge − conditional mean) / σ
#
#   - bin edge:         e_j ± d/2   (center of bin j, nudged out to its edge)
#   - conditional mean: ρ·e_i       (where the distribution is centered, given today's state i)
#   - σ:                std of income (ε) (not the stationary std)
#
#   Interior states  -> Φ(upper edge) − Φ(lower edge)   [mass between edges]
#   Leftmost state   ->  Φ(upper edge)                   [everything below it]
#   Rightmost state  ->  1 − Φ(lower edge)               [everything above it]
#
#   The left/right columns use only one CDF term so each row sums to exactly 1.
#
# TEST BEFORE TRUSTING
#   - sum(P, dims=2) ≈ 1 everywhere
#   - stationary dist of P has mean ≈ 1 in levels (since we exponentiate a
#     zero-mean log process) — if not, normalization bug
# =============================================================================

using Distributions


function tauchen(p::AiyagariParams)
    #Step 1: compute stationary distribtion
    σ_y = p.σ/sqrt(1 - p.ρ^2)

    #Step 2: create the grid for the state space
    e_min = -p.m * σ_y #centered around 0, so min is -m std devs
    e_max = p.m * σ_y #max is +m std devs
    e_grid = range(e_min, e_max, length=p.n_e) #evenly spaced grid points

    #step 3: compute transiition probabilites 
    P = zeros(p.n_e, p.n_e) #initialize transition matrix
    d = step(e_grid) #width of each bin (distance between grid points)

    for j in 1:p.n_e
        μ = p.ρ * e_grid[j] #conditional mean of next period's income given today's state j
        for k in 1:p.n_e
            if k == 1 #First column: probability of transitioning to the lowest income state
                P[j, k] = cdf(Normal(), (e_grid[1] + d/2 - μ) / p.σ) #cdf of upper edge of bin
            elseif k == p.n_e #Last column: probability of transitiing to the highest income state
                P[j, k] = 1 - cdf(Normal(), (e_grid[end]+ d/2 - μ) / p.σ) #cdf from lower edge of last bin
            else # Interior columns: probability of transition to interior income states
                P[j, k] = cdf(Normal(), (e_grid[k] + d/2 - μ) / p.σ) - #cdf subtracting the upper edge of the bin from the lower edge
                          cdf(Normal(), (e_grid[k] - d/2 - μ) / p.σ)
            end
        end
    end

    return e_grid, P

end 

