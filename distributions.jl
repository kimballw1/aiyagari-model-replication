# =============================================================================
# distributions.jl - Solve the stationary distribution
# =============================================================================

""" 
Goal: Given a policy function g_idx and transition matrix P, 
      find the stationary distribution of (asset, income) pairs

Note: this is not the simple stationary dist of income, becuase we also must simultaeously
      account for the distribution of assets across households, which is endogenously determined by the policy function g_idx.

Output: μ[i,j] = fraction of households at asset level i and income state j in the stationary distribution

"""

function solve_distribution(g_idx, P, p::AiyagariParams)
    (; n_a, n_e, tol_dist, max_iter_dist) = p

    μ = ones(n_a, n_e) ./ (n_a * n_e) # start with uniform distribution over states since its distrubtion across (a,e) pairs we are trying to solve for, not just income states

    for iter in 1:max_iter_dist
        μ_new = zeros(n_a, n_e) # new distribution after one iteration
        for i in 1:n_a
            for j in 1:n_e
                k  = g_idx[i, j] # optimal next-period asset index for household currently at (i, j)
                for j_next in 1:n_e # loop over possible next-period income states
                    μ_new[k, j_next] += P[j, j_next] * μ[i, j]
                end
            end 
        end 

        # check convergence
        err = maximum(abs.(μ_new .- μ))
        if iter % 10 == 0 || iter == 1
             println("Distribution iter $iter | error = $err")  
        end 
        if err < tol_dist
            println("Distribution converged in $iter iterations")
            break
        end

        μ = μ_new
    end

    return μ
end 

