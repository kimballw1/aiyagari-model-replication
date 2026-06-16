# =============================================================================
# distributions.jl - Solve the stationary distribution
# =============================================================================

""" 
solve_distribution

Goal: Given a policy function g_idx and transition matrix P, 
      find the stationary distribution of (asset, income) pairs

Output: μ[i,j] = fraction of households at asset level i and income state j in the stationary distribution

"""

function solve_distribution(g_idx, P, p::AiyagariParams;
                            tol = p.tol_dist, max_iter = p.max_iter_dist)
    (; n_a, n_e) = p

    # NOTE on the stopping rule. The loop stops on the per-step change
    # ‖μ_new − μ‖∞, not on the distance to the fixed point. On a slow-mixing
    # chain the per-step change is small simply because μ barely moves each step,
    # so a loose tol stops while μ is still far from μ*. The joint (a,e) chain
    # here mixes slowly (second eigenvalue ≈ 0.98), so the distance to μ* is
    # roughly 1/(1−|λ₂|) ≈ 50× the per-step change: tol = 1e-6 leaves μ* about
    # 5e-5 from the fixed point, which biases the wealth distribution, the Gini,
    # and the ergodicity reference. Pass a tighter tol for the final, reported
    # distribution (see solve_aiyagari). The fast 1e-6 default is fine inside the
    # equilibrium root-find, where only the mean of μ enters.

    # start with uniform distribution over states since its distrubtion across
    #(a,e) pairs we are trying to solve for, not just income states
    μ = ones(n_a, n_e) ./ (n_a * n_e)

    for iter in 1:max_iter
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
        μ = μ_new
        if iter % 10 == 0 || iter == 1
             println("Distribution iter $iter | error = $err")
        end
        if err < tol
            println("Distribution converged in $iter iterations")
            break
        end
    end

    return μ
end 

