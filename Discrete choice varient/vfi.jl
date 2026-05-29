# =============================================================================
# vfi.jl - Discrete Choice Value Function Iteration
# =============================================================================

""" 
Goal: Given prices (r,w) and grids, find:
     V[i,j] - value function: lifetime utality at state (asset i, income j)
     g_idx[i,j] - policy function: optimal next-period esset INDEX at each state 

Bellman equation at each state (a_i, e_j):
     V(a_i, e_j) = max over a_k { u(c) + β * E[V(a_k, e')] }
     where c = (1+r) * a_i + w * e_j - a_k (budget constraint)
     and E[V(a_k, e')] = dot(P[j,:], V[a_k, :]) (expectation over next period's income)

"""

function solve_vfi(a_grid, e_grid, P, r, w, p::AiyagariParams)
    (; β, y, n_a, n_e, tol_vfi, max_iter_vfi, howard_every, howard_steps) = p

    #  Utility function — returns -Inf for c ≤ 0, never NaN
    u(c) = c > 0 ? (y == 1.0 ? log(c) : c^(1 - y) / (1 - y)) : -Inf

    #  Precompute cash-on-hand: coh[i,j] = (1+r)*a_i + w*e_j
    #  Shape: n_a × n_e
    coh = (1 + r) .* a_grid .+ (w .* e_grid')   # broadcasting: column vec * row vec

    #  Initialize value function and policy
    V = zeros(n_a, n_e)
    V_new = zeros(n_a, n_e) # max possible lifetime utility for household at asset level i and income state j
    g_idx = ones(Int, n_a, n_e) # each entry is the index onto a_grid of the optimal savings choice for a house hold with currently at asset level i 

    #  Main VFI loop
    for iter in 1:max_iter_vfi

        # Expected continuation value: EV[k,j] = sum_{j'} P[j,j'] * V[k,j']
        # Recompute at the start of every iteration after V updates
        EV = V * P'    # n_a × n_e

        #  One full Bellman sweep over all states
        for j in 1:n_e # for each income state

            k_min = 1    # monotonicity: optimal k is non-decreasing in i

            for i in 1:n_a # for each asset level

                best_val = -Inf
                best_k   = k_min

                for k in k_min:n_a # for each savings choice (next period asset index)
                    c = coh[i,j] - a_grid[k]

                    if c <= 0
                        break   # infeasible, and all higher k are also infeasible
                    end

                    val = u(c) + β * EV[k,j] #bellman 

                    if val > best_val
                        best_val = val
                        best_k = k
                    else
                        break   # concavity: past the peak, stop searching
                    end
                end

                V_new[i,j] = best_val
                g_idx[i,j] = best_k
                k_min = best_k

            end
        end


        #  Howard's policy improvement
        #  Every howard_every iterations, evaluate the current policy
        #  for howard_steps iterations without re-optimizing
        if iter % howard_every == 0

            # Precompute consumption under current policy — fixed during Howard steps
            c_howard = zeros(n_a, n_e)
            for j in 1:n_e
                for i in 1:n_a
                    c_howard[i,j] = coh[i,j] - a_grid[g_idx[i,j]]
                end
            end

            for _ in 1:howard_steps
                EV_h = V_new * P'    # update EV with latest V

                for j in 1:n_e
                    for i in 1:n_a
                        V_new[i,j] = u(c_howard[i,j]) + β * EV_h[g_idx[i,j], j]
                    end
                end

            end
            V .= V_new  # sync before convergence check so err reflects Howard improvement, not the full jump from previous VFI iter
        end

        #  Convergence check
        err = norm(V_new - V, Inf)

        if iter % 10 == 0
            println("VFI iter $iter | error = $err")
        end

        if err < tol_vfi
            println("VFI converged in $iter iterations")
            break
        end

        V .= V_new    # update V in-place for next iteration
    end

    #  Recover consumption policy after convergence
    c = zeros(n_a, n_e)
    for j in 1:n_e
        for i in 1:n_a
            c[i,j] = coh[i,j] - a_grid[g_idx[i,j]]
        end
    end

    return V, g_idx, c #Value of each asset-income pair, 
end
