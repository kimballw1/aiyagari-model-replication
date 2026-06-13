# The Aiyagari Model as a Markov Chain — Interactive Solver

An interactive web page that solves the Aiyagari (1994) incomplete-markets model
in the browser and visualizes it as a finite Markov chain. A household faces an
uninsurable AR(1) income shock, discretized with the Rouwenhorst method; its
saving rule makes the pair (wealth, income) a Markov chain whose stationary
distribution is the economy's cross-section of wealth. Prices adjust until the
capital market clears.

## How to open

**Double-click `index.html`.** It opens in any modern web browser — no
installation and no server.

The page needs an internet connection only to load the plotting library
(Plotly, from a CDN). All of the economics — the value-function iteration, the
stationary distribution, and the search for the market-clearing interest rate —
is computed locally on your machine.

## What you can do

- Move the sliders (β, γ, ρ, σ, and the grid sizes) and press **Solve
  equilibrium** to re-solve the whole model.
- Read the equilibrium interest rate, capital, wage, and wealth Gini at the top.
- Explore the figures: the income transition matrix and its eigenvalues, the
  convergence of any starting population to the stationary distribution, the
  stationary wealth distribution and Lorenz curve, and the household policy
  functions.
- Section 5 of the page explains what each figure shows, what the interest rate
  means (including why it can be negative), how each parameter moves the
  equilibrium, and lists references.

## Files

- `index.html` — the complete, self-contained page (open this).
- `source/` — the same model written as separate, commented files
  (`aiyagari.js` is the model, `app.js` the interface, `worker.js` an optional
  background-thread version). These are for reading the code; to run that
  version, serve the folder over a local web server, e.g.
  `python3 -m http.server` inside `source/`, then open `http://localhost:8000`.

## Method

The solver ports a Julia reference implementation to JavaScript: Rouwenhorst
discretization of the AR(1) income process, value-function iteration with Howard
policy improvement, a stationary-distribution solve, and a bisection on the
interest rate to clear the capital market. It was cross-checked against the Julia
output at the default parameters (r* ≈ 0.025, K* ≈ 7.58).
