# Replications of Dynamic Economic Models

A big part of my interest in dynamic programming is reading and understanding the work of people much smarter than me. This growing repository is where I do that. Each folder holds one model: the replication code, the figures and analysis it produces, and a model walkthrough.

Everything is written in Julia. Each model is self-contained, and its README explains the model, the solution method, and how to run the code.

## Models so far

| Model | Paper | What's inside |
|---|---|---|
| [Aiyagari](Aiyagari%281984%29) | Aiyagari (1994), "Uninsured Idiosyncratic Risk and Aggregate Saving" | Heterogeneous agents with uninsurable income risk and a borrowing constraint; stationary general equilibrium via value function iteration and capital-market clearing |
| [McCall job search](mccall-search-model) | McCall (1970), "Economics of Information and Job Search" | The reservation wage computed two independent ways, plus separations, risk aversion, persistent offers, and simulated unemployment durations |
| [Rust engine replacement](rust-engine-replacement) | Rust (1987), "Optimal Replacement of GMC Bus Engines" | Solves and estimates the model: nested fixed-point (NFXP) maximum likelihood, a Monte-Carlo study, and a counterfactual policy experiment |
| [Real business cycle](rbc-model) | Kydland & Prescott (1982), "Time to Build and Aggregate Fluctuations" | The stochastic growth model solved by value function iteration, calibrated to quarterly U.S. data with an HP-filtered moment scorecard |


## Planned additions

- Sovereign default
- New Keynesian model
- HANK and RANK