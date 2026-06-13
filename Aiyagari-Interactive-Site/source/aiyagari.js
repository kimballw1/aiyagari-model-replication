// =============================================================================
// aiyagari.js  —  Faithful JavaScript port of the discrete-choice Aiyagari (1994)
//
// This mirrors the Julia reference implementation one-to-one:
//   rouwenhorst.jl   -> rouwenhorst()
//   grids.jl         -> createAssetGrid()
//   vfi.jl           -> solveVFI()       (Howard + monotonicity + concavity)
//   distributions.jl -> solveDistribution()
//   main.jl          -> solveAiyagari()  (market-clearing root find for r*)
//
// Everything here is pure (no DOM), so it runs identically on the main thread,
// in a Web Worker, or under Node for validation against the Julia output.
// =============================================================================

// ---- default parameters (match params.jl; n_a/a_max trimmed for interactivity)
const DEFAULTS = {
  beta: 0.96,   // discount factor
  gamma: 2.0,   // risk aversion (Julia calls this `y`)
  alpha: 0.36,  // capital share
  delta: 0.08,  // depreciation
  rho: 0.9,     // income persistence (AR(1))
  sigma: 0.2,   // std of innovations entering Rouwenhorst
  a_min: 0.0,   // ad-hoc borrowing limit (used only when borrow_limit !== "natural")
  borrow_limit: "natural", // "natural" = Aiyagari limit −w·e_min/r; "adhoc" = fixed a_min
  b_max: 30.0,  // cap on borrowing for the natural limit (keeps the grid finite when r is small)
  nbl_buffer: 1e-2, // borrow this fraction inside the natural limit so c > 0 at the constraint
  a_max: 80.0,  // top of asset grid
  n_a: 220,     // asset grid points (500 in Julia; 220 keeps the browser snappy)
  n_e: 7,       // income states
  E: 3,         // asset-grid curvature exponent
  tol_vfi: 1e-6,
  tol_dist: 1e-6,
  max_iter_vfi: 2000,
  max_iter_dist: 5000,
  howard_every: 20,
  howard_steps: 30,
};

// -----------------------------------------------------------------------------
// Rouwenhorst (1995) discretization of  z_t = rho*z_{t-1} + eps,  eps~N(0,sigma^2)
// Returns { z: log-income grid, P: n_e x n_e transition matrix }.
// Identical construction to rouwenhorst.jl.
// -----------------------------------------------------------------------------
function rouwenhorst(p) {
  const { n_e, rho, sigma } = p;
  if (n_e === 1) return { z: [0.0], P: [[1.0]] };

  const prob = (1.0 + rho) / 2.0;
  let P = [
    [prob, 1.0 - prob],
    [1.0 - prob, prob],
  ];

  for (let n = 3; n <= n_e; n++) {
    const Pn = Array.from({ length: n }, () => new Array(n).fill(0.0));
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - 1; j++) {
        Pn[i][j]         += prob * P[i][j];
        Pn[i][j + 1]     += (1 - prob) * P[i][j];
        Pn[i + 1][j]     += (1 - prob) * P[i][j];
        Pn[i + 1][j + 1] += prob * P[i][j];
      }
    }
    // normalize interior rows so each row sums to 1
    for (let i = 1; i < n - 1; i++) {
      for (let j = 0; j < n; j++) Pn[i][j] /= 2.0;
    }
    P = Pn;
  }

  const sigma_y = sigma / Math.sqrt(1.0 - rho * rho);
  const psi = Math.sqrt(n_e - 1) * sigma_y;
  const z = [];
  for (let i = 0; i < n_e; i++) z.push(-psi + (2 * psi * i) / (n_e - 1));
  return { z, P };
}

// Eigenvalues of the Rouwenhorst matrix are analytic: lambda_k = rho^k, k=0..n-1.
// (Second eigenvalue is exactly rho, so the spectral gap is 1 - rho.)
function rouwenhorstEigs(p) {
  const out = [];
  for (let k = 0; k < p.n_e; k++) out.push(Math.pow(p.rho, k));
  return out;
}

// Stationary distribution of a Markov matrix P by power iteration on P'.
function stationaryDist(P) {
  const n = P.length;
  let pi = new Array(n).fill(1.0 / n);
  for (let it = 0; it < 5000; it++) {
    const next = new Array(n).fill(0.0);
    for (let j = 0; j < n; j++)
      for (let j2 = 0; j2 < n; j2++) next[j2] += pi[j] * P[j][j2];
    let err = 0;
    for (let j = 0; j < n; j++) err = Math.max(err, Math.abs(next[j] - pi[j]));
    pi = next;
    if (err < 1e-14) break;
  }
  return pi;
}

// -----------------------------------------------------------------------------
// Curved asset grid:  a_min + (a_max - a_min) * u^E,  u in [0,1].  (grids.jl)
// -----------------------------------------------------------------------------
function createAssetGrid(p, a_min = p.a_min) {
  const { a_max, n_a, E } = p;
  const a = new Array(n_a);
  for (let i = 0; i < n_a; i++) {
    const u = i / (n_a - 1);
    a[i] = a_min + (a_max - a_min) * Math.pow(u, E);
  }
  return a;
}

// Aiyagari natural borrowing limit (only used when borrow_limit === 'natural').
function borrowingLimit(p, w, r, e_min) {
  if (p.borrow_limit !== "natural") return p.a_min;
  const phi = r > 0 ? (w * e_min) / r : Infinity;
  const b_max = p.b_max ?? 30.0;
  const buf = p.nbl_buffer ?? 1e-2;
  return -Math.min(b_max, (1 - buf) * phi);
}

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

// -----------------------------------------------------------------------------
// Value Function Iteration (vfi.jl): Howard improvement + monotonicity +
// concavity early-exit. Stores V, policy indices g_idx, and consumption c.
// V is laid out as V[i][j]; EV[k][j] = sum_j' P[j][j'] V[k][j'].
// -----------------------------------------------------------------------------
function solveVFI(a_grid, e_grid, P, r, w, p) {
  const { gamma, n_e } = p;
  const n_a = a_grid.length;
  const u = (c) =>
    c > 0 ? (gamma === 1.0 ? Math.log(c) : Math.pow(c, 1 - gamma) / (1 - gamma)) : -Infinity;

  // cash-on-hand coh[i][j] = (1+r) a_i + w e_j
  const coh = Array.from({ length: n_a }, (_, i) => {
    const row = new Float64Array(n_e);
    for (let j = 0; j < n_e; j++) row[j] = (1 + r) * a_grid[i] + w * e_grid[j];
    return row;
  });

  let V = Array.from({ length: n_a }, () => new Float64Array(n_e));
  let Vn = Array.from({ length: n_a }, () => new Float64Array(n_e));
  const g = Array.from({ length: n_a }, () => new Int32Array(n_e));

  const EV = (Varr) => {
    // EV[k][j] = sum_{j'} V[k][j'] P[j][j']
    const out = Array.from({ length: n_a }, () => new Float64Array(n_e));
    for (let k = 0; k < n_a; k++) {
      const Vk = Varr[k], Ok = out[k];
      for (let j = 0; j < n_e; j++) {
        let s = 0;
        for (let j2 = 0; j2 < n_e; j2++) s += Vk[j2] * P[j][j2];
        Ok[j] = s;
      }
    }
    return out;
  };

  let iters = 0;
  for (let iter = 1; iter <= p.max_iter_vfi; iter++) {
    iters = iter;
    const ev = EV(V);
    for (let j = 0; j < n_e; j++) {
      let kmin = 0; // monotonicity: optimal k non-decreasing in i
      for (let i = 0; i < n_a; i++) {
        let best = -Infinity, bestk = kmin;
        const cij = coh[i][j];
        for (let k = kmin; k < n_a; k++) {
          const c = cij - a_grid[k];
          if (c <= 0) break;                 // infeasible (and higher k too)
          const val = u(c) + p.beta * ev[k][j];
          if (val > best) { best = val; bestk = k; }
          else break;                        // concavity: past the peak
        }
        Vn[i][j] = best;
        g[i][j] = bestk;
        kmin = bestk;
      }
    }

    // Howard policy improvement every `howard_every` iterations
    if (iter % p.howard_every === 0) {
      const ch = Array.from({ length: n_a }, (_, i) => {
        const row = new Float64Array(n_e);
        for (let j = 0; j < n_e; j++) row[j] = coh[i][j] - a_grid[g[i][j]];
        return row;
      });
      for (let h = 0; h < p.howard_steps; h++) {
        const evh = EV(Vn);
        for (let j = 0; j < n_e; j++)
          for (let i = 0; i < n_a; i++)
            Vn[i][j] = u(ch[i][j]) + p.beta * evh[g[i][j]][j];
      }
    }

    let err = 0;
    for (let i = 0; i < n_a; i++)
      for (let j = 0; j < n_e; j++) err = Math.max(err, Math.abs(Vn[i][j] - V[i][j]));

    const tmp = V; V = Vn; Vn = tmp;        // swap (Vn becomes scratch)
    if (err < p.tol_vfi) break;
  }

  const c = Array.from({ length: n_a }, (_, i) => {
    const row = new Float64Array(n_e);
    for (let j = 0; j < n_e; j++) row[j] = coh[i][j] - a_grid[g[i][j]];
    return row;
  });
  return { V, g, c, iters };
}

// -----------------------------------------------------------------------------
// Stationary distribution over (a,e) given policy g and income matrix P.
// (distributions.jl) — push-forward iteration to a fixed point.
// -----------------------------------------------------------------------------
function solveDistribution(g, P, p) {
  const n_e = p.n_e, n_a = g.length;
  let mu = Array.from({ length: n_a }, () => new Float64Array(n_e).fill(1 / (n_a * n_e)));
  for (let iter = 0; iter < p.max_iter_dist; iter++) {
    const mn = Array.from({ length: n_a }, () => new Float64Array(n_e));
    for (let i = 0; i < n_a; i++)
      for (let j = 0; j < n_e; j++) {
        const k = g[i][j], m = mu[i][j];
        if (m === 0) continue;
        for (let j2 = 0; j2 < n_e; j2++) mn[k][j2] += P[j][j2] * m;
      }
    let err = 0;
    for (let i = 0; i < n_a; i++)
      for (let j = 0; j < n_e; j++) err = Math.max(err, Math.abs(mn[i][j] - mu[i][j]));
    mu = mn;
    if (err < p.tol_dist) break;
  }
  return mu;
}

// -----------------------------------------------------------------------------
// General equilibrium: find r* such that household capital supply = firm demand.
// Mirrors main.jl: K_demand from inverted MPK, firm_prices, r_high widening,
// then bisection on excess(r) over the bracket.
// -----------------------------------------------------------------------------
function solveAiyagari(params = {}, onProgress = () => {}) {
  const p = { ...DEFAULTS, ...params };
  const { z, P } = rouwenhorst(p);
  const e_grid = z.map(Math.exp);          // levels
  const e_min = Math.min(...e_grid);

  const pi_stat = stationaryDist(P);
  const L = dot(pi_stat, e_grid);

  const K_demand = (r) => L * Math.pow((r + p.delta) / p.alpha, 1 / (p.alpha - 1));
  const firmPrices = (K) => ({
    r: p.alpha * Math.pow(K / L, p.alpha - 1) - p.delta,
    w: (1 - p.alpha) * Math.pow(K / L, p.alpha),
  });

  const K_supply = (r, w) => {
    const a_grid = createAssetGrid(p, borrowingLimit(p, w, r, e_min));
    const { g } = solveVFI(a_grid, e_grid, P, r, w, p);
    const mu = solveDistribution(g, P, p);
    let K = 0;
    for (let i = 0; i < a_grid.length; i++)
      for (let j = 0; j < p.n_e; j++) K += a_grid[g[i][j]] * mu[i][j];
    return K;
  };

  const excess = (r) => {
    const Kd = K_demand(r);
    const { w } = firmPrices(Kd);
    return K_supply(r, w) - Kd;
  };

  const r_RA = 1 / p.beta - 1;
  const r_cap = r_RA - 1e-6;
  const r_low = -p.delta + 1e-4;
  let r_high = r_RA - (p.borrow_limit === "natural" ? 0.001 : 0.01);
  onProgress({ phase: "bracket", r_high });
  while (excess(r_high) < 0 && r_high < r_cap) {
    r_high = Math.min(r_cap, (r_high + r_RA) / 2);
    onProgress({ phase: "bracket", r_high });
  }

  // bisection
  let lo = r_low, hi = r_high, flo = excess(lo), r_eq = 0.5 * (lo + hi);
  for (let it = 0; it < 60; it++) {
    r_eq = 0.5 * (lo + hi);
    const f = excess(r_eq);
    onProgress({ phase: "bisect", iter: it, r: r_eq, excess: f });
    if (Math.abs(f) < 1e-5 || hi - lo < 1e-7) break;
    if (Math.sign(f) === Math.sign(flo)) { lo = r_eq; flo = f; }
    else hi = r_eq;
  }

  const K_eq = K_demand(r_eq);
  const { w: w_eq } = firmPrices(K_eq);
  const a_grid = createAssetGrid(p, borrowingLimit(p, w_eq, r_eq, e_min));
  const { V, g, c } = solveVFI(a_grid, e_grid, P, r_eq, w_eq, p);
  // Final μ* to a much tighter tolerance than the equilibrium-loop solves. The
  // joint chain mixes slowly (|λ2|≈0.98), so the default per-step-change stop at
  // 1e-6 halts early, leaving μ* ~1e-3 from the true fixed point. That biases the
  // displayed distribution AND makes the ergodicity trajectories pass *through*
  // the imperfect reference (a spurious dip). Iterating to 1e-11 fixes both.
  const mu = solveDistribution(g, P, { ...p, tol_dist: 1e-11, max_iter_dist: 20000 });

  return { p, r: r_eq, w: w_eq, K: K_eq, L, r_RA,
           z, P, pi_stat, e_grid, a_grid, V, g, c, mu };
}

// -----------------------------------------------------------------------------
// Post-solve analytics (analyze.jl): Lorenz/Gini, top shares, wealth moments,
// Markov spectral summary, ergodic forward iteration & convergence series.
// -----------------------------------------------------------------------------
function lorenzPts(vals, wts) {
  const ord = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
  const v = ord.map((i) => vals[i]);
  const wSum = wts.reduce((s, x) => s + x, 0);
  const w = ord.map((i) => wts[i] / wSum);
  const F = [0], L = [0];
  let denom = 0;
  for (let k = 0; k < v.length; k++) denom += v[k] * w[k];
  let cf = 0, cl = 0;
  for (let k = 0; k < v.length; k++) {
    cf += w[k]; cl += (v[k] * w[k]) / denom;
    F.push(cf); L.push(cl);
  }
  return { F, L };
}
function gini(vals, wts) {
  const { F, L } = lorenzPts(vals, wts);
  let area = 0;
  for (let k = 1; k < F.length; k++) area += (F[k] - F[k - 1]) * (L[k] + L[k - 1]) / 2;
  return 1 - 2 * area;
}

function analyze(res) {
  const { p, mu, a_grid, e_grid, pi_stat, P, rho } = res;
  const n_a = a_grid.length, n_e = p.n_e;

  // marginal wealth distribution
  const mu_a = new Array(n_a).fill(0);
  for (let i = 0; i < n_a; i++) for (let j = 0; j < n_e; j++) mu_a[i] += mu[i][j];
  const cdf = []; let acc = 0;
  for (let i = 0; i < n_a; i++) { acc += mu_a[i]; cdf.push(acc); }

  const mean_a = dot(a_grid, mu_a);
  const median_a = a_grid[cdf.findIndex((x) => x >= 0.5)];
  let var_a = 0;
  for (let i = 0; i < n_a; i++) var_a += mu_a[i] * (a_grid[i] - mean_a) ** 2;
  const std_a = Math.sqrt(var_a);

  // joint Gini
  const vals = [], wts = [];
  for (let j = 0; j < n_e; j++) for (let i = 0; i < n_a; i++) { vals.push(a_grid[i]); wts.push(mu[i][j]); }
  const G_w = gini(vals, wts);

  const p90 = cdf.findIndex((x) => x >= 0.9);
  const p99 = cdf.findIndex((x) => x >= 0.99);
  const share = (from) => { let s = 0; for (let i = from; i < n_a; i++) s += a_grid[i] * mu_a[i]; return s / mean_a; };
  const sh10 = share(p90), sh1 = share(p99);

  // Markov spectral summary (eigenvalues = rho^k analytically)
  const eigs = rouwenhorstEigs(p);
  const lam2 = Math.abs(eigs[1] ?? 0);
  const sgap = 1 - lam2;
  const t_mix = Math.ceil(Math.log(n_e / 0.02) / -Math.log(lam2));
  const income_gini = gini(e_grid, pi_stat);

  const wedge = res.r_RA - res.r;

  const { F, L } = lorenzPts(a_grid, mu_a);

  return {
    mu_a, cdf, mean_a, median_a, std_a, cv: std_a / mean_a, G_w,
    sh10, sh1, eigs, sgap, t_mix, income_gini, wedge,
    lorenz: { F, L },
  };
}

// Ergodicity: push a point-mass forward under (policy, P); record sup-norm
// distance to mu_star and snapshots of the marginal wealth dist for animation.
function ergodicForward(res, start, T, mu_star, snapEvery = 4, frameUntil = Infinity) {
  const { g, P, p, a_grid } = res;
  const n_a = a_grid.length, n_e = p.n_e;
  let mu = Array.from({ length: n_a }, () => new Float64Array(n_e));
  mu[start.i][start.j] = 1.0;
  const dist = [], frames = [];
  for (let t = 0; t < T; t++) {
    const mn = Array.from({ length: n_a }, () => new Float64Array(n_e));
    for (let i = 0; i < n_a; i++)
      for (let j = 0; j < n_e; j++) {
        const m = mu[i][j]; if (m === 0) continue;
        const k = g[i][j];
        for (let j2 = 0; j2 < n_e; j2++) mn[k][j2] += P[j][j2] * m;
      }
    // Total-variation distance d_TV = ½·Σ|μ_t − μ*|, the canonical metric for
    // Markov mixing (and what the reported TV-mixing time refers to). Summing all
    // components avoids the sup-norm's spurious log-scale spikes near convergence,
    // where a single max-deviation component crosses μ* and momentarily nulls.
    let d = 0;
    for (let i = 0; i < n_a; i++) for (let j = 0; j < n_e; j++) d += Math.abs(mn[i][j] - mu_star[i][j]);
    dist.push(0.5 * d);
    if (t < frameUntil && t % snapEvery === 0) {
      const marg = new Array(n_a).fill(0);
      for (let i = 0; i < n_a; i++) for (let j = 0; j < n_e; j++) marg[i] += mn[i][j];
      frames.push({ t, marg });
    }
    mu = mn;
  }
  return { dist, frames };
}

// Export for Node validation and ES-module import; attach to self for workers.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { DEFAULTS, rouwenhorst, rouwenhorstEigs, stationaryDist,
    createAssetGrid, solveVFI, solveDistribution, solveAiyagari, analyze,
    ergodicForward, lorenzPts, gini };
}
if (typeof self !== "undefined") {
  self.Aiyagari = { DEFAULTS, rouwenhorst, rouwenhorstEigs, stationaryDist,
    createAssetGrid, solveVFI, solveDistribution, solveAiyagari, analyze,
    ergodicForward, lorenzPts, gini };
}
