// worker.js — runs the Aiyagari solve off the main thread ("in the background")
// so the UI never freezes during VFI / market clearing. Streams progress back.
importScripts("aiyagari.js");

self.onmessage = (e) => {
  const params = e.data.params || {};
  try {
    const res = self.Aiyagari.solveAiyagari(params, (prog) => {
      self.postMessage({ type: "progress", prog });
    });
    const an = self.Aiyagari.analyze(res);

    // Ergodicity: forward-iterate three extreme starts toward the stationary mu.
    // T is long enough that even the slow asset-accumulation mode drives every
    // start down to the numerical floor (μ*'s own tolerance), so on the log plot
    // the three lines visibly flatten and meet at the same destination.
    // Animation frames are only collected over the first 180 steps (the visually
    // interesting transient), and only for the "poor" start.
    const T = 1200;
    const last = res.a_grid.length - 1, lastE = res.p.n_e - 1;
    const poor = self.Aiyagari.ergodicForward(res, { i: 0, j: 0 }, T, res.mu, 3, 180);
    const rich = self.Aiyagari.ergodicForward(res, { i: last, j: lastE }, T, res.mu, 1, 0);
    const mid  = self.Aiyagari.ergodicForward(res, { i: (last >> 1), j: (lastE >> 1) }, T, res.mu, 1, 0);

    // Slim payload: strip the big V; keep what the charts need.
    const n_e = res.p.n_e;
    const policy_savings = [], policy_cons = [];
    for (let j = 0; j < n_e; j++) {
      const sav = [], con = [];
      for (let i = 0; i < res.a_grid.length; i++) {
        sav.push(res.a_grid[res.g[i][j]]);
        con.push(res.c[i][j]);
      }
      policy_savings.push(sav);
      policy_cons.push(con);
    }

    self.postMessage({
      type: "done",
      payload: {
        p: res.p,
        r: res.r, w: res.w, K: res.K, L: res.L, r_RA: res.r_RA,
        a_grid: res.a_grid, e_grid: res.e_grid, z: res.z,
        P: res.P, pi_stat: res.pi_stat,
        analytics: an,
        policy_savings, policy_cons,
        ergodic: {
          poor: poor.dist, rich: rich.dist, mid: mid.dist,
          frames_poor: poor.frames, frames_rich: rich.frames,
        },
      },
    });
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.stack || err) });
  }
};
