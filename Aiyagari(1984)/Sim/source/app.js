// app.js — UI orchestration: read controls, run the solver in a Web Worker,
// render every figure with Plotly. Stochastics-forward presentation.

const PLOT_BG = "#ffffff", GRID = "#e9e7df", INK = "#1b1b1f", MUTED = "#71717a";
const ACCENT = "#2f5d86", ACCENT2 = "#b07d3f", GOOD = "#517a5d";
const HEAT = [[0, "#f4f3ee"], [1, "#2f5d86"]];  // restrained light→blue heatmap
const FONT = { family: "Inter,-apple-system,Segoe UI,sans-serif", color: INK, size: 12 };
const baseLayout = (over = {}) => ({
  paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG, font: FONT,
  margin: { l: 52, r: 16, t: 10, b: 44 }, showlegend: false,
  xaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: GRID },
  yaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: GRID },
  ...over,
});
const CFG = { displayModeBar: false, responsive: true };
const $ = (id) => document.getElementById(id);

// single-hue palette for income states (light slate → deep blue), low → high income
function palette(n) {
  const stops = [[168,192,212],[96,138,170],[47,93,134],[28,58,86]];
  const lerp = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
  const out = [];
  for (let k = 0; k < n; k++) {
    const x = (n === 1) ? 0 : k / (n - 1) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const c = lerp(stops[i], stops[i + 1], x - i);
    out.push(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  return out;
}

// ---- uniform-bin histogram helpers --------------------------------------
// The asset grid is curved (u^E): the first ~20 nodes sit inside [0, 0.05],
// so plotting raw mass *per node* smears the constrained pile across many
// zero-width bars and over-weights the sparse tail. Rebinning the stationary
// mass onto fixed-width wealth bins recovers the true right-skewed shape.
function makeBins(lo, hi, n) {
  const w = (hi - lo) / n, edges = [], ctr = [];
  for (let i = 0; i <= n; i++) edges.push(lo + i * w);
  for (let i = 0; i < n; i++) ctr.push(lo + (i + 0.5) * w);
  return { edges, ctr, w, lo, hi, n };
}
function rebin(a, mass, bins) {
  const y = new Array(bins.n).fill(0);
  for (let i = 0; i < a.length; i++) {
    if (mass[i] <= 0) continue;
    let b = Math.floor((a[i] - bins.lo) / bins.w);
    if (b < 0) b = 0;
    if (b >= bins.n) continue;   // let the thin tail beyond the display cap fade out
    y[b] += mass[i];
  }
  return y;
}

// ---- controls -> params
const fmt = { beta: 3, gamma: 1, rho: 2, sigma: 2, n_e: 0, n_a: 0 };
function bindControls() {
  for (const id of Object.keys(fmt)) {
    const el = $(id), out = $(id + "_v");
    const sync = () => { out.textContent = Number(el.value).toFixed(fmt[id]); };
    el.addEventListener("input", sync); sync();
  }
}
function readParams() {
  return {
    beta: +$("beta").value, gamma: +$("gamma").value, rho: +$("rho").value,
    sigma: +$("sigma").value, n_e: +$("n_e").value, n_a: +$("n_a").value,
    borrow_limit: $("borrow").value,   // "natural" or "adhoc"
  };
}
// Households hold negative wealth only under the natural limit. The standard
// Gini/Lorenz construction is not meaningful once wealth can be negative, so we
// suppress the Gini in that case and say so.
const hasBorrowers = (d) => d.a_grid[0] < -1e-9;

// ---- worker lifecycle
let worker = null, animTimer = null;
function solve() {
  if (worker) worker.terminate();
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  worker = new Worker("worker.js");
  const btn = $("solve"); btn.disabled = true;
  setStatus("solving — running VFI in a background thread…"); setProgress(0.08);

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") {
      if (m.prog.phase === "bracket") { setStatus("bracketing r*…"); setProgress(0.2); }
      else if (m.prog.phase === "bisect") {
        setProgress(0.3 + 0.6 * Math.min(1, m.prog.iter / 30));
        setStatus(`market clearing — r = ${(m.prog.r * 100).toFixed(3)}%  (excess K = ${m.prog.excess.toFixed(3)})`);
      }
    } else if (m.type === "done") {
      render(m.payload); setProgress(1); setStatus("solved ✓  equilibrium found"); btn.disabled = false;
      setTimeout(() => setProgress(0), 900);
    } else if (m.type === "error") {
      setStatus("error: " + m.message); btn.disabled = false;
    }
  };
  worker.postMessage({ params: readParams() });
}
const setStatus = (t) => ($("status").textContent = t);
const setProgress = (f) => ($("progressfill").style.width = (f * 100) + "%");

// ---- master render
let LAST = null;
function render(d) {
  LAST = d;
  const an = d.analytics;
  // cards
  $("c_r").textContent = (d.r * 100).toFixed(3) + "%";
  $("c_wedge").textContent = (an.wedge * 100).toFixed(2) + " pp";
  $("c_K").textContent = d.K.toFixed(3);
  $("c_w").textContent = d.w.toFixed(3);
  $("c_gini").textContent = hasBorrowers(d) ? "n/a" : an.G_w.toFixed(3);

  // Shared uniform bins for the wealth density + mixing animation, so the
  // animated frames and the stationary target are directly comparable.
  const i995 = an.cdf.findIndex((x) => x >= 0.995);
  const dispMax = Math.min(d.p.a_max, d.a_grid[i995 >= 0 ? i995 : d.a_grid.length - 1]);
  d._bins = makeBins(d.a_grid[0], Math.max(dispMax, d.a_grid[0] + 1), 60);

  drawTransition(d);
  drawEigs(d, an);
  drawPi(d);
  drawIncomeStates(d);
  drawErgodic(d);
  drawAnimFrame(d, 0);
  drawWealth(d);
  drawLorenz(d, an);
  drawPolicies(d);
}

// ---- 1. stochastic core
function drawTransition(d) {
  const labels = d.P.map((_, j) => "s" + (j + 1));
  Plotly.react("fig_P", [{
    z: d.P, x: labels, y: labels, type: "heatmap", colorscale: HEAT,
    zmin: 0, zmax: Math.max(...d.P.flat()),
    colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: MUTED } },
    hovertemplate: "P(%{y}→%{x}) = %{z:.3f}<extra></extra>",
  }], baseLayout({
    margin: { l: 40, r: 10, t: 10, b: 40 },
    yaxis: { autorange: "reversed", gridcolor: GRID }, xaxis: { gridcolor: GRID },
  }), CFG);
}
function drawEigs(d, an) {
  const k = an.eigs.map((_, i) => i);
  Plotly.react("fig_eig", [
    { x: k, y: an.eigs, type: "bar", marker: { color: ACCENT }, name: "λ" },
    { x: k, y: k.map((i) => Math.pow(d.p.rho, i)), mode: "markers", type: "scatter",
      marker: { color: ACCENT2, size: 9, symbol: "circle-open", line: { width: 2 } }, name: "ρ^k" },
  ], baseLayout({
    showlegend: true, legend: { font: { color: MUTED }, x: 0.6, y: 1 },
    xaxis: { title: "k", gridcolor: GRID }, yaxis: { title: "eigenvalue", gridcolor: GRID, range: [0, 1.05] },
  }), CFG);
  $("spectral").innerHTML =
    `<div>2nd eigenvalue |λ₂|</div><div><b>${Math.abs(an.eigs[1]).toFixed(4)}</b> (= ρ)</div>` +
    `<div>spectral gap 1−|λ₂|</div><div><b>${an.sgap.toFixed(4)}</b></div>` +
    `<div>TV-mixing time</div><div>≤ <b>${an.t_mix}</b> steps</div>` +
    `<div>income Gini</div><div><b>${an.income_gini.toFixed(4)}</b></div>`;
}
function drawPi(d) {
  const labels = d.pi_stat.map((_, j) => "s" + (j + 1));
  Plotly.react("fig_pi", [{
    x: labels, y: d.pi_stat, type: "bar", marker: { color: GOOD },
    hovertemplate: "π(%{x}) = %{y:.4f}<extra></extra>",
  }], baseLayout({ yaxis: { title: "probability", gridcolor: GRID } }), CFG);
}
function drawIncomeStates(d) {
  Plotly.react("fig_e", [{
    x: d.e_grid.map((_, j) => j + 1), y: d.e_grid, mode: "lines+markers", type: "scatter",
    line: { color: ACCENT2, width: 2 }, marker: { size: 7, color: ACCENT2 },
  }], baseLayout({ xaxis: { title: "state", gridcolor: GRID }, yaxis: { title: "income e = exp(z)", gridcolor: GRID } }), CFG);
}

// ---- 2. ergodicity
function drawErgodic(d) {
  // Clip at μ*'s numerical noise floor (the settled tail level). Below it the
  // TV distance is meaningless and a trajectory can pass spuriously close to the
  // imperfect reference, producing a one-point downward needle on the log scale.
  const tail = [...d.ergodic.poor.slice(-80), ...d.ergodic.rich.slice(-80), ...d.ergodic.mid.slice(-80)].sort((a, b) => a - b);
  const floorVal = tail[Math.floor(tail.length / 2)] * 0.9;
  const log = (a) => a.map((x) => Math.log10(Math.max(x, floorVal) + 1e-300));
  const t = d.ergodic.poor.map((_, i) => i);
  Plotly.react("fig_erg", [
    { x: t, y: log(d.ergodic.poor), mode: "lines", line: { color: "#a8553c", width: 2 }, name: "all poor" },
    { x: t, y: log(d.ergodic.rich), mode: "lines", line: { color: ACCENT, width: 2 }, name: "all rich" },
    { x: t, y: log(d.ergodic.mid), mode: "lines", line: { color: GOOD, width: 2, dash: "dash" }, name: "midpoint" },
  ], baseLayout({
    showlegend: true, legend: { font: { color: MUTED }, x: 0.55, y: 1 },
    xaxis: { title: "iteration t", gridcolor: GRID },
    yaxis: { title: "log₁₀ d_TV(μₜ, μ*)", gridcolor: GRID },
  }), CFG);
}
function drawAnimFrame(d, fi) {
  const frames = d.ergodic.frames_poor;
  const f = frames[Math.min(fi, frames.length - 1)];
  $("anim_t").textContent = f.t;
  const bins = d._bins;
  const yt = rebin(d.a_grid, d.analytics.mu_a, bins);   // stationary target
  const yc = rebin(d.a_grid, f.marg, bins);             // current frame
  Plotly.react("fig_anim", [
    { x: bins.ctr, y: yt, type: "bar", width: bins.w * 0.92,
      marker: { color: "rgba(81,122,93,.32)" }, name: "μ*", hoverinfo: "skip" },
    { x: bins.ctr, y: yc, type: "bar", width: bins.w * 0.92,
      marker: { color: ACCENT }, name: "μₜ", hovertemplate: "a≈%{x:.2f}<br>share=%{y:.3f}<extra></extra>" },
  ], baseLayout({
    barmode: "overlay", bargap: 0,
    xaxis: { title: "wealth a", gridcolor: GRID, range: [bins.lo, bins.hi] },
    yaxis: { title: "share of households", gridcolor: GRID },
  }), CFG);
}
function playAnim() {
  if (!LAST) return;
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  let fi = 0; const n = LAST.ergodic.frames_poor.length;
  animTimer = setInterval(() => {
    drawAnimFrame(LAST, fi); fi++;
    if (fi >= n) { clearInterval(animTimer); animTimer = null; }
  }, 90);
}

// ---- 3. wealth distribution
function drawWealth(d) {
  const bins = d._bins;
  const y = rebin(d.a_grid, d.analytics.mu_a, bins);
  Plotly.react("fig_wealth", [{
    x: bins.ctr, y, type: "bar", width: bins.w * 0.92,
    marker: { color: ACCENT, line: { width: 0 } },
    hovertemplate: "a≈%{x:.2f}<br>share=%{y:.3f}<extra></extra>",
  }], baseLayout({
    bargap: 0,
    xaxis: { title: "wealth a", gridcolor: GRID, range: [bins.lo, bins.hi] },
    yaxis: { title: "share of households", gridcolor: GRID },
  }), CFG);
}
function drawLorenz(d, an) {
  Plotly.react("fig_lorenz", [
    { x: an.lorenz.F, y: an.lorenz.L, mode: "lines", line: { color: ACCENT, width: 2.5 }, name: "Lorenz" },
    { x: [0, 1], y: [0, 1], mode: "lines", line: { color: MUTED, width: 1, dash: "dash" }, name: "equality" },
  ], baseLayout({
    xaxis: { title: "cumulative population share", gridcolor: GRID, range: [0, 1] },
    yaxis: { title: "cumulative wealth share", gridcolor: GRID, range: [0, 1] },
  }), CFG);
  const giniCell = hasBorrowers(d)
    ? `<div>wealth Gini</div><div><b>n/a</b> <span style="opacity:.6">(negative wealth)</span></div>`
    : `<div>wealth Gini</div><div><b>${an.G_w.toFixed(4)}</b></div>`;
  $("ineq").innerHTML = giniCell +
    `<div>top 10% share</div><div><b>${(an.sh10 * 100).toFixed(1)}%</b></div>` +
    `<div>borrowers (a&lt;0)</div><div><b>${(d.analytics.mu_a.reduce((s, m, i) => s + (d.a_grid[i] < -1e-9 ? m : 0), 0) * 100).toFixed(1)}%</b></div>` +
    `<div>mean / median</div><div><b>${an.mean_a.toFixed(2)} / ${an.median_a.toFixed(2)}</b></div>`;
}

// ---- 4. policies
function drawPolicies(d) {
  const clr = palette(d.p.n_e);
  const sav = d.policy_savings.map((y, j) => ({
    x: d.a_grid, y, mode: "lines", line: { color: clr[j], width: 1.8 }, name: "e" + (j + 1),
  }));
  sav.push({ x: d.a_grid, y: d.a_grid, mode: "lines", line: { color: MUTED, width: 1, dash: "dash" }, name: "45°" });
  Plotly.react("fig_sav", sav, baseLayout({
    xaxis: { title: "current assets a", gridcolor: GRID }, yaxis: { title: "next assets a′", gridcolor: GRID },
  }), CFG);

  const cons = d.policy_cons.map((y, j) => ({
    x: d.a_grid, y, mode: "lines", line: { color: clr[j], width: 1.8 }, name: "e" + (j + 1),
  }));
  Plotly.react("fig_cons", cons, baseLayout({
    xaxis: { title: "current assets a", gridcolor: GRID }, yaxis: { title: "consumption c", gridcolor: GRID },
  }), CFG);
}

// ---- boot
bindControls();
$("solve").addEventListener("click", solve);
$("play").addEventListener("click", playAnim);
$("reset").addEventListener("click", () => {
  const defs = { beta: 0.96, gamma: 2, rho: 0.9, sigma: 0.2, n_e: 7, n_a: 300 };
  for (const k in defs) { $(k).value = defs[k]; $(k).dispatchEvent(new Event("input")); }
  $("borrow").value = "natural";
  solve();
});
solve(); // auto-solve baseline on load
