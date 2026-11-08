"use strict";
// Pattern Lens: plays a 4K chart and shows the classifier's per-section scores as it goes.

const $ = (id) => document.getElementById(id);
const CATS = [
  ["Chordjack", "Rice", "#e8503a"], ["Streams", "Rice", "#f08a4b"], ["Jumpstream/Handstream", "Rice", "#d9402f"],
  ["Technical", "Rice", "#f5b87a"], ["Release", "LN", "#2fb584"], ["Coordination", "LN", "#3fbfd0"],
  ["Density/Inverse", "LN", "#7fe0bb"], ["Reading", "SV", "#b680dc"], ["Memo", "SV", "#d9a2f0"],
];
const SHORT = { "Jumpstream/Handstream": "Jump/Handstream", "Density/Inverse": "LN Density/Inverse" };
const SLOTS = [["RC", "Rice", "#e8503a"], ["LN", "LN", "#2fb584"], ["HB", "Hybrid", "#f2b53c"], ["SV", "SV", "#b680dc"]];
const LOCAL = {
  jack: ["Jack", "#ff5b6b"], chordjack: ["Chordjack", "#ff2f7d"], stream: ["Stream", "#ffb03d"],
  jumpstream: ["Jumpstream", "#ef6a45"], handstream: ["Handstream", "#d9402f"], trill: ["Trill", "#c6e86b"],
  ln: ["Long note", "#2fb584"], chord: ["Chord", "#c9a8ee"], single: ["Single", "#d7dbe6"],
};
const COLS4 = ["#e7e9ee", "#6fb8ff", "#6fb8ff", "#e7e9ee"];

const S = {
  info: null, diff: 0, chart: null, audio: null, playing: false, t: 0, t0Perf: 0, tAtStart: 0,
  rate: 1, speed: 1.3, win: "notes", sv: true, color: "model", ticks: false, offset: 0,
  X: null, noteX: null, beats: null, beatX: null, rows: null, lastTickT: -1e9,
  ac: null, raf: 0, dirtyTimeline: true,
};

// ------------------------------------------------------------------ loading

function toast(msg, ms = 5000) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), ms);
}
function busy(on, text = "Analysing chart") {
  $("busy").classList.toggle("hidden", !on);
  $("busyText").textContent = text;
}
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

async function loadFile(file) {
  showPlayer();
  busy(true, "Uploading " + file.name);
  const fd = new FormData();
  fd.append("file", file);
  try { await openSession(await api("api/upload", { method: "POST", body: fd })); }
  catch (e) { busy(false); toast("Could not load the file: " + e.message); }
}
async function loadUrl(url) {
  showPlayer();
  busy(true, "Downloading the beatmapset");
  try { await openSession(await api("api/fetch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) })); }
  catch (e) { busy(false); toast("Could not load the map: " + e.message); }
}
function showPlayer() {
  $("empty").classList.add("hidden");
  $("player").classList.remove("hidden");
  resize();
}

async function openSession(info, diff) {
  pause();
  S.info = info;
  $("songTitle").textContent = `${info.artist} - ${info.title}`;
  $("songSub").textContent = `mapped by ${info.creator}` + (info.audio ? "" : " · no audio (load the .osz for sound)");
  const sel = $("diffSelect");
  sel.innerHTML = "";
  for (const d of info.diffs) {
    const o = document.createElement("option");
    o.value = d.i;
    o.textContent = `${d.version} · ${d.keys}K · ${d.notes} notes · ${Math.floor(d.length_s / 60)}:${String(d.length_s % 60).padStart(2, "0")}`;
    sel.appendChild(o);
  }
  for (const a of Object.values(S.tracks || {})) { a.pause(); a.src = ""; }
  S.tracks = {};
  S.audio = null;
  const pick = diff ?? info.diffs[0].i;
  sel.value = pick;
  await openDiff(pick);
}

// each difficulty plays the audio file its .osu names (rate edits, cut versions and so on)
function useTrack(d) {
  const k = !d ? null : d.audio !== undefined ? d.audio : S.info.audio ? 0 : null;  // older sessions: one track
  if (S.audio) S.audio.pause();
  S.audio = null;
  if (k !== null) {
    if (!S.tracks[k]) {
      const a = new Audio(`api/audio/${S.info.sid}/${k}`);
      a.preload = "auto";
      a.preservesPitch = true;
      a.addEventListener("ended", pause);
      S.tracks[k] = a;
    }
    S.audio = S.tracks[k];
  }
  $("noAudio").classList.toggle("hidden", !!S.audio);
  $("songSub").textContent = `mapped by ${S.info.creator}` + (S.audio ? "" : S.info.audio
    ? " · this difficulty's audio file isn't in the set" : " · no audio (load the .osz for sound)");
}

async function openDiff(i) {
  pause();
  S.diff = +i;
  busy(true);
  try {
    S.chart = await api(`api/chart/${S.info.sid}/${i}`);
  } catch (e) { busy(false); toast("Analysis failed: " + e.message); return; }
  busy(false);
  const d = S.info.diffs.find((x) => x.i === +i);
  useTrack(d);
  const fw = document.querySelector(".fieldwrap");
  if (d && d.has_bg) { fw.style.backgroundImage = `url(api/bg/${S.info.sid}/${i})`; fw.classList.add("bg"); }
  else { fw.style.backgroundImage = ""; fw.classList.remove("bg"); }
  history.replaceState(null, "", `?s=${S.info.sid}&d=${i}`);
  prepare();
  prepareNotes();
  buildPanel();
  S.t = S.audio ? 0 : S.chart.start - 1500;
  if (S.audio) S.audio.currentTime = 0;
  S.dirtyTimeline = true;
  draw();
  if (S.win !== "notes" && !(S.chart.analysis.windows_by_size || {})[S.win]) setWindow(S.win);
}

// ------------------------------------------------------------------ chart preparation

function prepare() {
  const c = S.chart;
  // rows for click sounds
  const rows = [];
  for (const t of c.notes.t) if (!rows.length || t - rows[rows.length - 1] > 2) rows.push(t);
  S.rows = rows;
  // beat lines
  const beats = [];
  const tp = c.timing.length ? c.timing : [[c.start, 500]];
  for (let i = 0; i < tp.length; i++) {
    const [t0, bl] = tp[i];
    const t1 = i + 1 < tp.length ? tp[i + 1][0] : c.end + 4000;
    if (bl < 20) continue;
    let k = 0;
    for (let t = t0; t < t1 && beats.length < 40000; t += bl, k++) beats.push([t, k % 4 === 0]);
  }
  // lead-in beat lines before the first timing point
  if (tp.length && tp[0][1] >= 20) for (let t = tp[0][0] - tp[0][1], k = -1; t > -3000; t -= tp[0][1], k--) beats.unshift([t, ((k % 4) + 4) % 4 === 0]);
  S.beats = beats;
  buildScroll();
}

function buildScroll() {
  const c = S.chart;
  const st = c.scroll.t, v = c.scroll.v;
  const cum = new Float64Array(st.length);
  for (let i = 1; i < st.length; i++) cum[i] = cum[i - 1] + (st[i] - st[i - 1]) * v[i - 1];
  S.X = (t) => {
    if (!S.sv) return t;
    if (t <= st[0]) return (t - st[0]) * (v[0] || 1);
    let lo = 0, hi = st.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (st[m] <= t) lo = m; else hi = m - 1; }
    return cum[lo] + (t - st[lo]) * v[lo];
  };
  S.noteX = Float64Array.from(c.notes.t, S.X);
  S.noteEX = Float64Array.from(c.notes.e, S.X);
  S.beatX = Float64Array.from(S.beats, (b) => S.X(b[0]));
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

// ---- note model: per-note probabilities, averaged over the notes just played
function noteMode() { return S.win === "notes" && !!S.NM; }
function TH() {
  const a = S.chart.analysis;
  return noteMode() ? { ...a.thresholds, ...S.NM.thresholds } : a.thresholds;
}
function prepareNotes() {
  const nm = S.chart.analysis.notes;
  S.NM = null;
  if (!nm) return;
  const raw = atob(nm.p), T = nm.tasks.length;
  const q = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) q[i] = raw.charCodeAt(i);
  const idx = {};
  nm.tasks.forEach((t, j) => { idx[t] = j; });
  // timeline: the same trailing score as the bars, sampled every 500 ms
  const c = S.chart, bins = {}, bt = [], HOP = 500;
  for (let x = Math.floor(c.start / HOP) * HOP; x <= c.end + TRAIL_MS; x += HOP) bt.push(x);
  S.NM = { q, T };
  nm.tasks.forEach((t, j) => { bins[t] = bt.map((x) => trailing(j, x)); });
  S.NM = { tasks: nm.tasks, idx, q, T, thresholds: nm.thresholds, report: nm.report, chart: nm.chart,
           bins: { t: bt, series: bins, hop: HOP, lead: 0 } };
}
// trailing average over the notes already played: weight exp(-age / TAU) over the last TRAIL_MS, so a
// section only lights up once its notes reach the hit line, and fades out over ~1 s after it ends
const TRAIL_MS = 2000, TAU_MS = 700, FULL_W = 3;
function trailing(j, t) {
  const nt = S.chart.notes.t, q = S.NM.q, T = S.NM.T;
  const i1 = lowerBound(nt, t + 30), i0 = lowerBound(nt, t - TRAIL_MS);
  if (i1 - i0 < 2) return null;
  let sw = 0, sp = 0;
  for (let i = i0; i < i1; i++) {
    const w = Math.exp(-Math.max(0, t - nt[i]) / TAU_MS);
    sw += w; sp += w * q[i * T + j];
  }
  return (sp / sw / 255) * Math.min(1, sw / FULL_W);
}
function noteScore(name, t) {
  const j = S.NM.idx[name];
  if (j === undefined) return undefined;
  return trailing(j, t);
}
function catSeries() {
  if (noteMode()) return S.NM.bins;
  const w = W();
  return { t: w.t, series: w.series, hop: w.t.length > 1 ? w.t[1] - w.t[0] : 1000, lead: lead() };
}
// strongest category of one note, for "by model" colouring
function noteCategoryName(i) {
  let best = null, br = 1;
  const th = TH();
  for (const [name] of CATS) {
    const j = S.NM.idx[name];
    if (j === undefined) continue;
    const r = S.NM.q[i * S.NM.T + j] / 255 / (th[name] ?? 0.5);
    if (r >= br) { br = r; best = name; }
  }
  return best;
}
function noteCategory(i) {
  const n = noteCategoryName(i);
  return n && CATS.find((c) => c[0] === n)[2];
}

// "strong" cut for a section: the window score that best separates labelled charts by their top section
// (calibrated for 16 s windows only; shorter windows use the chart cut + 0.2)
function peakCut(name) {
  const a = S.chart.analysis;
  if (noteMode()) return Math.min(0.95, (TH()[name] ?? 0.5) + 0.2);
  const cal = W().win_ms === 16000 && a.window_thresholds ? a.window_thresholds[name] : undefined;
  return cal ?? Math.min(0.95, (a.thresholds[name] ?? 0.5) + 0.2);
}

// the window series for the selected length; the window read at playhead t is the one that ends at t
// ([t - len, t]), so nothing is scored before it has been played
function W() {
  const a = S.chart.analysis;
  return (a.windows_by_size && a.windows_by_size[S.win]) || a.windows;  // note mode: slot from the default windows
}
function lead() { return -W().win_ms / 2; }

// window score at time t, linearly interpolated between window centres
function scoreAt(name, t) {
  if (noteMode()) { const v = noteScore(name, t); if (v !== undefined) return v; }
  const w = W(), s = w.series[name];
  t += lead();
  if (!s || !w.t.length) return null;
  const ts = w.t;
  if (t < ts[0] - 4000 || t > ts[ts.length - 1] + 4000) return null;
  const i = lowerBound(ts, t);
  if (i <= 0) return s[0];
  if (i >= ts.length) return s[ts.length - 1];
  if (ts[i] - ts[i - 1] > 3000) return t - ts[i - 1] < ts[i] - t ? s[i - 1] : s[i];
  const f = (t - ts[i - 1]) / (ts[i] - ts[i - 1]);
  return s[i - 1] * (1 - f) + s[i] * f;
}

// ------------------------------------------------------------------ panel

function buildPanel() {
  const a = S.chart.analysis;
  const thr = TH();
  const sr = $("slotRow");
  sr.innerHTML = "";
  for (const [k, name, col] of SLOTS) {
    const d = document.createElement("div");
    d.className = "slot";
    d.style.color = col;
    d.innerHTML = `<span>${name}</span><b>–</b><span class="fill"></span>`;
    d.dataset.k = k;
    sr.appendChild(d);
  }
  const bars = $("catBars");
  bars.innerHTML = "";
  for (const [name, fam, col] of CATS) {
    if (!(name in W().series) && !(noteMode() && name in S.NM.idx)) continue;
    const el = document.createElement("div");
    el.className = "bar";
    el.dataset.name = name;
    el.style.color = col;
    const pk = peakCut(name);
    const rep = noteMode() ? S.NM.report : a.report;
    el.title = `${name}: held-out chart AP ${rep[name]?.ap ?? "?"}. Lit from ${(thr[name] ?? 0.5).toFixed(2)} (the chart-level cut), strong from ${pk.toFixed(2)} (the section score that best separates labelled charts)`;
    el.innerHTML = `<span class="name"><span class="fam" style="color:${col}">${fam}</span>${SHORT[name] || name}</span>
      <span class="track"><span class="fill" style="background:${col};width:0%"></span><span class="thr" style="left:${(thr[name] ?? 0.5) * 100}%"></span><span class="thr peak" style="left:${pk * 100}%"></span></span>
      <span class="val">–</span>`;
    bars.appendChild(el);
  }
  // whole-chart summary (chart-level cuts)
  const ch = a.chart, sc = ch.scores, cthr = a.thresholds;
  const slot = SLOTS.find((s) => s[0] === ch.slot);
  const cats = ch.categories.map((n) => {
    const col = (CATS.find((c) => c[0] === n) || [0, 0, "#ccc"])[2];
    return `<span class="pill" style="color:${col}">${SHORT[n] || n}</span>`;
  }).join(" ") || '<span class="muted">none above the cut</span>';
  const tags = Object.entries(sc).filter(([k, p]) => k.startsWith("tag:") && p >= (cthr[k] ?? 0.5))
    .sort((x, y) => y[1] / cthr[y[0]] - x[1] / cthr[x[0]]).map(([k]) => `<span class="pill" style="color:#c9cfdc">${k.slice(4)}</span>`).join(" ")
    || '<span class="muted">none</span>';
  const pct = (k) => sc[k] == null ? "–" : Math.round(sc[k] * 100) + "%";
  const keys = S.chart.keys;
  $("chartSummary").innerHTML = `
    <div class="row"><span class="k">Slot</span><span class="pill" style="color:${slot[2]}">${ch.slot_label || slot[1]}</span>
      <span class="muted">${SLOTS.map((s) => `${s[1]} ${Math.round((ch.slot_probs[s[0]] || 0) * 100)}%`).join(" · ")}</span></div>
    <div class="row"><span class="k">Categories</span>${cats}</div>
    <div class="row"><span class="k">Stamina</span><span>${pct("Stamina")}</span><span class="k" style="width:auto;margin-left:12px">Tiebreaker</span><span>${pct("TB")}</span></div>
    <div class="row"><span class="k">User tags</span>${tags}</div>
    ${keys !== 4 ? `<div class="warn">This is a ${keys}K chart. The heads were tuned for 4K, so treat these scores as rough.</div>` : ""}`;
}

let lastPanel = 0;
function updatePanel(t) {
  const now = performance.now();
  if (now - lastPanel < 60) return;
  lastPanel = now;
  const a = S.chart.analysis, thr = TH();
  const m = Math.max(0, t) / 1000;
  $("clock").textContent = `${Math.floor(m / 60)}:${(m % 60).toFixed(1).padStart(4, "0")}`;
  let best = null, bestP = -1;
  const sp = {};
  for (const [k] of SLOTS) { sp[k] = scoreAt("slot:" + k, t); if (sp[k] != null && sp[k] > bestP) { bestP = sp[k]; best = k; } }
  for (const el of $("slotRow").children) {
    const p = sp[el.dataset.k];
    el.querySelector("b").textContent = p == null ? "–" : Math.round(p * 100) + "%";
    el.querySelector(".fill").style.transform = `scaleX(${p ?? 0})`;
    el.classList.toggle("top", el.dataset.k === best);
  }
  for (const el of $("catBars").children) {
    const n = el.dataset.name, p = scoreAt(n, t);
    el.querySelector(".fill").style.width = `${(p ?? 0) * 100}%`;
    el.querySelector(".val").textContent = p == null ? "–" : p.toFixed(2);
    el.classList.toggle("on", p != null && p >= (thr[n] ?? 0.5));
    el.classList.toggle("strong", p != null && p >= peakCut(n));
  }
  // tags: the ones above their cut, else the three closest
  const tags = (noteMode() ? S.NM.tasks : Object.keys(W().series)).filter((k) => k.startsWith("tag:"))
    .map((k) => [k, scoreAt(k, t)]).filter((x) => x[1] != null)
    .map(([k, p]) => [k, p, p / (thr[k] ?? 0.5)]).sort((x, y) => y[2] - x[2]);
  const on = tags.filter((x) => x[2] >= 1);
  const show = on.length ? on.slice(0, 8) : tags.slice(0, 3);
  $("tagChips").innerHTML = show.length ? show.map(([k, p, r]) =>
    `<span class="chip ${r >= 1 ? "" : "dim"}">${k.slice(4)}<span class="n">${p.toFixed(2)}</span></span>`).join("")
    : '<span class="muted" style="font-size:12px">no notes in this section</span>';
  // local note labels on screen
  const c = S.chart, x0 = S.X(t), span = visibleSpan();
  const i0 = lowerBound(S.noteX, x0), i1 = lowerBound(S.noteX, x0 + span);
  const cnt = {}, tot = i1 - i0;
  if (S.color === "model" && S.NM) {  // each note's strongest category (above its cut)
    for (let i = i0; i < i1; i++) { const n = noteCategoryName(i) || "none"; cnt[n] = (cnt[n] || 0) + 1; }
    $("localChips").innerHTML = Object.entries(cnt).sort((x, y) => y[1] - x[1]).map(([k, n]) => {
      const col = k === "none" ? "#5b6170" : CATS.find((x) => x[0] === k)[2];
      return `<span class="chip"><i style="background:${col}"></i>${k === "none" ? "no category" : SHORT[k] || k}<span class="n">${Math.round((100 * n) / tot)}%</span></span>`;
    }).join("") || '<span class="muted" style="font-size:12px">nothing on screen</span>';
    return;
  }
  for (let i = i0; i < i1; i++) cnt[c.notes.k[i]] = (cnt[c.notes.k[i]] || 0) + 1;
  $("localChips").innerHTML = Object.entries(cnt).sort((x, y) => y[1] - x[1]).map(([k, n]) =>
    `<span class="chip"><i style="background:${LOCAL[k][1]}"></i>${LOCAL[k][0]}<span class="n">${Math.round((100 * n) / tot)}%</span></span>`).join("")
    || '<span class="muted" style="font-size:12px">nothing on screen</span>';
}

// ------------------------------------------------------------------ playfield

const field = $("field"), fctx = field.getContext("2d");
const tl = $("timeline"), tctx = tl.getContext("2d");
let DPR = 1;
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  for (const cv of [field, tl]) {
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(1, Math.round(r.width * DPR));
    cv.height = Math.max(1, Math.round(r.height * DPR));
  }
  S.dirtyTimeline = true;
  if (S.chart) draw();
}
window.addEventListener("resize", resize);

function pxPerMs() { return (field.height * S.speed) / 1500; }
function visibleSpan() { return field.height / pxPerMs(); }

function drawField(t) {
  const W = field.width, H = field.height, c = S.chart, K = c.keys;
  fctx.clearRect(0, 0, W, H);
  const hasBg = document.querySelector(".fieldwrap").classList.contains("bg");
  fctx.fillStyle = hasBg ? "rgba(10,18,51,0.84)" : "#0a1233";
  fctx.fillRect(0, 0, W, H);
  const laneW = Math.min(W * 0.9 / K, 90 * DPR), x0 = (W - laneW * K) / 2;
  const hitY = H - 70 * DPR, ppm = pxPerMs(), X = S.X, xt = X(t);
  // lanes
  for (let k = 0; k < K; k++) {
    fctx.fillStyle = k % 2 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.045)";
    fctx.fillRect(x0 + k * laneW, 0, laneW, H);
  }
  // beat lines
  const b0 = lowerBound(S.beatX, xt - 80 / ppm), b1 = lowerBound(S.beatX, xt + hitY / ppm + 10);
  for (let i = b0; i < b1; i++) {
    const y = hitY - (S.beatX[i] - xt) * ppm;
    fctx.fillStyle = S.beats[i][1] ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.07)";
    fctx.fillRect(x0, y, laneW * K, S.beats[i][1] ? 2 * DPR : DPR);
  }
  // hit line and lane glow
  const glow = new Float32Array(K);
  // notes: walk back far enough to catch long notes still being held
  let i0 = lowerBound(S.noteX, xt - hitY / ppm - 400);
  const i1 = lowerBound(S.noteX, xt + hitY / ppm + 20 * DPR / ppm);
  const nt = c.notes, pad = 4 * DPR, nh = Math.max(10 * DPR, laneW * 0.24), r = 5 * DPR;
  for (let i = Math.max(0, i0 - 200); i < i1; i++) {
    const isLn = nt.e[i] > nt.t[i];
    if (!isLn && nt.t[i] < t - 25) continue;
    if (isLn && nt.e[i] < t - 120) continue;
    const k = nt.c[i], lx = x0 + k * laneW + pad, lw = laneW - 2 * pad;
    const col = S.color === "pattern" ? LOCAL[nt.k[i]][1]
      : S.color === "model" ? (S.NM ? noteCategory(i) || "#5b6170" : LOCAL[nt.k[i]][1])
      : (K === 4 ? COLS4[k] : (k % 2 ? "#6fb8ff" : "#e7e9ee"));
    let y = hitY - (S.noteX[i] - xt) * ppm;
    const dt = nt.t[i] - t;
    if (Math.abs(dt) < 45) glow[k] = Math.max(glow[k], 1 - Math.abs(dt) / 45);
    if (isLn) {
      let yEnd = hitY - (S.noteEX[i] - xt) * ppm;
      if (nt.t[i] <= t && nt.e[i] >= t) { y = hitY; glow[k] = Math.max(glow[k], 0.6); }
      fctx.fillStyle = col + "55";
      fctx.fillRect(lx + lw * 0.18, yEnd, lw * 0.64, y - yEnd);
      fctx.fillStyle = col;
      roundRect(lx + lw * 0.18, yEnd - nh * 0.25, lw * 0.64, nh * 0.5, r * 0.6);
    }
    if (y < -nh || y > H + nh) continue;
    fctx.fillStyle = col;
    roundRect(lx, y - nh, lw, nh, r);
  }
  // receptors
  for (let k = 0; k < K; k++) {
    const lx = x0 + k * laneW + pad, lw = laneW - 2 * pad;
    fctx.strokeStyle = `rgba(255,255,255,${0.25 + 0.6 * glow[k]})`;
    fctx.lineWidth = 2 * DPR;
    fctx.beginPath();
    fctx.roundRect ? fctx.roundRect(lx, hitY - nh, lw, nh, r) : fctx.rect(lx, hitY - nh, lw, nh);
    fctx.stroke();
    if (glow[k] > 0) {
      const g = fctx.createLinearGradient(0, hitY - 200 * DPR, 0, hitY);
      g.addColorStop(0, "rgba(232,80,58,0)");
      g.addColorStop(1, `rgba(232,80,58,${0.28 * glow[k]})`);
      fctx.fillStyle = g;
      fctx.fillRect(x0 + k * laneW, hitY - 200 * DPR, laneW, 200 * DPR);
    }
  }
  // live category strip along the side: which categories are "on" right now
  const thr = TH();
  let yy = 12 * DPR;
  fctx.font = `${11 * DPR}px Inter, system-ui, sans-serif`;
  fctx.textAlign = "left";
  for (const [name, , col] of CATS) {
    const p = scoreAt(name, t);
    if (p == null || p < (thr[name] ?? 0.5)) continue;
    const strong = p >= peakCut(name);
    const label = (SHORT[name] || name) + (strong ? " ●" : "");
    const w = fctx.measureText(label).width + 16 * DPR;
    fctx.fillStyle = "rgba(10,18,51,0.75)";
    roundRect(8 * DPR, yy, w, 20 * DPR, 10 * DPR);
    fctx.fillStyle = col;
    fctx.fillText(label, 16 * DPR, yy + 14 * DPR);
    yy += 25 * DPR;
  }
}

function roundRect(x, y, w, h, r) {
  fctx.beginPath();
  if (fctx.roundRect) fctx.roundRect(x, y, w, Math.max(h, 0.1), Math.min(r, Math.abs(h) / 2, w / 2));
  else fctx.rect(x, y, w, h);
  fctx.fill();
}

// ------------------------------------------------------------------ timeline

let tlCache = null;
function tlGeom() {
  const W = tl.width, H = tl.height, left = 132 * DPR, right = 10 * DPR;
  const c = S.chart, t0 = Math.min(c.start, 0), t1 = c.end + 1000;
  return { W, H, left, right, t0, t1, sx: (t) => left + ((t - t0) / (t1 - t0)) * (W - left - right) };
}
function drawTimelineBase() {
  const g = tlGeom(), c = S.chart, a = c.analysis, w = W(), L = lead();
  const off = document.createElement("canvas");
  off.width = g.W; off.height = g.H;
  const x = off.getContext("2d");
  const rowsN = CATS.length + 2;
  const rh = (g.H - 8 * DPR) / rowsN;
  x.font = `${11 * DPR}px Inter, system-ui, sans-serif`;
  x.textBaseline = "middle";
  // density strip
  const dens = new Float32Array(400);
  for (const t of c.notes.t) dens[Math.min(399, Math.max(0, Math.floor(((t - g.t0) / (g.t1 - g.t0)) * 400)))]++;
  const dm = Math.max(...dens) || 1;
  x.fillStyle = "#8b93a7";
  x.fillText("Note density", 6 * DPR, rh / 2);
  for (let i = 0; i < 400; i++) {
    const h = (dens[i] / dm) * (rh - 4 * DPR);
    const xs = g.sx(g.t0 + (i / 400) * (g.t1 - g.t0)), xe = g.sx(g.t0 + ((i + 1) / 400) * (g.t1 - g.t0));
    x.fillStyle = "rgba(231,233,238,0.55)";
    x.fillRect(xs, rh - 2 * DPR - h, Math.max(1, xe - xs), h);
  }
  // slot strip
  x.fillStyle = "#8b93a7";
  x.fillText("Slot", 6 * DPR, rh * 1.5);
  const hop = w.t.length > 1 ? w.t[1] - w.t[0] : 1000;
  for (let i = 0; i < w.t.length; i++) {
    let best = SLOTS[0], bp = -1;
    for (const s of SLOTS) { const p = w.series["slot:" + s[0]][i]; if (p > bp) { bp = p; best = s; } }
    x.globalAlpha = 0.35 + 0.65 * bp;
    x.fillStyle = best[2];
    x.fillRect(g.sx(w.t[i] - L - hop / 2), rh + 2 * DPR, Math.max(1, g.sx(w.t[i] - L + hop / 2) - g.sx(w.t[i] - L - hop / 2)), rh - 4 * DPR);
  }
  x.globalAlpha = 1;
  CATS.forEach(([name, , col], j) => {
    const y = rh * (j + 2);
    x.fillStyle = "#8b93a7";
    x.fillText(SHORT[name] || name, 6 * DPR, y + rh / 2);
    x.fillStyle = "rgba(255,255,255,0.03)";
    x.fillRect(g.left, y + 2 * DPR, g.W - g.left - g.right, rh - 4 * DPR);
    const cs = catSeries(), s = cs.series[name];
    if (!s) return;
    const thr = TH()[name] ?? 0.5;
    const lo = 0.5 * thr, hi = Math.max(peakCut(name), thr + 0.05);
    for (let i = 0; i < cs.t.length; i++) {
      if (s[i] == null || s[i] < lo) continue;
      x.globalAlpha = Math.min(1, 0.06 + 0.94 * Math.pow(Math.min(1, (s[i] - lo) / (hi - lo)), 1.5));
      x.fillStyle = col;
      const a0 = cs.t[i] - cs.lead - cs.hop / 2, a1 = cs.t[i] - cs.lead + cs.hop / 2;
      x.fillRect(g.sx(a0), y + 2 * DPR, Math.max(1, g.sx(a1) - g.sx(a0)), rh - 4 * DPR);
    }
    x.globalAlpha = 1;
  });
  // minute ticks
  x.fillStyle = "rgba(139,147,167,0.8)";
  for (let m = 0; m * 60000 < g.t1; m++) {
    const xs = g.sx(m * 60000);
    if (xs < g.left) continue;
    x.fillRect(xs, g.H - 6 * DPR, DPR, 6 * DPR);
  }
  tlCache = off;
  S.dirtyTimeline = false;
}
function drawTimeline(t) {
  if (S.dirtyTimeline || !tlCache) drawTimelineBase();
  const g = tlGeom();
  tctx.clearRect(0, 0, g.W, g.H);
  tctx.drawImage(tlCache, 0, 0);
  const win = noteMode() ? TRAIL_MS : W().win_ms;
  tctx.fillStyle = "rgba(255,255,255,0.07)";
  tctx.fillRect(g.sx(t - win), 0, g.sx(t) - g.sx(t - win), g.H);
  tctx.fillStyle = "#e8503a";
  tctx.fillRect(g.sx(t) - DPR, 0, 2 * DPR, g.H);
}
tl.addEventListener("click", (e) => {
  if (!S.chart) return;
  const r = tl.getBoundingClientRect(), g = tlGeom();
  const px = (e.clientX - r.left) * DPR;
  if (px < g.left) return;
  seek(g.t0 + ((px - g.left) / (g.W - g.left - g.right)) * (g.t1 - g.t0));
});

// ------------------------------------------------------------------ clock and transport

function now() {
  if (!S.playing) return S.t;
  if (S.audio) return S.audio.currentTime * 1000 - S.offset;
  return S.tAtStart + (performance.now() - S.t0Perf) * S.rate;
}
function audioCtx() {
  if (!S.ac) S.ac = new (window.AudioContext || window.webkitAudioContext)();
  return S.ac;
}
function click(strong) {
  const ac = audioCtx(), o = ac.createOscillator(), g = ac.createGain();
  o.frequency.value = strong ? 1500 : 1100;
  g.gain.setValueAtTime(0.12, ac.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + 0.04);
  o.connect(g).connect(ac.destination);
  o.start();
  o.stop(ac.currentTime + 0.05);
}
async function play() {
  if (!S.chart || S.playing) return;
  if (S.ticks || !S.audio) audioCtx().resume();
  if (S.audio) {
    S.audio.playbackRate = S.rate;
    S.audio.currentTime = Math.max(0, (S.t + S.offset) / 1000);
    try { await S.audio.play(); } catch (e) { toast("The browser blocked audio: " + e.message); return; }
  }
  S.t0Perf = performance.now();
  S.tAtStart = S.t;
  S.lastTickT = S.t;
  S.playing = true;
  $("playBtn").textContent = "❚❚";
  loop();
}
function pause() {
  if (!S.playing) return;
  S.t = now();
  S.playing = false;
  if (S.audio) S.audio.pause();
  $("playBtn").textContent = "▶";
  cancelAnimationFrame(S.raf);
}
function seek(t) {
  const was = S.playing;
  pause();
  S.t = Math.max(Math.min(t, S.chart.end + 1000), Math.min(S.chart.start - 3000, 0));
  draw();
  if (was) play();
}
function loop() {
  if (!S.playing) return;
  const t = now();
  if (t > S.chart.end + 2500 && !S.audio) { pause(); return; }
  if (S.ticks || !S.audio) {
    const a = lowerBound(S.rows, S.lastTickT), b = lowerBound(S.rows, t);
    if (b > a && b - a < 4) click(false);
    S.lastTickT = t;
  }
  S.t = t;
  draw(t);
  S.raf = requestAnimationFrame(loop);
}
function draw(t = S.t) {
  if (!S.chart) return;
  drawField(t);
  drawTimeline(t);
  updatePanel(t);
}

// ------------------------------------------------------------------ controls

$("playBtn").onclick = () => (S.playing ? pause() : play());
$("rateSel").onchange = (e) => { S.rate = +e.target.value; if (S.playing) { pause(); play(); } };
$("speedIn").oninput = (e) => { S.speed = +e.target.value; draw(); };
$("svChk").onchange = (e) => { S.sv = e.target.checked; buildScroll(); draw(); };
$("winSel").onchange = (e) => setWindow(e.target.value);
async function setWindow(size) {
  S.win = size;
  if (!S.chart) return;
  const a = S.chart.analysis;
  a.windows_by_size = a.windows_by_size || {};
  if (size === "notes" && !S.NM) { toast("This server has no note-level model yet; showing 8 s windows."); }
  if (size !== "notes" && !a.windows_by_size[size]) {
    const sid = S.info.sid, diff = S.diff;
    busy(true, `Scoring ${size / 1000} s windows`);
    try {
      const w = await api(`api/windows/${sid}/${diff}/${size}`);
      if (S.info.sid !== sid || S.diff !== diff) return;  // the user moved on meanwhile
      a.windows_by_size[size] = w;
    } catch (e) { toast("Could not score that window length: " + e.message); }
    finally { busy(false); }
  }
  S.dirtyTimeline = true;
  buildPanel();
  draw();
}
$("colorSel").onchange = (e) => { S.color = e.target.value; draw(); };
$("tickChk").onchange = (e) => { S.ticks = e.target.checked; };
$("offIn").onchange = (e) => { S.offset = +e.target.value || 0; if (S.playing) { pause(); play(); } };
$("diffSelect").onchange = (e) => openDiff(e.target.value);
$("fetchForm").onsubmit = (e) => { e.preventDefault(); const v = $("urlInput").value.trim(); if (v) loadUrl(v); };
$("fileInput").onchange = (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ""; };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type === "text") return;
  if (!S.chart) return;
  if (e.code === "Space") { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.code === "ArrowRight") seek(now() + 5000);
  else if (e.code === "ArrowLeft") seek(now() - 5000);
  else if (e.code === "ArrowUp") { e.preventDefault(); S.speed = Math.min(3, S.speed + 0.1); $("speedIn").value = S.speed; draw(); }
  else if (e.code === "ArrowDown") { e.preventDefault(); S.speed = Math.max(0.4, S.speed - 0.1); $("speedIn").value = S.speed; draw(); }
});
for (const ev of ["dragover", "dragenter"]) document.addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.add("over"); });
for (const ev of ["dragleave", "drop"]) document.addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.remove("over"); });
document.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

// deep links: ?s=<session>&d=<diff> (while the session lives) or ?set=<beatmapset id or link>
(async () => {
  const q = new URLSearchParams(location.search);
  if (q.get("s")) {
    try { showPlayer(); busy(true); await openSession(await api(`api/session/${q.get("s")}`), q.get("d")); return; }
    catch (e) { busy(false); $("player").classList.add("hidden"); $("empty").classList.remove("hidden"); toast("That session expired. Load the map again."); }
  }
  if (q.get("set")) { $("urlInput").value = q.get("set"); loadUrl(q.get("set")); }
})();
