"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const pct = (p) => `${Math.round(p * 100)}%`;
const ordinal = (n) => { const r = Math.round(n), s = ["th", "st", "nd", "rd"], v = r % 100; return r + (s[(v - 20) % 10] || s[v] || s[0]); };
const PART = { head: "Head", shoulder: "Shoulders", hands: "Hands", hips: "Hips", feet: "Feet" };

let stages = [];
let choice = null;        // {kind: "file", file} | {kind: "sample", id, label}
let pollTimer = null;

function show(view) {
  for (const v of ["view-input", "view-progress", "view-results"]) $(v).hidden = v !== view;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- server status ----------
async function pollHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    stages = h.stages;
    const s = $("status");
    if (h.error) { s.className = "status status-error"; $("status-text").textContent = "Models failed to load"; return; }
    if (h.ready) {
      s.className = "status status-ready";
      $("status-text").textContent = h.llm ? `Ready · coach: ${h.llm}` : "Ready";
      return;
    }
    s.className = "status status-loading";
    $("status-text").textContent = `Warming up: ${h.stage}…`;
  } catch { $("status-text").textContent = "Connecting…"; }
  setTimeout(pollHealth, 1500);
}

// ---------- input ----------
function setChoice(c) {
  choice = c;
  $("analyse").disabled = !c;
  $("input-error").hidden = true;
  document.querySelectorAll(".sample").forEach((b) => b.classList.toggle("on", c && c.kind === "sample" && b.dataset.id === c.id));
  const isFile = c && c.kind === "file";
  $("file-chosen").hidden = !isFile;
  $("dropzone").hidden = isFile;
  if (isFile) {
    $("file-name").textContent = c.file.name;
    $("file-size").textContent = `${(c.file.size / 1048576).toFixed(1)} MB`;
    const v = $("file-preview");
    if (v.src) URL.revokeObjectURL(v.src);
    v.src = URL.createObjectURL(c.file);
    v.play().catch(() => {});
  }
}

function pickFile(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name)) {
    $("input-error").textContent = "That doesn't look like a video file."; $("input-error").hidden = false; return;
  }
  setChoice({ kind: "file", file });
}

async function loadSamples() {
  const box = $("samples");
  try {
    const list = await (await fetch("/api/samples")).json();
    box.textContent = "";
    if (!list.length) { box.append(el("p", "muted small", "No sample clips found on this machine.")); return; }
    for (const s of list) {
      const b = el("button", "sample"); b.type = "button"; b.dataset.id = s.id;
      const img = el("img"); img.src = s.thumb; img.alt = ""; img.loading = "lazy";
      const cap = el("span", "cap");
      cap.append(el("b", null, s.label ? labelName(s.label) : "Clip"), el("span", "muted", s.source));
      b.append(img, cap, el("span", "check", "✓"));
      b.title = "Dataset label: " + (s.label ? labelName(s.label) : "unknown");
      b.onclick = () => setChoice({ kind: "sample", id: s.id, label: s.label, source: s.source });
      box.append(b);
    }
  } catch { box.textContent = ""; box.append(el("p", "muted small", "Couldn't load samples.")); }
}
const LABELS = { cut: "Cut", defence: "Defence", drive: "Drive", flick_glance: "Flick / Glance", lofted: "Lofted", pull_hook: "Pull / Hook", scoop: "Scoop", sweep: "Sweep", other: "Other" };
const labelName = (k) => LABELS[k] || k;

async function startAnalysis() {
  if (!choice) return;
  $("analyse").disabled = true;
  try {
    let res;
    if (choice.kind === "file") {
      const fd = new FormData(); fd.append("file", choice.file);
      res = await fetch("/api/jobs", { method: "POST", body: fd });
    } else {
      res = await fetch("/api/jobs/sample", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_id: choice.id }) });
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    beginProgress(data.job_id, choice.kind === "file" ? choice.file.name : `Sample clip (${choice.source})`);
  } catch (e) {
    $("input-error").textContent = e.message; $("input-error").hidden = false; $("analyse").disabled = false;
  }
}

// ---------- progress ----------
function beginProgress(jobId, name) {
  $("progress-name").textContent = name;
  $("progress-error").hidden = true; $("progress-back").hidden = true;
  const ol = $("steps"); ol.textContent = "";
  for (const s of stages) { const li = el("li"); li.dataset.key = s.key; li.append(el("span", "ico"), el("span", null, s.label)); ol.append(li); }
  setProgress("read", 0, "queued");
  show("view-progress");
  clearTimeout(pollTimer);
  const poll = async () => {
    try {
      const j = await (await fetch(`/api/jobs/${jobId}`)).json();
      if (j.status === "done") { setProgress("done", 1); setTimeout(() => renderResults(jobId, name, j.result), 350); return; }
      if (j.status === "error") { failProgress(j.error); return; }
      setProgress(j.stage, j.progress || 0, j.status, j.queue_position);
    } catch { /* transient; keep polling */ }
    pollTimer = setTimeout(poll, 1000);
  };
  poll();
}

function setProgress(stage, frac, status, queuePos) {
  const idx = stage === "done" ? stages.length : stages.findIndex((s) => s.key === stage);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.className = i < idx ? "done" : i === idx && status !== "queued" ? "active" : "";
    li.querySelector(".ico").textContent = i < idx ? "✓" : "";
  });
  $("progress-bar").style.width = pct(frac);
  $("progress-pct").textContent = pct(frac);
  $("progress-note").textContent = status === "queued"
    ? (queuePos > 1 ? `Waiting in line (position ${queuePos})…` : "Waiting for the models to finish loading…")
    : "Usually one to two minutes on a laptop CPU.";
}

function failProgress(msg) {
  $("progress-error").textContent = msg || "Analysis failed."; $("progress-error").hidden = false;
  $("progress-back").hidden = false;
  document.querySelectorAll("#steps li.active").forEach((li) => (li.className = ""));
}

// ---------- results ----------
function renderResults(jobId, name, r) {
  $("res-name").textContent = name;
  const media = (f) => `/api/jobs/${jobId}/media/${f}`;

  const w = $("warnings"); w.textContent = "";
  for (const msg of r.warnings || []) w.append(el("div", "notice", msg));

  // video + key moments
  const v = $("res-video");
  v.src = media(r.media.video);
  v.onloadedmetadata = () => { v.playbackRate = currentSpeed(); v.play().catch(() => {}); };
  const kf = $("keyframes"); kf.textContent = "";
  for (const k of r.media.keyframes) {
    const b = el("button", "kf"); b.type = "button";
    const img = el("img"); img.src = media(k.file); img.alt = k.label;
    const cap = el("span"); cap.append(el("b", null, k.label), el("span", "muted", `${k.time_s.toFixed(2)} s`));
    b.append(img, cap);
    b.onclick = () => { v.pause(); v.currentTime = k.time_s + 0.001; };
    kf.append(b);
  }

  // shot
  const s = r.shot;
  $("shot-name").textContent = s.display;
  const tag = $("shot-tag");
  const unsure = s.confidence < 0.6;
  tag.className = unsure ? "tag unsure" : "tag";
  tag.textContent = unsure ? `Unsure · ${pct(s.confidence)}` : `${pct(s.confidence)} confident`;
  const pb = $("shot-probs"); pb.textContent = "";
  s.probs.slice(0, 4).forEach((p, i) => {
    const row = el("div", i === 0 ? "prob top" : "prob");
    const tr = el("div", "track"); const f = el("div", "fill"); f.style.width = pct(p.p); tr.append(f);
    row.append(el("span", null, p.display), tr, el("span", "v", pct(p.p)));
    pb.append(row);
  });

  // technique
  const t = r.technique;
  $("tech-score").textContent = t.overall.toFixed(1);
  const ring = $("ring-fg"); ring.style.strokeDashoffset = 327;
  requestAnimationFrame(() => requestAnimationFrame(() => (ring.style.strokeDashoffset = 327 * (1 - t.overall / 10))));
  const refName = t.reference_shot === "all" ? "all rated shots" : `rated ${labelName(t.reference_shot).toLowerCase()}s`;
  const tp = $("tech-pct"); tp.textContent = "";
  if (t.overall_pct != null) tp.append("Better than ", el("b", null, `${Math.round(t.overall_pct)}%`), ` of ${refName} in the coaching dataset.`);
  const parts = $("parts"); parts.textContent = "";
  for (const p of t.parts) {
    const row = el("div", "part");
    const tr = el("div", "track"); const f = el("div", "fill"); f.style.width = `${p.score * 10}%`; tr.append(f);
    row.append(el("span", null, PART[p.part]), tr, el("span", "v", p.score.toFixed(1)));
    parts.append(row);
  }

  renderMetrics(r.metrics);
  renderCoach(r.coach);
  renderDetails(r);
  show("view-results");
}

function renderMetrics(m) {
  const box = $("metrics"); box.textContent = "";
  const ref = m.reference_shot && m.reference_shot !== "all" ? `${labelName(m.reference_shot).toLowerCase()}s` : "shots";
  const tile = (label, value, unit, p, lowTxt, highTxt) => {
    const d = el("div", "metric" + (value == null ? " na" : ""));
    d.append(el("div", "label", label));
    if (value == null) {
      d.append(el("div", "value", "Not available"));
      d.append(el("p", "note", m.calibrated ? "Couldn't be measured reliably on this clip." : "The crease or stumps weren't clear enough to convert pixels into real distances (this works on about 1 in 5 broadcast clips)."));
      return d;
    }
    const val = el("div", "value", value.toFixed(unit === "cm" ? 0 : 1)); val.append(el("small", null, unit));
    d.append(val);
    if (p != null) {
      const band = el("div", "band"); const mk = el("span", "marker"); mk.style.left = `${Math.min(99, Math.max(1, p))}%`; band.append(mk);
      const labels = el("div", "band-labels"); labels.append(el("span", null, lowTxt), el("span", null, "typical"), el("span", null, highTxt));
      d.append(band, labels, el("p", "note", `${ordinal(p)} percentile among ${ref} in the dataset.`));
    }
    return d;
  };
  box.append(tile("Stride length", m.stride_cm, "cm", m.stride_pct, "shorter", "longer"));
  box.append(tile("Peak hand speed", m.swing_mps, "m/s", m.swing_pct, "slower", "faster"));
  if (m.calibrated) box.append(el("p", "metrics-foot", `Scale from the ${m.method === "crease" ? "painted crease lines" : "stumps"} (${m.cm_per_px.toFixed(2)} cm per pixel at the batter's crease). Hand speed is measured at the wrists, not the bat tip.`));
}

function renderCoach(c) {
  $("coach-source").textContent = c.source === "template" ? "Rule-based summary" : `Written by ${c.source} · runs locally`;
  const box = $("coach-text"); box.textContent = "";
  let list = null;
  for (const raw of c.text.split("\n")) {
    const line = raw.trim();
    if (!line) { list = null; continue; }
    const heading = line.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
    if (heading && !line.startsWith("-")) {
      list = null; box.append(el("h4", null, heading[1].replace(/:$/, "")));
      if (heading[2]) box.append(inline("p", heading[2]));
      continue;
    }
    if (/^[-*•]\s+/.test(line)) {
      if (!list) { list = el("ul"); box.append(list); }
      list.append(inline("li", line.replace(/^[-*•]\s+/, "")));
      continue;
    }
    list = null; box.append(inline("p", line));
  }
  const ul = $("findings"); ul.textContent = "";
  for (const f of c.findings) { const li = el("li", f.tone); li.append(el("b", null, f.title), document.createTextNode(f.detail)); ul.append(li); }
}

function inline(tag, text) {           // **bold** only; everything else stays plain text
  const node = el(tag);
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) node.append(el("strong", null, part.slice(2, -2)));
    else if (part) node.append(document.createTextNode(part));
  });
  return node;
}

function renderDetails(r) {
  const dl = $("details"); dl.textContent = "";
  const b = r.batter, w = b.window;
  const rows = [
    ["Batter identification", `${pct(b.finder_p)} confidence`],
    ["Batter tracked in shot window", pct(b.found_in_window)],
    ["Pose confidence (mean)", b.pose_confidence.toFixed(2)],
    ["Shot window", `${w.start_s.toFixed(2)}–${w.end_s.toFixed(2)} s`],
    ["Contact (estimated)", `${w.contact_s.toFixed(2)} s`],
    ["Video", `${r.video.width}×${r.video.height} · ${r.video.fps.toFixed(0)} fps · ${r.video.seconds.toFixed(1)} s`],
    ["Distance calibration", r.metrics.calibrated ? (r.metrics.method === "crease" ? "Crease lines" : "Stumps") : "Not available"],
    ["Technique compared with", r.technique.reference_shot === "all" ? "All rated shots" : `Rated ${labelName(r.technique.reference_shot).toLowerCase()}s`],
  ];
  for (const [k, val] of rows) { const d = el("div"); d.append(el("dt", null, k), el("dd", null, val)); dl.append(d); }
}

function currentSpeed() { return parseFloat(document.querySelector(".speed .on").dataset.speed); }

// ---------- wiring ----------
function init() {
  const dz = $("dropzone"), input = $("file");
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.onchange = () => pickFile(input.files[0]);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
  $("file-clear").onclick = () => { input.value = ""; setChoice(null); };
  $("analyse").onclick = startAnalysis;
  $("again").onclick = $("progress-back").onclick = () => { $("res-video").pause(); show("view-input"); setChoice(choice); };
  document.querySelectorAll(".speed button").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".speed button").forEach((x) => x.classList.toggle("on", x === b));
    $("res-video").playbackRate = currentSpeed();
  }));
  pollHealth();
  loadSamples();
}
init();
