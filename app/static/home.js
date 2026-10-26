"use strict";
// Home page only needs the live status badge in the top bar -- same polling logic as app.js,
// kept separate so the landing page doesn't load the heavier upload/results code it never uses.
async function pollHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    const s = document.getElementById("status");
    const text = document.getElementById("status-text");
    if (h.error) { s.className = "status status-error"; text.textContent = "Models failed to load"; return; }
    if (h.ready) { s.className = "status status-ready"; text.textContent = h.llm ? `Ready · coach: ${h.llm}` : "Ready"; return; }
    s.className = "status status-loading";
    text.textContent = `Warming up: ${h.stage}…`;
  } catch {
    document.getElementById("status-text").textContent = "Connecting…";
  }
  setTimeout(pollHealth, 1500);
}
pollHealth();
