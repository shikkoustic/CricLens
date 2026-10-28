// Home page: theme switch, the ball that flies the page, and the section animations.
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------- theme ----------
const toggle = document.getElementById("theme-toggle");
if (toggle) {
  toggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("criclens-theme", next); } catch { /* private window */ }
  });
}

// ---------- hero clip ----------
// Broadcast footage served from data/, so it may not exist on this machine: only reveal the
// viewport's video once it can actually play, and drop it quietly if it cannot.
const clip = document.getElementById("hero-clip");
if (clip) {
  clip.addEventListener("canplay", () => { clip.classList.add("ready"); clip.play().catch(() => {}); }, { once: true });
  clip.addEventListener("error", () => clip.closest(".hero-frame")?.classList.add("no-clip"), { once: true });
  clip.src = "/api/hero.mp4";
}

// ---------- the ball flies the page ----------
// One SVG path spans the whole document; scroll position picks the point along it, so the ball
// arcs from section to section and the dashed trail fills in behind it like a replay graphic.
const path = document.getElementById("rail-path");
const trail = document.getElementById("rail-trail");
const ball = document.getElementById("rail-ball");
if (path && ball && !reduced) {
  const len = path.getTotalLength();
  trail.style.strokeDasharray = `${len} ${len}`;
  let queued = false;
  const draw = () => {
    queued = false;
    const max = document.documentElement.scrollHeight - innerHeight;
    const p = max > 0 ? Math.min(1, Math.max(0, scrollY / max)) : 0;
    const pt = path.getPointAtLength(p * len);
    // the path is in a 0-100 viewBox stretched over the viewport
    ball.style.transform = `translate3d(${pt.x}vw, ${pt.y}vh, 0) rotate(${p * 1440}deg)`;
    trail.style.strokeDashoffset = `${len * (1 - p)}`;
    ball.classList.toggle("lit", p > 0.02);
  };
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(draw); } }, { passive: true });
  addEventListener("resize", draw);
  draw();
} else if (ball) {
  ball.remove();
  document.getElementById("rail")?.remove();
}

// ---------- sections animate in once ----------
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}, { rootMargin: "0px 0px -14% 0px", threshold: 0.15 });
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

// The technique dial counts up to a believable score the first time it is seen.
const dial = document.getElementById("dial-fg");
const dialVal = document.getElementById("dial-val");
if (dial && dialVal) {
  const C = 2 * Math.PI * 64;
  dial.style.strokeDasharray = `${C}`;
  dial.style.strokeDashoffset = `${C}`;
  const target = 7.4;
  const run = () => {
    if (reduced) { dial.style.strokeDashoffset = `${C * (1 - target / 10)}`; dialVal.textContent = target.toFixed(1); return; }
    const t0 = performance.now(), ms = 1100;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / ms);
      const eased = 1 - Math.pow(1 - k, 3);
      dial.style.strokeDashoffset = `${C * (1 - (target / 10) * eased)}`;
      dialVal.textContent = (target * eased).toFixed(1);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const dio = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { run(); dio.disconnect(); }
  }, { threshold: 0.4 });
  dio.observe(dial.closest(".dial-demo"));
}

// ---------- warming-up pill ----------
const status = document.getElementById("status");
const statusText = document.getElementById("status-text");
if (status) {
  const poll = async () => {
    try {
      const h = await (await fetch("/api/health")).json();
      if (h.error) { status.hidden = false; status.className = "status status-error"; statusText.textContent = "Models failed to load"; return; }
      if (h.ready) { status.hidden = true; return; }
      status.hidden = false; status.className = "status status-loading"; statusText.textContent = "Warming up…";
    } catch { /* server not up yet */ }
    setTimeout(poll, 2000);
  };
  poll();
}
