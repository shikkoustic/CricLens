// Home page: hero clip, scroll reveals, and a ball that travels down the pitch as you scroll.

// The hero clip is broadcast footage served from data/, so it may not exist on every machine.
// Only show the video once it can actually play; otherwise the gradient behind it stands alone.
const hero = document.getElementById("hero-video");
if (hero) {
  hero.addEventListener("canplay", () => { hero.classList.add("ready"); hero.play().catch(() => {}); }, { once: true });
  hero.addEventListener("error", () => hero.remove(), { once: true });
  hero.src = "/api/hero.mp4";
}

// Sections fade up once, when they first come into view.
const reveal = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) { e.target.classList.add("in"); reveal.unobserve(e.target); }
  }
}, { rootMargin: "0px 0px -12% 0px", threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((el) => reveal.observe(el));

// The ball runs the length of the pitch strip as that section crosses the viewport: the delivery
// itself is the progress indicator for "release -> follow-through".
const strip = document.querySelector(".pitch-strip");
const ball = document.querySelector(".travel-ball");
if (strip && ball && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  let ticking = false;
  const place = () => {
    ticking = false;
    const r = strip.getBoundingClientRect();
    const span = innerHeight + r.height;
    const p = Math.min(1, Math.max(0, (innerHeight - r.top) / span));
    ball.style.setProperty("--p", p.toFixed(4));
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(place); } }, { passive: true });
  addEventListener("resize", place);
  place();
}

// Warming-up pill: shown only while the models load, then it gets out of the way.
const status = document.getElementById("status");
const statusText = document.getElementById("status-text");
if (status) {
  const poll = async () => {
    try {
      const h = await (await fetch("/api/health")).json();
      if (h.error) { status.hidden = false; status.className = "status status-error"; statusText.textContent = "Models failed to load"; return; }
      if (h.ready) { status.hidden = true; return; }
      status.hidden = false; status.className = "status status-loading"; statusText.textContent = "Warming up…";
    } catch { /* server not up yet; try again */ }
    setTimeout(poll, 2000);
  };
  poll();
}
