// Home page: theme switch, the hero clip, the section reveals and the pitch strip.
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
// Licensed stock footage (see the credit comment in home.html), committed as a static asset --
// unlike a dataset sample, it's fine to ship. Still only reveal the viewport once it can
// actually play, and drop it quietly if it somehow cannot (e.g. blocked by the browser).
const clip = document.getElementById("hero-clip");
if (clip) {
  clip.addEventListener("canplay", () => { clip.classList.add("ready"); clip.play().catch(() => {}); }, { once: true });
  clip.addEventListener("error", () => clip.closest(".hero-frame")?.classList.add("no-clip"), { once: true });
  clip.src = "/static/hero.mp4";
}

// ---------- sections animate in once ----------
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}, { rootMargin: "0px 0px -14% 0px", threshold: 0.15 });
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

// ---------- the technique dial ----------
// Settles on a believable score the first time it is seen, and afterwards follows whatever
// the net below it produces.
const dialFg = document.getElementById("dial-fg");
const dialVal = document.getElementById("dial-val");
const DIAL_C = 2 * Math.PI * 64;
let dialNow = 0;

const paintDial = (v) => {
  dialFg.style.strokeDashoffset = `${DIAL_C * (1 - v / 10)}`;
  dialVal.textContent = v.toFixed(1);
};
const setDial = (target, ms = 900) => {
  if (!dialFg || !dialVal) return;
  const from = dialNow;
  dialNow = target;
  if (reduced) { paintDial(target); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    const eased = 1 - Math.pow(1 - k, 3);
    paintDial(from + (target - from) * eased);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
};

if (dialFg && dialVal) {
  dialFg.style.strokeDasharray = `${DIAL_C}`;
  paintDial(0);
  const dio = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { setDial(7.4, 1100); dio.disconnect(); }
  }, { threshold: 0.4 });
  dio.observe(dialFg.closest(".dial-demo"));
}

// ---------- the net ----------
// One ball, timed by hand. How close the swing is to the moment the ball arrives sets the
// score on the dial above. It is a toy: the real number comes off a clip on the Analyse page.
const net = document.getElementById("net");
if (net && dialFg && !reduced) {
  const ball = document.getElementById("np-ball");
  const bat = document.getElementById("np-bat");
  const stumpsN = document.getElementById("np-stumps");
  const bails = [document.getElementById("np-bail-a"), document.getElementById("np-bail-b")];
  const verdict = document.getElementById("net-verdict");
  const go = document.getElementById("net-go");
  const goLabel = document.getElementById("net-go-label");

  const FLIGHT = 1250;        // ms from the bowler's hand to the stumps
  const CONTACT = 0.76;       // where down that flight the ball meets the bat
  const EDGE = 0.17;          // further out than this and you miss it altogether
  const REST = "rotate(34deg)";

  let raf = 0, t0 = 0, swing = -1, live = false, hit = false;

  const lane = () => net.querySelector(".net-pitch").clientWidth;
  // left edge of the lane is the bowler's hand, right edge is the stumps; the bat sits at
  // roughly CONTACT of the way across, so the two line up without being tuned separately
  const ballAt = (p) => {
    const w = lane();
    const x = 42 + ((w - 30 - 42) * Math.min(p, 1));
    const bounce = Math.abs(Math.sin(p * Math.PI * 1.4)) * 14 * (1 - p * 0.5);
    return [x, bounce];
  };

  const reset = () => {
    cancelAnimationFrame(raf);
    live = false; hit = false; swing = -1;
    net.classList.remove("live");
    bat.style.transform = REST;
    stumpsN.style.transform = "";
    bails.forEach((b) => { b.style.transform = ""; b.style.opacity = ""; });
    const [x, y] = ballAt(0);
    ball.style.transform = `translate3d(${x}px, ${-y}px, 0)`;
    ball.style.opacity = "0";
  };

  const knockStumps = () => {
    stumpsN.style.transition = "transform .25s cubic-bezier(.2,.8,.3,1)";
    stumpsN.style.transform = "rotate(11deg)";
    bails.forEach((b, i) => {
      b.style.transition = "transform .5s cubic-bezier(.2,.7,.3,1)";
      b.style.transform = `translate3d(${i ? 26 : 16}px, ${i ? -16 : -30}px, 0) rotate(${i ? -220 : 260}deg)`;
    });
  };

  const finish = (err) => {
    live = false;
    net.classList.remove("live");
    const missed = err === null;
    const off = missed ? 1 : Math.abs(err);
    const quality = Math.max(0, 1 - off / 0.3);
    const score = missed ? 1.8 : 2.5 + quality * 7.2;
    setDial(+score.toFixed(1), 700);
    verdict.textContent =
      missed ? "Bowled — no shot offered."
      : off < 0.05 ? "Middled it. Right under your eyes."
      : err < 0 ? (off < 0.13 ? "A touch early, but it came off." : "Too early — the bat face turned.")
      : (off < 0.13 ? "A fraction late, still safe." : "Late — squared up on it.");
    goLabel.textContent = "Bowl again";
  };

  const tick = (now) => {
    const p = (now - t0) / FLIGHT;

    if (hit) {
      // driven away: the ball leaves on its own line
      const k = Math.min(1, (now - hit) / 420);
      const [x, y] = ballAt(swing);
      ball.style.transform = `translate3d(${x - k * lane() * 0.9}px, ${-(y + k * 48)}px, 0)`;
      ball.style.opacity = `${1 - k}`;
      if (k >= 1) { finish(swing - CONTACT); return; }
      raf = requestAnimationFrame(tick);
      return;
    }

    if (p >= 1) {
      knockStumps();
      ball.style.opacity = "0";
      finish(swing < 0 ? null : swing - CONTACT);
      return;
    }

    const [x, y] = ballAt(p);
    ball.style.transform = `translate3d(${x}px, ${-y}px, 0)`;
    raf = requestAnimationFrame(tick);
  };

  const playShot = () => {
    if (!live || swing >= 0) return;
    swing = (performance.now() - t0) / FLIGHT;
    bat.style.transition = "transform .16s cubic-bezier(.2,.9,.3,1)";
    bat.style.transform = "rotate(-42deg)";
    if (Math.abs(swing - CONTACT) <= EDGE) hit = performance.now();
  };

  const bowl = () => {
    reset();
    ball.style.opacity = "1";
    bat.style.transition = "";
    stumpsN.style.transition = "";
    bails.forEach((b) => (b.style.transition = ""));
    verdict.textContent = "…";
    goLabel.textContent = "Bowling";
    live = true;
    net.classList.add("live");
    t0 = performance.now();
    raf = requestAnimationFrame(tick);
  };

  go.addEventListener("click", (e) => { e.preventDefault(); live ? playShot() : bowl(); });
  net.addEventListener("pointerdown", (e) => { if (live && e.target !== go && !go.contains(e.target)) playShot(); });
  addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    // Leave the key alone whenever something focusable owns it: on a focused button Space
    // already fires a click, so intercepting here would bowl twice.
    if (document.activeElement && document.activeElement.closest(
        "button, a, input, textarea, select, [contenteditable]")) return;
    // and only while the net is actually on screen
    const r = net.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    e.preventDefault();
    live ? playShot() : bowl();
  });

  reset();
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

// ---------- the pitch strip ----------
// A band along the foot of the window. Each of the three stages gets one beat, scrubbed by
// how far that section has come up the screen: the beat finishes exactly as the section
// lands, and runs backwards if you scroll back up. Decoration only — nothing here is
// announced, and none of it is in the tab order.
const pitch = document.getElementById("pitch");
if (pitch && !reduced) {
  const byId = (id) => document.getElementById(id);
  const batter = byId("act-batter"), player = byId("act-player"), wicket = byId("act-wicket");
  const ball = byId("act-ball"), bailA = byId("act-bail-a"), bailB = byId("act-bail-b");
  const shadowFig = byId("act-shadow-a"), shadowBall = byId("act-shadow-b");
  const bubble = byId("act-bubble");
  const bat = batter.querySelector(".s-bat");
  const arm = player.querySelector(".s-arm");
  const stumps = wicket.querySelector(".s-stumps");

  // one beat per stage, in the order they appear
  const beats = [...document.querySelectorAll("main .stage")].slice(0, 3)
    .map((section, i) => ({ section, kind: ["shot", "throw", "drill"][i] }));

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  // progress of one move inside a beat
  const at = (p, from, to) => clamp01((p - from) / (to - from));
  const put = (node, x, y) => { node.style.transform = `translate3d(${x}px, ${-(y || 0)}px, 0)`; };
  const vis = (node, on) => { node.style.display = on ? "" : "none"; };
  // the ball turns in quarters: a pixel sprite rotated off-axis just goes blurry
  const roll = (node, x, y, turns) =>
    { node.style.transform = `translate3d(${x}px, ${-y}px, 0) rotate(${Math.round(turns * 4) * 90}deg)`; };

  const IMPACT = 0.9;   // when the ball reaches the stumps

  let queued = false;
  const draw = () => {
    queued = false;
    const W = innerWidth, vh = innerHeight;

    // find the stage currently arriving, and how far in it is
    let beat = null, p = 0;
    for (const b of beats) {
      const t = clamp01((vh - b.section.getBoundingClientRect().top) / (vh * 0.62));
      if (t > 0.001 && t < 0.999) { beat = b; p = t; break; }
    }
    pitch.classList.toggle("on", !!beat);
    if (!beat) return;

    const kind = beat.kind;
    vis(batter, kind === "shot" || kind === "drill");
    vis(player, kind !== "shot");
    vis(wicket, kind === "throw");
    vis(bubble, kind === "drill");

    let ballX = 0, ballY = 0, travel = 0, ballOn = true;

    if (kind === "shot") {
      // the striker drives, and the ball runs away square along the floor
      const figX = W * 0.76;
      put(batter, figX, 0);
      put(shadowFig, figX - 6, 0);
      shadowFig.style.opacity = 1;
      bat.style.transform = `rotate(${lerp(40, -46, at(p, 0.02, 0.24))}deg)`;
      travel = at(p, 0.18, 1);
      ballX = lerp(figX - 6, -40, travel);
      ballOn = travel > 0 && travel < 1;

    } else if (kind === "drill") {
      // throwdowns, the drill the coach note actually asks for: one is fed in, the
      // striker plays it back past the feeder, and the coach has something to say
      const coachX = W * 0.13, strikerX = W * 0.76;
      put(player, coachX, 0);
      put(batter, strikerX, 0);
      put(shadowFig, coachX - 6, 0);
      shadowFig.style.opacity = 1;
      // underarm, so it reads differently from the throw in the beat before
      arm.style.transform = `rotate(${lerp(64, -26, at(p, 0.04, 0.26))}deg)`;
      bat.style.transform = `rotate(${lerp(38, -44, at(p, 0.52, 0.68))}deg)`;

      const fed = at(p, 0.2, 0.56);
      const back = at(p, 0.58, 0.96);
      if (back > 0) {
        travel = back;
        ballX = lerp(strikerX - 14, -40, back);
        ballY = Math.sin(back * Math.PI) * 30;
        ballOn = back < 1;
      } else {
        travel = fed;
        ballX = lerp(coachX + 32, strikerX - 14, fed);
        ballY = Math.abs(Math.sin(fed * Math.PI * 1.2)) * 24 * (1 - fed * 0.4);
        ballOn = fed > 0;
      }

      const say = at(p, 0.72, 0.9);
      bubble.style.opacity = `${say}`;
      bubble.style.transform =
        `translate3d(${coachX + 30}px, ${-(46 + say * 8)}px, 0) scale(${0.6 + say * 0.6})`;

    } else {
      // in from the deep: the arm comes over, and the throw takes the stumps
      const figX = W * 0.1, wicketX = W * 0.82;
      put(player, figX, 0);
      put(wicket, wicketX, 0);
      put(shadowFig, figX - 6, 0);
      shadowFig.style.opacity = 1;
      arm.style.transform = `rotate(${lerp(-150, 34, at(p, 0.04, 0.3))}deg)`;
      travel = at(p, 0.26, IMPACT);
      ballX = lerp(figX + 30, wicketX + 10, travel);
      ballY = Math.sin(travel * Math.PI) * 42;
      ballOn = travel > 0;

      const k = at(p, IMPACT, 1);
      stumps.style.transform = `rotate(${k * 10}deg)`;
      bailA.style.transform = `translate3d(${k * 26}px, ${-k * 36}px, 0) rotate(${k * 300}deg)`;
      bailB.style.transform = `translate3d(${k * 44}px, ${-k * 20}px, 0) rotate(${-k * 240}deg)`;
    }

    roll(ball, ballX, ballY, travel * 2.2);
    ball.style.opacity = ballOn ? "1" : "0";
    put(shadowBall, ballX - 18, 0);
    shadowBall.style.opacity = ballOn ? `${0.45 * (1 - ballY / 60)}` : "0";
    shadowBall.style.width = "34px";
  };


  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(draw); } }, { passive: true });
  addEventListener("resize", draw);
  draw();
}

// ---------- the stroke list opens a clip of that stroke ----------
// The clips are the real held-out test-split samples /api/samples already serves for the
// Analyse page, so nothing new is shipped and nothing is staged. On a machine without the
// clips the endpoint comes back empty and the dialog says so rather than failing silently.
const shotDialog = document.getElementById("shot-dialog");
if (shotDialog && typeof shotDialog.showModal === "function") {
  const vid = document.getElementById("sd-video");
  const title = document.getElementById("sd-title");
  const note = document.getElementById("sd-note");
  const meta = document.getElementById("sd-meta");
  const next = document.getElementById("sd-next");

  let byShot = null;          // label -> [sample, ...], fetched once
  let current = [], at = 0, opener = null;

  const loadSamples = async () => {
    if (byShot) return byShot;
    byShot = {};
    try {
      for (const s of await (await fetch("/api/samples")).json()) {
        if (s.label) (byShot[s.label] ||= []).push(s);
      }
    } catch { /* leave it empty; show() reports it */ }
    return byShot;
  };

  const show = (i) => {
    if (!current.length) {
      vid.hidden = true;
      note.hidden = false;
      note.textContent = "No clip for this stroke on this machine. Run scripts/fetch_data.sh to pull the samples.";
      meta.textContent = "";
      next.hidden = true;
      return;
    }
    at = (i + current.length) % current.length;
    const s = current[at];
    vid.hidden = false;
    note.hidden = true;
    next.hidden = current.length < 2;
    vid.src = `/api/samples/${encodeURIComponent(s.id)}/video`;
    vid.play().catch(() => { /* autoplay refused; the clip is still scrubbable */ });
    meta.textContent = `${s.source} · clip ${at + 1} of ${current.length} · held out from training`;
  };

  const open = async (btn) => {
    opener = btn;
    title.textContent = btn.querySelector("b").textContent;
    note.hidden = false;
    note.textContent = "Loading…";
    vid.hidden = true;
    meta.textContent = "";
    next.hidden = true;
    shotDialog.showModal();
    const map = await loadSamples();
    current = map[btn.dataset.shot] || [];
    show(0);
  };

  document.querySelectorAll(".shots .shot").forEach((btn) =>
    btn.addEventListener("click", () => open(btn)));

  next.addEventListener("click", () => show(at + 1));
  document.getElementById("sd-close").addEventListener("click", () => shotDialog.close());
  // clicking the backdrop (ie. outside the panel) closes it too
  shotDialog.addEventListener("click", (e) => { if (e.target === shotDialog) shotDialog.close(); });
  shotDialog.addEventListener("close", () => {
    vid.pause();
    vid.removeAttribute("src");
    vid.load();
    if (opener) opener.focus();
  });
}
