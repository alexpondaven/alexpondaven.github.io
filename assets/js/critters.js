// The /colony/ critters. The paragraph is the environment: each letter is a
// physical object with a home slot in the text. Critters run a simple job
// loop (pick a letter, haul it to the pile, later haul it back home), but
// every actual movement — approach, arrival, dodging your cursor — is
// steered by a tiny MLP trained offline (assets/worldmodel/critter_train.js).
const stage = document.getElementById('colony-stage');
const statusEl = document.getElementById('colony-status');

if (stage) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    if (statusEl) statusEl.textContent = 'colony is resting (reduced motion is on) — the text is safe';
  } else {
    init();
  }
}

async function init() {
  let policy;
  try {
    const r = await fetch('/assets/worldmodel/critter_policy.json');
    policy = await r.json();
  } catch (e) {
    if (statusEl) statusEl.textContent = 'could not load the steering net';
    return;
  }
  const { accCap: ACC_CAP, friction: FRICTION } = policy.meta;
  const net = policy.net;
  const N_PARAMS = net.W1.length + net.b1.length + net.W2.length + net.b2.length;

  function forward(x) {
    const { W1, b1, W2, b2, in: inDim, hidden, out: outDim } = net;
    const a1 = new Float64Array(hidden);
    for (let h = 0; h < hidden; h++) {
      let acc = b1[h];
      for (let i = 0; i < inDim; i++) acc += x[i] * W1[i * hidden + h];
      a1[h] = acc > 0 ? acc : 0;
    }
    const o = new Float64Array(outDim);
    for (let k = 0; k < outDim; k++) {
      let acc = b2[k];
      for (let h = 0; h < hidden; h++) acc += a1[h] * W2[h * outDim + k];
      o[k] = acc;
    }
    return o;
  }

  // --- turn the paragraph into a world of letters ---------------------------
  const textEl = stage.querySelector('.colony-text');
  const raw = textEl.textContent;
  textEl.textContent = '';
  const letters = [];  // {ch, span, home:{x,y}px, at:'home'|'held'|'ground', pos:{x,y}, angle, claimed, word}
  const words = [];    // arrays of letter indices
  let word = null;
  for (const ch of raw) {
    if (/\s/.test(ch)) {
      textEl.appendChild(document.createTextNode(ch));
      word = null;
      continue;
    }
    const span = document.createElement('span');
    span.className = 'colony-ch';
    span.textContent = ch;
    textEl.appendChild(span);
    if (!word) { word = []; words.push(word); }
    word.push(letters.length);
    letters.push({ ch, span, home: { x: 0, y: 0 }, at: 'home', pos: { x: 0, y: 0 }, angle: 0, claimed: false, word });
  }

  let W = 0, H = 0, font = '16px sans-serif', inkColor = '#333';
  const canvas = document.getElementById('colony-canvas');
  const ctx = canvas.getContext('2d');
  function measure() {
    const rect = stage.getBoundingClientRect();
    W = rect.width; H = rect.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cs = getComputedStyle(textEl);
    font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    inkColor = cs.color;
    for (const L of letters) {
      const r = L.span.getBoundingClientRect();
      L.home.x = r.left - rect.left + r.width / 2;
      L.home.y = r.top - rect.top + r.height / 2;
    }
  }
  measure();
  if ('ResizeObserver' in window) new ResizeObserver(measure).observe(stage);

  // normalized [-1,1] world <-> stage px
  const toNorm = (p) => ({ x: (p.x / W) * 2 - 1, y: (p.y / H) * 2 - 1 });
  const toPx = (n) => ({ x: ((n.x + 1) / 2) * W, y: ((n.y + 1) / 2) * H });

  // pile slots along the bottom
  const PILE_Y = () => H - 34;
  function pileSlot(i) {
    const perRow = Math.max(8, Math.floor((W - 60) / 22));
    return {
      x: 30 + (i % perRow) * 22 + (Math.random() - 0.5) * 6,
      y: PILE_Y() - Math.floor(i / perRow) * 26 + (Math.random() - 0.5) * 5,
    };
  }
  let pileCount = 0;
  const inPileZone = (p) => p.y > H - 95;

  // --- cursor (the predator) -------------------------------------------------
  const cursor = { x: 2.2, y: 2.2, vx: 0, vy: 0, px: 2.2, py: 2.2, active: false };
  stage.addEventListener('pointermove', (e) => {
    const rect = stage.getBoundingClientRect();
    const n = toNorm({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    cursor.x = n.x; cursor.y = n.y; cursor.active = true;
  });
  stage.addEventListener('pointerleave', () => { cursor.active = false; });

  // a click scares them: nearby critters bolt and drop whatever they carry
  stage.addEventListener('pointerdown', (e) => {
    const rect = stage.getBoundingClientRect();
    const n = toNorm({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    for (const c of critters) {
      const dx = c.x - n.x, dy = c.y - n.y;
      const d = Math.hypot(dx, dy);
      if (d < 0.5 && d > 1e-6) {
        const kick = 0.09 * (1 - d / 0.5);
        c.vx += (dx / d) * kick;
        c.vy += (dy / d) * kick;
        if (c.carrying >= 0) {   // panic-drop the letter right here
          const L = letters[c.carrying];
          L.at = 'ground';
          L.pos = toPx({ x: c.x, y: c.y });
          L.angle = (Math.random() - 0.5) * 0.9;
          L.claimed = false;
          c.carrying = -1;
          c.job = null;
        }
      }
    }
  });

  // --- critters ---------------------------------------------------------------
  const PALETTE = ['#818cf8', '#a78bfa', '#6ee7b7', '#fbbf24', '#f9a8d4', '#7dd3fc', '#fda4af'];
  const critters = PALETTE.map((color, i) => ({
    x: (Math.random() * 2 - 1) * 0.8, y: (Math.random() * 2 - 1) * 0.8,
    vx: 0, vy: 0,
    color,
    carrying: -1,
    job: null,               // {letter, phase:'fetch'|'deliver', dest:{x,y}px}
    wanderT: 0, wander: { x: 0, y: 0 },
    blinkT: 60 + Math.random() * 120, blink: 0,
    seed: i * 977,
  }));

  let mode = 'dismantle';    // 'dismantle' | 'rebuild' | 'rest'
  let restT = 0;

  function claimJob(c) {
    if (mode === 'rest') return null;
    // scattered letters (panic drops outside the pile) get priority in both modes
    const ground = letters.filter((L) => L.at === 'ground' && !L.claimed);
    if (mode === 'dismantle') {
      const stray = ground.find((L) => !inPileZone(L.pos));
      if (stray) {
        stray.claimed = true;
        return { letter: stray, phase: 'fetch', dest: pileSlot(pileCount++) };
      }
      // dismantle words from their last letter — looks like nibbling
      const live = words.filter((w) => w.some((i) => letters[i].at === 'home' && !letters[i].claimed));
      if (!live.length) return null;
      const w = live[Math.floor(Math.random() * live.length)];
      for (let k = w.length - 1; k >= 0; k--) {
        const L = letters[w[k]];
        if (L.at === 'home' && !L.claimed) {
          L.claimed = true;
          return { letter: L, phase: 'fetch', dest: pileSlot(pileCount++) };
        }
      }
      return null;
    }
    // rebuild: any grounded letter goes home
    if (!ground.length) return null;
    const L = ground[Math.floor(Math.random() * ground.length)];
    L.claimed = true;
    return { letter: L, phase: 'fetch', dest: { x: L.home.x, y: L.home.y } };
  }

  function jobTarget(c) {
    const L = c.job.letter;
    if (c.job.phase === 'fetch') return L.at === 'home' ? L.home : L.pos;
    return c.job.dest;
  }

  function stepColony() {
    // cursor velocity (per model step)
    if (!cursor.active) { cursor.x = 2.2; cursor.y = 2.2; }
    cursor.vx = cursor.x - cursor.px; cursor.vy = cursor.y - cursor.py;
    cursor.px = cursor.x; cursor.py = cursor.y;

    // colony mode transitions
    const homeCount = letters.filter((L) => L.at === 'home').length;
    const groundCount = letters.filter((L) => L.at === 'ground').length;
    if (mode === 'dismantle' && homeCount === 0 && groundCount === letters.length) {
      mode = 'rest'; restT = 75; // ~2.5s of gloating
    } else if (mode === 'rebuild' && homeCount === letters.length) {
      mode = 'rest'; restT = 75;
      pileCount = 0;
    } else if (mode === 'rest' && --restT <= 0) {
      mode = homeCount === letters.length ? 'dismantle' : 'rebuild';
    }

    for (const c of critters) {
      // job bookkeeping
      if (!c.job) c.job = claimJob(c);
      let target;
      if (c.job) {
        target = toNorm(jobTarget(c));
      } else {
        if (--c.wanderT <= 0) {
          c.wanderT = 60 + Math.random() * 90;
          c.wander = { x: (Math.random() * 2 - 1) * 0.8, y: (Math.random() * 2 - 1) * 0.8 };
        }
        target = c.wander;
      }

      // the net makes every movement decision (feature layout must match
      // critter_train.js exactly, unit directions included)
      const tx = target.x - c.x, ty = target.y - c.y;
      const dt = Math.hypot(tx, ty);
      const utx = dt > 1e-6 ? tx / dt : 0, uty = dt > 1e-6 ? ty / dt : 0;
      const fx = cursor.x - c.x, fy = cursor.y - c.y;
      const df = Math.hypot(fx, fy);
      const ufx = df > 1e-6 ? fx / df : 0, ufy = df > 1e-6 ? fy / df : 0;
      const closing = df > 1e-6 ? -(fx * (cursor.vx - c.vx) + fy * (cursor.vy - c.vy)) / df : 0;
      const o = forward([c.x, c.y, c.vx, c.vy, tx, ty, dt, utx, uty, fx, fy, df, ufx, ufy, closing]);
      let ax = o[0], ay = o[1];
      const a = Math.hypot(ax, ay);
      if (a > ACC_CAP) { ax *= ACC_CAP / a; ay *= ACC_CAP / a; }
      c.vx = (c.vx + ax) * FRICTION;
      c.vy = (c.vy + ay) * FRICTION;
      c.x = Math.max(-1, Math.min(1, c.x + c.vx));
      c.y = Math.max(-1, Math.min(1, c.y + c.vy));

      // arrivals — checked after the move, with a ring comfortably outside
      // the net's learned hover offset (~0.05)
      const dtNow = Math.hypot(target.x - c.x, target.y - c.y);
      if (c.job && dtNow < 0.085) {
        const L = c.job.letter;
        if (c.job.phase === 'fetch') {
          if (L.at === 'home') L.span.style.visibility = 'hidden';
          L.at = 'held';
          c.carrying = letters.indexOf(L);
          c.job.phase = 'deliver';
        } else {
          const goingHome = Math.hypot(c.job.dest.x - L.home.x, c.job.dest.y - L.home.y) < 2;
          if (goingHome) {
            L.at = 'home';
            L.span.style.visibility = '';
          } else {
            L.at = 'ground';
            L.pos = { ...c.job.dest };
            L.angle = (Math.random() - 0.5) * 0.5;
          }
          L.claimed = false;
          c.carrying = -1;
          c.job = null;
        }
      }

      // blinking
      if (--c.blinkT <= 0) { c.blink = 5; c.blinkT = 90 + Math.random() * 150; }
      if (c.blink > 0) c.blink--;
    }

    if (statusEl) {
      const hoarded = letters.filter((L) => L.at !== 'home').length;
      const verb = mode === 'dismantle' ? 'dismantling' : mode === 'rebuild' ? 'repairing' : 'plotting';
      statusEl.textContent = `${critters.length} critters, one shared ${N_PARAMS.toLocaleString()}-param steering net · ${verb} · ${hoarded}/${letters.length} letters stolen`;
    }
  }

  // --- drawing -----------------------------------------------------------------
  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // grounded letters
    ctx.fillStyle = inkColor;
    for (const L of letters) {
      if (L.at !== 'ground') continue;
      ctx.save();
      ctx.translate(L.pos.x, L.pos.y);
      ctx.rotate(L.angle);
      ctx.globalAlpha = 0.85;
      ctx.fillText(L.ch, 0, 0);
      ctx.restore();
    }

    for (const c of critters) {
      const p = toPx(c);
      const speed = Math.hypot(c.vx, c.vy);
      const ang = Math.atan2(c.vy, c.vx);
      const squash = Math.min(0.35, speed * 9);
      const bob = Math.sin(now / 130 + c.seed) * 1.3;

      // shadow
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + 8, 7.5, 2.6, 0, 0, Math.PI * 2);
      ctx.fill();

      // body
      ctx.save();
      ctx.translate(p.x, p.y + bob * 0.4);
      ctx.rotate(ang);
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.ellipse(0, 0, 8 * (1 + squash), 8 * (1 - squash * 0.6), 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // eyes (always upright — cuter)
      const look = speed > 0.002 ? { x: Math.cos(ang), y: Math.sin(ang) } : { x: 0, y: 0.3 };
      for (const side of [-1, 1]) {
        const ex = p.x + side * 3.4, ey = p.y - 2.5 + bob * 0.4;
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        if (c.blink > 0) {
          ctx.fillRect(ex - 2, ey - 0.6, 4, 1.2);
        } else {
          ctx.arc(ex, ey, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#1e1b4b';
          ctx.beginPath();
          ctx.arc(ex + look.x * 1.1, ey + look.y * 1.1, 1.25, 0, Math.PI * 2);
        }
        ctx.fill();
      }

      // carried letter, held overhead with a little sway
      if (c.carrying >= 0) {
        const L = letters[c.carrying];
        ctx.save();
        ctx.translate(p.x, p.y - 15 + bob);
        ctx.rotate(Math.sin(now / 210 + c.seed) * 0.16);
        ctx.fillStyle = inkColor;
        ctx.fillText(L.ch, 0, 0);
        ctx.restore();
      }
    }
  }

  // --- loop ---------------------------------------------------------------------
  const STEP_MS = 33;
  let last = 0, running = true;
  document.addEventListener('visibilitychange', () => { running = !document.hidden; });
  function frame(now) {
    if (running) {
      if (now - last >= STEP_MS) { last = now; stepColony(); }
      draw(now);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
