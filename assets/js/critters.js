// The /colony/ ecosystem. The paragraph is the environment: every letter is
// a physical object with a home slot in the text. Three castes share ONE
// tiny steering MLP (assets/worldmodel/critter_train.js) — same brain,
// different bodies and goals:
//   workers  — dismantle the text into a pile, then rebuild it
//   monsters — eat letters; what they swallow is gone until they pop
//   slingers — fetch letters and throw them at monsters (3 hits = pop)
// Job choices are a plain state machine; every movement decision (approach,
// arrival, fleeing) is the net's. The cursor is a predator to the small
// castes, and clicking a monster bonks it like a thrown letter.
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

  // One steering step for any body. `threat` may be null (fed as a far-away,
  // on-manifold pseudo-cursor so the flee channel stays quiet).
  function steer(c, target, threat, accScale) {
    const tx = target.x - c.x, ty = target.y - c.y;
    const dt = Math.hypot(tx, ty);
    const utx = dt > 1e-6 ? tx / dt : 0, uty = dt > 1e-6 ? ty / dt : 0;
    const th = threat || { x: c.x + 1.2, y: c.y + 1.2, vx: 0, vy: 0 };
    const fx = th.x - c.x, fy = th.y - c.y;
    const df = Math.hypot(fx, fy);
    const ufx = df > 1e-6 ? fx / df : 0, ufy = df > 1e-6 ? fy / df : 0;
    const closing = df > 1e-6 ? -(fx * ((th.vx || 0) - c.vx) + fy * ((th.vy || 0) - c.vy)) / df : 0;
    const o = forward([c.x, c.y, c.vx, c.vy, tx, ty, dt, utx, uty, fx, fy, df, ufx, ufy, closing]);
    let ax = o[0], ay = o[1];
    const a = Math.hypot(ax, ay);
    const cap = ACC_CAP * accScale;
    if (a > cap) { ax *= cap / a; ay *= cap / a; }
    c.vx = (c.vx + ax) * FRICTION;
    c.vy = (c.vy + ay) * FRICTION;
    // keep bodies fully inside the stage — a critter cowering ON the border
    // gets its body clipped by the frame and looks like a rendering bug
    c.x = Math.max(-0.955, Math.min(0.955, c.x + c.vx));
    c.y = Math.max(-0.955, Math.min(0.955, c.y + c.vy));
    return dt;
  }

  // --- turn the paragraph into a world of letters ---------------------------
  const textEl = stage.querySelector('.colony-text');
  const raw = textEl.textContent;
  textEl.textContent = '';
  const letters = [];  // {ch, span, home{px}, at:'home'|'held'|'ground'|'eaten'|'flying', pos{px}, angle, claimed}
  const words = [];
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
    letters.push({ ch, span, home: { x: 0, y: 0 }, at: 'home', pos: { x: 0, y: 0 }, angle: 0, claimed: false });
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

  const toNorm = (p) => ({ x: (p.x / W) * 2 - 1, y: (p.y / H) * 2 - 1 });
  const toPx = (n) => ({ x: ((n.x + 1) / 2) * W, y: ((n.y + 1) / 2) * H });

  function pileSlot(i) {
    const perRow = Math.max(8, Math.floor((W - 60) / 22));
    return {
      x: 30 + (i % perRow) * 22 + (Math.random() - 0.5) * 6,
      y: H - 34 - Math.floor(i / perRow) * 26 + (Math.random() - 0.5) * 5,
    };
  }
  let pileCount = 0;
  const inPileZone = (p) => p.y > H - 95;

  // --- cursor -----------------------------------------------------------------
  const cursor = { x: 2.2, y: 2.2, vx: 0, vy: 0, px: 2.2, py: 2.2, active: false };
  stage.addEventListener('pointermove', (e) => {
    const rect = stage.getBoundingClientRect();
    const n = toNorm({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    cursor.x = n.x; cursor.y = n.y; cursor.active = true;
  });
  stage.addEventListener('pointerleave', () => { cursor.active = false; });

  stage.addEventListener('pointerdown', (e) => {
    const rect = stage.getBoundingClientRect();
    const n = toNorm({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    // bonk a monster?
    for (const m of monsters) {
      if (Math.hypot(m.x - n.x, m.y - n.y) < 0.14) {
        hitMonster(m);
        return;
      }
    }
    // otherwise scare the small folk
    for (const c of critters) {
      const dx = c.x - n.x, dy = c.y - n.y;
      const d = Math.hypot(dx, dy);
      if (d < 0.5 && d > 1e-6) {
        const kick = 0.09 * (1 - d / 0.5);
        c.vx += (dx / d) * kick;
        c.vy += (dy / d) * kick;
        if (c.carrying >= 0) dropCarried(c);
      }
    }
  });

  function dropCarried(c) {
    const L = letters[c.carrying];
    L.at = 'ground';
    L.pos = toPx({ x: c.x, y: c.y });
    L.angle = (Math.random() - 0.5) * 0.9;
    L.claimed = false;
    c.carrying = -1;
    c.job = null;
  }

  // --- castes -------------------------------------------------------------------
  const WORKER_COLORS = ['#818cf8', '#a78bfa', '#6ee7b7', '#f9a8d4', '#7dd3fc'];
  const SLINGER_COLORS = ['#fb923c', '#fbbf24'];
  const critters = [
    ...WORKER_COLORS.map((color, i) => makeSmall('worker', color, i)),
    ...SLINGER_COLORS.map((color, i) => makeSmall('slinger', color, i + 5)),
  ];
  function makeSmall(role, color, i) {
    return {
      role, color,
      x: (Math.random() * 2 - 1) * 0.8, y: (Math.random() * 2 - 1) * 0.8,
      vx: 0, vy: 0,
      carrying: -1, job: null,
      wanderT: 0, wander: { x: 0, y: 0 },
      blinkT: 60 + Math.random() * 120, blink: 0,
      throwCooldown: 0,
      seed: i * 977,
    };
  }
  const monsters = [makeMonster(0), makeMonster(1)];
  function makeMonster(i) {
    const edge = Math.random() < 0.5 ? -1 : 1;
    return {
      x: edge * 0.95, y: (Math.random() * 2 - 1) * 0.8,
      vx: 0, vy: 0,
      belly: [],            // letter indices it has eaten
      hits: 0,
      target: -1,           // letter index being hunted
      cooldown: 60 + Math.random() * 60,
      flash: 0, mouth: 0,
      wanderT: 0, wander: null,
      seed: 313 + i * 733,
    };
  }
  const projectiles = [];   // {letter, x, y, vx, vy, spin}
  const bursts = [];        // {x, y, t} pop effects
  const stats = { eaten: 0, thrown: 0, hits: 0, pops: 0 };
  window.__colony = stats;  // for tests
  window.__colonyDebug = { critters: () => critters.map((c) => ({ role: c.role, x: +c.x.toFixed(2), y: +c.y.toFixed(2), carrying: c.carrying, cooldown: c.throwCooldown, job: c.job ? { phase: c.job.phase, at: c.job.letter.at } : null })), monsters: () => monsters.map((m) => ({ x: +m.x.toFixed(2), y: +m.y.toFixed(2), belly: m.belly.length, hits: m.hits, cooldown: m.cooldown })) };

  function hitMonster(m) {
    m.hits++;
    m.flash = 8;
    stats.hits++;
    if (m.hits >= 4) {
      // pop: release everything it swallowed, respawn at an edge
      const p = toPx(m);
      bursts.push({ x: p.x, y: p.y, t: 14 });
      for (const li of m.belly) {
        const L = letters[li];
        L.at = 'ground';
        L.pos = { x: p.x + (Math.random() - 0.5) * 70, y: p.y + (Math.random() - 0.5) * 50 };
        L.angle = (Math.random() - 0.5) * 1.2;
        L.claimed = false;
      }
      m.belly = [];
      m.hits = 0;
      m.target = -1;
      const edge = Math.random() < 0.5 ? -1 : 1;
      m.x = edge * 0.98; m.y = (Math.random() * 2 - 1) * 0.8;
      m.vx = 0; m.vy = 0;
      m.cooldown = 150 + Math.random() * 120;
      stats.pops++;
    }
  }

  // --- jobs for the small castes ---------------------------------------------
  let mode = 'dismantle';
  let restT = 0;

  function claimWorkerJob() {
    if (mode === 'rest') return null;
    const ground = letters.filter((L) => L.at === 'ground' && !L.claimed);
    if (mode === 'dismantle') {
      const stray = ground.find((L) => !inPileZone(L.pos));
      if (stray) {
        stray.claimed = true;
        return { letter: stray, phase: 'fetch', dest: pileSlot(pileCount++) };
      }
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
    if (!ground.length) return null;
    const L = ground[Math.floor(Math.random() * ground.length)];
    L.claimed = true;
    return { letter: L, phase: 'fetch', dest: { x: L.home.x, y: L.home.y } };
  }

  function claimAmmo() {
    // slingers prefer loose letters, then the pile, then the text itself —
    // sacrificing words for defense
    const ground = letters.filter((L) => L.at === 'ground' && !L.claimed);
    let L = ground.find((G) => !inPileZone(G.pos)) || ground[0];
    if (!L) L = letters.find((G) => G.at === 'home' && !G.claimed);
    if (!L) return null;
    L.claimed = true;
    return { letter: L, phase: 'fetch' };
  }

  const jobValid = (job) => {
    const s = job.letter.at;
    return job.phase === 'deliver' || s === 'home' || s === 'ground';
  };

  function stepColony() {
    if (!cursor.active) { cursor.x = 2.2; cursor.y = 2.2; }
    cursor.vx = cursor.x - cursor.px; cursor.vy = cursor.y - cursor.py;
    cursor.px = cursor.x; cursor.py = cursor.y;

    // colony mode
    const homeCount = letters.filter((L) => L.at === 'home').length;
    const groundCount = letters.filter((L) => L.at === 'ground').length;
    // dismantle ends when the text is gone; rebuild ends when there is
    // nothing left to pick up (whatever's missing is in monster bellies or
    // someone's hands — the colony can't finish repairs until monsters pop)
    if (mode === 'dismantle' && homeCount === 0) {
      mode = 'rest'; restT = 75;
    } else if (mode === 'rebuild' && groundCount === 0) {
      mode = 'rest'; restT = 75;
      if (homeCount === letters.length) pileCount = 0;
    } else if (mode === 'rest' && --restT <= 0) {
      mode = homeCount > letters.length * 0.6 ? 'dismantle' : 'rebuild';
    }

    // nearest threat to a small critter: the cursor or any monster
    function threatFor(c) {
      let best = cursor.active ? cursor : null;
      let bd = best ? Math.hypot(cursor.x - c.x, cursor.y - c.y) : Infinity;
      for (const m of monsters) {
        const d = Math.hypot(m.x - c.x, m.y - c.y);
        if (d < bd) { bd = d; best = m; }
      }
      return best;
    }

    // --- small castes ---------------------------------------------------------
    for (const c of critters) {
      if (c.job && !jobValid(c.job)) { c.job.letter.claimed = false; c.job = null; }
      if (c.throwCooldown > 0) c.throwCooldown--;

      let target = null;
      if (c.role === 'worker') {
        if (!c.job) c.job = claimWorkerJob();
        if (c.job) target = toNorm(c.job.phase === 'fetch' ? (c.job.letter.at === 'home' ? c.job.letter.home : c.job.letter.pos) : c.job.dest);
      } else { // slinger
        if (c.carrying < 0 && !c.job && c.throwCooldown <= 0) c.job = claimAmmo();
        if (c.job && c.carrying < 0) {
          target = toNorm(c.job.letter.at === 'home' ? c.job.letter.home : c.job.letter.pos);
        } else if (c.carrying >= 0) {
          // armed: take a standoff position near the hungriest monster
          let m = monsters[0], bd = Infinity;
          for (const mm of monsters) {
            const d = Math.hypot(mm.x - c.x, mm.y - c.y);
            if (d < bd) { bd = d; m = mm; }
          }
          const away = Math.hypot(c.x - m.x, c.y - m.y) || 1;
          target = {
            x: m.x + ((c.x - m.x) / away) * 0.4,
            y: m.y + ((c.y - m.y) / away) * 0.4,
          };
          // in range and lined up? throw!
          if (bd < 0.55 && c.throwCooldown <= 0) {
            const L = letters[c.carrying];
            const aim = { x: m.x + m.vx * 8, y: m.y + m.vy * 8 };
            const dx = aim.x - c.x, dy = aim.y - c.y;
            const dl = Math.hypot(dx, dy) || 1;
            projectiles.push({ letter: c.carrying, x: c.x, y: c.y, vx: (dx / dl) * 0.055, vy: (dy / dl) * 0.055, spin: 0 });
            L.at = 'flying';
            L.claimed = false;
            c.carrying = -1;
            c.job = null;
            c.throwCooldown = 100 + Math.random() * 50;
            stats.thrown++;
          }
        }
      }
      if (!target) {
        if (--c.wanderT <= 0) {
          c.wanderT = 60 + Math.random() * 90;
          c.wander = { x: (Math.random() * 2 - 1) * 0.8, y: (Math.random() * 2 - 1) * 0.8 };
        }
        target = c.wander;
      }

      // slingers hold their ground near monsters; workers fear everything
      const threat = c.role === 'worker' ? threatFor(c) : (cursor.active ? cursor : null);
      steer(c, target, threat, 1);
      const dtNow = Math.hypot(target.x - c.x, target.y - c.y);

      // arrivals
      if (c.job && dtNow < 0.085) {
        const L = c.job.letter;
        if (c.job.phase === 'fetch') {
          if (L.at === 'home') L.span.style.visibility = 'hidden';
          L.at = 'held';
          c.carrying = letters.indexOf(L);
          c.job.phase = 'deliver';
          if (c.role === 'slinger') c.job = null; // armed — standoff logic takes over
        } else if (c.role === 'worker') {
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

      if (--c.blinkT <= 0) { c.blink = 5; c.blinkT = 90 + Math.random() * 150; }
      if (c.blink > 0) c.blink--;
    }

    // --- monsters ---------------------------------------------------------------
    for (const m of monsters) {
      if (m.flash > 0) m.flash--;
      let target;
      if (m.cooldown > 0) {
        m.cooldown--;
        m.target = -1;
        if (--m.wanderT <= 0 || !m.wander) {
          m.wanderT = 50 + Math.random() * 70;
          m.wander = { x: (Math.random() * 2 - 1) * 0.7, y: (Math.random() * 2 - 1) * 0.7 };
        }
        target = m.wander;
      } else {
        // hunt the nearest edible letter (home or ground; held ones are safe)
        if (m.target < 0 || !(letters[m.target].at === 'home' || letters[m.target].at === 'ground')) {
          let bd = Infinity;
          m.target = -1;
          const mp = toPx(m);
          for (let i = 0; i < letters.length; i++) {
            const L = letters[i];
            if (L.at !== 'home' && L.at !== 'ground') continue;
            const p = L.at === 'home' ? L.home : L.pos;
            const d = Math.hypot(p.x - mp.x, p.y - mp.y);
            if (d < bd) { bd = d; m.target = i; }
          }
        }
        if (m.target >= 0) {
          const L = letters[m.target];
          target = toNorm(L.at === 'home' ? L.home : L.pos);
        } else {
          target = { x: 0, y: 0 };
        }
      }
      const dt = steer(m, target, null, 0.62); // heavier, slower body
      m.mouth = Math.max(0, Math.min(1, (0.35 - dt) / 0.3));

      if (m.cooldown <= 0 && m.target >= 0 && dt < 0.1) {
        const L = letters[m.target];
        if (L.at === 'home') L.span.style.visibility = 'hidden';
        L.at = 'eaten';
        L.claimed = false;
        m.belly.push(m.target);
        m.target = -1;
        m.cooldown = 90 + Math.random() * 90;
        stats.eaten++;
      }
    }

    // --- projectiles --------------------------------------------------------------
    for (let i = projectiles.length - 1; i >= 0; i--) {
      const p = projectiles[i];
      p.x += p.vx; p.y += p.vy;
      p.spin += 0.35;
      let done = false;
      for (const m of monsters) {
        if (Math.hypot(m.x - p.x, m.y - p.y) < 0.11) {
          hitMonster(m);
          done = true;
          break;
        }
      }
      if (done || Math.abs(p.x) > 1.02 || Math.abs(p.y) > 1.02) {
        const L = letters[p.letter];
        L.at = 'ground';
        const lp = toPx({ x: Math.max(-0.97, Math.min(0.97, p.x)), y: Math.max(-0.97, Math.min(0.97, p.y)) });
        L.pos = { x: lp.x, y: lp.y };
        L.angle = (Math.random() - 0.5) * 1.2;
        projectiles.splice(i, 1);
      }
    }

    if (statusEl) {
      const eaten = monsters.reduce((s, m) => s + m.belly.length, 0);
      const verb = mode === 'dismantle' ? 'dismantling' : mode === 'rebuild' ? 'repairing' : 'plotting';
      statusEl.textContent =
        `5 workers + 2 slingers + 2 monsters, one shared ${N_PARAMS}-param steering net · ${verb} · ` +
        `${eaten} letters in monster bellies · ${stats.pops} monsters popped`;
    }
  }

  // --- drawing --------------------------------------------------------------------
  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // grounded letters
    for (const L of letters) {
      if (L.at !== 'ground') continue;
      ctx.save();
      ctx.translate(L.pos.x, L.pos.y);
      ctx.rotate(L.angle);
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = inkColor;
      ctx.fillText(L.ch, 0, 0);
      ctx.restore();
    }

    // flying letters
    for (const p of projectiles) {
      const pp = toPx(p);
      ctx.save();
      ctx.translate(pp.x, pp.y);
      ctx.rotate(p.spin);
      ctx.fillStyle = inkColor;
      ctx.fillText(letters[p.letter].ch, 0, 0);
      ctx.restore();
    }

    // pop bursts
    for (let i = bursts.length - 1; i >= 0; i--) {
      const b = bursts[i];
      const k = 1 - b.t / 14;
      ctx.strokeStyle = `rgba(167,139,250,${1 - k})`;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(b.x, b.y, 10 + k * 34, 0, Math.PI * 2);
      ctx.stroke();
      if (--b.t <= 0) bursts.splice(i, 1);
    }

    // monsters
    for (const m of monsters) {
      const p = toPx(m);
      const speed = Math.hypot(m.vx, m.vy);
      const ang = Math.atan2(m.vy, m.vx);
      const r = 13 + Math.min(5, m.belly.length * 0.5);
      const wob = Math.sin(now / 160 + m.seed) * 1.6;

      ctx.fillStyle = 'rgba(0,0,0,0.22)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + r * 0.72, r * 0.85, 3.2, 0, 0, Math.PI * 2);
      ctx.fill();

      // body
      ctx.save();
      ctx.translate(p.x, p.y + wob * 0.4);
      ctx.rotate(ang * 0.15);
      ctx.fillStyle = m.flash > 0 ? '#ede9fe' : '#7c3aed';
      ctx.beginPath();
      ctx.ellipse(0, 0, r * (1 + Math.min(0.25, speed * 6)), r * 0.92, 0, 0, Math.PI * 2);
      ctx.fill();
      // horns
      ctx.fillStyle = m.flash > 0 ? '#c4b5fd' : '#4c1d95';
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(s * r * 0.45, -r * 0.75);
        ctx.lineTo(s * r * 0.7, -r * 1.35);
        ctx.lineTo(s * r * 0.85, -r * 0.6);
        ctx.closePath();
        ctx.fill();
      }
      // angry eyes
      for (const s of [-1, 1]) {
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(s * r * 0.35, -r * 0.15, 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#1e1b4b';
        ctx.beginPath();
        ctx.arc(s * r * 0.35 + (speed > 0.002 ? Math.cos(ang) : 0) * 1.3, -r * 0.15 + (speed > 0.002 ? Math.sin(ang) : 0.4) * 1.3, 1.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = m.flash > 0 ? '#c4b5fd' : '#4c1d95';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(s * r * 0.12, -r * 0.48);
        ctx.lineTo(s * r * 0.6, -r * 0.28);
        ctx.stroke();
      }
      // mouth opens as it closes on a snack
      ctx.fillStyle = '#2e1065';
      ctx.beginPath();
      ctx.ellipse(0, r * 0.35, r * 0.32, r * 0.08 + m.mouth * r * 0.3, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // hit pips
      for (let i = 0; i < m.hits; i++) {
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(p.x - 12 + i * 8, p.y - r - 8, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // small castes
    for (const c of critters) {
      const p = toPx(c);
      const speed = Math.hypot(c.vx, c.vy);
      const ang = Math.atan2(c.vy, c.vx);
      const squash = Math.min(0.35, speed * 9);
      const bob = Math.sin(now / 130 + c.seed) * 1.3;

      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + 8, 7.5, 2.6, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(p.x, p.y + bob * 0.4);
      ctx.rotate(ang);
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.ellipse(0, 0, 8 * (1 + squash), 8 * (1 - squash * 0.6), 0, 0, Math.PI * 2);
      ctx.fill();
      if (c.role === 'slinger') {           // headband
        ctx.strokeStyle = 'rgba(120,53,15,0.85)';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(0, 0, 8 * (1 + squash) * 0.92, -2.6, -0.5);
        ctx.stroke();
      }
      ctx.restore();

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

      if (c.carrying >= 0) {
        const L = letters[c.carrying];
        ctx.save();
        ctx.translate(p.x, p.y - 15 + bob);
        // slingers wind up: their letter spins faster the closer a monster is
        const spin = c.role === 'slinger' ? now / 90 : Math.sin(now / 210 + c.seed) * 0.16;
        ctx.rotate(c.role === 'slinger' ? spin % (Math.PI * 2) : spin);
        ctx.fillStyle = inkColor;
        ctx.fillText(L.ch, 0, 0);
        ctx.restore();
      }
    }
  }

  // --- loop --------------------------------------------------------------------------
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
