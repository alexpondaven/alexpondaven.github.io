// The /colony/ ecosystem, now with continual learning. The paragraph is the
// environment; every letter is a physical object with a home slot.
//
// Castes (all movement steered by ONE frozen tiny MLP — see
// assets/worldmodel/critter_train.js):
//   workers (4) — dismantle the text into a pile, then rebuild it
//   slinger (1) — throws letters at the monster during raids
//   monster (1) — raids every so often and eats letters until driven off
//   elder   (1) — the critic: watches outcomes and hands out rewards
//
// The LEARNING happens in the decision layer, live in your browser:
//   - the slinger's aim (how far to lead a moving monster) starts naive and
//     hill-climbs from the elder's hit/miss rewards
//   - workers pick jobs with a tiny linear value model (distance, risk,
//     stray-ness) updated online from delivery/theft/drop rewards
// Learned state persists in localStorage, so the colony gets better across
// visits. The steering net itself stays frozen.
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
    c.x = Math.max(-0.955, Math.min(0.955, c.x + c.vx));
    c.y = Math.max(-0.955, Math.min(0.955, c.y + c.vy));
    return dt;
  }

  // --- letters ----------------------------------------------------------------
  const textEl = stage.querySelector('.colony-text');
  const raw = textEl.textContent;
  textEl.textContent = '';
  const letters = [];
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
    letters.push({ ch, span, home: { x: 0, y: 0 }, at: 'home', pos: { x: 0, y: 0 }, angle: 0, claimed: false, owner: null });
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

  // --- continual learning state (persists across visits) -----------------------
  const LEARN_KEY = 'colony-learn-v1';
  let learn = {
    aim: { lead: 0, sigma: 5, throws: 0, hits: 0, recent: [] },   // slinger
    jobW: [0.5, -0.5, -0.5, 0.3],                                  // worker value weights
    elder: { plus: 0, minus: 0 },
  };
  try {
    const saved = JSON.parse(localStorage.getItem(LEARN_KEY));
    if (saved && saved.aim && saved.jobW) learn = saved;
  } catch (e) { /* fresh colony */ }
  function saveLearn() {
    try { localStorage.setItem(LEARN_KEY, JSON.stringify(learn)); } catch (e) { /* private mode */ }
  }
  setInterval(saveLearn, 10000);
  document.addEventListener('visibilitychange', () => { if (document.hidden) saveLearn(); });

  const randn = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  // job value model: features -> scalar. Linear on purpose: four weights are
  // enough to express "prefer close, safe letters", they learn in seconds,
  // and you can read the learned policy right out of the numbers.
  function jobFeatures(c, L) {
    const p = L.at === 'home' ? L.home : L.pos;
    const n = toNorm(p);
    const dist = Math.hypot(n.x - c.x, n.y - c.y) / 2.8;            // 0..1
    let risk = 0;
    if (monster) risk = Math.max(0, 1 - Math.hypot(n.x - monster.x, n.y - monster.y) / 0.55);
    const stray = L.at === 'ground' && !inPileZone(L.pos) ? 1 : 0;
    return [1, dist, risk, stray];
  }
  const jobValue = (f) => f.reduce((s, v, i) => s + v * learn.jobW[i], 0);
  function rewardWorker(c, r) {
    if (!c.lastJobF) return;
    const v = jobValue(c.lastJobF);
    for (let i = 0; i < learn.jobW.length; i++) {
      learn.jobW[i] = Math.max(-3, Math.min(3, learn.jobW[i] + 0.08 * (r - v) * c.lastJobF[i]));
    }
    emitReward(c, r);
  }

  // the elder's visible verdicts
  const sparks = [];   // {x, y, txt, good, t}
  function emitReward(c, r) {
    const p = toPx(c);
    if (sparks.length > 5) sparks.shift();
    sparks.push({ x: p.x, y: p.y - 16, txt: (r > 0 ? '+' : '−') + Math.abs(r).toFixed(1).replace(/\.0$/, ''), good: r > 0, t: 45 });
    if (r > 0) learn.elder.plus++; else learn.elder.minus++;
    elder.interest = { x: c.x, y: c.y };
  }

  // --- cursor ------------------------------------------------------------------
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
    if (monster && Math.hypot(monster.x - n.x, monster.y - n.y) < 0.14) {
      hitMonster();
      return;
    }
    for (const c of critters) {
      const dx = c.x - n.x, dy = c.y - n.y;
      const d = Math.hypot(dx, dy);
      if (d < 0.45 && d > 1e-6) {
        const kick = 0.08 * (1 - d / 0.45);
        c.vx += (dx / d) * kick;
        c.vy += (dy / d) * kick;
        if (c.carrying >= 0) {
          const L = letters[c.carrying];
          L.at = 'ground';
          L.pos = toPx({ x: c.x, y: c.y });
          L.angle = (Math.random() - 0.5) * 0.9;
          L.claimed = false;
          if (c.role === 'worker') rewardWorker(c, -0.5);
          L.owner = null;
          c.carrying = -1;
          c.job = null;
        }
      }
    }
  });

  // --- castes -------------------------------------------------------------------
  const WORKER_COLORS = ['#818cf8', '#a78bfa', '#6ee7b7', '#7dd3fc'];
  const critters = [
    ...WORKER_COLORS.map((color, i) => makeSmall('worker', color, i)),
    makeSmall('slinger', '#fb923c', 5),
  ];
  const elder = { role: 'elder', x: 0.6, y: 0.6, vx: 0, vy: 0, interest: null, wanderT: 0, wander: { x: 0.5, y: 0.5 }, blinkT: 200, blink: 0, seed: 4242 };
  function makeSmall(role, color, i) {
    return {
      role, color,
      x: (Math.random() * 2 - 1) * 0.8, y: (Math.random() * 2 - 1) * 0.8,
      vx: 0, vy: 0,
      carrying: -1, job: null, lastJobF: null,
      wanderT: 0, wander: { x: 0, y: 0 },
      blinkT: 60 + Math.random() * 120, blink: 0,
      throwCooldown: 0, thrownLead: 0,
      seed: i * 977,
    };
  }

  // one monster, arriving in RAIDS with calm in between
  let monster = null;
  let raidTimer = 500 + Math.random() * 400;   // first raid ~20-30s in
  function spawnMonster() {
    const edge = Math.random() < 0.5 ? -1 : 1;
    monster = {
      x: edge * 0.95, y: (Math.random() * 2 - 1) * 0.7,
      vx: 0, vy: 0,
      belly: [], hits: 0, target: -1,
      cooldown: 40,
      flash: 0, mouth: 0,
      wanderT: 0, wander: null,
      seed: 313,
    };
  }
  const projectiles = [];
  const bursts = [];
  const stats = { eaten: 0, thrown: 0, hits: 0, pops: 0, delivered: 0 };
  window.__colony = stats;
  window.__colonyLearn = learn;

  function hitMonster() {
    if (!monster) return;
    monster.hits++;
    monster.flash = 8;
    stats.hits++;
    if (monster.hits >= 3) {
      const p = toPx(monster);
      bursts.push({ x: p.x, y: p.y, t: 16 });
      for (const li of monster.belly) {
        const L = letters[li];
        L.at = 'ground';
        L.pos = { x: p.x + (Math.random() - 0.5) * 70, y: p.y + (Math.random() - 0.5) * 50 };
        L.angle = (Math.random() - 0.5) * 1.2;
        L.claimed = false;
      }
      monster = null;
      raidTimer = 1200 + Math.random() * 900;   // 40-70s of calm
      stats.pops++;
    }
  }

  // --- jobs ---------------------------------------------------------------------
  let mode = 'dismantle';
  let restT = 0;

  // workers CHOOSE among candidate jobs with the learned value model
  function claimWorkerJob(c) {
    if (mode === 'rest') return null;
    const cands = [];
    const ground = letters.filter((L) => L.at === 'ground' && !L.claimed);
    const strays = ground.filter((L) => !inPileZone(L.pos));
    if (mode === 'dismantle') {
      for (const s of strays.slice(0, 2)) cands.push({ letter: s, dest: () => pileSlot(pileCount++) });
      const live = words.filter((w) => w.some((i) => letters[i].at === 'home' && !letters[i].claimed));
      for (let k = 0; k < 3 && live.length; k++) {
        const w = live[Math.floor(Math.random() * live.length)];
        for (let j = w.length - 1; j >= 0; j--) {
          const L = letters[w[j]];
          if (L.at === 'home' && !L.claimed && !cands.some((cd) => cd.letter === L)) {
            cands.push({ letter: L, dest: () => pileSlot(pileCount++) });
            break;
          }
        }
      }
    } else {
      for (let k = 0; k < 4 && ground.length; k++) {
        const L = ground[Math.floor(Math.random() * ground.length)];
        if (!cands.some((cd) => cd.letter === L)) cands.push({ letter: L, dest: () => ({ x: L.home.x, y: L.home.y }) });
      }
    }
    if (!cands.length) return null;
    // softmax over learned values
    const feats = cands.map((cd) => jobFeatures(c, cd.letter));
    const vals = feats.map(jobValue);
    const mx = Math.max(...vals);
    const exps = vals.map((v) => Math.exp((v - mx) / 0.25));
    const Z = exps.reduce((a, b) => a + b, 0);
    let pick = Math.random() * Z, idx = 0;
    for (; idx < exps.length - 1; idx++) { pick -= exps[idx]; if (pick <= 0) break; }
    const chosen = cands[idx];
    chosen.letter.claimed = true;
    chosen.letter.owner = c;
    c.lastJobF = feats[idx];
    return { letter: chosen.letter, phase: 'fetch', dest: chosen.dest() };
  }

  function claimAmmo(c) {
    const ground = letters.filter((L) => L.at === 'ground' && !L.claimed);
    let L = ground.find((G) => !inPileZone(G.pos)) || ground[0];
    if (!L) L = letters.find((G) => G.at === 'home' && !G.claimed);
    if (!L) return null;
    L.claimed = true;
    L.owner = c;
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

    const homeCount = letters.filter((L) => L.at === 'home').length;
    const groundCount = letters.filter((L) => L.at === 'ground').length;
    if (mode === 'dismantle' && homeCount === 0) {
      mode = 'rest'; restT = 75;
    } else if (mode === 'rebuild' && groundCount === 0) {
      mode = 'rest'; restT = 75;
      if (homeCount === letters.length) pileCount = 0;
    } else if (mode === 'rest' && --restT <= 0) {
      mode = homeCount > letters.length * 0.6 ? 'dismantle' : 'rebuild';
    }

    // raid pacing
    if (!monster && --raidTimer <= 0) spawnMonster();

    function threatFor(c) {
      let best = cursor.active ? cursor : null;
      let bd = best ? Math.hypot(cursor.x - c.x, cursor.y - c.y) : Infinity;
      if (monster) {
        const d = Math.hypot(monster.x - c.x, monster.y - c.y);
        if (d < bd) { bd = d; best = monster; }
      }
      return best;
    }

    // --- small castes ----------------------------------------------------------
    for (const c of critters) {
      if (c.job && !jobValid(c.job)) { c.job.letter.claimed = false; c.job = null; }
      if (c.throwCooldown > 0) c.throwCooldown--;

      let target = null;
      if (c.role === 'worker') {
        if (!c.job) c.job = claimWorkerJob(c);
        if (c.job) target = toNorm(c.job.phase === 'fetch' ? (c.job.letter.at === 'home' ? c.job.letter.home : c.job.letter.pos) : c.job.dest);
      } else { // slinger
        if (c.carrying < 0 && !c.job && c.throwCooldown <= 0) c.job = claimAmmo(c);
        if (c.job && c.carrying < 0) {
          target = toNorm(c.job.letter.at === 'home' ? c.job.letter.home : c.job.letter.pos);
        } else if (c.carrying >= 0 && monster) {
          const away = Math.hypot(c.x - monster.x, c.y - monster.y) || 1;
          target = {
            x: monster.x + ((c.x - monster.x) / away) * 0.4,
            y: monster.y + ((c.y - monster.y) / away) * 0.4,
          };
          if (away < 0.55 && c.throwCooldown <= 0) {
            // LEARNED aim: lead the target by a learned number of steps,
            // plus exploration noise the elder's rewards will shrink
            const L = letters[c.carrying];
            const lead = learn.aim.lead + randn() * learn.aim.sigma;
            c.thrownLead = lead;
            const aim = { x: monster.x + monster.vx * lead, y: monster.y + monster.vy * lead };
            const dx = aim.x - c.x, dy = aim.y - c.y;
            const dl = Math.hypot(dx, dy) || 1;
            projectiles.push({ letter: c.carrying, x: c.x, y: c.y, vx: (dx / dl) * 0.05, vy: (dy / dl) * 0.05, spin: 0, lead, thrower: c });
            L.at = 'flying';
            L.claimed = false;
            L.owner = null;
            c.carrying = -1;
            c.job = null;
            c.throwCooldown = 70 + Math.random() * 40;
            stats.thrown++;
            learn.aim.throws++;
          }
        } else if (c.carrying >= 0) {
          // calm: stand guard near the middle with the ammo ready
          if (--c.wanderT <= 0) {
            c.wanderT = 120 + Math.random() * 120;
            c.wander = { x: (Math.random() * 2 - 1) * 0.4, y: (Math.random() * 2 - 1) * 0.4 };
          }
          target = c.wander;
        }
      }
      if (!target) {
        if (--c.wanderT <= 0) {
          c.wanderT = 60 + Math.random() * 90;
          c.wander = { x: (Math.random() * 2 - 1) * 0.8, y: (Math.random() * 2 - 1) * 0.8 };
        }
        target = c.wander;
      }

      const threat = c.role === 'worker' ? threatFor(c) : (cursor.active ? cursor : null);
      steer(c, target, threat, 1);
      const dtNow = Math.hypot(target.x - c.x, target.y - c.y);

      if (c.job && dtNow < 0.085) {
        const L = c.job.letter;
        if (c.job.phase === 'fetch') {
          if (L.at === 'home') L.span.style.visibility = 'hidden';
          L.at = 'held';
          c.carrying = letters.indexOf(L);
          c.job.phase = 'deliver';
          if (c.role === 'slinger') c.job = null;
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
          L.owner = null;
          stats.delivered++;
          rewardWorker(c, 1);
          c.carrying = -1;
          c.job = null;
        }
      }

      if (--c.blinkT <= 0) { c.blink = 5; c.blinkT = 90 + Math.random() * 150; }
      if (c.blink > 0) c.blink--;
    }

    // --- monster -----------------------------------------------------------------
    if (monster) {
      const m = monster;
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
        target = m.target >= 0 ? toNorm(letters[m.target].at === 'home' ? letters[m.target].home : letters[m.target].pos) : { x: 0, y: 0 };
      }
      const dt = steer(m, target, null, 0.62);
      m.mouth = Math.max(0, Math.min(1, (0.35 - dt) / 0.3));

      if (m.cooldown <= 0 && m.target >= 0 && dt < 0.1) {
        const L = letters[m.target];
        if (L.owner && L.owner.role === 'worker') rewardWorker(L.owner, -1.5);
        if (L.at === 'home') L.span.style.visibility = 'hidden';
        L.at = 'eaten';
        L.claimed = false;
        L.owner = null;
        m.belly.push(m.target);
        m.target = -1;
        m.cooldown = 100 + Math.random() * 80;
        stats.eaten++;
      }
    }

    // --- projectiles: outcomes feed the aim learner --------------------------------
    for (let i = projectiles.length - 1; i >= 0; i--) {
      const p = projectiles[i];
      p.x += p.vx; p.y += p.vy;
      p.spin += 0.3;
      let outcome = null;
      if (monster && Math.hypot(monster.x - p.x, monster.y - p.y) < 0.09) outcome = 'hit';
      else if (Math.abs(p.x) > 1.02 || Math.abs(p.y) > 1.02) outcome = 'miss';
      if (outcome) {
        const L = letters[p.letter];
        L.at = 'ground';
        const lp = toPx({ x: Math.max(-0.97, Math.min(0.97, p.x)), y: Math.max(-0.97, Math.min(0.97, p.y)) });
        L.pos = { x: lp.x, y: lp.y };
        L.angle = (Math.random() - 0.5) * 1.2;
        if (outcome === 'hit') {
          hitMonster();
          learn.aim.hits++;
          // pull the lead toward what just worked, calm the exploration
          learn.aim.lead += 0.35 * (p.lead - learn.aim.lead);
          learn.aim.sigma = Math.max(1, learn.aim.sigma * 0.9);
          emitReward(p.thrower, 1);
        } else {
          learn.aim.sigma = Math.min(8, learn.aim.sigma * 1.05);
          emitReward(p.thrower, -1);
        }
        learn.aim.recent.push(outcome === 'hit' ? 1 : 0);
        if (learn.aim.recent.length > 20) learn.aim.recent.shift();
        projectiles.splice(i, 1);
      }
    }

    // --- elder: drift toward whatever just happened --------------------------------
    {
      let target;
      if (elder.interest) {
        target = elder.interest;
        if (Math.hypot(target.x - elder.x, target.y - elder.y) < 0.2) elder.interest = null;
      } else {
        if (--elder.wanderT <= 0) {
          elder.wanderT = 150 + Math.random() * 150;
          elder.wander = { x: (Math.random() * 2 - 1) * 0.5, y: (Math.random() * 2 - 1) * 0.5 };
        }
        target = elder.wander;
      }
      steer(elder, target, null, 0.45);
      if (--elder.blinkT <= 0) { elder.blink = 6; elder.blinkT = 150 + Math.random() * 200; }
      if (elder.blink > 0) elder.blink--;
    }

    if (statusEl) {
      const recent = learn.aim.recent;
      const rate = recent.length ? Math.round((recent.reduce((a, b) => a + b, 0) / recent.length) * 100) : null;
      const aimTxt = learn.aim.throws
        ? `slinger aim: lead ${learn.aim.lead.toFixed(1)} (${rate === null ? '—' : rate + '% of last ' + recent.length} · lifetime ${learn.aim.hits}/${learn.aim.throws})`
        : 'slinger aim: untrained';
      const raidTxt = monster ? 'RAID' : `next raid ~${Math.ceil(raidTimer / 30)}s`;
      statusEl.textContent =
        `${mode === 'dismantle' ? 'dismantling' : mode === 'rebuild' ? 'repairing' : 'resting'} · ${raidTxt} · ` +
        `${aimTxt} · elder verdicts +${learn.elder.plus}/−${learn.elder.minus} · learning saved locally`;
    }
  }

  // --- drawing ---------------------------------------------------------------------
  // rendered positions are smoothed so 30Hz steering reads as glide, not jitter
  function smooth(e) {
    const p = toPx(e);
    e.sx = e.sx === undefined ? p.x : e.sx + (p.x - e.sx) * 0.35;
    e.sy = e.sy === undefined ? p.y : e.sy + (p.y - e.sy) * 0.35;
    return { x: e.sx, y: e.sy };
  }

  function drawEyes(p, look, blink, dark) {
    for (const side of [-1, 1]) {
      const ex = p.x + side * 3.4, ey = p.y - 2.5;
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      if (blink > 0) {
        ctx.fillRect(ex - 2, ey - 0.6, 4, 1.2);
      } else {
        ctx.arc(ex, ey, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = dark;
        ctx.beginPath();
        ctx.arc(ex + look.x * 1.1, ey + look.y * 1.1, 1.25, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

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

    for (const p of projectiles) {
      const pp = toPx(p);
      ctx.save();
      ctx.translate(pp.x, pp.y);
      ctx.rotate(p.spin);
      ctx.fillStyle = inkColor;
      ctx.fillText(letters[p.letter].ch, 0, 0);
      ctx.restore();
    }

    for (let i = bursts.length - 1; i >= 0; i--) {
      const b = bursts[i];
      const k = 1 - b.t / 16;
      ctx.strokeStyle = `rgba(167,139,250,${0.7 * (1 - k)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(b.x, b.y, 10 + k * 30, 0, Math.PI * 2);
      ctx.stroke();
      if (--b.t <= 0) bursts.splice(i, 1);
    }

    // reward sparks
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i];
      s.y -= 0.5;
      const a = Math.min(1, s.t / 20);
      ctx.fillStyle = s.good ? `rgba(52,211,153,${a})` : `rgba(251,113,133,${a})`;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(s.txt, s.x, s.y);
      ctx.font = font;
      if (--s.t <= 0) sparks.splice(i, 1);
    }

    if (monster) {
      const m = monster;
      const p = smooth(m);
      const speed = Math.hypot(m.vx, m.vy);
      const ang = Math.atan2(m.vy, m.vx);
      const r = 13 + Math.min(5, m.belly.length * 0.5);

      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + r * 0.72, r * 0.85, 3.2, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.fillStyle = m.flash > 0 ? '#ede9fe' : '#7c3aed';
      ctx.beginPath();
      ctx.ellipse(0, 0, r * (1 + Math.min(0.18, speed * 5)), r * 0.92, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = m.flash > 0 ? '#c4b5fd' : '#4c1d95';
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(s * r * 0.45, -r * 0.75);
        ctx.lineTo(s * r * 0.7, -r * 1.35);
        ctx.lineTo(s * r * 0.85, -r * 0.6);
        ctx.closePath();
        ctx.fill();
      }
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
      ctx.fillStyle = '#2e1065';
      ctx.beginPath();
      ctx.ellipse(0, r * 0.35, r * 0.32, r * 0.08 + m.mouth * r * 0.3, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      for (let i = 0; i < m.hits; i++) {
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(p.x - 8 + i * 8, p.y - r - 8, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (const c of critters) {
      const p = smooth(c);
      const speed = Math.hypot(c.vx, c.vy);
      const ang = Math.atan2(c.vy, c.vx);
      const squash = Math.min(0.2, speed * 6);
      // bob only while idling — moving critters glide
      const idle = Math.max(0, 1 - speed * 30);
      const bob = Math.sin(now / 260 + c.seed) * 0.8 * idle;

      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + 8, 7.5, 2.6, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(p.x, p.y + bob);
      ctx.rotate(ang * Math.min(1, speed * 40));
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.ellipse(0, 0, 8 * (1 + squash), 8 * (1 - squash * 0.6), 0, 0, Math.PI * 2);
      ctx.fill();
      if (c.role === 'slinger') {
        ctx.strokeStyle = 'rgba(120,53,15,0.85)';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(0, 0, 7.4, -2.6, -0.5);
        ctx.stroke();
      }
      ctx.restore();

      const look = speed > 0.004 ? { x: Math.cos(ang), y: Math.sin(ang) } : { x: 0, y: 0.3 };
      drawEyes({ x: p.x, y: p.y + bob }, look, c.blink, '#1e1b4b');

      if (c.carrying >= 0) {
        const L = letters[c.carrying];
        ctx.save();
        ctx.translate(p.x, p.y - 15 + bob);
        ctx.rotate(Math.sin(now / 300 + c.seed) * 0.12);
        ctx.fillStyle = inkColor;
        ctx.fillText(L.ch, 0, 0);
        ctx.restore();
      }
    }

    // the elder: bigger, ivory, crowned
    {
      const p = smooth(elder);
      const speed = Math.hypot(elder.vx, elder.vy);
      const ang = Math.atan2(elder.vy, elder.vx);
      const bob = Math.sin(now / 320) * 0.7;
      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + 10, 9.5, 3, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.save();
      ctx.translate(p.x, p.y + bob);
      ctx.fillStyle = '#e7e5e4';
      ctx.beginPath();
      ctx.ellipse(0, 0, 10.5, 9.5, 0, 0, Math.PI * 2);
      ctx.fill();
      // crown
      ctx.fillStyle = '#fbbf24';
      ctx.beginPath();
      ctx.moveTo(-6, -8);
      ctx.lineTo(-6, -13);
      ctx.lineTo(-3, -10);
      ctx.lineTo(0, -14);
      ctx.lineTo(3, -10);
      ctx.lineTo(6, -13);
      ctx.lineTo(6, -8);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      const look = speed > 0.003 ? { x: Math.cos(ang), y: Math.sin(ang) } : { x: 0, y: 0.2 };
      drawEyes({ x: p.x, y: p.y + bob + 1 }, look, elder.blink, '#44403c');
    }
  }

  // --- loop ------------------------------------------------------------------------
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
