// The /colony/ word-forge, with LOCAL OBSERVABILITY and a real economy.
//
// The paragraph is a pantry of letters. Each wordsmith can only see the
// letters inside its sensing ring (drawn faintly in its color) — so what it
// can spell depends on where it stands. Before proposing, it SCOUTS: each
// wordsmith learns its own spatial value map (a coarse grid of "how much
// net reward have words composed from this region earned me?") and walks to
// a promising cell before looking around and committing to a word.
//
// The economy: the elder's verdict is only GROSS income. Net reward =
// verdict − effort (steps spent hauling letters to the lane). Exotic
// letters make novel words but cost real travel; two wordsmiths harvesting
// the same region steal each other's tiles and eat the loss — so the value
// maps push them apart into foraging territories. That spatial division of
// labor is the strategy this page exists to let you watch emerge.
//
// The elder is a NOVELTY CRITIC with a heritable taste genome (novelty,
// flow, length): unseen words score high, repeats decay hard. Wordsmiths
// learn word-construction by REINFORCE (bigram + length preferences) on
// NET reward, so cost-of-assembly shapes the language itself.
//
// Above it all, the dynasty: wordsmiths age, die and hatch heirs with
// mutated brains (value maps included); elders reign and are succeeded
// along a fitness-selected lineage by the Ancestor. Movement for everyone
// is ONE frozen tiny steering MLP (assets/worldmodel/critter_train.js).
// Everything learned persists in localStorage.
const stage = document.getElementById('colony-stage');
const statusEl = document.getElementById('colony-status');
const eventEl = document.getElementById('colony-event');
const panelEl = document.getElementById('colony-panel');

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

  // --- the pantry: the paragraph as letters -----------------------------------
  const textEl = stage.querySelector('.colony-text');
  const raw = textEl.textContent;
  textEl.textContent = '';
  const letters = [];
  for (const ch of raw) {
    if (/\s/.test(ch)) {
      textEl.appendChild(document.createTextNode(ch));
      continue;
    }
    const span = document.createElement('span');
    span.className = 'colony-ch';
    span.textContent = ch;
    textEl.appendChild(span);
    letters.push({ ch, low: ch.toLowerCase(), span, home: { x: 0, y: 0 }, at: 'home', pos: { x: 0, y: 0 }, claimed: false });
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

  // offering lanes: one per worker, in the bottom band
  const N_WORKERS = 3;
  const laneY = (i) => H - 112 + i * 38;
  const laneSlot = (i, k) => ({ x: W / 2 - 70 + k * 22, y: laneY(i) });

  const randn = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const roman = (n) => {
    const T = [[100, 'C'], [90, 'XC'], [50, 'L'], [40, 'XL'], [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I']];
    let s = '';
    for (const [v, r] of T) while (n >= v) { s += r; n -= v; }
    return s || 'I';
  };
  function announce(txt) {
    if (!eventEl) return;
    eventEl.textContent = txt;
    eventEl.style.opacity = '1';
    clearTimeout(announce.t);
    announce.t = setTimeout(() => { eventEl.style.opacity = '0.4'; }, 7000);
  }

  // --- dynasty state (persists) --------------------------------------------------
  const AZ = 'abcdefghijklmnopqrstuvwxyz';
  const LEARN_KEY = 'colony-wordforge-v2';
  // the scouting grid: 4x3 cells over the text region of the stage
  const CELL_COLS = 4, CELL_ROWS = 3;
  const CELL_Y_MIN = -0.95, CELL_Y_MAX = 0.35;
  const cellOf = (x, y) => {
    const col = Math.max(0, Math.min(CELL_COLS - 1, Math.floor(((x + 1) / 2) * CELL_COLS)));
    const row = Math.max(0, Math.min(CELL_ROWS - 1, Math.floor(((y - CELL_Y_MIN) / (CELL_Y_MAX - CELL_Y_MIN)) * CELL_ROWS)));
    return row * CELL_COLS + col;
  };
  const cellCenter = (idx) => ({
    x: -1 + ((idx % CELL_COLS) + 0.5) * (2 / CELL_COLS),
    y: CELL_Y_MIN + (Math.floor(idx / CELL_COLS) + 0.5) * ((CELL_Y_MAX - CELL_Y_MIN) / CELL_ROWS),
  });
  const freshBrain = () => ({
    big: Array.from({ length: 27 }, () => Array.from({ length: 26 }, () => randn() * 0.1)),
    lenLog: [0, 0, 0, 0],   // word lengths 3..6
    base: 0.5,              // REINFORCE baseline
    cells: Array.from({ length: CELL_COLS * CELL_ROWS }, () => 0), // learned value of scouting each region
    avgR: 0, words: 0,      // lifetime accounting for the panel
    gen: 1,
  });
  const freshTaste = () => ({ novelty: 1, flow: 0.7, length: 0.7 });
  let learn = {
    brains: Array.from({ length: N_WORKERS }, freshBrain),
    elder: { taste: freshTaste(), ordinal: 1 },
    reign: { steps: 0, newWords: 0, lettersPlaced: 0 },
    lineage: { ancestors: [], sigmaMut: 0.3, fitHistory: [] },
    archive: {},            // word -> times seen (the novelty memory)
    best: null,             // {word, score}
  };
  try {
    const saved = JSON.parse(localStorage.getItem(LEARN_KEY));
    if (saved && saved.brains && saved.elder && saved.archive) learn = saved;
  } catch (e) { /* fresh dynasty */ }
  function saveLearn() {
    try { localStorage.setItem(LEARN_KEY, JSON.stringify(learn)); } catch (e) { /* private mode */ }
  }
  setInterval(saveLearn, 10000);
  document.addEventListener('visibilitychange', () => { if (document.hidden) saveLearn(); });
  window.__colonyLearn = learn;

  // reign fitness — the ground truth: how much NEW vocabulary this elder's
  // taste coaxed out of the colony, per minute
  const reignFitness = (r) => (2 * r.newWords + 0.05 * r.lettersPlaced) / Math.max(0.5, r.steps / 1800);

  const isVowel = (ch) => 'aeiou'.includes(ch);
  const flowOf = (w) => {
    if (w.length < 2) return 0;
    let alt = 0;
    for (let i = 1; i < w.length; i++) if (isVowel(w[i]) !== isVowel(w[i - 1])) alt++;
    return alt / (w.length - 1);
  };

  // the elder's verdict — novelty first, shaped by its heritable taste
  function judgeWord(word) {
    const t = elder.taste;
    const seen = learn.archive[word] || 0;
    const novelty = 1 / (1 + seen);
    const flow = flowOf(word);
    const lenScore = (word.length - 3) / 3;
    const wsum = t.novelty + t.flow + t.length;
    const score = 3 * (t.novelty * novelty + t.flow * flow + t.length * lenScore) / Math.max(0.3, wsum);
    learn.archive[word] = seen + 1;
    // keep the archive bounded
    const keys = Object.keys(learn.archive);
    if (keys.length > 400) for (const k of keys.slice(0, 40)) delete learn.archive[k];
    if (seen === 0) learn.reign.newWords++;
    if (!learn.best || score > learn.best.score) learn.best = { word, score: +score.toFixed(2) };
    return { score, novel: seen === 0 };
  }

  // --- word proposal & REINFORCE --------------------------------------------------
  // LOCAL OBSERVABILITY: a wordsmith only sees letters inside its sensing
  // ring, so the pool it can spell from depends on where it stands
  const SENSE = 0.5;
  function localCounts(c) {
    const counts = {};
    for (const L of letters) {
      if (L.at !== 'home' || L.claimed || !AZ.includes(L.low)) continue;
      const n = toNorm(L.home);
      if (Math.hypot(n.x - c.x, n.y - c.y) < SENSE) counts[L.low] = (counts[L.low] || 0) + 1;
    }
    return counts;
  }
  function sampleMasked(logits, counts) {
    // softmax over letters that are still available in the pantry
    let Z = 0;
    const p = new Array(26).fill(0);
    for (let i = 0; i < 26; i++) {
      if ((counts[AZ[i]] || 0) > 0) { p[i] = Math.exp(Math.max(-8, Math.min(8, logits[i]))); Z += p[i]; }
    }
    if (Z === 0) return null;
    let pick = Math.random() * Z;
    for (let i = 0; i < 26; i++) { pick -= p[i]; if (p[i] > 0 && pick <= 0) return { i, prob: p[i] / Z }; }
    return null;
  }
  function proposeWord(brain, counts) {
    // length 3..6 via learned preferences
    const lp = brain.lenLog.map((v) => Math.exp(v));
    const lz = lp.reduce((a, b) => a + b, 0);
    let pick = Math.random() * lz, li = 0;
    for (; li < 3; li++) { pick -= lp[li]; if (pick <= 0) break; }
    const targetLen = 3 + li;
    let prev = 26; // start token
    const chosen = [], trans = [];
    for (let k = 0; k < targetLen; k++) {
      const s = sampleMasked(brain.big[prev], counts);
      if (!s) break;
      const ch = AZ[s.i];
      chosen.push(ch);
      trans.push({ prev, next: s.i, prob: s.prob });
      counts[ch]--;
      prev = s.i;
    }
    if (chosen.length < 3) return null;
    return { word: chosen.join(''), trans, lenIdx: li };
  }
  function reinforce(brain, proposal, score) {
    const adv = score - brain.base;
    brain.base = 0.9 * brain.base + 0.1 * score;
    for (const t of proposal.trans) {
      brain.big[t.prev][t.next] = Math.max(-4, Math.min(4, brain.big[t.prev][t.next] + 0.5 * adv * (1 - t.prob)));
    }
    const lp = brain.lenLog.map((v) => Math.exp(v));
    const lz = lp.reduce((a, b) => a + b, 0);
    brain.lenLog[proposal.lenIdx] = Math.max(-3, Math.min(3, brain.lenLog[proposal.lenIdx] + 0.3 * adv * (1 - lp[proposal.lenIdx] / lz)));
  }

  // --- cursor ------------------------------------------------------------------------
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
    for (const c of critters) {
      if (!c.alive) continue;
      const dx = c.x - n.x, dy = c.y - n.y;
      const d = Math.hypot(dx, dy);
      if (d < 0.45 && d > 1e-6) {
        const kick = 0.08 * (1 - d / 0.45);
        c.vx += (dx / d) * kick;
        c.vy += (dy / d) * kick;
        if (c.carrying >= 0) {   // startled: the letter lands where it falls
          const L = letters[c.carrying];
          L.at = 'ground';
          L.pos = toPx({ x: c.x, y: c.y });
          c.carrying = -1;
        }
      }
    }
  });

  // --- castes -------------------------------------------------------------------------
  const WORKER_COLORS = ['#818cf8', '#6ee7b7', '#f9a8d4'];
  function makeWorker(i, brain) {
    return {
      role: 'worker', color: WORKER_COLORS[i], lane: i, brain,
      alive: true, fade: 1, hatch: 0,
      age: 0, lifespan: 5400 + Math.random() * 2700,
      energy: 0.8,            // fed by the elder's morsels; effort drains it
      x: (Math.random() * 2 - 1) * 0.7, y: (Math.random() * 2 - 1) * 0.5,
      vx: 0, vy: 0,
      state: 'rest', restT: 30 + Math.random() * 60,
      scoutCell: -1, dwellT: 0, effort: 0, eating: -1,
      proposal: null, letterIdx: 0, carrying: -1, placed: [],
      lastWord: null,         // {word, score, cost, net} for the panel
      wanderT: 0, wander: { x: 0, y: 0 },
      blinkT: 60 + Math.random() * 120, blink: 0,
      seed: i * 977,
    };
  }
  const critters = Array.from({ length: N_WORKERS }, (_, i) => makeWorker(i, learn.brains[i]));
  const eggs = [];

  const elder = {
    role: 'elder', alive: true, fade: 1,
    x: 0.5, y: 0.5, vx: 0, vy: 0,
    taste: learn.elder.taste,
    queue: [],              // lanes awaiting judgment
    judging: -1, judgeT: 0,
    wanderT: 0, wander: { x: 0.4, y: 0.4 },
    blinkT: 200, blink: 0, hop: 0,
  };
  let interregnum = 0, ancestorGlow = 0, panelT = 0;

  const verdicts = [];      // {x, y, txt, word, savored, t}
  const flights = [];       // letters flying home: {letter, fx, fy, t, dur}
  const morsels = [];       // food the elder conjures: {x, y, vx, vy, settled}
  const stats = { proposed: 0, judged: 0, savored: 0, newWords: 0, successions: 0, hatched: 0, morselsMade: 0, morselsEaten: 0, starved: 0, contentions: 0 };
  window.__colony = stats;

  // a pleased elder PRODUCES: morsels of food arc out of a good verdict, and
  // eating them is the only way wordsmiths refill the energy that hauling
  // burns — the reward signal is also the food chain
  function conjureMorsels(score, from) {
    const n = Math.max(0, Math.min(4, Math.round(score)));
    for (let i = 0; i < n; i++) {
      const ang = Math.PI * (0.9 + Math.random() * 1.2);
      morsels.push({
        x: from.x, y: from.y,
        vx: Math.cos(ang) * (1.2 + Math.random()), vy: -2 - Math.random() * 1.6,
        settled: false,
      });
      stats.morselsMade++;
    }
    if (morsels.length > 24) morsels.splice(0, morsels.length - 24);
    return n;
  }

  // --- the Ancestor: succession ---------------------------------------------------------
  function endReign() {
    const fit = reignFitness(learn.reign);
    const entry = { ord: learn.elder.ordinal, taste: { ...elder.taste }, fit: +fit.toFixed(2), reignS: Math.round(learn.reign.steps / 30) };
    const anc = learn.lineage.ancestors;
    anc.push(entry);
    anc.sort((a, b) => b.fit - a.fit);
    learn.lineage.ancestors = [...anc.slice(0, 6), ...anc.filter((a) => !anc.slice(0, 6).includes(a)).slice(-4)];
    learn.lineage.fitHistory.push(entry.fit);
    if (learn.lineage.fitHistory.length > 12) learn.lineage.fitHistory.shift();
    const h = learn.lineage.fitHistory;
    if (h.length >= 4) {
      const prev = (h[h.length - 4] + h[h.length - 3]) / 2;
      const now = (h[h.length - 2] + h[h.length - 1]) / 2;
      learn.lineage.sigmaMut = now > prev + 0.2
        ? Math.max(0.08, learn.lineage.sigmaMut * 0.85)
        : Math.min(0.7, learn.lineage.sigmaMut * 1.15);
    }
    elder.alive = false;
    elder.queue = [];
    elder.judging = -1;
    interregnum = 140;
    ancestorGlow = 140;
    stats.successions++;
    announce(`Elder ${roman(entry.ord)}'s reign ends — ${entry.fit >= 0 ? '+' : ''}${entry.fit} new-word fitness`);
  }
  function crownSuccessor() {
    const anc = learn.lineage.ancestors;
    let parent = anc[0];
    if (anc.length > 1) {
      const mx = Math.max(...anc.map((a) => a.fit));
      const exps = anc.map((a) => Math.exp((a.fit - mx) / 2));
      const Z = exps.reduce((a, b) => a + b, 0);
      let pick = Math.random() * Z;
      for (let i = 0; i < anc.length; i++) { pick -= exps[i]; if (pick <= 0) { parent = anc[i]; break; } }
    }
    const t = {};
    for (const k of Object.keys(parent.taste)) t[k] = Math.max(0.05, Math.min(2, parent.taste[k] + randn() * learn.lineage.sigmaMut));
    learn.elder.ordinal++;
    learn.elder.taste = t;
    elder.taste = t;
    elder.alive = true;
    elder.fade = 0;
    learn.reign = { steps: 0, newWords: 0, lettersPlaced: 0 };
    // words finished during the interregnum are still waiting to be judged
    for (const c of critters) {
      if (c.alive && c.state === 'await' && c.proposal) elder.queue.push(c.lane);
    }
    announce(`Elder ${roman(learn.elder.ordinal)} is crowned — line of Elder ${roman(parent.ord)} · taste: novelty ${t.novelty.toFixed(1)}, flow ${t.flow.toFixed(1)}, length ${t.length.toFixed(1)}`);
  }

  // --- worker lifecycle -------------------------------------------------------------------
  function retire(c, quiet) {
    abandonProposal(c);
    c.alive = false;
    const living = critters.filter((o) => o.alive && o.age > 300);
    const best = living.sort((a, b) => b.brain.avgR - a.brain.avgR)[0];
    const src = best ? best.brain : c.brain;
    const heir = {
      big: src.big.map((row) => row.map((v) => Math.max(-4, Math.min(4, v + randn() * 0.12)))),
      lenLog: src.lenLog.map((v) => Math.max(-3, Math.min(3, v + randn() * 0.1))),
      base: 0.5,
      cells: src.cells.map((v) => v * 0.7 + randn() * 0.1),  // territory lore, half-remembered
      avgR: 0, words: 0,
      gen: src.gen + 1,
    };
    eggs.push({ x: toPx(c).x, y: toPx(c).y, t: 160, slot: c.lane, brain: heir });
    if (!quiet) announce(`a wordsmith of gen ${c.brain.gen} passes on — its heir is laid`);
  }
  function hatch(egg) {
    const nc = makeWorker(egg.slot, egg.brain);
    const n = toNorm({ x: egg.x, y: egg.y });
    nc.x = Math.max(-0.9, Math.min(0.9, n.x));
    nc.y = Math.max(-0.9, Math.min(0.9, n.y));
    nc.hatch = 90;
    critters[egg.slot] = nc;
    learn.brains[egg.slot] = egg.brain;
    stats.hatched++;
  }
  function abandonProposal(c) {
    if (c.carrying >= 0) {
      const L = letters[c.carrying];
      L.at = 'ground';
      L.pos = toPx({ x: c.x, y: c.y });
      c.carrying = -1;
    }
    if (c.proposal) {
      for (const li of c.proposal.claimIdx) {
        const L = letters[li];
        L.claimed = false;
        if (L.at === 'ground' || L.at === 'placed') sendHome(L);
      }
      c.proposal = null;
      c.placed = [];
    }
  }
  function sendHome(L) {
    flights.push({ letter: L, fx: L.pos.x, fy: L.pos.y, t: 0, dur: 24 + Math.random() * 12 });
    L.at = 'flying';
  }

  // claim letter tiles for a proposed word — only within the sensing ring,
  // nearest tile first. Fails if a rival claimed them in the meantime.
  function claimTiles(word, c) {
    const idx = [];
    for (const ch of word) {
      let found = -1, bd = Infinity;
      for (let i = 0; i < letters.length; i++) {
        const L = letters[i];
        if (L.at !== 'home' || L.claimed || L.low !== ch) continue;
        const n = toNorm(L.home);
        const d = Math.hypot(n.x - c.x, n.y - c.y);
        if (d < SENSE && d < bd) { bd = d; found = i; }
      }
      if (found < 0) {
        for (const i of idx) letters[i].claimed = false;
        return null;
      }
      letters[found].claimed = true;
      idx.push(found);
    }
    return idx;
  }

  function stepColony() {
    if (!cursor.active) { cursor.x = 2.2; cursor.y = 2.2; }
    cursor.vx = cursor.x - cursor.px; cursor.vy = cursor.y - cursor.py;
    cursor.px = cursor.x; cursor.py = cursor.y;

    if (elder.alive) {
      learn.reign.steps++;
      if (learn.reign.steps > 3600) endReign();          // ~2 min reigns
    } else if (interregnum > 0 && --interregnum <= 0) {
      crownSuccessor();
    }
    if (ancestorGlow > 0) ancestorGlow--;

    // letter flights home
    for (let i = flights.length - 1; i >= 0; i--) {
      const f = flights[i];
      f.t++;
      const k = Math.min(1, f.t / f.dur);
      const e = 1 - Math.pow(1 - k, 3);
      f.letter.pos = { x: f.fx + (f.letter.home.x - f.fx) * e, y: f.fy + (f.letter.home.y - f.fy) * e };
      if (k >= 1) {
        f.letter.at = 'home';
        f.letter.claimed = false;
        f.letter.span.style.visibility = '';
        flights.splice(i, 1);
      }
    }

    for (let i = eggs.length - 1; i >= 0; i--) {
      if (--eggs[i].t <= 0) { hatch(eggs[i]); eggs.splice(i, 1); }
    }

    // --- morsel physics: little arcs, then they rest on the floor ------------------------
    for (const m of morsels) {
      if (m.settled) continue;
      m.vy += 0.18;
      m.x += m.vx; m.y += m.vy;
      if (m.y > H - 14) { m.y = H - 14; m.settled = true; }
      if (m.x < 10) m.x = 10;
      if (m.x > W - 10) m.x = W - 10;
    }

    // --- wordsmiths: scout -> propose (locally) -> fetch/place -> await verdict ----------
    for (const c of critters) {
      if (!c.alive) { c.fade = Math.max(0, c.fade - 0.02); continue; }
      if (c.hatch > 0) c.hatch--;
      c.age++;
      // metabolism: living costs a little, hauling costs more
      c.energy -= 0.00016 + (c.state === 'fetch' || c.state === 'place' ? 0.0004 : 0);
      if (c.energy <= 0) {
        stats.starved++;
        announce(`a gen-${c.brain.gen} wordsmith starved — the elder's table was too bare`);
        retire(c, true);
        continue;
      }
      if (c.age > c.lifespan && c.state !== 'await') { retire(c); continue; }

      // hungry and food on the floor? eating takes priority over art
      if (c.eating < 0 && c.energy < 0.55 && (c.state === 'rest' || c.state === 'scout')) {
        let bi = -1, bd = Infinity;
        for (let i = 0; i < morsels.length; i++) {
          if (!morsels[i].settled) continue;
          const n = toNorm(morsels[i]);
          const d = Math.hypot(n.x - c.x, n.y - c.y);
          if (d < bd) { bd = d; bi = i; }
        }
        if (bi >= 0) { c.eating = bi; }
      }

      let target = null;
      if (c.eating >= 0) {
        const m = morsels[c.eating];
        if (!m) { c.eating = -1; } else {
          target = toNorm(m);
        }
      } else if (c.state === 'rest') {
        if (--c.restT <= 0) {
          // pick a region to scout: softmax over this wordsmith's learned map
          const vals = c.brain.cells;
          const mx = Math.max(...vals);
          const exps = vals.map((v) => Math.exp((v - mx) / 0.35));
          const Z = exps.reduce((a, b) => a + b, 0);
          let pick = Math.random() * Z, idx = 0;
          for (; idx < exps.length - 1; idx++) { pick -= exps[idx]; if (pick <= 0) break; }
          c.scoutCell = idx;
          c.dwellT = 25;
          c.state = 'scout';
        }
      } else if (c.state === 'scout') {
        target = cellCenter(c.scoutCell);
        if (Math.hypot(target.x - c.x, target.y - c.y) < 0.18 && --c.dwellT <= 0) c.state = 'propose';
      } else if (c.state === 'propose') {
        const prop = proposeWord(c.brain, localCounts(c));
        const claimIdx = prop && claimTiles(prop.word, c);
        if (prop && claimIdx) {
          c.proposal = { ...prop, claimIdx, cell: c.scoutCell };
          c.letterIdx = 0;
          c.placed = [];
          c.effort = 0;
          c.state = 'fetch';
          stats.proposed++;
        } else {
          // nothing spellable here (or rivals claimed the tiles): the REGION
          // disappointed — mark the map, not the grammar
          const cell = c.scoutCell >= 0 ? c.scoutCell : cellOf(c.x, c.y);
          c.brain.cells[cell] += 0.3 * (-0.4 - c.brain.cells[cell]);
          if (prop && !claimIdx) stats.contentions++;
          c.state = 'rest';
          c.restT = 30 + Math.random() * 40;
        }
      } else if (c.state === 'fetch') {
        c.effort++;
        const L = letters[c.proposal.claimIdx[c.letterIdx]];
        target = toNorm(L.at === 'home' ? L.home : L.pos);
      } else if (c.state === 'place') {
        c.effort++;
        target = toNorm(laneSlot(c.lane, c.letterIdx));
      } else if (c.state === 'await') {
        target = toNorm({ x: laneSlot(c.lane, -2).x - 20, y: laneY(c.lane) });
      }
      if (!target) {
        if (--c.wanderT <= 0) {
          c.wanderT = 60 + Math.random() * 90;
          c.wander = { x: (Math.random() * 2 - 1) * 0.7, y: (Math.random() * 2 - 1) * 0.5 };
        }
        target = c.wander;
      }

      const dt = steer(c, target, cursor.active ? cursor : null, c.hatch > 0 ? 0.7 : 1);

      if (c.eating >= 0 && dt < 0.07) {
        const m = morsels[c.eating];
        if (m) {
          morsels.splice(c.eating, 1);
          // splices shift everyone's indices — re-aim any rival eaters
          for (const o of critters) { if (o.eating > c.eating) o.eating--; else if (o !== c && o.eating === c.eating) o.eating = -1; }
          c.energy = Math.min(1, c.energy + 0.28);
          stats.morselsEaten++;
        }
        c.eating = -1;
      } else if (c.state === 'fetch' && dt < 0.085) {
        const li = c.proposal.claimIdx[c.letterIdx];
        const L = letters[li];
        if (L.at === 'home') L.span.style.visibility = 'hidden';
        L.at = 'held';
        c.carrying = li;
        c.state = 'place';
      } else if (c.state === 'place' && dt < 0.085 && c.carrying >= 0) {
        const L = letters[c.carrying];
        L.at = 'placed';
        L.pos = laneSlot(c.lane, c.letterIdx);
        c.placed.push(c.carrying);
        c.carrying = -1;
        learn.reign.lettersPlaced++;
        c.letterIdx++;
        if (c.letterIdx >= c.proposal.claimIdx.length) {
          c.state = 'await';
          elder.queue.push(c.lane);
        } else {
          c.state = 'fetch';
        }
      }

      if (--c.blinkT <= 0) { c.blink = 5; c.blinkT = 90 + Math.random() * 150; }
      if (c.blink > 0) c.blink--;
    }

    // --- the elder: walk to complete words, judge them ----------------------------------
    if (elder.alive) {
      elder.fade = Math.min(1, elder.fade + 0.02);
      if (elder.judging < 0 && elder.queue.length) elder.judging = elder.queue.shift();
      let target;
      if (elder.judging >= 0) {
        const lane = elder.judging;
        const mid = laneSlot(lane, 2);
        target = toNorm({ x: mid.x, y: mid.y - 26 });
        const d = steer(elder, target, null, 0.5);
        const worker = critters[lane];
        if (d < 0.1 && worker && worker.alive && worker.state === 'await' && worker.proposal) {
          if (++elder.judgeT > 30) {      // a solemn one-second inspection
            const word = worker.proposal.word;
            const { score, novel } = judgeWord(word);
            const savored = score >= 1.8;
            // the ECONOMY: effort spent hauling is subtracted from the
            // verdict, and the NET is what teaches both the grammar and the
            // scouting map — cheap local words beat exotic marathons unless
            // the novelty premium truly pays
            const cost = Math.min(2.5, worker.effort / 450);
            const net = score - cost;
            reinforce(worker.brain, worker.proposal, net);
            const cell = worker.proposal.cell >= 0 ? worker.proposal.cell : cellOf(worker.x, worker.y);
            worker.brain.cells[cell] += 0.3 * (net - worker.brain.cells[cell]);
            worker.brain.words++;
            worker.brain.avgR += 0.15 * (net - worker.brain.avgR);
            worker.lastWord = { word, score: +score.toFixed(1), cost: +cost.toFixed(1), net: +net.toFixed(1) };
            // a pleased elder produces: food arcs out of the verdict
            const fed = conjureMorsels(score, { x: mid.x, y: laneY(lane) - 20 });
            verdicts.push({ x: mid.x, y: laneY(lane) - 34, txt: `${net >= 0 ? '+' : ''}${net.toFixed(1)} (${score.toFixed(1)}−${cost.toFixed(1)})`, word, savored, t: 110 });
            stats.judged++;
            if (novel) stats.newWords++;
            if (savored) {
              stats.savored++;
              elder.hop = 12;
              announce(`Elder ${roman(learn.elder.ordinal)} savored “${word}” — ${fed} morsels for the colony${novel ? ' (never seen before)' : ''}`);
            } else if (Math.random() < 0.3) {
              announce(`“${word}” did not impress Elder ${roman(learn.elder.ordinal)} (net ${net >= 0 ? '+' : ''}${net.toFixed(1)})`);
            }
            for (const li of worker.proposal.claimIdx) sendHome(letters[li]);
            worker.proposal = null;
            worker.placed = [];
            worker.state = 'rest';
            worker.restT = 40 + Math.random() * 50;
            elder.judging = -1;
            elder.judgeT = 0;
          }
        } else if (!worker || !worker.alive || worker.state !== 'await') {
          elder.judging = -1;   // the presenter died mid-ceremony
          elder.judgeT = 0;
        }
      } else {
        if (--elder.wanderT <= 0) {
          elder.wanderT = 150 + Math.random() * 150;
          elder.wander = { x: (Math.random() * 2 - 1) * 0.5, y: (Math.random() * 2 - 1) * 0.4 };
        }
        steer(elder, elder.wander, null, 0.45);
      }
      if (--elder.blinkT <= 0) { elder.blink = 6; elder.blinkT = 150 + Math.random() * 200; }
      if (elder.blink > 0) elder.blink--;
      if (elder.hop > 0) elder.hop--;
    } else {
      elder.fade = Math.max(0, elder.fade - 0.03);
    }

    if (statusEl) {
      const t = elder.taste;
      const vocab = Object.keys(learn.archive).length;
      const reignTxt = elder.alive
        ? `Elder ${roman(learn.elder.ordinal)} (taste: novelty ${t.novelty.toFixed(1)} · flow ${t.flow.toFixed(1)} · length ${t.length.toFixed(1)})`
        : 'interregnum — the Ancestor deliberates';
      statusEl.textContent =
        `${reignTxt} · vocabulary ${vocab} words · best “${learn.best ? learn.best.word : '—'}” ${learn.best ? '+' + learn.best.score : ''} · ${morsels.length} morsels on the floor · saved locally`;
    }

    // the strategy panel: each wordsmith's learned niche, in the open
    if (panelEl && (panelT = (panelT + 1) % 30) === 0) {
      const CELL_NAMES = ['NW', 'N', 'NE', 'far NE', 'W', 'mid', 'E', 'far E', 'SW', 'S', 'SE', 'far SE'];
      panelEl.innerHTML = critters.map((c) => {
        if (!c.alive) return `<div><span class="cp-dot" style="background:${WORKER_COLORS[c.lane]}"></span>an egg incubates…</div>`;
        const fav = c.brain.cells.indexOf(Math.max(...c.brain.cells));
        const last = c.lastWord
          ? `last <span class="cp-word">“${c.lastWord.word}”</span> ${c.lastWord.net >= 0 ? '+' : ''}${c.lastWord.net} (verdict ${c.lastWord.score} − haul ${c.lastWord.cost})`
          : 'has not presented yet';
        return `<div><span class="cp-dot" style="background:${c.color}"></span>` +
          `gen ${c.brain.gen} · energy ${(c.energy * 100).toFixed(0)}% · forages ${CELL_NAMES[fav] || fav} · avg ${c.brain.avgR >= 0 ? '+' : ''}${c.brain.avgR.toFixed(2)}/word · ${last}</div>`;
      }).join('');
    }
  }

  // --- drawing -----------------------------------------------------------------------------
  function smooth(e) {
    const p = toPx(e);
    e.sx = e.sx === undefined ? p.x : e.sx + (p.x - e.sx) * 0.35;
    e.sy = e.sy === undefined ? p.y : e.sy + (p.y - e.sy) * 0.35;
    return { x: e.sx, y: e.sy };
  }
  function drawEyes(p, look, blink, dark, scale = 1) {
    for (const side of [-1, 1]) {
      const ex = p.x + side * 3.4 * scale, ey = p.y - 2.5 * scale;
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      if (blink > 0) {
        ctx.fillRect(ex - 2 * scale, ey - 0.6, 4 * scale, 1.2);
      } else {
        ctx.arc(ex, ey, 2.5 * scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = dark;
        ctx.beginPath();
        ctx.arc(ex + look.x * 1.1, ey + look.y * 1.1, 1.25 * scale, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // lanes: ghost of each proposed word + placed letters
    for (const c of critters) {
      if (!c.alive || !c.proposal) continue;
      // lane marker in the worker's color
      ctx.fillStyle = c.color;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.arc(laneSlot(c.lane, -1).x - 14, laneY(c.lane), 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = inkColor;
      for (let k = 0; k < c.proposal.word.length; k++) {
        if (k < c.letterIdx) continue;    // already placed (drawn solid below)
        const s = laneSlot(c.lane, k);
        ctx.fillText(c.proposal.word[k], s.x, s.y);
      }
      ctx.globalAlpha = 1;
    }

    // grounded / placed letters
    for (const L of letters) {
      if (L.at !== 'ground' && L.at !== 'placed' && L.at !== 'flying') continue;
      ctx.globalAlpha = L.at === 'flying' ? 0.8 : 0.95;
      ctx.fillStyle = inkColor;
      ctx.fillText(L.ch, L.pos.x, L.pos.y);
      ctx.globalAlpha = 1;
    }

    // verdicts
    for (let i = verdicts.length - 1; i >= 0; i--) {
      const v = verdicts[i];
      v.y -= 0.35;
      const a = Math.min(1, v.t / 40);
      ctx.globalAlpha = a;
      ctx.font = 'bold 15px sans-serif';
      ctx.fillStyle = v.savored ? '#f59e0b' : 'rgba(120,120,140,0.9)';
      ctx.fillText(`${v.txt}  “${v.word}”`, v.x, v.y);
      if (v.savored) {
        const k = 1 - v.t / 110;
        ctx.strokeStyle = `rgba(245,158,11,${(1 - k) * 0.5})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(v.x, v.y + 20, 14 + k * 26, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.font = font;
      ctx.globalAlpha = 1;
      if (--v.t <= 0) verdicts.splice(i, 1);
    }

    // eggs
    for (const egg of eggs) {
      const wob = Math.sin(now / 90) * (egg.t < 40 ? 2.2 : 0.7);
      ctx.save();
      ctx.translate(egg.x + wob * 0.4, egg.y);
      ctx.rotate(wob * 0.03);
      ctx.fillStyle = '#f5f5f4';
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.beginPath();
      ctx.ellipse(0, 0, 5.5, 7, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // morsels: the elder's produce, resting on the floor
    for (const m of morsels) {
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(m.x, m.y, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.beginPath();
      ctx.arc(m.x - 1, m.y - 1, 1, 0, Math.PI * 2);
      ctx.fill();
    }

    // workers
    for (const c of critters) {
      if (!c.alive && c.fade <= 0) continue;
      const p = smooth(c);

      // the sensing ring — what this wordsmith can actually SEE — plus a
      // marker on the region it has chosen to scout
      if (c.alive) {
        ctx.strokeStyle = c.color;
        ctx.globalAlpha = 0.16;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, (SENSE / 2) * W, (SENSE / 2) * H, 0, 0, Math.PI * 2);
        ctx.stroke();
        if (c.state === 'scout' && c.scoutCell >= 0) {
          const cc = toPx(cellCenter(c.scoutCell));
          ctx.globalAlpha = 0.35;
          ctx.setLineDash([3, 5]);
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(cc.x, cc.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.globalAlpha = 1;
      }
      const speed = Math.hypot(c.vx, c.vy);
      const ang = Math.atan2(c.vy, c.vx);
      const grow = c.hatch > 0 ? 1 - (c.hatch / 90) * 0.45 : 1;
      const squash = Math.min(0.2, speed * 6);
      const idle = Math.max(0, 1 - speed * 30);
      const bob = Math.sin(now / 260 + c.seed) * 0.8 * idle;
      ctx.globalAlpha = c.alive ? 1 : c.fade;
      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + 8 * grow, 7.5 * grow, 2.6, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.save();
      ctx.translate(p.x, p.y + bob);
      ctx.rotate(ang * Math.min(1, speed * 40));
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.ellipse(0, 0, 8 * grow * (1 + squash), 8 * grow * (1 - squash * 0.6), 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      const look = speed > 0.004 ? { x: Math.cos(ang), y: Math.sin(ang) } : { x: 0, y: 0.3 };
      drawEyes({ x: p.x, y: p.y + bob }, look, c.blink, '#1e1b4b', grow);
      // energy bar: green fading to red as hunger bites
      if (c.alive) {
        const e = Math.max(0, Math.min(1, c.energy));
        ctx.fillStyle = 'rgba(0,0,0,0.15)';
        ctx.fillRect(p.x - 7, p.y + 12, 14, 2);
        ctx.fillStyle = e > 0.5 ? '#34d399' : e > 0.25 ? '#fbbf24' : '#fb7185';
        ctx.fillRect(p.x - 7, p.y + 12, 14 * e, 2);
      }
      if (c.carrying >= 0) {
        const L = letters[c.carrying];
        ctx.save();
        ctx.translate(p.x, p.y - 15 + bob);
        ctx.rotate(Math.sin(now / 300 + c.seed) * 0.12);
        ctx.fillStyle = inkColor;
        ctx.fillText(L.ch, 0, 0);
        ctx.restore();
      }
      ctx.globalAlpha = 1;
    }

    // the elder
    if (elder.fade > 0.01) {
      const p = smooth(elder);
      const speed = Math.hypot(elder.vx, elder.vy);
      const ang = Math.atan2(elder.vy, elder.vx);
      const hop = elder.hop > 0 ? -Math.sin((elder.hop / 12) * Math.PI) * 6 : 0;
      const bob = Math.sin(now / 320) * 0.7 + hop;
      ctx.globalAlpha = elder.fade;
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
      ctx.globalAlpha = 1;
    }

    // the Ancestor's apparition
    if (ancestorGlow > 0) {
      const a = Math.sin((ancestorGlow / 140) * Math.PI) * 0.18;
      ctx.globalAlpha = a;
      ctx.fillStyle = '#a8a29e';
      ctx.beginPath();
      ctx.ellipse(W / 2, H * 0.4, 46, 40, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(W / 2 - 20, H * 0.4 - 36);
      ctx.lineTo(W / 2 - 20, H * 0.4 - 48);
      ctx.lineTo(W / 2 - 10, H * 0.4 - 40);
      ctx.lineTo(W / 2, H * 0.4 - 52);
      ctx.lineTo(W / 2 + 10, H * 0.4 - 40);
      ctx.lineTo(W / 2 + 20, H * 0.4 - 48);
      ctx.lineTo(W / 2 + 20, H * 0.4 - 36);
      ctx.stroke();
      ctx.strokeStyle = '#57534e';
      ctx.lineWidth = 2.5;
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(W / 2 + s * 16 - 7, H * 0.4 - 6);
        ctx.quadraticCurveTo(W / 2 + s * 16, H * 0.4 - 1, W / 2 + s * 16 + 7, H * 0.4 - 6);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  // --- loop --------------------------------------------------------------------------------
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
