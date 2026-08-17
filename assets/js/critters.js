// The /colony/ letterlings. Everything on this page lives a life made of
// text, in a paragraph that is womb, pantry and grave at once:
//
//   EGG        a pearl on one of the paragraph's letters. It hatches and
//              that letter pops out of the text as a...
//   LARVA      a caterpillar whose BODY IS ITS LETTERS. It crawls the page
//              eating letters that extend its body along real-word
//              prefixes ("ca" hunts a "t" or an "r" — the thought bubble
//              shows what it's seeking). Stuck larvae relocate, or shed
//              their tail letter in frustration.
//   CHRYSALIS  the moment its body spells a real word, it curls up and
//              pulses in a silk cocoon...
//   BUTTERFLY  ...and emerges as a word on wings, looping over the text.
//              Longer words fly longer and grander.
//   SEEDING    when its flight ends it gives every letter back to the
//              paragraph — its own empty slots heal — and leaves an egg or
//              two behind. Hatchlings inherit their parent's appetite, and
//              the family book remembers every word that ever lived here.
//
// A masked SNATCHER stalks the margins and steals tail letters from
// caterpillars for its corner hoard (click it and it spills everything).
// Your cursor spooks all of them. Movement for every creature is ONE
// frozen tiny steering MLP (assets/worldmodel/critter_train.js) — the
// same net that powers /play/ and /arena/. The family book persists in
// localStorage.
const stage = document.getElementById('colony-stage');
const statusEl = document.getElementById('colony-status');
const eventEl = document.getElementById('colony-event');
const panelEl = document.getElementById('colony-panel');

if (stage) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    if (statusEl) statusEl.textContent = 'the letterlings are resting (reduced motion is on) — the text is safe';
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
    c.x = Math.max(-0.96, Math.min(0.96, c.x + c.vx));
    c.y = Math.max(-0.96, Math.min(0.96, c.y + c.vy));
    return dt;
  }

  // --- the lexicon -------------------------------------------------------------
  const WORDS = ('cat care card core corn call calm came camp cane cape cart case cast cave cell cent city clay clip coal coat code coin cold cool cope copy cord cost cozy crew crop cube cure curl dark dart data dawn deal dear deep deer desk dial dice diet dish dive dome done door dose down draw drop drum dust duty each earn ease east easy echo edge else even ever face fact fade fair fall fame farm fast fate fear feed feel fern film find fine fire firm fish five flag flat flew flow fold folk fond food fool foot fore fork form fort four free from fuel full fund gain game gate gaze gear gene gift give glad glow goat gold golf gone good grew grid grin grip grow gulf hand hang hard harm hate have hawk head heal heap hear heat held herb here hero hide high hill hint hire hold hole home hope horn hose host hour huge hunt hurt icon idea inch into iron item jazz join joke jury just keen keep kind king kiss kite knee knew know lace lake lamp land lane last late lava lawn lead leaf lean leap left lend lens less life lift like lime line link lion list live load loaf loan lock loft long look loop lord lose loss lost loud love luck lung made mail main make mane many mare mark mask mast mate maze meal mean meat melt mesh mess mild mile milk mill mind mine mint miss mist mode mood moon more moss most moth move much mule must myth name near neat neck need nest news nice nine node none noon nose note noun oven over pace pack page paid pain pair pale palm park part pass past path peak pear peel pine pink pipe plan play plot plow poem poet pole pond pool port pose post pour pray prey pull pure push quit race rack rage rail rain rake rank rare rate read real reap rear rent rest rice rich ride ring ripe rise risk road roam rock rode role roll roof room root rope rose ruby rule rush rust safe sage sail salt same sand save scan seal seat seed seek seem seen self sell send sent shed ship shoe shop show shut side sign silk sing sink site size skin slow snow soap soft soil sold sole some song soon sort soul soup spin spot star stay stem step stir stop such suit sure swan swim tale talk tall tame tank tape task team tear tell tend tent term test than that them then they thin this tide tile time tiny toad told tone took tool torn tour town trap tray tree trim trip true tube tune turn twin type unit upon urge used user vase vast very vine vote wage wait wake walk wall want ward warm wash wave weak wear week well went were west what when whom wide wife wild will wind wine wing wire wise wish with wolf wood wool word wore work worm worn wrap yard yarn year zone').split(' ');
  const WORDSET = new Set(WORDS);
  const PREFIX = new Map(); // prefix -> array of next letters
  for (const w of WORDS) {
    for (let i = 1; i < w.length; i++) {
      const p = w.slice(0, i);
      if (!PREFIX.has(p)) PREFIX.set(p, new Set());
      PREFIX.get(p).add(w[i]);
    }
  }
  for (const [k, v] of PREFIX) PREFIX.set(k, [...v]);
  const FIRSTS = new Set(WORDS.map((w) => w[0]));

  // --- the paragraph as a world ---------------------------------------------------
  const textEl = stage.querySelector('.colony-text');
  const raw = textEl.textContent;
  textEl.textContent = '';
  const letters = []; // {ch, low, span, home{px}, at:'home'|'body'|'flying'|'hoard', pos{px}, egg}
  for (const ch of raw) {
    if (/\s/.test(ch)) {
      textEl.appendChild(document.createTextNode(ch));
      continue;
    }
    const span = document.createElement('span');
    span.className = 'colony-ch';
    span.textContent = ch;
    textEl.appendChild(span);
    letters.push({ ch, low: ch.toLowerCase(), span, home: { x: 0, y: 0 }, at: 'home', pos: { x: 0, y: 0 }, egg: null });
  }

  let W = 0, H = 0, font = '16px sans-serif', fontPx = 16, inkColor = '#333';
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
    fontPx = parseFloat(cs.fontSize) || 16;
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

  const randn = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  function announce(txt) {
    if (!eventEl) return;
    eventEl.textContent = txt;
    eventEl.style.opacity = '1';
    clearTimeout(announce.t);
    announce.t = setTimeout(() => { eventEl.style.opacity = '0.4'; }, 7000);
  }

  // --- persistence: the family book -------------------------------------------------
  const LEARN_KEY = 'colony-letterlings-v1';
  let book = { lived: [], appetiteTail: 4 };
  try {
    const saved = JSON.parse(localStorage.getItem(LEARN_KEY));
    if (saved && Array.isArray(saved.lived)) book = saved;
  } catch (e) { /* first spring */ }
  function saveBook() {
    try { localStorage.setItem(LEARN_KEY, JSON.stringify(book)); } catch (e) { /* private mode */ }
  }
  setInterval(saveBook, 10000);
  document.addEventListener('visibilitychange', () => { if (document.hidden) saveBook(); });
  window.__colonyBook = book;

  // --- cursor -------------------------------------------------------------------------
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
    if (Math.hypot(snatcher.x - n.x, snatcher.y - n.y) < 0.16) {
      let spilled = 0;
      for (const L of letters) {
        if (L.at === 'hoard') { sendHome(L); spilled++; }
      }
      snatcher.state = 'flee';
      snatcher.fleeT = 110;
      snatcher.targetC = null;
      stats.bonks++;
      announce(spilled ? `you bonked the Snatcher — ${spilled} stolen letter${spilled > 1 ? 's fly' : ' flies'} home!` : 'you bonked the Snatcher — its hoard was already empty');
    }
  });

  // --- flights: letters travelling home ----------------------------------------------
  const flights = [];
  function sendHome(L) {
    flights.push({ letter: L, fx: L.pos.x, fy: L.pos.y, t: 0, dur: 30 + Math.random() * 20 });
    L.at = 'flying';
  }

  // --- creatures ----------------------------------------------------------------------
  const PASTELS = ['#818cf8', '#6ee7b7', '#f9a8d4', '#fbbf24', '#7dd3fc', '#a78bfa', '#fda4af'];
  const hue = (word) => PASTELS[[...word].reduce((a, ch) => a + ch.charCodeAt(0), 0) % PASTELS.length];
  const creatures = [];
  const POP_TARGET = 5;
  let nextId = 1;

  function spawnEgg(lineage, appetite, preferIdx) {
    let hostIdx = preferIdx;
    if (hostIdx === undefined || hostIdx < 0) {
      const homes = letters.map((L, i) => ({ L, i })).filter((o) => o.L.at === 'home' && !o.L.egg && FIRSTS.has(o.L.low));
      if (!homes.length) return false;
      hostIdx = homes[Math.floor(Math.random() * homes.length)].i;
    }
    letters[hostIdx].egg = {
      t: 240 + Math.random() * 240,
      lineage: (lineage || []).slice(0, 3),
      appetite: Math.max(3, Math.min(6, Math.round(appetite || book.appetiteTail || 4))),
    };
    return true;
  }

  function hatch(hostIdx) {
    const L = letters[hostIdx];
    const egg = L.egg;
    L.egg = null;
    if (L.at !== 'home' || !FIRSTS.has(L.low)) return; // the host letter left; the egg is lost
    L.at = 'body';
    L.span.style.visibility = 'hidden';
    const n = toNorm(L.home);
    creatures.push({
      id: nextId++,
      stage: 'larva',
      body: [hostIdx],
      appetite: egg.appetite,
      lineage: egg.lineage,
      x: n.x, y: n.y, vx: 0, vy: 0,
      trail: [], seekLetters: [], targetIdx: -1,
      stuckT: 0, chewT: 0, age: 0,
      cocoonT: 0, flightT: 0, seedIdx: 0,
      wanderT: 0, wander: { x: 0, y: 0 },
      blinkT: 80 + Math.random() * 120, blink: 0,
      color: PASTELS[(nextId + 2) % PASTELS.length],
      seed: nextId * 977,
    });
    stats.hatched++;
    announce(`an egg hatched on “${L.ch}”${egg.lineage.length ? ` — line of “${egg.lineage[0]}”` : ''}`);
  }

  const bodyWord = (c) => c.body.map((i) => letters[i].low).join('');
  function extensions(c) {
    const w = bodyWord(c);
    return PREFIX.get(w) || [];
  }
  function findLetterTile(c, wanted, maxDist) {
    let best = -1, bd = Infinity;
    for (let i = 0; i < letters.length; i++) {
      const L = letters[i];
      if (L.at !== 'home' || L.egg || !wanted.includes(L.low)) continue;
      const n = toNorm(L.home);
      const d = Math.hypot(n.x - c.x, n.y - c.y);
      if (d < bd) { bd = d; best = i; }
    }
    return bd <= (maxDist || Infinity) ? best : (maxDist ? -1 : best);
  }
  function shedTail(c) {
    if (c.body.length <= 1) return;
    const li = c.body.pop();
    const L = letters[li];
    L.at = 'flying';
    const tp = tilePos(c, c.body.length); // roughly where the tail was
    L.pos = { x: tp.x, y: tp.y };
    sendHome(L);
    L.pos = { x: tp.x, y: tp.y };
  }
  function dissolve(c, reason) {
    for (let k = c.body.length - 1; k >= 0; k--) {
      const li = c.body[k];
      const L = letters[li];
      const tp = tilePos(c, k);
      L.pos = { x: tp.x, y: tp.y };
      sendHome(L);
    }
    c.dead = true;
    if (reason) announce(reason);
  }

  // caterpillar geometry: segments follow the head's trail at fixed spacing
  const SEG_PX = () => fontPx * 0.82;
  function tilePos(c, k) {
    if (k === 0) return toPx(c);
    const need = k * SEG_PX();
    let acc = 0;
    const head = toPx(c);
    let prev = head;
    for (let i = c.trail.length - 1; i >= 0; i--) {
      const p = c.trail[i];
      const d = Math.hypot(p.x - prev.x, p.y - prev.y);
      if (acc + d >= need) {
        const f = (need - acc) / (d || 1);
        return { x: prev.x + (p.x - prev.x) * f, y: prev.y + (p.y - prev.y) * f };
      }
      acc += d;
      prev = p;
    }
    return prev;
  }

  // --- the Snatcher ---------------------------------------------------------------------
  const snatcher = {
    x: 0.9, y: -0.9, vx: 0, vy: 0,
    state: 'lurk', targetC: null, raidT: 300 + Math.random() * 200, fleeT: 0,
    blinkT: 120, blink: 0,
  };
  const NEST = () => ({ x: W - 40, y: 26 });
  function hoardCount() { return letters.filter((L) => L.at === 'hoard').length; }

  const stats = { hatched: 0, pupated: 0, emerged: 0, seeded: 0, shed: 0, snatched: 0, bonks: 0 };
  window.__colony = stats;

  // --- the world turns -------------------------------------------------------------------
  function stepColony() {
    if (!cursor.active) { cursor.x = 2.2; cursor.y = 2.2; }
    cursor.vx = cursor.x - cursor.px; cursor.vy = cursor.y - cursor.py;
    cursor.px = cursor.x; cursor.py = cursor.y;

    // letters flying home
    for (let i = flights.length - 1; i >= 0; i--) {
      const f = flights[i];
      f.t++;
      const k = Math.min(1, f.t / f.dur);
      const e = 1 - Math.pow(1 - k, 3);
      f.letter.pos = { x: f.fx + (f.letter.home.x - f.fx) * e, y: f.fy + (f.letter.home.y - f.fy) * e };
      if (k >= 1) {
        f.letter.at = 'home';
        f.letter.span.style.visibility = '';
        flights.splice(i, 1);
      }
    }

    // eggs incubate
    for (let i = 0; i < letters.length; i++) {
      const L = letters[i];
      if (!L.egg) continue;
      if (--L.egg.t <= 0) hatch(i);
    }

    // keep the population going
    const living = creatures.filter((c) => !c.dead).length;
    const eggCount = letters.filter((L) => L.egg).length;
    if (living + eggCount < POP_TARGET && Math.random() < 0.02) {
      spawnEgg(book.lived.slice(-1), book.appetiteTail, -1);
    }

    // --- each letterling lives its stage -------------------------------------------------
    for (const c of creatures) {
      if (c.dead) continue;
      c.age++;

      if (c.stage === 'larva') {
        // record the trail for the body segments
        const hp = toPx(c);
        const lastT = c.trail[c.trail.length - 1];
        if (!lastT || Math.hypot(hp.x - lastT.x, hp.y - lastT.y) > 2) {
          c.trail.push({ x: hp.x, y: hp.y });
          if (c.trail.length > 220) c.trail.shift();
        }

        // a good meal deserves a moment: wiggle in place after each letter
        if (c.chewT > 0) {
          c.chewT--;
          c.vx *= 0.8; c.vy *= 0.8;
          continue;
        }

        const w = bodyWord(c);
        const isWord = w.length >= 3 && WORDSET.has(w);
        const opts = extensions(c);
        c.seekLetters = opts.slice(0, 3);

        // time to pupate? (a full word, and either sated or out of options)
        if (isWord && (w.length >= c.appetite || !opts.length)) {
          c.stage = 'chrysalis';
          c.cocoonT = 200 + w.length * 30;
          stats.pupated++;
          announce(`“${w}” is complete — it spins a cocoon`);
          continue;
        }
        if (!opts.length) {
          // dead-end prefix that isn't a word: shed the tail and try again
          shedTail(c);
          stats.shed++;
          continue;
        }

        // hunt the nearest extending letter
        if (c.targetIdx < 0 || letters[c.targetIdx].at !== 'home' || letters[c.targetIdx].egg || !opts.includes(letters[c.targetIdx].low)) {
          c.targetIdx = findLetterTile(c, opts);
        }
        let target;
        if (c.targetIdx >= 0) {
          target = toNorm(letters[c.targetIdx].home);
          c.stuckT = 0;
        } else {
          // nothing edible anywhere (all claimed/eggs) — wander hungrily
          if (++c.stuckT > 400) {
            if (c.body.length > 1) { shedTail(c); stats.shed++; }
            else if (c.age > 2400) { dissolve(c, `a lone “${w}” gave up and slipped back into the text`); }
            c.stuckT = 0;
          }
          if (--c.wanderT <= 0) {
            c.wanderT = 60 + Math.random() * 60;
            c.wander = { x: (Math.random() * 2 - 1) * 0.7, y: (Math.random() * 2 - 1) * 0.6 };
          }
          target = c.wander;
        }
        const dt = steer(c, target, cursor.active ? cursor : null, 0.28);
        if (c.targetIdx >= 0 && dt < 0.07) {
          const L = letters[c.targetIdx];
          if (L.at === 'home' && !L.egg) {
            L.at = 'body';
            L.span.style.visibility = 'hidden';
            c.body.push(c.targetIdx);
            c.chewT = 75 + Math.random() * 45;
          }
          c.targetIdx = -1;
        }
        // old larva that never finished: return to the text
        if (c.age > 5400) dissolve(c, `“${bodyWord(c)}” grew old before it grew whole — its letters fly home`);
      } else if (c.stage === 'chrysalis') {
        c.vx = c.vy = 0;
        if (--c.cocoonT <= 0) {
          c.stage = 'butterfly';
          const w = bodyWord(c);
          c.flightT = 500 + w.length * 120;
          c.color = hue(w);
          stats.emerged++;
          book.lived.push(w);
          if (book.lived.length > 60) book.lived.shift();
          book.appetiteTail = Math.max(3, Math.min(6, w.length + (Math.random() < 0.3 ? 1 : 0)));
          announce(`“${w}” has wings!`);
        }
      } else if (c.stage === 'butterfly') {
        if (--c.wanderT <= 0 || !c.wander) {
          c.wanderT = 50 + Math.random() * 70;
          c.wander = { x: (Math.random() * 2 - 1) * 0.75, y: -0.2 - Math.random() * 0.7 };
        }
        steer(c, c.wander, cursor.active ? cursor : null, 1.1);
        if (--c.flightT <= 0) {
          c.stage = 'seeding';
          announce(`“${bodyWord(c)}”'s flight is over — it returns to the paragraph`);
        }
      } else if (c.stage === 'seeding') {
        // glide toward the middle of the text, then give everything back
        const mid = toNorm({ x: W / 2, y: H * 0.32 });
        const dt = steer(c, mid, null, 0.8);
        if (dt < 0.25) {
          const w = bodyWord(c);
          for (const li of c.body) {
            const L = letters[li];
            const p = toPx(c);
            L.pos = { x: p.x + (Math.random() - 0.5) * 30, y: p.y + (Math.random() - 0.5) * 20 };
            sendHome(L);
          }
          const nEggs = w.length >= 5 ? 2 : 1;
          for (let k = 0; k < nEggs; k++) spawnEgg([w, ...c.lineage], w.length + (Math.random() < 0.4 ? 1 : 0), -1);
          c.dead = true;
          stats.seeded++;
          announce(`“${w}” dissolved into the text and left ${nEggs} egg${nEggs > 1 ? 's' : ''} behind`);
        }
      }

      if (--c.blinkT <= 0) { c.blink = 5; c.blinkT = 90 + Math.random() * 150; }
      if (c.blink > 0) c.blink--;
    }
    // bury the dead
    for (let i = creatures.length - 1; i >= 0; i--) if (creatures[i].dead) creatures.splice(i, 1);

    // --- the Snatcher ---------------------------------------------------------------------
    {
      const S = snatcher;
      let target = null;
      if (S.state === 'flee') {
        target = { x: 0.9, y: -0.9 };
        if (--S.fleeT <= 0) { S.state = 'lurk'; S.raidT = 350 + Math.random() * 300; }
      } else if (S.state === 'lurk') {
        target = toNorm(NEST());
        if (--S.raidT <= 0) {
          const marks = creatures.filter((c) => !c.dead && c.stage === 'larva' && c.body.length >= 2);
          if (marks.length) {
            S.targetC = marks[Math.floor(Math.random() * marks.length)];
            S.state = 'stalk';
          } else {
            S.raidT = 90;
          }
        }
      } else if (S.state === 'stalk') {
        const m = S.targetC;
        if (!m || m.dead || m.stage !== 'larva' || m.body.length < 2) {
          S.state = 'lurk';
          S.raidT = 400 + Math.random() * 300;
          S.targetC = null;
        } else {
          const tail = tilePos(m, m.body.length - 1);
          const tn = toNorm(tail);
          target = tn;
          const d = Math.hypot(tn.x - S.x, tn.y - S.y);
          const cd = cursor.active ? Math.hypot(cursor.x - S.x, cursor.y - S.y) : Infinity;
          if (cd < 0.25) {
            S.state = 'flee';
            S.fleeT = 80;
            S.targetC = null;
          } else if (d < 0.09) {
            // SNATCH the tail letter
            const li = m.body.pop();
            const L = letters[li];
            L.at = 'hoard';
            const nest = NEST();
            const n = hoardCount();
            L.pos = { x: nest.x - (n % 5) * 12, y: nest.y + Math.floor(n / 5) * 14 };
            m.targetIdx = -1;
            stats.snatched++;
            announce(`the Snatcher stole “${L.ch}” right off “${bodyWord(m)}${L.low}”'s tail!`);
            S.state = 'flee';
            S.fleeT = 90;
            S.targetC = null;
          }
        }
      }
      if (target) {
        const threat = cursor.active && Math.hypot(cursor.x - S.x, cursor.y - S.y) < 0.5 ? cursor : null;
        steer(S, target, threat, S.state === 'stalk' ? 0.85 : 0.6);
      }
      if (--S.blinkT <= 0) { S.blink = 5; S.blinkT = 120 + Math.random() * 150; }
      if (S.blink > 0) S.blink--;
    }

    // --- status + panel -----------------------------------------------------------------
    if (statusEl) {
      const larvae = creatures.filter((c) => c.stage === 'larva').length;
      const cocoons = creatures.filter((c) => c.stage === 'chrysalis').length;
      const flying = creatures.filter((c) => c.stage === 'butterfly' || c.stage === 'seeding').length;
      const eggN = letters.filter((L) => L.egg).length;
      statusEl.textContent =
        `${eggN} egg${eggN === 1 ? '' : 's'} · ${larvae} larva${larvae === 1 ? '' : 'e'} · ${cocoons} cocoon${cocoons === 1 ? '' : 's'} · ${flying} in flight · ` +
        `${book.lived.length} words have lived here${book.lived.length ? ` — latest “${book.lived[book.lived.length - 1]}”` : ''} · hoard ${hoardCount()}`;
    }
    if (panelEl && (panelT = (panelT + 1) % 30) === 0) {
      const rows = creatures.map((c) => {
        const w = bodyWord(c);
        const line = c.lineage.length ? ` · line of <span class="cp-word">“${c.lineage[0]}”</span>` : '';
        if (c.stage === 'larva') {
          const seeking = c.seekLetters.length ? `hunting <b>${c.seekLetters.join('</b> or <b>')}</b>` : 'foraging';
          return `<div><span class="cp-dot" style="background:${c.color}"></span>larva <span class="cp-word">“${w}”</span> (wants ${c.appetite} letters) · ${seeking}${line}</div>`;
        }
        if (c.stage === 'chrysalis') return `<div><span class="cp-dot" style="background:${c.color}"></span>cocoon <span class="cp-word">“${w}”</span> · emerging in ${Math.ceil(c.cocoonT / 30)}s${line}</div>`;
        if (c.stage === 'butterfly') return `<div><span class="cp-dot" style="background:${c.color}"></span>butterfly <span class="cp-word">“${w}”</span> · ${Math.ceil(c.flightT / 30)}s of flight left${line}</div>`;
        return `<div><span class="cp-dot" style="background:${c.color}"></span><span class="cp-word">“${w}”</span> is coming home${line}</div>`;
      });
      const recent = book.lived.slice(-6).map((w) => `“${w}”`).join(', ');
      rows.push(`<div><span class="cp-dot" style="background:#475569"></span>the Snatcher · ${snatcher.state === 'stalk' ? 'ON THE HUNT' : snatcher.state} · ${hoardCount()} letters hoarded · click it to bonk</div>`);
      if (recent) rows.push(`<div style="opacity:0.8">family book: ${recent}</div>`);
      panelEl.innerHTML = rows.join('');
    }
  }
  let panelT = 0;

  // --- drawing ------------------------------------------------------------------------------
  function smooth(e) {
    const p = toPx(e);
    e.sx = e.sx === undefined ? p.x : e.sx + (p.x - e.sx) * 0.35;
    e.sy = e.sy === undefined ? p.y : e.sy + (p.y - e.sy) * 0.35;
    return { x: e.sx, y: e.sy };
  }

  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // eggs: pearls sitting on their host letters
    for (const L of letters) {
      if (!L.egg) continue;
      const shimmer = 0.75 + Math.sin(now / 300 + L.home.x) * 0.25;
      ctx.fillStyle = `rgba(226,222,255,${shimmer})`;
      ctx.strokeStyle = 'rgba(129,140,248,0.7)';
      ctx.beginPath();
      ctx.ellipse(L.home.x + fontPx * 0.32, L.home.y - fontPx * 0.38, 3.2, 3.8, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // flying letters
    for (const L of letters) {
      if (L.at !== 'flying') continue;
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = inkColor;
      ctx.fillText(L.ch, L.pos.x, L.pos.y);
      ctx.globalAlpha = 1;
    }

    // the hoard in the Snatcher's corner nest
    for (const L of letters) {
      if (L.at !== 'hoard') continue;
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = inkColor;
      ctx.save();
      ctx.translate(L.pos.x, L.pos.y);
      ctx.rotate(((L.home.x * 7) % 10 - 5) * 0.06);
      ctx.fillText(L.ch, 0, 0);
      ctx.restore();
      ctx.globalAlpha = 1;
    }

    // --- creatures ---------------------------------------------------------------------------
    for (const c of creatures) {
      const p = smooth(c);
      const w = bodyWord(c);

      if (c.stage === 'larva') {
        const speed = Math.hypot(c.vx, c.vy);
        // body segments, tail to head, inchworm ripple
        for (let k = c.body.length - 1; k >= 0; k--) {
          const tp = k === 0 ? p : tilePos(c, k);
          const ripple = Math.sin(now / 140 - k * 0.9 + c.seed) * 2 * Math.min(1, speed * 60 + 0.25);
          ctx.fillStyle = 'rgba(0,0,0,0.12)';
          ctx.beginPath();
          ctx.ellipse(tp.x, tp.y + fontPx * 0.42, fontPx * 0.34, 2.2, 0, 0, Math.PI * 2);
          ctx.fill();
          // tile
          ctx.fillStyle = c.color;
          ctx.globalAlpha = 0.28;
          ctx.beginPath();
          ctx.arc(tp.x, tp.y + ripple * 0.4, fontPx * 0.44, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.fillStyle = inkColor;
          ctx.fillText(letters[c.body[k]].ch, tp.x, tp.y + ripple * 0.4);
        }
        // face on the head tile
        const ang = Math.atan2(c.vy, c.vx);
        const look = speed > 0.003 ? { x: Math.cos(ang), y: Math.sin(ang) } : { x: 0, y: 0.3 };
        for (const side of [-1, 1]) {
          const ex = p.x + side * 3.6, ey = p.y - fontPx * 0.52;
          ctx.fillStyle = '#fff';
          ctx.beginPath();
          if (c.blink > 0) {
            ctx.fillRect(ex - 1.8, ey - 0.5, 3.6, 1);
          } else {
            ctx.arc(ex, ey, 2.1, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#1e1b4b';
            ctx.beginPath();
            ctx.arc(ex + look.x, ey + look.y, 1.05, 0, Math.PI * 2);
          }
          ctx.fill();
        }
        // antennae
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 1.2;
        for (const side of [-1, 1]) {
          ctx.beginPath();
          ctx.moveTo(p.x + side * 2, p.y - fontPx * 0.6);
          ctx.quadraticCurveTo(p.x + side * 5, p.y - fontPx * 0.95, p.x + side * 7, p.y - fontPx * 0.85);
          ctx.stroke();
        }
        // the thought bubble: what it's hunting
        if (c.seekLetters.length && c.stage === 'larva') {
          const bx = p.x + fontPx * 1.1, by = p.y - fontPx * 1.35;
          ctx.fillStyle = 'rgba(255,255,255,0.85)';
          ctx.strokeStyle = 'rgba(120,120,140,0.5)';
          ctx.lineWidth = 1;
          const txt = c.seekLetters.slice(0, 2).join(' ');
          const bw = 14 + txt.length * 6;
          ctx.beginPath();
          ctx.roundRect ? ctx.roundRect(bx - bw / 2, by - 9, bw, 18, 8) : ctx.rect(bx - bw / 2, by - 9, bw, 18);
          ctx.fill();
          ctx.stroke();
          ctx.beginPath();
      ctx.arc(bx - bw / 2 - 3, by + 9, 1.6, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = 'rgba(90,90,110,0.95)';
          ctx.font = `600 ${Math.round(fontPx * 0.7)}px sans-serif`;
          ctx.fillText(txt + '?', bx, by + 0.5);
          ctx.font = font;
        }
      } else if (c.stage === 'chrysalis') {
        const pulse = 1 + Math.sin(now / 220 + c.seed) * 0.06;
        const soon = c.cocoonT < 90;
        const rw = (w.length * fontPx * 0.34 + 10) * pulse;
        // glow before emergence
        if (soon) {
          ctx.fillStyle = `rgba(251,191,36,${0.12 + Math.sin(now / 90) * 0.08})`;
          ctx.beginPath();
          ctx.ellipse(p.x, p.y, rw + 12, rw * 0.62 + 12, 0, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = 'rgba(231,229,228,0.92)';
        ctx.strokeStyle = 'rgba(168,162,158,0.8)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, rw, rw * 0.62, 0.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // silk threads
        ctx.strokeStyle = 'rgba(168,162,158,0.45)';
        for (let s = -1; s <= 1; s++) {
          ctx.beginPath();
          ctx.ellipse(p.x, p.y, rw * (0.75 + s * 0.12), rw * 0.62 * (0.7 + s * 0.1), 0.2 + s * 0.35, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.fillStyle = 'rgba(87,83,78,0.75)';
        ctx.font = `${Math.round(fontPx * 0.8)}px ${font.split('px')[1] || 'serif'}`;
        ctx.fillText(w, p.x, p.y);
        ctx.font = font;
      } else { // butterfly / seeding
        const flap = Math.sin(now / 110 + c.seed);
        const bobY = Math.sin(now / 300 + c.seed) * 3;
        const wingW = fontPx * (0.8 + w.length * 0.12);
        const wingH = fontPx * (0.9 + w.length * 0.1);
        ctx.save();
        ctx.translate(p.x, p.y + bobY);
        const ang = Math.atan2(c.vy, c.vx);
        ctx.rotate(Math.hypot(c.vx, c.vy) > 0.004 ? ang * 0.12 : 0);
        for (const side of [-1, 1]) {
          ctx.save();
          ctx.scale(side * (0.35 + Math.abs(flap) * 0.65), 1);
          ctx.fillStyle = c.color;
          ctx.globalAlpha = 0.55;
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.quadraticCurveTo(wingW * 0.9, -wingH * 0.9, wingW, -wingH * 0.25);
          ctx.quadraticCurveTo(wingW * 0.95, wingH * 0.15, 0, wingH * 0.12);
          ctx.quadraticCurveTo(wingW * 0.55, wingH * 0.6, wingW * 0.25, wingH * 0.75);
          ctx.quadraticCurveTo(wingW * 0.1, wingH * 0.4, 0, wingH * 0.15);
          ctx.closePath();
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.restore();
        }
        // the word is the body
        ctx.fillStyle = inkColor;
        ctx.fillText(w, 0, 0);
        // tiny antennae
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 1.2;
        for (const side of [-1, 1]) {
          ctx.beginPath();
          ctx.moveTo(side * 2, -fontPx * 0.5);
          ctx.quadraticCurveTo(side * 6, -fontPx * 0.95, side * 9, -fontPx * 0.9);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // --- the Snatcher ---------------------------------------------------------------------
    {
      const S = snatcher;
      const p = smooth(S);
      const speed = Math.hypot(S.vx, S.vy);
      const ang = Math.atan2(S.vy, S.vx);
      ctx.globalAlpha = S.state === 'lurk' ? 0.55 : 1;
      ctx.fillStyle = 'rgba(0,0,0,0.16)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + 8, 8, 2.6, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(ang * Math.min(1, speed * 40));
      ctx.fillStyle = '#475569';
      ctx.beginPath();
      ctx.ellipse(0, 0, 8.5 * (1 + Math.min(0.2, speed * 6)), 8, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(p.x - 7, p.y - 5, 14, 5);
      for (const side of [-1, 1]) {
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath();
        if (S.blink > 0) {
          ctx.fillRect(p.x + side * 3.4 - 2, p.y - 2.8, 4, 1.2);
        } else {
          ctx.arc(p.x + side * 3.4, p.y - 2.5, 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#0f172a';
          ctx.beginPath();
          const look = speed > 0.003 ? { x: Math.cos(ang), y: Math.sin(ang) } : { x: 0, y: 0.3 };
          ctx.arc(p.x + side * 3.4 + look.x, p.y - 2.5 + look.y, 1, 0, Math.PI * 2);
        }
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }

  // --- first spring ---------------------------------------------------------------------------
  for (let i = 0; i < 3; i++) spawnEgg(book.lived.slice(-1), book.appetiteTail, -1);
  // stagger the first hatches so the page comes alive quickly
  let staggered = 0;
  for (const L of letters) {
    if (L.egg) { L.egg.t = 60 + staggered * 120; staggered++; }
  }

  const STEP_MS = 33;
  let last = 0, visible = true, onscreen = true;
  let running = true;
  const updateRunning = () => { running = visible && onscreen; };
  document.addEventListener('visibilitychange', () => { visible = !document.hidden; updateRunning(); });
  // several games share one page — sleep whenever scrolled out of view
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((es) => {
      for (const en of es) onscreen = en.isIntersecting;
      updateRunning();
    }, { threshold: 0.02 }).observe(stage);
  }
  function frame(now) {
    if (running) {
      if (now - last >= STEP_MS) { last = now; stepColony(); }
      draw(now);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
