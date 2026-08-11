// Trains the steering brain for the /colony/ critters: a tiny MLP that
// takes [own state, where I'm going, where the cursor is] and outputs an
// acceleration. The critters' BEHAVIOR (pick a letter, haul it to the pile,
// put it back) is a simple state machine on the page — but every actual
// movement decision flows through this net: how to approach a target, how
// to slow into an arrival, how hard to flee the cursor, how to respect the
// edges. Behavior-cloned from a scripted steering demonstrator, then
// self-forced so the net's own rollouts stay stable. Hand-rolled, no ML
// libraries. Run with: node critter_train.js
'use strict';
const fs = require('fs');

// World: [-1,1]^2, point-mass critter.
const FRICTION = 0.90;
const ACC_CAP = 0.012;
const MAX_SPEED = 0.045;
const FLEE_RADIUS = 0.35;

// --- scripted demonstrator: classic steering behaviors ----------------------
function demonstrator(c, target, cursor) {
  // seek with arrival: desired velocity shrinks near the target
  const tx = target.x - c.x, ty = target.y - c.y;
  const dt = Math.hypot(tx, ty) || 1e-6;
  const speed = Math.min(MAX_SPEED, dt * 0.6); // assertive arrival: keep a
  // real approach speed until close, or the cloned net inherits a mushy
  // endgame and parks short of the target
  let ax = (tx / dt) * speed - c.vx;
  let ay = (ty / dt) * speed - c.vy;

  // flee the cursor, weight ramps up as it closes in
  const fx = c.x - cursor.x, fy = c.y - cursor.y;
  const df = Math.hypot(fx, fy) || 1e-6;
  if (df < FLEE_RADIUS) {
    const w = ((FLEE_RADIUS - df) / FLEE_RADIUS) * 0.06;
    ax += (fx / df) * w - c.vx * 0.3;
    ay += (fy / df) * w - c.vy * 0.3;
  }

  // stay off the walls
  const M = 0.10, K = 0.03;
  if (c.x < -1 + M) ax += ((-1 + M) - c.x) * K / M;
  if (c.x > 1 - M) ax += ((1 - M) - c.x) * K / M;
  if (c.y < -1 + M) ay += ((-1 + M) - c.y) * K / M;
  if (c.y > 1 - M) ay += ((1 - M) - c.y) * K / M;

  const a = Math.hypot(ax, ay);
  if (a > ACC_CAP) { ax = (ax / a) * ACC_CAP; ay = (ay / a) * ACC_CAP; }
  return [ax, ay];
}

function stepBody(c, ax, ay) {
  c.vx = (c.vx + ax) * FRICTION;
  c.vy = (c.vy + ay) * FRICTION;
  c.x += c.vx;
  c.y += c.vy;
  c.x = Math.max(-1, Math.min(1, c.x));
  c.y = Math.max(-1, Math.min(1, c.y));
}

// --- features (must match assets/js/critters.js exactly) --------------------
// Unit directions are fed explicitly: the steering label is speed(dt) times
// tx/dt — a ratio of two vanishing numbers near the target that a one-layer
// ReLU net cannot form on its own (the same lesson as the arena's closing-
// speed feature). With the units as inputs, the label is almost linear.
function features(c, target, cursor) {
  const tx = target.x - c.x, ty = target.y - c.y;
  const dt = Math.hypot(tx, ty);
  const utx = dt > 1e-6 ? tx / dt : 0, uty = dt > 1e-6 ? ty / dt : 0;
  const fx = cursor.x - c.x, fy = cursor.y - c.y;
  const df = Math.hypot(fx, fy);
  const ufx = df > 1e-6 ? fx / df : 0, ufy = df > 1e-6 ? fy / df : 0;
  const closing = df > 1e-6 ? -(fx * (cursor.vx - c.vx) + fy * (cursor.vy - c.vy)) / df : 0;
  return [c.x, c.y, c.vx, c.vy, tx, ty, dt, utx, uty, fx, fy, df, ufx, ufy, closing];
}
const FEAT = 15, HID = 24, OUT = 2;

// --- data: episodes with wandering targets and a sometimes-hostile cursor ---
function randRange(a, b) { return a + Math.random() * (b - a); }
function makeCursor() {
  return { x: randRange(-1, 1), y: randRange(-1, 1), vx: 0, vy: 0, chase: Math.random() < 0.4 };
}
function buildData(nEpisodes, steps) {
  const X = [], Y = [];
  for (let e = 0; e < nEpisodes; e++) {
    const c = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9), vx: 0, vy: 0 };
    let target = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9) };
    const cursor = makeCursor();
    for (let t = 0; t < steps; t++) {
      // cursor: random walk, sometimes actively chasing the critter
      const gx = cursor.chase ? c.x : cursor.x + randRange(-0.2, 0.2);
      const gy = cursor.chase ? c.y : cursor.y + randRange(-0.2, 0.2);
      cursor.vx = (cursor.vx + (gx - cursor.x) * 0.01) * 0.9;
      cursor.vy = (cursor.vy + (gy - cursor.y) * 0.01) * 0.9;
      cursor.x += cursor.vx; cursor.y += cursor.vy;
      if (Math.random() < 0.01) cursor.chase = !cursor.chase;
      if (Math.random() < 0.015 || Math.hypot(target.x - c.x, target.y - c.y) < 0.05) {
        target = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9) };
      }
      X.push(features(c, target, cursor));
      const [ax, ay] = demonstrator(c, target, cursor);
      Y.push([ax, ay]);
      stepBody(c, ax, ay);
    }
  }

  // Stall states: demonstration rollouts almost never contain "nearly
  // stationary, short of the target" (the demonstrator is always moving),
  // so behavior cloning leaves that region undefined — and the net's
  // default there turned out to be "do nothing", a stable fixed point that
  // parked every critter ~0.2 away from its letter. Sample that hole
  // explicitly, labeled by the demonstrator (which says: push on).
  for (let i = 0; i < 80000; i++) {
    const c = {
      x: randRange(-0.95, 0.95), y: randRange(-0.95, 0.95),
      vx: randRange(-0.008, 0.008), vy: randRange(-0.008, 0.008),
    };
    let target;
    if (Math.random() < 0.5) {
      const ang = Math.random() * Math.PI * 2, d = 0.03 + Math.random() * 0.4;
      target = {
        x: Math.max(-0.95, Math.min(0.95, c.x + Math.cos(ang) * d)),
        y: Math.max(-0.95, Math.min(0.95, c.y + Math.sin(ang) * d)),
      };
    } else {
      target = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9) };
    }
    const cursor = { x: randRange(-1.3, 1.3), y: randRange(-1.3, 1.3), vx: 0, vy: 0 };
    X.push(features(c, target, cursor));
    Y.push(demonstrator(c, target, cursor));
  }

  return { X, Y };
}

// --- MLP + Adam (same recipe as the other trainers) --------------------------
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function zeros(n) { return new Float64Array(n); }
function makeNet(inDim, hidden, outDim) {
  const W1 = new Float64Array(inDim * hidden), W2 = new Float64Array(hidden * outDim);
  for (let i = 0; i < W1.length; i++) W1[i] = randn() * Math.sqrt(2 / inDim);
  for (let i = 0; i < W2.length; i++) W2[i] = randn() * Math.sqrt(2 / hidden);
  return { W1, b1: zeros(hidden), W2, b2: zeros(outDim), inDim, hidden, outDim };
}
function forward(net, x) {
  const { W1, b1, W2, b2, inDim, hidden, outDim } = net;
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
function trainNet(net, X, Y, epochs, lr0, label) {
  const { inDim, hidden, outDim } = net;
  const adam = {};
  for (const k of ['W1', 'b1', 'W2', 'b2']) adam[k] = { m: zeros(net[k].length), v: zeros(net[k].length) };
  const B = 64;
  let step = 0;
  const order = X.map((_, i) => i);
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const lr = lr0 * Math.pow(0.85, ep);
    let loss = 0, seen = 0;
    for (let s = 0; s + B <= order.length; s += B) {
      const g = { W1: zeros(net.W1.length), b1: zeros(hidden), W2: zeros(net.W2.length), b2: zeros(outDim) };
      for (let bi = 0; bi < B; bi++) {
        const idx = order[s + bi];
        const x = X[idx], y = Y[idx];
        const z1 = new Float64Array(hidden), a1 = new Float64Array(hidden);
        for (let h = 0; h < hidden; h++) {
          let acc = net.b1[h];
          for (let i2 = 0; i2 < inDim; i2++) acc += x[i2] * net.W1[i2 * hidden + h];
          z1[h] = acc; a1[h] = acc > 0 ? acc : 0;
        }
        const dout = new Float64Array(outDim);
        for (let k = 0; k < outDim; k++) {
          let acc = net.b2[k];
          for (let h = 0; h < hidden; h++) acc += a1[h] * net.W2[h * outDim + k];
          const diff = acc - y[k];
          loss += diff * diff;
          dout[k] = (2 * diff) / (B * outDim);
        }
        seen++;
        const dh = new Float64Array(hidden);
        for (let k = 0; k < outDim; k++) {
          g.b2[k] += dout[k];
          for (let h = 0; h < hidden; h++) {
            g.W2[h * outDim + k] += a1[h] * dout[k];
            dh[h] += net.W2[h * outDim + k] * dout[k];
          }
        }
        for (let h = 0; h < hidden; h++) {
          if (z1[h] <= 0) continue;
          g.b1[h] += dh[h];
          for (let i2 = 0; i2 < inDim; i2++) g.W1[i2 * hidden + h] += x[i2] * dh[h];
        }
      }
      step++;
      for (const k of ['W1', 'b1', 'W2', 'b2']) {
        const p = net[k], gr = g[k], a = adam[k];
        for (let i2 = 0; i2 < p.length; i2++) {
          a.m[i2] = 0.9 * a.m[i2] + 0.1 * gr[i2];
          a.v[i2] = 0.999 * a.v[i2] + 0.001 * gr[i2] * gr[i2];
          p[i2] -= lr * (a.m[i2] / (1 - Math.pow(0.9, step))) / (Math.sqrt(a.v[i2] / (1 - Math.pow(0.999, step))) + 1e-8);
        }
      }
    }
    console.log(`  ${label} epoch ${ep + 1}/${epochs} loss=${(loss / (seen * outDim)).toExponential(3)}`);
  }
}

// Self-forcing: roll out the NET, label visited states with the demonstrator.
function selfForcingData(net, nEpisodes, steps) {
  const X = [], Y = [];
  for (let e = 0; e < nEpisodes; e++) {
    const c = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9), vx: 0, vy: 0 };
    let target = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9) };
    const cursor = makeCursor();
    for (let t = 0; t < steps; t++) {
      cursor.x += cursor.vx; cursor.y += cursor.vy;
      cursor.vx = (cursor.vx + randRange(-0.002, 0.002)) * 0.95;
      cursor.vy = (cursor.vy + randRange(-0.002, 0.002)) * 0.95;
      if (Math.random() < 0.015 || Math.hypot(target.x - c.x, target.y - c.y) < 0.05) {
        target = { x: randRange(-0.9, 0.9), y: randRange(-0.9, 0.9) };
      }
      const f = features(c, target, cursor);
      X.push(f);
      Y.push(demonstrator(c, target, cursor));
      const o = forward(net, f);           // advance with the LEARNED brain
      const a = Math.hypot(o[0], o[1]);
      const s = a > ACC_CAP ? ACC_CAP / a : 1;
      stepBody(c, o[0] * s, o[1] * s);
    }
  }
  return { X, Y };
}

function main() {
  console.log('Generating steering demonstrations...');
  const { X, Y } = buildData(600, 300);
  console.log(`examples: ${X.length}`);
  const net = makeNet(FEAT, HID, OUT);
  trainNet(net, X, Y, 6, 0.004, 'steer');

  for (let round = 1; round <= 3; round++) {
    const sf = selfForcingData(net, 120, 250);
    console.log(` self-forcing round ${round}: ${sf.X.length} examples`);
    trainNet(net, sf.X, sf.Y, 2, 0.0006, ' sf');
  }

  // Sanity 1: net rollout reaches a static target and settles
  {
    const c = { x: -0.8, y: -0.8, vx: 0, vy: 0 };
    const target = { x: 0.6, y: 0.5 };
    const cursor = { x: -2, y: -2, vx: 0, vy: 0 }; // far away
    let reached = -1;
    for (let t = 0; t < 400; t++) {
      const o = forward(net, features(c, target, cursor));
      const a = Math.hypot(o[0], o[1]);
      const s = a > ACC_CAP ? ACC_CAP / a : 1;
      stepBody(c, o[0] * s, o[1] * s);
      if (reached < 0 && Math.hypot(c.x - target.x, c.y - target.y) < 0.06) reached = t;
    }
    const settled = Math.hypot(c.x - target.x, c.y - target.y);
    console.log(`Sanity seek: reached target at step ${reached}, final offset ${settled.toFixed(3)}`);
  }
  // Sanity 2: cursor parked next to the critter — it must clear out
  {
    const c = { x: 0, y: 0, vx: 0, vy: 0 };
    const target = { x: 0.05, y: 0 };
    const cursor = { x: -0.08, y: 0, vx: 0, vy: 0 };
    let minD = Infinity, maxD = 0;
    for (let t = 0; t < 150; t++) {
      const o = forward(net, features(c, target, cursor));
      const a = Math.hypot(o[0], o[1]);
      const s = a > ACC_CAP ? ACC_CAP / a : 1;
      stepBody(c, o[0] * s, o[1] * s);
      const d = Math.hypot(c.x - cursor.x, c.y - cursor.y);
      minD = Math.min(minD, d); maxD = Math.max(maxD, d);
    }
    console.log(`Sanity flee: distance from cursor grew to ${maxD.toFixed(2)} (never below ${minD.toFixed(2)})`);
  }

  const out = {
    meta: { feat: FEAT, hidden: HID, out: OUT, accCap: ACC_CAP, friction: FRICTION },
    net: { in: net.inDim, hidden: net.hidden, out: net.outDim, W1: Array.from(net.W1), b1: Array.from(net.b1), W2: Array.from(net.W2), b2: Array.from(net.b2) },
  };
  fs.writeFileSync(__dirname + '/critter_policy.json', JSON.stringify(out));
  console.log('Saved critter_policy.json,', fs.statSync(__dirname + '/critter_policy.json').size, 'bytes');
}
main();
