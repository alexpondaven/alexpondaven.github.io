// Trains the two tiny MLPs that power the /chase/ game: a player net and a
// hunter net. As on /arena/, ALL motion on the page is net output — but here
// the same nets are ALSO the hunter's imagination: it plans by rolling
// candidate futures through them (model-predictive control in the browser),
// and the page draws every future it considers. Hand-rolled training, no ML
// libraries. Run with: node chase_train.js
'use strict';

// ---------------------------------------------------------------------------
// Ground-truth physics used to GENERATE data (never shipped): two driven
// bodies in a [-1,1]^2 arena. They never collide — contact is a game event,
// not dynamics — so each body's world model is independent.
// ---------------------------------------------------------------------------
const FRICTION = 0.90;
const WALL_BOUNCE = 0.65;
const R = 0.07;
const ACC_P = 0.010;                 // the player is quicker...
const ACC_H = 0.0072;                // ...so the hunter can only win by planning
const DIAG = Math.SQRT1_2;
const P_DIRS = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]];
const H_DIRS = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [-DIAG, -DIAG], [DIAG, -DIAG], [-DIAG, DIAG], [DIAG, DIAG]];

function wall(e) {
  if (e.x < -1 + R) { e.x = -1 + R; e.vx = -e.vx * WALL_BOUNCE; }
  if (e.x > 1 - R) { e.x = 1 - R; e.vx = -e.vx * WALL_BOUNCE; }
  if (e.y < -1 + R) { e.y = -1 + R; e.vy = -e.vy * WALL_BOUNCE; }
  if (e.y > 1 - R) { e.y = 1 - R; e.vy = -e.vy * WALL_BOUNCE; }
}
function simStep(e, dirs, acc, action) {
  const [ax, ay] = dirs[action];
  e.vx = (e.vx + ax * acc) * FRICTION;
  e.vy = (e.vy + ay * acc) * FRICTION;
  e.x += e.vx;
  e.y += e.vy;
  wall(e);
}

const moved = (e0, e1) =>
  Math.abs(e1.x - e0.x) + Math.abs(e1.y - e0.y) > 1e-4 || Math.hypot(e1.vx, e1.vy) > 0.003;

function randEntity() {
  return { x: (Math.random() * 2 - 1) * 0.85, y: (Math.random() * 2 - 1) * 0.85, vx: (Math.random() * 2 - 1) * 0.05, vy: (Math.random() * 2 - 1) * 0.05 };
}
function stickyActions(steps, n) {
  const seq = [];
  let a = 0, hold = 0;
  for (let t = 0; t < steps; t++) {
    if (hold <= 0) { a = Math.floor(Math.random() * n); hold = 1 + Math.floor(Math.random() * 14); }
    hold--;
    seq.push(a);
  }
  return seq;
}

// features: [own(4), action onehot(n), gate is output 5]
const GATE_SCALE = 0.05;
function record(X, Y, before, after, action, nActions) {
  const onehot = new Array(nActions).fill(0);
  onehot[action] = 1;
  X.push([before.x, before.y, before.vx, before.vy, ...onehot]);
  Y.push([after.x - before.x, after.y - before.y, after.vx - before.vx, after.vy - before.vy,
    (moved(before, after) || action !== 0) ? GATE_SCALE : 0]);
}

function buildData(dirs, acc, nActions) {
  const X = [], Y = [];
  // free rollouts
  for (let e = 0; e < 500; e++) {
    const b = randEntity();
    const acts = stickyActions(240, nActions);
    for (let t = 0; t < 240; t++) {
      const before = { ...b };
      simStep(b, dirs, acc, acts[t]);
      record(X, Y, before, b, acts[t], nActions);
    }
  }
  // boundary recovery: states past the walls must pull back in
  for (let i = 0; i < 20000; i++) {
    const beyond = () => (Math.random() < 0.5 ? -1 : 1) * (0.85 + Math.random() * 0.4);
    const mixed = () => (Math.random() < 0.5 ? beyond() : Math.random() * 2 - 1);
    const b = { x: mixed(), y: mixed(), vx: (Math.random() * 2 - 1) * 0.15, vy: (Math.random() * 2 - 1) * 0.15 };
    const a = Math.floor(Math.random() * nActions);
    const before = { ...b };
    simStep(b, dirs, acc, a);
    record(X, Y, before, b, a, nActions);
  }
  // rest anchors: idle at rest must be EXACTLY still. These need real
  // weight in the dataset — with free rollouts almost always moving, a
  // token rest set leaves the gate head learning "always open"
  for (let i = 0; i < 80000; i++) {
    const slow = Math.random() < 0.25;
    const b = {
      x: (Math.random() * 2 - 1) * 0.92, y: (Math.random() * 2 - 1) * 0.92,
      vx: slow ? (Math.random() * 2 - 1) * 0.02 : 0, vy: slow ? (Math.random() * 2 - 1) * 0.02 : 0,
    };
    const a = Math.random() < 0.8 ? 0 : Math.floor(Math.random() * nActions);
    const before = { ...b };
    simStep(b, dirs, acc, a);
    record(X, Y, before, b, a, nActions);
  }
  return { X, Y };
}

// --- MLP + Adam (the house recipe) ------------------------------------------
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
          z1[h] = acc;
          a1[h] = acc > 0 ? acc : 0;
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

let GATE_T = GATE_SCALE / 2;
function applyGated(e, d) {
  if (d[4] < GATE_T) { e.vx = 0; e.vy = 0; return; }
  e.x += d[0]; e.y += d[1]; e.vx += d[2]; e.vy += d[3];
}

// Per-net gate calibration: rather than trusting GATE_SCALE/2, measure what
// the trained head actually outputs at exact rest vs in motion and split
// the difference. Cheap insurance against a marginal gate.
function calibrateGate(net, nActions) {
  let rest = 0, move = 0;
  for (let i = 0; i < 300; i++) {
    const x = [(Math.random() * 2 - 1) * 0.9, (Math.random() * 2 - 1) * 0.9, 0, 0];
    for (let k = 0; k < nActions; k++) x.push(k === 0 ? 1 : 0);
    rest += forward(net, x)[4];
    const a = 1 + Math.floor(Math.random() * (nActions - 1));
    const y = [(Math.random() * 2 - 1) * 0.9, (Math.random() * 2 - 1) * 0.9, (Math.random() * 2 - 1) * 0.06, (Math.random() * 2 - 1) * 0.06];
    for (let k = 0; k < nActions; k++) y.push(k === a ? 1 : 0);
    move += forward(net, y)[4];
  }
  rest /= 300; move /= 300;
  const t = (rest + move) / 2;
  console.log(` gate calibration: rest=${rest.toFixed(4)} move=${move.toFixed(4)} -> threshold ${t.toFixed(4)}`);
  return t;
}

// self-forcing: roll the net, label visited states with the simulator
function selfForce(net, dirs, acc, nActions, rounds) {
  for (let r = 0; r < rounds; r++) {
    const X = [], Y = [];
    for (let e = 0; e < 150; e++) {
      let b = randEntity();
      const acts = stickyActions(200, nActions);
      for (let t = 0; t < 200; t++) {
        const gt = { ...b };
        simStep(gt, dirs, acc, acts[t]);
        record(X, Y, b, gt, acts[t], nActions);
        const onehot = new Array(nActions).fill(0);
        onehot[acts[t]] = 1;
        const d = forward(net, [b.x, b.y, b.vx, b.vy, ...onehot]);
        b = { ...b };
        applyGated(b, d);
        b.x = Math.max(-1.05, Math.min(1.05, b.x));
        b.y = Math.max(-1.05, Math.min(1.05, b.y));
      }
    }
    console.log(` self-forcing round ${r + 1}: ${X.length} examples`);
    trainNet(net, X, Y, 2, 0.0006, ' sf');
  }
}

function sanity(net, dirs, acc, nActions, label) {
  // driven rollout containment + idle stillness
  let b = { x: -0.5, y: -0.5, vx: 0, vy: 0 };
  const acts = stickyActions(600, nActions);
  let maxAbs = 0;
  for (let t = 0; t < 600; t++) {
    const onehot = new Array(nActions).fill(0);
    onehot[acts[t]] = 1;
    applyGated(b, forward(net, [b.x, b.y, b.vx, b.vy, ...onehot]));
    maxAbs = Math.max(maxAbs, Math.abs(b.x), Math.abs(b.y));
  }
  const idle = { x: 0.3, y: -0.2, vx: 0, vy: 0 };
  for (let t = 0; t < 300; t++) {
    const onehot = new Array(nActions).fill(0);
    onehot[0] = 1;
    applyGated(idle, forward(net, [idle.x, idle.y, idle.vx, idle.vy, ...onehot]));
  }
  const drift = Math.hypot(idle.x - 0.3, idle.y + 0.2);
  console.log(`${label}: 600-step driven max|pos|=${maxAbs.toFixed(2)}, 300-step idle drift=${drift.toFixed(4)}`);
}

function main() {
  console.log('Training player net (5 actions)...');
  const pd = buildData(P_DIRS, ACC_P, 5);
  console.log(` examples: ${pd.X.length}`);
  const playerNet = makeNet(4 + 5, 72, 5);
  trainNet(playerNet, pd.X, pd.Y, 8, 0.006, 'player');
  selfForce(playerNet, P_DIRS, ACC_P, 5, 2);
  const playerGateT = calibrateGate(playerNet, 5);
  GATE_T = playerGateT;
  sanity(playerNet, P_DIRS, ACC_P, 5, 'player');

  console.log('Training hunter net (9 actions)...');
  const hd = buildData(H_DIRS, ACC_H, 9);
  console.log(` examples: ${hd.X.length}`);
  const hunterNet = makeNet(4 + 9, 72, 5);
  trainNet(hunterNet, hd.X, hd.Y, 8, 0.006, 'hunter');
  selfForce(hunterNet, H_DIRS, ACC_H, 9, 2);
  const hunterGateT = calibrateGate(hunterNet, 9);
  GATE_T = hunterGateT;
  sanity(hunterNet, H_DIRS, ACC_H, 9, 'hunter');

  const pack = (net) => ({ in: net.inDim, hidden: net.hidden, out: net.outDim, W1: Array.from(net.W1), b1: Array.from(net.b1), W2: Array.from(net.W2), b2: Array.from(net.b2) });
  require('fs').writeFileSync(__dirname + '/chase_weights.json', JSON.stringify({
    meta: { playerGateT: playerGateT, hunterGateT: hunterGateT, playerActions: 5, hunterActions: 9 },
    player: pack(playerNet),
    hunter: pack(hunterNet),
  }));
  console.log('Saved chase_weights.json,', require('fs').statSync(__dirname + '/chase_weights.json').size, 'bytes');
}
main();
