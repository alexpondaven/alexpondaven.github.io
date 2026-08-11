// Trains the two tiny MLPs that power the /arena/ game: a ball net and a
// crate net (the crate net is applied to every crate — shared parameters,
// like subject tokens in a multi-entity world model). ALL in-game motion,
// including ball-crate and crate-crate collisions, is predicted by these
// nets; the arena page contains no physics code. Hand-rolled training, no
// ML libraries. Run with: node arena_train.js
'use strict';

// ---------------------------------------------------------------------------
// Ground-plane physics used to GENERATE data (never shipped to the page):
// a driven ball and pushable crates in a [-1,1]^2 arena.
// ---------------------------------------------------------------------------
const ACC = 0.015;
const BALL_FRICTION = 0.94;
const CRATE_FRICTION = 0.85;
const WALL_BOUNCE = 0.6;
const BALL_R = 0.08;
const CRATE_R = 0.10;      // crates treated as circles for contact
const N_CRATES = 3;
const ACTION_DIR = [
  [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
];
const N_ACTIONS = ACTION_DIR.length;

function wall(e, r) {
  if (e.x < -1 + r) { e.x = -1 + r; e.vx = -e.vx * WALL_BOUNCE; }
  if (e.x > 1 - r) { e.x = 1 - r; e.vx = -e.vx * WALL_BOUNCE; }
  if (e.y < -1 + r) { e.y = -1 + r; e.vy = -e.vy * WALL_BOUNCE; }
  if (e.y > 1 - r) { e.y = 1 - r; e.vy = -e.vy * WALL_BOUNCE; }
}

// One physics step for the whole arena. Mutates entities.
function simStep(ball, crates, action) {
  const [ax, ay] = ACTION_DIR[action];
  ball.vx = (ball.vx + ax * ACC) * BALL_FRICTION;
  ball.vy = (ball.vy + ay * ACC) * BALL_FRICTION;
  ball.x += ball.vx; ball.y += ball.vy;
  wall(ball, BALL_R);

  for (const c of crates) {
    c.vx *= CRATE_FRICTION; c.vy *= CRATE_FRICTION;
    c.x += c.vx; c.y += c.vy;
    wall(c, CRATE_R);
  }

  // ball pushes crates
  for (const c of crates) {
    const dx = c.x - ball.x, dy = c.y - ball.y;
    const d = Math.hypot(dx, dy), rsum = BALL_R + CRATE_R;
    if (d > 1e-6 && d < rsum) {
      const nx = dx / d, ny = dy / d;
      const overlap = rsum - d;
      // separate + impart momentum
      c.x += nx * overlap * 0.7; c.y += ny * overlap * 0.7;
      ball.x -= nx * overlap * 0.3; ball.y -= ny * overlap * 0.3;
      const rel = (ball.vx - c.vx) * nx + (ball.vy - c.vy) * ny;
      if (rel > 0) {
        c.vx += nx * rel * 0.8; c.vy += ny * rel * 0.8;
        ball.vx -= nx * rel * 0.4; ball.vy -= ny * rel * 0.4;
      }
      wall(c, CRATE_R); wall(ball, BALL_R);
    }
  }
  // crate-crate separation
  for (let i = 0; i < crates.length; i++) {
    for (let j = i + 1; j < crates.length; j++) {
      const a = crates[i], b = crates[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.hypot(dx, dy), rsum = 2 * CRATE_R;
      if (d > 1e-6 && d < rsum) {
        const nx = dx / d, ny = dy / d, ov = (rsum - d) / 2;
        a.x -= nx * ov; a.y -= ny * ov;
        b.x += nx * ov; b.y += ny * ov;
        const rel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
        if (rel > 0) {
          a.vx -= nx * rel * 0.5; a.vy -= ny * rel * 0.5;
          b.vx += nx * rel * 0.5; b.vy += ny * rel * 0.5;
        }
        wall(a, CRATE_R); wall(b, CRATE_R);
      }
    }
  }
}

function randEntity(margin) {
  return {
    x: (Math.random() * 2 - 1) * (1 - margin),
    y: (Math.random() * 2 - 1) * (1 - margin),
    vx: 0, vy: 0,
  };
}

function stickyActions(steps) {
  const seq = [];
  let action = 0, hold = 0;
  for (let t = 0; t < steps; t++) {
    if (hold <= 0) { action = Math.floor(Math.random() * N_ACTIONS); hold = 1 + Math.floor(Math.random() * 15); }
    hold--;
    seq.push(action);
  }
  return seq;
}

function nearestCrate(ball, crates, excludeIdx) {
  let best = null, bd = Infinity;
  for (let i = 0; i < crates.length; i++) {
    if (i === excludeIdx) continue;
    const c = crates[i];
    const d = Math.hypot(c.x - ball.x, c.y - ball.y);
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}

function rel(a, b) { // b relative to a
  return [b.x - a.x, b.y - a.y, b.vx - a.vx, b.vy - a.vy];
}

// Distance and closing speed of `other` toward `e`, as explicit input
// features. The motion gate needs "is something bearing down on me" — a
// PRODUCT of relative direction and relative velocity, which a one-hidden-
// layer ReLU net cannot synthesize (same lesson as /play/'s confidence
// dial). Computed as inputs, the predicate becomes linear and learnable.
function proximity(e, other) {
  const dx = other.x - e.x, dy = other.y - e.y;
  const d = Math.hypot(dx, dy);
  const closing = d > 1e-6 ? -(dx * (other.vx - e.vx) + dy * (other.vy - e.vy)) / d : 0;
  return [d, closing];
}

// Shared input builders — training, self-forcing, sanity rollouts and the
// page runtime (arena.js) must all agree on this exact layout.
function ballInput(ball, crates, action) {
  const onehot = new Array(N_ACTIONS).fill(0);
  onehot[action] = 1;
  const nb = nearestCrate(ball, crates, -1);
  return [ball.x, ball.y, ball.vx, ball.vy, ...onehot, ...rel(ball, nb), ...proximity(ball, nb)];
}
function crateInput(c, i, crates, ball) {
  const other = nearestCrate(c, crates, i);
  return [c.x, c.y, c.vx, c.vy, ...rel(c, ball), ...rel(c, other), ...proximity(c, ball)];
}

// ---------------------------------------------------------------------------
// Data: ball examples [ball(4), action onehot(5), rel nearest crate(4)] -> delta(4)
//       crate examples [crate(4), rel ball(4), rel nearest other crate(4)] -> delta(4)
// ---------------------------------------------------------------------------
// Record one transition into the four dataset arrays.
//
// The 5th target is a MOTION GATE: 1 (scaled) if the entity actually moved
// this step, 0 if it sat still. MSE training can never drive the delta
// outputs to EXACTLY zero at rest (a ~5e-4 residual is far below the loss
// floor), and the velocity feedback loop amplifies that residual into a
// steady conveyor-belt drift. But an all-or-nothing gate is exactly what a
// tiny ReLU net learns well (the /play/ model taught us it can't do
// proportional dials but nails binary switches) — so the runtime applies
// the delta only when the net itself says "this thing is moving".
const GATE_SCALE = 0.05;
const moved = (e0, e1) =>
  Math.abs(e1.x - e0.x) + Math.abs(e1.y - e0.y) > 1e-4 || Math.hypot(e1.vx, e1.vy) > 0.003;
// "About to be struck": ball near and closing in. Contact onset lasts 1-2
// frames, so labeling the gate purely by "did it move" leaves the net
// regressing the hit PROBABILITY over ambiguous approach states (~0.3),
// which never clears the open threshold. This predicate is a crisp,
// deterministic function of the crate net's own inputs, so it's learnable
// to near-perfection — the gate opens on approach (where true deltas are
// still ~0, harmless) and is wide open by the moment of impact.
const incoming = (c, ball) => {
  const [d, closing] = proximity(c, ball);
  return d < 0.32 && closing > 0.03;
};
function record(BX, BY, CX, CY, before, after, action) {
  const ballGate = (moved(before.ball, after.ball) || action !== 0) ? GATE_SCALE : 0;
  BX.push(ballInput(before.ball, before.crates, action));
  BY.push([after.ball.x - before.ball.x, after.ball.y - before.ball.y, after.ball.vx - before.ball.vx, after.ball.vy - before.ball.vy, ballGate]);
  for (let i = 0; i < N_CRATES; i++) {
    const c0 = before.crates[i], c1 = after.crates[i];
    CX.push(crateInput(c0, i, before.crates, before.ball));
    CY.push([c1.x - c0.x, c1.y - c0.y, c1.vx - c0.vx, c1.vy - c0.vy, (moved(c0, c1) || incoming(c0, before.ball)) ? GATE_SCALE : 0]);
  }
}

function snapshot(ball, crates) {
  return { ball: { ...ball }, crates: crates.map((c) => ({ ...c })) };
}

function buildData(nEpisodes, steps) {
  const BX = [], BY = [], CX = [], CY = [];
  for (let e = 0; e < nEpisodes; e++) {
    const ball = randEntity(0.15);
    const crates = Array.from({ length: N_CRATES }, () => randEntity(0.2));
    const acts = stickyActions(steps);
    for (let t = 0; t < steps; t++) {
      const before = snapshot(ball, crates);
      simStep(ball, crates, acts[t]);
      record(BX, BY, CX, CY, before, snapshot(ball, crates), acts[t]);
    }
  }

  // Contact-rich episodes: fast collisions last only 1-2 frames, so plain
  // rollouts barely sample them and the learned impulse comes out mushy —
  // crates creep instead of getting properly shoved. Spawn the ball right
  // next to a crate with approach velocity and record short bursts, so
  // impact transitions get real representation in the data.
  for (let e = 0; e < 6000; e++) {
    const crates = Array.from({ length: N_CRATES }, () => randEntity(0.2));
    const target = crates[Math.floor(Math.random() * N_CRATES)];
    const angle = Math.random() * Math.PI * 2;
    const dist = 0.2 + Math.random() * 0.1;
    const speed = 0.05 + Math.random() * 0.18;
    const ball = {
      x: target.x + Math.cos(angle) * dist,
      y: target.y + Math.sin(angle) * dist,
      vx: -Math.cos(angle) * speed,
      vy: -Math.sin(angle) * speed,
    };
    const acts = stickyActions(10);
    for (let t = 0; t < 10; t++) {
      const before = snapshot(ball, crates);
      simStep(ball, crates, acts[t]);
      record(BX, BY, CX, CY, before, snapshot(ball, crates), acts[t]);
    }
  }

  // Boundary recovery: learned rollouts drift entities slightly past the
  // walls (a regime plain rollouts never visit, so the nets had no
  // restoring force there and escapes cascaded). Sample states in and
  // beyond the wall band; simStep's wall clamp makes the target "get back
  // inside", teaching an inward pull outside the arena.
  for (let i = 0; i < 25000; i++) {
    const beyond = () => {
      const s = Math.random() < 0.5 ? -1 : 1;
      return s * (0.85 + Math.random() * 0.45); // 0.85 .. 1.3, both signs
    };
    const mixed = () => (Math.random() < 0.5 ? beyond() : (Math.random() * 2 - 1));
    const ball = { x: mixed(), y: mixed(), vx: (Math.random() * 2 - 1) * 0.15, vy: (Math.random() * 2 - 1) * 0.15 };
    const crates = Array.from({ length: N_CRATES }, () => ({
      x: mixed(), y: mixed(), vx: (Math.random() * 2 - 1) * 0.1, vy: (Math.random() * 2 - 1) * 0.1,
    }));
    const action = Math.floor(Math.random() * N_ACTIONS);
    const before = snapshot(ball, crates);
    simStep(ball, crates, action);
    record(BX, BY, CX, CY, before, snapshot(ball, crates), action);
  }

  // Contact ONSET: the frame where a moving ball first strikes a RESTING
  // crate is the whole game (that's how pushing starts), yet it's a sliver
  // of rollout data — and the motion gate has to flip open exactly there,
  // against a sea of "resting crate → gate closed" examples. Dedicated
  // single-step examples (hits and near-misses both) pin that boundary.
  for (let i = 0; i < 40000; i++) {
    const crates = [];
    while (crates.length < N_CRATES) {
      const c = { x: (Math.random() * 2 - 1) * 0.85, y: (Math.random() * 2 - 1) * 0.85, vx: 0, vy: 0 };
      if (crates.every((o) => Math.hypot(o.x - c.x, o.y - c.y) > 0.25)) crates.push(c);
    }
    const t = crates[Math.floor(Math.random() * N_CRATES)];
    const ang = Math.random() * Math.PI * 2;
    const dist = 0.17 + Math.random() * 0.16;   // just at contact .. near miss
    const speed = 0.02 + Math.random() * 0.22;
    const ball = {
      x: t.x + Math.cos(ang) * dist, y: t.y + Math.sin(ang) * dist,
      vx: -Math.cos(ang) * speed, vy: -Math.sin(ang) * speed,
    };
    const action = Math.floor(Math.random() * N_ACTIONS);
    const before = snapshot(ball, crates);
    simStep(ball, crates, action);
    record(BX, BY, CX, CY, before, snapshot(ball, crates), action);
  }

  // Rest anchors: an untouched crate must stay EXACTLY still. A tiny
  //  systematic bias in the crate net (well under the loss floor per step)
  // compounds over a rollout into a visible conveyor-belt drift that walks
  // every crate into a corner. Oversample no-contact, at-rest / near-rest
  // states so "output zero" is carved in, not left to chance.
  for (let i = 0; i < 50000; i++) {
    const sep = (list, p, d) => list.every((o) => Math.hypot(o.x - p.x, o.y - p.y) > d);
    const still = () => {
      const slow = Math.random() < 0.5;
      return {
        x: (Math.random() * 2 - 1) * 0.9, y: (Math.random() * 2 - 1) * 0.9,
        vx: slow ? (Math.random() * 2 - 1) * 0.02 : 0,
        vy: slow ? (Math.random() * 2 - 1) * 0.02 : 0,
      };
    };
    const crates = [];
    while (crates.length < N_CRATES) {
      const c = still();
      if (sep(crates, c, 0.3)) crates.push(c);
    }
    let ball;
    do { ball = still(); } while (!sep(crates, ball, 0.3));
    const action = Math.random() < 0.7 ? 0 : Math.floor(Math.random() * N_ACTIONS);
    const before = snapshot(ball, crates);
    simStep(ball, crates, action);
    record(BX, BY, CX, CY, before, snapshot(ball, crates), action);
  }

  return { BX, BY, CX, CY };
}

// Self-forcing for the arena: roll out the LEARNED nets, and from every
// state they actually visit (drifted or not), label with the simulator's
// one-step correction — so the nets learn to pull their own rollout back
// toward real dynamics instead of compounding away.
function selfForcingData(ballNet, crateNet, nEpisodes, steps) {
  const BX = [], BY = [], CX = [], CY = [];
  for (let e = 0; e < nEpisodes; e++) {
    let ball = randEntity(0.15);
    let crates = Array.from({ length: N_CRATES }, () => randEntity(0.2));
    const acts = stickyActions(steps);
    for (let t = 0; t < steps; t++) {
      const before = snapshot(ball, crates);
      // ground-truth one-step from the visited state
      const gtBall = { ...ball }, gtCrates = crates.map((c) => ({ ...c }));
      simStep(gtBall, gtCrates, acts[t]);
      record(BX, BY, CX, CY, before, { ball: gtBall, crates: gtCrates }, acts[t]);

      // advance with the LEARNED nets
      const bd = forward(ballNet, ballInput(ball, crates, acts[t]));
      const cds = crates.map((c, i) => forward(crateNet, crateInput(c, i, crates, ball)));
      ball = applyGated(ball, bd);
      crates = crates.map((c, i) => applyGated(c, cds[i]));
    }
  }
  return { BX, BY, CX, CY };
}

// Advance an entity by a net's output, honoring the motion gate — the exact
// rule the page runtime uses (arena.js must stay in sync with this).
const GATE_THRESHOLD = GATE_SCALE / 2;
function applyGated(e, d) {
  if (d[4] < GATE_THRESHOLD) return { x: e.x, y: e.y, vx: 0, vy: 0 };
  return { x: e.x + d[0], y: e.y + d[1], vx: e.vx + d[2], vy: e.vy + d[3] };
}

// ---------------------------------------------------------------------------
// Minimal MLP trainer (same recipe as train.js).
// ---------------------------------------------------------------------------
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
function trainNet(net, X, Y, epochs, lr0, decay, label) {
  const { inDim, hidden, outDim } = net;
  const adam = {};
  for (const k of ['W1', 'b1', 'W2', 'b2']) adam[k] = { m: zeros(net[k].length), v: zeros(net[k].length) };
  let step = 0, lr = lr0;
  const idx = Array.from({ length: X.length }, (_, i) => i);
  const B = 64;
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    let lossSum = 0, nb = 0;
    for (let s = 0; s < idx.length; s += B) {
      const batch = idx.slice(s, s + B);
      const g = { W1: zeros(net.W1.length), b1: zeros(net.b1.length), W2: zeros(net.W2.length), b2: zeros(net.b2.length) };
      let loss = 0;
      for (const j of batch) {
        const x = X[j], y = Y[j];
        const z1 = new Float64Array(hidden), a1 = new Float64Array(hidden);
        for (let h = 0; h < hidden; h++) {
          let acc = net.b1[h];
          for (let i2 = 0; i2 < inDim; i2++) acc += x[i2] * net.W1[i2 * hidden + h];
          z1[h] = acc; a1[h] = acc > 0 ? acc : 0;
        }
        const o = new Float64Array(outDim), dz2 = new Float64Array(outDim);
        for (let k = 0; k < outDim; k++) {
          let acc = net.b2[k];
          for (let h = 0; h < hidden; h++) acc += a1[h] * net.W2[h * outDim + k];
          o[k] = acc;
          const diff = acc - y[k];
          loss += diff * diff;
          dz2[k] = 2 * diff / batch.length;
        }
        const da1 = new Float64Array(hidden);
        for (let k = 0; k < outDim; k++) {
          g.b2[k] += dz2[k];
          for (let h = 0; h < hidden; h++) {
            g.W2[h * outDim + k] += a1[h] * dz2[k];
            da1[h] += net.W2[h * outDim + k] * dz2[k];
          }
        }
        for (let h = 0; h < hidden; h++) {
          if (z1[h] <= 0) continue;
          g.b1[h] += da1[h];
          for (let i2 = 0; i2 < inDim; i2++) g.W1[i2 * hidden + h] += x[i2] * da1[h];
        }
      }
      step++;
      const b1c = 1 - Math.pow(0.9, step), b2c = 1 - Math.pow(0.999, step);
      for (const k of ['W1', 'b1', 'W2', 'b2']) {
        const p = net[k], gr = g[k], st = adam[k];
        for (let i2 = 0; i2 < p.length; i2++) {
          st.m[i2] = 0.9 * st.m[i2] + 0.1 * gr[i2];
          st.v[i2] = 0.999 * st.v[i2] + 0.001 * gr[i2] * gr[i2];
          p[i2] -= lr * (st.m[i2] / b1c) / (Math.sqrt(st.v[i2] / b2c) + 1e-8);
        }
      }
      lossSum += loss / (batch.length * outDim); nb++;
    }
    lr *= decay;
    console.log(`  ${label} epoch ${ep + 1}/${epochs} loss=${(lossSum / nb).toExponential(3)}`);
  }
}

// ---------------------------------------------------------------------------
function main() {
  console.log('Generating arena rollouts...');
  const { BX, BY, CX, CY } = buildData(400, 200); // 80k ball, 240k crate examples
  console.log(`ball examples: ${BX.length}, crate examples: ${CX.length}`);

  const ballNet = makeNet(4 + N_ACTIONS + 4 + 2, 96, 5);  // 15 -> 96 -> 4 delta + gate
  const crateNet = makeNet(4 + 4 + 4 + 2, 96, 5);         // 14 -> 96 -> 4 delta + gate

  console.log('Training ball net...');
  trainNet(ballNet, BX, BY, 14, 0.008, 0.85, 'ball');
  console.log('Training crate net...');
  trainNet(crateNet, CX, CY, 8, 0.008, 0.85, 'crate');

  console.log('Self-forcing rounds...');
  for (let round = 1; round <= 3; round++) {
    const sf = selfForcingData(ballNet, crateNet, 150, 120);
    // mix with a random replay slice of the base data (incl. boundary
    // recovery examples) so nothing gets forgotten
    const pick = (X, n) => {
      const idx = Array.from({ length: n }, () => Math.floor(Math.random() * X.length));
      return idx;
    };
    const bi = pick(BX, sf.BX.length), ci = pick(CX, sf.CX.length);
    console.log(` round ${round}: ${sf.BX.length} ball + ${sf.CX.length} crate self-forced examples`);
    trainNet(ballNet, sf.BX.concat(bi.map((i) => BX[i])), sf.BY.concat(bi.map((i) => BY[i])), 2, 0.0005, 0.9, ' sf-ball');
    trainNet(crateNet, sf.CX.concat(ci.map((i) => CX[i])), sf.CY.concat(ci.map((i) => CY[i])), 2, 0.0005, 0.9, ' sf-crate');
  }

  // Sanity: learned rollout — drive the ball into crates for 800 steps,
  // everything must stay in the arena, crates must actually move when hit.
  {
    const ball = { x: -0.7, y: -0.7, vx: 0, vy: 0 };
    const crates = [{ x: 0, y: 0, vx: 0, vy: 0 }, { x: 0.4, y: 0.2, vx: 0, vy: 0 }, { x: -0.2, y: 0.5, vx: 0, vy: 0 }];
    const acts = stickyActions(800);
    let maxAbs = 0, crateMove = 0;
    const c0 = crates.map((c) => ({ ...c }));
    for (let t = 0; t < 800; t++) {
      const bd = forward(ballNet, ballInput(ball, crates, acts[t]));
      const crateDeltas = crates.map((c, i) => forward(crateNet, crateInput(c, i, crates, ball)));
      Object.assign(ball, applyGated(ball, bd));
      crates.forEach((c, i) => Object.assign(c, applyGated(c, crateDeltas[i])));
      for (const e of [ball, ...crates]) maxAbs = Math.max(maxAbs, Math.abs(e.x), Math.abs(e.y));
    }
    crates.forEach((c, i) => { crateMove += Math.hypot(c.x - c0[i].x, c.y - c0[i].y); });
    console.log(`Learned 800-step rollout: max |pos|=${maxAbs.toFixed(2)} total crate displacement=${crateMove.toFixed(2)}`);

    // Idle drift check: nobody touches anything for 400 steps — with the
    // motion gate this must be EXACTLY zero movement.
    const iBall = { x: -0.8, y: 0.8, vx: 0, vy: 0 };
    const iCrates = [{ x: -0.1, y: -0.1, vx: 0, vy: 0 }, { x: 0.35, y: 0.3, vx: 0, vy: 0 }, { x: -0.45, y: -0.35, vx: 0, vy: 0 }];
    const iStart = iCrates.map((c) => ({ x: c.x, y: c.y }));
    let drift = 0;
    for (let t = 0; t < 400; t++) {
      Object.assign(iBall, applyGated(iBall, forward(ballNet, ballInput(iBall, iCrates, 0))));
      iCrates.forEach((c, i) => {
        Object.assign(c, applyGated(c, forward(crateNet, crateInput(c, i, iCrates, iBall))));
      });
    }
    iCrates.forEach((c, i) => { drift += Math.hypot(c.x - iStart[i].x, c.y - iStart[i].y); });
    drift += Math.hypot(iBall.x + 0.8, iBall.y - 0.8);
    console.log(`Idle 400-step drift (gate on): ${drift.toFixed(4)}`);
  }

  const out = {
    meta: { actions: N_ACTIONS, gateThreshold: GATE_THRESHOLD },
    ball: { in: ballNet.inDim, hidden: ballNet.hidden, out: ballNet.outDim, W1: Array.from(ballNet.W1), b1: Array.from(ballNet.b1), W2: Array.from(ballNet.W2), b2: Array.from(ballNet.b2) },
    crate: { in: crateNet.inDim, hidden: crateNet.hidden, out: crateNet.outDim, W1: Array.from(crateNet.W1), b1: Array.from(crateNet.b1), W2: Array.from(crateNet.W2), b2: Array.from(crateNet.b2) },
  };
  require('fs').writeFileSync(__dirname + '/arena_weights.json', JSON.stringify(out));
  console.log('Saved arena_weights.json, size:', require('fs').statSync(__dirname + '/arena_weights.json').size, 'bytes');
}

main();
