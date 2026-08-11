// Trains a tiny MLP "world model": given (state, action) it predicts the
// change in state one physics step later. No external deps — everything is
// hand-rolled (forward/backward pass, Adam) since the network is tiny
// (9 -> 64 -> 4, ~900 params). Run with: node train.js
'use strict';

// ---------------------------------------------------------------------------
// Toy environment: a single ball in a [-1,1]x[-1,1] box with friction and
// wall bounces, nudged by one of 5 discrete actions (none/up/down/left/right).
// ---------------------------------------------------------------------------
const ACC = 0.012;
const FRICTION = 0.94;
const BOUNCE = 0.8;
const ACTION_DIR = [
  [0, 0],   // none
  [0, -1],  // up
  [0, 1],   // down
  [-1, 0],  // left
  [1, 0],   // right
];
const N_ACTIONS = ACTION_DIR.length;

function physicsStep(state, action) {
  let [x, y, vx, vy] = state;
  const [ax, ay] = ACTION_DIR[action];
  vx = (vx + ax * ACC) * FRICTION;
  vy = (vy + ay * ACC) * FRICTION;
  x += vx;
  y += vy;
  if (x < -1) { x = -1; vx = -vx * BOUNCE; }
  if (x > 1) { x = 1; vx = -vx * BOUNCE; }
  if (y < -1) { y = -1; vy = -vy * BOUNCE; }
  if (y > 1) { y = 1; vy = -vy * BOUNCE; }
  return [x, y, vx, vy];
}

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Dataset: rollout episodes with "sticky" random actions (hold a key for a
// few steps, like a real user would) so the model sees realistic transitions.
// ---------------------------------------------------------------------------
function buildDataset(nEpisodes, stepsPerEpisode) {
  const X = []; // [x,y,vx,vy, onehot(action)] length 9
  const Y = []; // delta state length 4
  for (let e = 0; e < nEpisodes; e++) {
    let state = [
      (Math.random() * 2 - 1) * 0.8,
      (Math.random() * 2 - 1) * 0.8,
      0, 0,
    ];
    let action = 0;
    let holdRemaining = 0;
    for (let t = 0; t < stepsPerEpisode; t++) {
      if (holdRemaining <= 0) {
        action = Math.floor(Math.random() * N_ACTIONS);
        holdRemaining = 1 + Math.floor(Math.random() * 12);
      }
      holdRemaining--;
      const next = physicsStep(state, action);
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      X.push([...state, ...onehot]);
      Y.push([next[0] - state[0], next[1] - state[1], next[2] - state[2], next[3] - state[3]]);
      state = next;
    }
  }

  // Explicitly anchor the "at rest" region: idle (v=0, action=none) truly
  // maps to zero delta everywhere in the box. Without heavy coverage here,
  // the autoregressive rollout tends to pick up a tiny velocity bias near
  // the origin that snowballs every frame into a runaway drift *before the
  // user has even pressed a key* — which reads as broken rather than the
  // "gradual drift under play" this demo is actually trying to show.
  for (let i = 0; i < 30000; i++) {
    const x = (Math.random() * 2 - 1) * 0.95;
    const y = (Math.random() * 2 - 1) * 0.95;
    X.push([x, y, 0, 0, 1, 0, 0, 0, 0]); // action=none onehot
    Y.push([0, 0, 0, 0]);
  }
  // Also anchor low-velocity decay-toward-rest under action=none, across a
  // spread of small velocities, so the model learns friction contracts
  // toward zero rather than amplifying near v≈0. Weighted toward the
  // smallest velocities (where autoregressive rollouts spend the most time
  // once nearly settled) via a squared random draw.
  for (let i = 0; i < 30000; i++) {
    const x = (Math.random() * 2 - 1) * 0.95;
    const y = (Math.random() * 2 - 1) * 0.95;
    const magX = Math.random() * Math.random() * 0.08;
    const magY = Math.random() * Math.random() * 0.08;
    const vx = (Math.random() < 0.5 ? -1 : 1) * magX;
    const vy = (Math.random() < 0.5 ? -1 : 1) * magY;
    const next = physicsStep([x, y, vx, vy], 0);
    X.push([x, y, vx, vy, 1, 0, 0, 0, 0]);
    Y.push([next[0] - x, next[1] - y, next[2] - vx, next[3] - vy]);
  }

  return { X, Y };
}

// ---------------------------------------------------------------------------
// Tiny 2-layer MLP: in(9) -> hidden(64, ReLU) -> out(4, linear), trained with
// hand-rolled backprop + Adam.
// ---------------------------------------------------------------------------
const IN = 4 + N_ACTIONS; // 9
const HIDDEN = 96;
const OUT = 4;

function zeros(n) { return new Float64Array(n); }
function randMat(rows, cols, scale) {
  const m = new Float64Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = randn() * scale;
  return m;
}

function initModel() {
  return {
    W1: randMat(IN, HIDDEN, Math.sqrt(2 / IN)),
    b1: zeros(HIDDEN),
    W2: randMat(HIDDEN, OUT, Math.sqrt(2 / HIDDEN)),
    b2: zeros(OUT),
  };
}

function initAdam(model) {
  const s = {};
  for (const k of Object.keys(model)) {
    s[k] = { m: zeros(model[k].length), v: zeros(model[k].length) };
  }
  return s;
}

function adamStep(model, grads, state, t, lr) {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  for (const k of Object.keys(model)) {
    const p = model[k], g = grads[k], s = state[k];
    for (let i = 0; i < p.length; i++) {
      s.m[i] = beta1 * s.m[i] + (1 - beta1) * g[i];
      s.v[i] = beta2 * s.v[i] + (1 - beta2) * g[i] * g[i];
      const mHat = s.m[i] / (1 - Math.pow(beta1, t));
      const vHat = s.v[i] / (1 - Math.pow(beta2, t));
      p[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }
}

// Forward+backward over a minibatch. xs: array of length-IN arrays, ys: array
// of length-OUT arrays. Returns {loss, grads}.
function trainBatch(model, xs, ys) {
  const B = xs.length;
  const { W1, b1, W2, b2 } = model;

  const Z1 = new Float64Array(B * HIDDEN);
  const A1 = new Float64Array(B * HIDDEN);
  const Z2 = new Float64Array(B * OUT);

  for (let b = 0; b < B; b++) {
    const x = xs[b];
    for (let h = 0; h < HIDDEN; h++) {
      let acc = b1[h];
      for (let i = 0; i < IN; i++) acc += x[i] * W1[i * HIDDEN + h];
      Z1[b * HIDDEN + h] = acc;
      A1[b * HIDDEN + h] = acc > 0 ? acc : 0;
    }
    for (let o = 0; o < OUT; o++) {
      let acc = b2[o];
      for (let h = 0; h < HIDDEN; h++) acc += A1[b * HIDDEN + h] * W2[h * OUT + o];
      Z2[b * OUT + o] = acc;
    }
  }

  let loss = 0;
  const dZ2 = new Float64Array(B * OUT);
  for (let b = 0; b < B; b++) {
    for (let o = 0; o < OUT; o++) {
      const diff = Z2[b * OUT + o] - ys[b][o];
      loss += diff * diff;
      dZ2[b * OUT + o] = (2 * diff) / B;
    }
  }
  loss /= (B * OUT);

  const dW2 = zeros(HIDDEN * OUT), db2 = zeros(OUT);
  const dA1 = new Float64Array(B * HIDDEN);
  for (let b = 0; b < B; b++) {
    for (let o = 0; o < OUT; o++) {
      const g = dZ2[b * OUT + o];
      db2[o] += g;
      for (let h = 0; h < HIDDEN; h++) {
        dW2[h * OUT + o] += A1[b * HIDDEN + h] * g;
        dA1[b * HIDDEN + h] += W2[h * OUT + o] * g;
      }
    }
  }

  const dZ1 = new Float64Array(B * HIDDEN);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < HIDDEN; h++) {
      dZ1[b * HIDDEN + h] = Z1[b * HIDDEN + h] > 0 ? dA1[b * HIDDEN + h] : 0;
    }
  }

  const dW1 = zeros(IN * HIDDEN), db1 = zeros(HIDDEN);
  for (let b = 0; b < B; b++) {
    const x = xs[b];
    for (let h = 0; h < HIDDEN; h++) {
      const g = dZ1[b * HIDDEN + h];
      db1[h] += g;
      for (let i = 0; i < IN; i++) dW1[i * HIDDEN + h] += x[i] * g;
    }
  }

  return { loss, grads: { W1: dW1, b1: db1, W2: dW2, b2: db2 } };
}

function shuffleInPlace(idx) {
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
}

function main() {
  console.log('Building dataset...');
  const { X, Y } = buildDataset(400, 150); // 60k transitions
  console.log(`Dataset: ${X.length} transitions`);

  const model = initModel();
  const adam = initAdam(model);

  const BATCH = 64;
  const EPOCHS = 60;
  let lr = 0.01;
  let step = 0;

  const idx = Array.from({ length: X.length }, (_, i) => i);

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    shuffleInPlace(idx);
    let epochLoss = 0, nBatches = 0;
    for (let i = 0; i < idx.length; i += BATCH) {
      const batchIdx = idx.slice(i, i + BATCH);
      const xs = batchIdx.map((j) => X[j]);
      const ys = batchIdx.map((j) => Y[j]);
      const { loss, grads } = trainBatch(model, xs, ys);
      step++;
      adamStep(model, grads, adam, step, lr);
      epochLoss += loss;
      nBatches++;
    }
    lr *= 0.94; // decay
    console.log(`epoch ${epoch + 1}/${EPOCHS}  loss=${(epochLoss / nBatches).toExponential(3)}  lr=${lr.toFixed(4)}`);
  }

  // Sanity check 1: idle for 3 seconds (~180 frames @ 60fps) with no action
  // pressed at all — this should stay essentially at rest, not run away.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    let maxDrift = 0;
    for (let t = 0; t < 180; t++) {
      state = physicsStep(state, 0);
      const delta = forward(model, [...modelState, 1, 0, 0, 0, 0]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const dx = state[0] - modelState[0], dy = state[1] - modelState[1];
      maxDrift = Math.max(maxDrift, Math.sqrt(dx * dx + dy * dy));
    }
    console.log(`Idle (no key pressed) max drift over 180 frames: ${maxDrift.toFixed(4)}`);
  }

  // Sanity check 2: hold "right" the whole time — this is allowed to drift
  // (esp. near the wall bounce), that's the intended teaching moment once
  // the user is actually playing.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    const action = 4;
    let totalDrift = 0;
    for (let t = 0; t < 100; t++) {
      state = physicsStep(state, action);
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const delta = forward(model, [...modelState, ...onehot]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const dx = state[0] - modelState[0], dy = state[1] - modelState[1];
      totalDrift += Math.sqrt(dx * dx + dy * dy);
    }
    console.log(`Mean drift over 100-step "hold right" rollout: ${(totalDrift / 100).toFixed(4)}`);
  }

  const out = {
    meta: { in: IN, hidden: HIDDEN, out: OUT, actions: N_ACTIONS },
    W1: Array.from(model.W1), b1: Array.from(model.b1),
    W2: Array.from(model.W2), b2: Array.from(model.b2),
  };
  require('fs').writeFileSync(__dirname + '/weights.json', JSON.stringify(out));
  console.log('Saved weights.json, size:', require('fs').statSync(__dirname + '/weights.json').size, 'bytes');
}

function forward(model, x) {
  const { W1, b1, W2, b2 } = model;
  const a1 = new Float64Array(HIDDEN);
  for (let h = 0; h < HIDDEN; h++) {
    let acc = b1[h];
    for (let i = 0; i < IN; i++) acc += x[i] * W1[i * HIDDEN + h];
    a1[h] = acc > 0 ? acc : 0;
  }
  const out = new Float64Array(OUT);
  for (let o = 0; o < OUT; o++) {
    let acc = b2[o];
    for (let h = 0; h < HIDDEN; h++) acc += a1[h] * W2[h * OUT + o];
    out[o] = acc;
  }
  return out;
}

main();
