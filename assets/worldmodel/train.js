// Trains a tiny MLP "world model": given (state, action, and an optional
// "anchor" correction) it predicts the change in state one physics step
// later. No external deps — everything is hand-rolled (forward/backward
// pass, Adam) since the network is tiny. Run with: node train.js
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
// Sticky random action sequence: hold a "key" for 1-12 steps like a human.
function stickyActions(steps) {
  const seq = [];
  let action = 0, hold = 0;
  for (let t = 0; t < steps; t++) {
    if (hold <= 0) {
      action = Math.floor(Math.random() * N_ACTIONS);
      hold = 1 + Math.floor(Math.random() * 12);
    }
    hold--;
    seq.push(action);
  }
  return seq;
}

function buildDataset(nEpisodes, stepsPerEpisode) {
  const X = []; // [x,y,vx,vy, onehot(action), token(4), flag(1)] length 14
  const Y = []; // [delta state(4), next-action onehot(N_ACTIONS)] length 9
  for (let e = 0; e < nEpisodes; e++) {
    let state = [
      (Math.random() * 2 - 1) * 0.8,
      (Math.random() * 2 - 1) * 0.8,
      0, 0,
    ];
    // Precompute the episode's action sequence so every step also knows the
    // NEXT action — that's the behavior-prior target. (Sticky actions make
    // it learnable: mostly "keep holding", occasionally "switch".)
    const acts = stickyActions(stepsPerEpisode + 1);
    for (let t = 0; t < stepsPerEpisode; t++) {
      const action = acts[t];
      const next = physicsStep(state, action);
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const nextOnehot = new Array(N_ACTIONS).fill(0);
      nextOnehot[acts[t + 1]] = ACTION_TARGET_SCALE;
      X.push([...state, ...onehot, ...NO_ANCHOR]);
      Y.push([next[0] - state[0], next[1] - state[1], next[2] - state[2], next[3] - state[3], ...nextOnehot]);
      state = next;
    }
  }

  // Explicitly anchor the "at rest" region: idle (v=0, action=none) truly
  // maps to zero delta everywhere in the box. Without heavy coverage here,
  // the autoregressive rollout tends to pick up a tiny velocity bias near
  // the origin that snowballs every frame into a runaway drift *before the
  // user has even pressed a key* — which reads as broken rather than the
  // "gradual drift under play" this demo is actually trying to show.
  // These synthetic examples have no real "next action", so their action
  // target is the uniform distribution — the honest conditional mean of a
  // fresh random draw, contributing no bias to the behavior prior.
  for (let i = 0; i < 30000; i++) {
    const x = (Math.random() * 2 - 1) * 0.95;
    const y = (Math.random() * 2 - 1) * 0.95;
    X.push([x, y, 0, 0, 1, 0, 0, 0, 0, ...NO_ANCHOR]); // action=none onehot
    Y.push([0, 0, 0, 0, ...UNIFORM_ACTION]);
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
    X.push([x, y, vx, vy, 1, 0, 0, 0, 0, ...NO_ANCHOR]);
    Y.push([next[0] - x, next[1] - y, next[2] - vx, next[3] - vy, ...UNIFORM_ACTION]);
  }

  return { X, Y };
}

// ---------------------------------------------------------------------------
// Tiny 2-layer MLP, trained with hand-rolled backprop + Adam. Input layout:
//   [state(4), action onehot(N_ACTIONS), GT token state(4), token flag(1)]
// The token is normally all zeros with flag=0, meaning "no reference,
// behave normally." When flag=1 and the token holds the true state, the
// network is trained to steer its prediction gently toward that token —
// see buildGuidanceDataset.
// ---------------------------------------------------------------------------
const ANCHOR_DIM = 4;
const CONF_DIM = 1;
const NO_ANCHOR = [0, 0, 0, 0, 0]; // token(4) + flag(1), all zero
const IN = 4 + N_ACTIONS + ANCHOR_DIM + CONF_DIM; // 14
const HIDDEN = 96;
// Output = state delta (4) + a behavior prior: a distribution over the NEXT
// action (N_ACTIONS), regressed against one-hots with the same MSE loss —
// MSE against one-hot labels converges to the conditional class
// probabilities, so sampling from the (clamped, renormalized) head gives a
// learned policy. It's trained on the same sticky-random play data, so it
// predicts how a player behaves: mostly keep holding the current key,
// sometimes switch. That lets a ball keep playing itself.
const OUT = 4 + N_ACTIONS; // 9
// Action targets are scaled way down so the single MSE loss isn't dominated
// by them: raw one-hots (~1.0) dwarf state deltas (~0.01) squared-error-wise
// and would trade away dynamics precision. Sampling at inference clamps and
// renormalizes the head, so the scale cancels out exactly.
const ACTION_TARGET_SCALE = 0.02;
// Honest action target for synthetic examples that have no real next action.
const UNIFORM_ACTION = new Array(N_ACTIONS).fill(ACTION_TARGET_SCALE / N_ACTIONS);

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

function trainEpochs(model, adam, X, Y, epochs, lrStart, lrDecay, stepRef) {
  const BATCH = 64;
  const idx = Array.from({ length: X.length }, (_, i) => i);
  let lr = lrStart;
  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleInPlace(idx);
    let epochLoss = 0, nBatches = 0;
    for (let i = 0; i < idx.length; i += BATCH) {
      const batchIdx = idx.slice(i, i + BATCH);
      const xs = batchIdx.map((j) => X[j]);
      const ys = batchIdx.map((j) => Y[j]);
      const { loss, grads } = trainBatch(model, xs, ys);
      stepRef.step++;
      adamStep(model, grads, adam, stepRef.step, lr);
      epochLoss += loss;
      nBatches++;
    }
    lr *= lrDecay;
    console.log(`  epoch ${epoch + 1}/${epochs}  loss=${(epochLoss / nBatches).toExponential(3)}  lr=${lr.toFixed(5)}`);
  }
}

// ---------------------------------------------------------------------------
// Self forcing: teacher forcing (train only on ground-truth trajectories)
// creates exposure bias — the model never sees its own mistakes during
// training, only during inference, so small errors compound the moment it
// has to consume its own output as the next input. Here we periodically
// roll the *current* model out on its own predictions, then label every
// state it actually visited (drifted or not) with the correct physics
// delta from that state. Training on that teaches the model to correct
// itself back toward reality instead of drifting further — DAgger applied
// to a next-state predictor.
function selfForcingBatch(model, nEpisodes, stepsPerEpisode) {
  const X = [], Y = [];
  for (let e = 0; e < nEpisodes; e++) {
    let modelState = [
      (Math.random() * 2 - 1) * 0.8,
      (Math.random() * 2 - 1) * 0.8,
      0, 0,
    ];
    // Mix episode "styles" so self-forcing directly practices correcting
    // the two situations this demo actually shows visitors: sitting idle,
    // and holding one direction into a wall — not just varied sticky play.
    const style = Math.random();
    const episodeLen = style < 0.6 ? stepsPerEpisode : stepsPerEpisode * 2;
    let fixedAction = -1;
    if (style < 0.6) {
      // sticky random (realistic mixed play) — handled per-step below
    } else if (style < 0.8) {
      fixedAction = 0; // long idle
    } else {
      fixedAction = 1 + Math.floor(Math.random() * (N_ACTIONS - 1)); // long single-direction hold
    }

    // Precompute the action sequence (one extra for next-action targets).
    const acts = fixedAction >= 0
      ? new Array(episodeLen + 1).fill(fixedAction)
      : stickyActions(episodeLen + 1);

    for (let t = 0; t < episodeLen; t++) {
      const action = acts[t];
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const nextOnehot = new Array(N_ACTIONS).fill(0);
      nextOnehot[acts[t + 1]] = ACTION_TARGET_SCALE;

      // The "correct" thing to do from wherever the model actually is now.
      const trueNext = physicsStep(modelState, action);
      X.push([...modelState, ...onehot, ...NO_ANCHOR]);
      Y.push([
        trueNext[0] - modelState[0], trueNext[1] - modelState[1],
        trueNext[2] - modelState[2], trueNext[3] - modelState[3],
        ...nextOnehot,
      ]);

      // Advance using the MODEL's own prediction, not ground truth — this
      // is what makes it self-forcing rather than teacher-forced. (The
      // input must be the FULL 14 dims incl. the zeroed token: an earlier
      // version passed 9 and the missing entries read as undefined -> NaN,
      // which the ReLU silently flushed to 0 — the "rollout" was advancing
      // by the output biases alone.)
      const delta = forward(model, [...modelState, ...onehot, ...NO_ANCHOR]);
      modelState = [
        modelState[0] + delta[0], modelState[1] + delta[1],
        modelState[2] + delta[2], modelState[3] + delta[3],
      ];
    }
  }
  return { X, Y };
}

// ---------------------------------------------------------------------------
// Learned guidance toward a ground-truth token: self-forcing reduces how
// fast error compounds per step, but a network this small still drifts
// unboundedly given long enough. Real long-horizon video models fight this
// by conditioning on a real reference (a keyframe token) and LEARNING to
// steer their rollout toward it. Same trick here: the input carries an
// extra token — the true [x,y,vx,vy] plus a presence flag — and training
// uses conditioning dropout:
//   - token zeroed, flag 0: plain dynamics (covered by buildDataset /
//     selfForcingBatch — nothing new needed here).
//   - token = the TRUE current state, flag 1: the state the network acts
//     from is deliberately displaced (as if it had drifted there on its
//     own), and the target is its own physics step PLUS a fixed gentle
//     fraction (PULL_RATE) of the gap to the token — position and
//     velocity, so the learned correction carries motion.
// At inference the demo passes the real state as the token every step, so
// the pull you see on screen is the network's own output.
//
// Design lesson, learned the hard way: an earlier version exposed the pull
// strength as a continuous confidence input, expecting the network to
// modulate the correction proportionally. Across two training attempts a
// ~1,800-parameter net collapsed that dial to an all-or-nothing gate
// (conf >= ~0.5 snapped hard, conf <= ~0.3 did nothing) — scaling one
// input by another is a multiplicative interaction a tiny ReLU net fights
// to represent. With a FIXED rate the correction is just a linear function
// of (token - own state), which fits easily. The proportionality the demo
// needs lives in the error vector, not in a learned dial.
//
// Token states must cover the velocities real play actually produces
// (terminal speed under a held key is ~0.19, plus sign flips at wall
// bounces). An early version sampled token velocities only in ±0.05, so at
// speed the velocity part of the correction was out-of-distribution: the
// model corrected position but dropped the token's motion. Drawing half
// the tokens from genuine physics rollouts fixes the velocity carryover.
const PULL_RATE = 0.15;
function collectManifoldStates(n) {
  const states = [];
  let s = [(Math.random() * 2 - 1) * 0.8, (Math.random() * 2 - 1) * 0.8, 0, 0];
  let action = 0, hold = 0;
  while (states.length < n) {
    if (hold <= 0) { action = Math.floor(Math.random() * N_ACTIONS); hold = 1 + Math.floor(Math.random() * 15); }
    hold--;
    s = physicsStep(s, action);
    states.push(s.slice());
    if (Math.random() < 0.005) { // occasional episode reset for coverage
      s = [(Math.random() * 2 - 1) * 0.8, (Math.random() * 2 - 1) * 0.8, 0, 0];
    }
  }
  return states;
}

function buildGuidanceDataset(n) {
  const X = [], Y = [];
  const manifold = collectManifoldStates(Math.ceil(n / 2));
  for (let i = 0; i < n; i++) {
    // Half on-manifold rollout states (realistic velocity/position combos,
    // bounces included), half uniform draws for coverage.
    const trueState = (i % 2 === 0)
      ? manifold[Math.floor(Math.random() * manifold.length)].slice()
      : [
          (Math.random() * 2 - 1) * 0.95,
          (Math.random() * 2 - 1) * 0.95,
          (Math.random() * 2 - 1) * 0.2,
          (Math.random() * 2 - 1) * 0.2,
        ];
    const action = Math.floor(Math.random() * N_ACTIONS);
    const onehot = new Array(N_ACTIONS).fill(0);
    onehot[action] = 1;

    // Simulate "the model thinks it's somewhere else" — a displaced own
    // state. Displacement magnitudes are weighted toward the small errors a
    // guided rollout actually lives at, with a long tail for recovery from
    // bad drift. Velocity displacement spans the full realistic range so
    // corrections are learned for badly-wrong motion, not just position.
    const posNoise = () => (Math.random() * 2 - 1) * (Math.random() < 0.6 ? 0.15 : 0.7);
    const velNoise = () => (Math.random() * 2 - 1) * (Math.random() < 0.6 ? 0.03 : 0.2);
    const ownState = [
      trueState[0] + posNoise(), trueState[1] + posNoise(),
      trueState[2] + velNoise(), trueState[3] + velNoise(),
    ];

    // Target: the physics step from where the model actually is, plus a
    // fixed gentle fraction of the gap to the token. Everything the demo's
    // guided mode does is this learned rule.
    const ownNext = physicsStep(ownState, action);
    const target = [0, 1, 2, 3].map((k) =>
      (ownNext[k] - ownState[k]) + PULL_RATE * (trueState[k] - ownState[k]));

    X.push([...ownState, ...onehot, ...trueState, 1]);
    Y.push([...target, ...UNIFORM_ACTION]); // single transitions: no real next action
  }
  return { X, Y };
}

function main() {
  console.log('Building base (teacher-forced) dataset...');
  const base = buildDataset(400, 150); // 120k transitions incl. rest anchoring
  console.log(`Base dataset: ${base.X.length} transitions`);

  const model = initModel();
  const adam = initAdam(model);
  const stepRef = { step: 0 };

  console.log('Phase 1: teacher-forced pretraining');
  trainEpochs(model, adam, base.X, base.Y, 40, 0.01, 0.93, stepRef);

  console.log('Phase 2: self-forcing rounds (correcting the model\'s own drift)');
  const SELF_FORCE_ROUNDS = 6;
  const baseIdxPool = Array.from({ length: base.X.length }, (_, i) => i);
  for (let round = 1; round <= SELF_FORCE_ROUNDS; round++) {
    const sf = selfForcingBatch(model, 250, 40); // ~14k self-visited states
    shuffleInPlace(baseIdxPool);
    const replay = baseIdxPool.slice(0, sf.X.length);
    const mixX = sf.X.concat(replay.map((j) => base.X[j]));
    const mixY = sf.Y.concat(replay.map((j) => base.Y[j]));
    console.log(`round ${round}/${SELF_FORCE_ROUNDS}: ${sf.X.length} self-forced + ${replay.length} replayed base examples`);
    trainEpochs(model, adam, mixX, mixY, 3, 0.00015, 0.9, stepRef);
  }

  console.log('Phase 3: learned guidance toward the GT token');
  {
    const guide = buildGuidanceDataset(40000);
    shuffleInPlace(baseIdxPool);
    const replay = baseIdxPool.slice(0, guide.X.length);
    const mixX = guide.X.concat(replay.map((j) => base.X[j]));
    const mixY = guide.Y.concat(replay.map((j) => base.Y[j]));
    console.log(`${guide.X.length} guidance examples + ${replay.length} replayed base examples`);
    trainEpochs(model, adam, mixX, mixY, 6, 0.0015, 0.9, stepRef);
  }

  // Sanity check 1: idle for 180 model steps (~9s at the browser's 20Hz
  // step rate) with no action pressed at all, no grounding pulses — this
  // should stay essentially at rest, not run away.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    let maxDrift = 0;
    for (let t = 0; t < 180; t++) {
      state = physicsStep(state, 0);
      const delta = forward(model, [...modelState, 1, 0, 0, 0, 0, ...NO_ANCHOR]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const dx = state[0] - modelState[0], dy = state[1] - modelState[1];
      maxDrift = Math.max(maxDrift, Math.sqrt(dx * dx + dy * dy));
    }
    console.log(`Idle (no key pressed) max drift over 180 frames: ${maxDrift.toFixed(4)}`);
  }

  // Sanity check 2: hold "right" the whole time, no grounding — this is
  // allowed to drift (esp. near the wall bounce), that's the intended
  // teaching moment once the user is actually playing.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    const action = 4;
    let totalDrift = 0;
    for (let t = 0; t < 100; t++) {
      state = physicsStep(state, action);
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const delta = forward(model, [...modelState, ...onehot, ...NO_ANCHOR]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const dx = state[0] - modelState[0], dy = state[1] - modelState[1];
      totalDrift += Math.sqrt(dx * dx + dy * dy);
    }
    console.log(`Mean drift over 100-step "hold right" rollout, no grounding: ${(totalDrift / 100).toFixed(4)}`);
  }

  // Sanity check 3: what the browser's guided mode actually ships — the GT
  // token passed with flag=1 EVERY step, correction fully produced by the
  // network. Idle should settle to a small bounded offset (a memoryless
  // net can only learn proportional-style control, which leaves a small
  // steady-state error against its own bias — eliminating it would need
  // memory for integral action). Play drift stays clearly visible.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    let lateDrift = 0, lateN = 0;
    for (let t = 0; t < 600; t++) {
      const token = [...state, 1];
      state = physicsStep(state, 0);
      const delta = forward(model, [...modelState, 1, 0, 0, 0, 0, ...token]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      if (t >= 500) { lateDrift += Math.hypot(state[0] - modelState[0], state[1] - modelState[1]); lateN++; }
    }
    console.log(`Idle, learned token guidance (browser demo): late-100-step avg drift=${(lateDrift / lateN).toFixed(4)}`);
  }

  // Sanity check 4: sticky random play with the same learned guidance.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    let action = 0, hold = 0;
    let totalDrift = 0, maxDrift = 0;
    for (let t = 0; t < 300; t++) {
      if (hold <= 0) { action = Math.floor(Math.random() * N_ACTIONS); hold = 1 + Math.floor(Math.random() * 15); }
      hold--;
      const token = [...state, 1];
      state = physicsStep(state, action);
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const delta = forward(model, [...modelState, ...onehot, ...token]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const d = Math.hypot(state[0] - modelState[0], state[1] - modelState[1]);
      totalDrift += d; maxDrift = Math.max(maxDrift, d);
    }
    console.log(`Sticky-random play, 300 steps, learned token guidance: avg=${(totalDrift / 300).toFixed(4)} max=${maxDrift.toFixed(4)}`);
  }

  // Sanity check 5: the behavior prior. Sample actions from the head and let
  // a ball play ITSELF (unguided) for 600 steps: it should keep moving
  // (mean speed well above zero), roam a good chunk of the box, and hold
  // actions stickily (P(keep) near the data's ~0.87) rather than jittering.
  {
    let state = [0, 0, 0, 0];
    let action = 0;
    let sameCount = 0, moveSum = 0;
    let minX = 1, maxX = -1, minY = 1, maxY = -1;
    for (let t = 0; t < 600; t++) {
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const o = forward(model, [...state, ...onehot, ...NO_ANCHOR]);
      state = [state[0] + o[0], state[1] + o[1], state[2] + o[2], state[3] + o[3]];
      moveSum += Math.hypot(o[0], o[1]);
      minX = Math.min(minX, state[0]); maxX = Math.max(maxX, state[0]);
      minY = Math.min(minY, state[1]); maxY = Math.max(maxY, state[1]);
      // sample next action from the clamped, renormalized head
      const probs = [];
      let z = 0;
      for (let a = 0; a < N_ACTIONS; a++) { const p = Math.max(0, o[4 + a]); probs.push(p); z += p; }
      let nextAction = Math.floor(Math.random() * N_ACTIONS);
      if (z > 1e-9) {
        let r = Math.random() * z;
        for (let a = 0; a < N_ACTIONS; a++) { r -= probs[a]; if (r <= 0) { nextAction = a; break; } }
      }
      if (nextAction === action) sameCount++;
      action = nextAction;
    }
    console.log(`Behavior prior, 600-step autonomous rollout: mean speed=${(moveSum / 600).toFixed(4)} roamX=${(maxX - minX).toFixed(2)} roamY=${(maxY - minY).toFixed(2)} P(keep action)=${(sameCount / 600).toFixed(2)}`);
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
