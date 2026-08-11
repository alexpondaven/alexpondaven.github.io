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
      X.push([...state, ...onehot, ...NO_ANCHOR]);
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
    X.push([x, y, 0, 0, 1, 0, 0, 0, 0, ...NO_ANCHOR]); // action=none onehot
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
    X.push([x, y, vx, vy, 1, 0, 0, 0, 0, ...NO_ANCHOR]);
    Y.push([next[0] - x, next[1] - y, next[2] - vx, next[3] - vy]);
  }

  return { X, Y };
}

// ---------------------------------------------------------------------------
// Tiny 2-layer MLP, trained with hand-rolled backprop + Adam. Input layout:
//   [state(4), action onehot(N_ACTIONS), anchor state(4), anchor confidence(1)]
// The anchor is normally all zeros with confidence=0, meaning "no outside
// correction, behave normally." When confidence=1 and anchor holds a real
// state, the network is trained to snap its prediction toward that anchor
// regardless of how far its own state has drifted — see buildGroundingDataset.
// ---------------------------------------------------------------------------
const ANCHOR_DIM = 4;
const CONF_DIM = 1;
const NO_ANCHOR = [0, 0, 0, 0, 0]; // anchor(4) + confidence(1), all zero
const IN = 4 + N_ACTIONS + ANCHOR_DIM + CONF_DIM; // 14
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

    let action = 0, holdRemaining = 0;
    for (let t = 0; t < episodeLen; t++) {
      if (fixedAction >= 0) {
        action = fixedAction;
      } else if (holdRemaining <= 0) {
        action = Math.floor(Math.random() * N_ACTIONS);
        holdRemaining = 1 + Math.floor(Math.random() * 12);
      }
      holdRemaining--;
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;

      // The "correct" thing to do from wherever the model actually is now.
      const trueNext = physicsStep(modelState, action);
      X.push([...modelState, ...onehot, ...NO_ANCHOR]);
      Y.push([
        trueNext[0] - modelState[0], trueNext[1] - modelState[1],
        trueNext[2] - modelState[2], trueNext[3] - modelState[3],
      ]);

      // Advance using the MODEL's own prediction, not ground truth — this
      // is what makes it self-forcing rather than teacher-forced.
      const delta = forward(model, [...modelState, ...onehot]);
      modelState = [
        modelState[0] + delta[0], modelState[1] + delta[1],
        modelState[2] + delta[2], modelState[3] + delta[3],
      ];
    }
  }
  return { X, Y };
}

// ---------------------------------------------------------------------------
// Periodic grounding: self-forcing reduces how fast error compounds per
// step, but a network this small will still drift, unboundedly, given long
// enough. Real long-horizon video world models fight this by periodically
// re-conditioning on a real reference (a keyframe, a sparse ground-truth
// pose) instead of purely on their own rollout. We teach the same trick:
// an extra "anchor" state + confidence input the network can optionally be
// told to trust. Training data:
//   - confidence 0, anchor zeroed: same as ordinary dynamics (already
//     covered by buildDataset/selfForcingBatch — nothing new needed here).
//   - confidence in (0,1], anchor = the TRUE current state, but the state
//     the network is asked to act from is deliberately displaced (as if it
//     had drifted there on its own). The target blends between "keep
//     drifting from where I am" and "snap toward what the anchor says
//     reality actually is," weighted by confidence.
// Honest outcome: this mechanism trains, but not the way it was designed
// to. At ~1,800 parameters the network never learns a PROPORTIONAL pull —
// across two attempts (uniform confidence sampling, then sampling heavily
// biased toward low confidence) it collapses to an all-or-nothing gate:
// conf >= ~0.5 snaps hard toward the anchor, conf <= ~0.3 does nothing.
// The snap works (verified by probes and the sanity checks below), but a
// snap looks like teleporting, and applying it in small frequent doses
// suppressed the drift so completely the demo lost its point. So the
// browser demo leaves the anchor input zeroed and applies soft guidance
// as a plain state blend outside the network instead (GUIDE_RATE in
// assets/js/worldmodel.js) — which carries velocity exactly and moves the
// ball no faster than its natural motion. The gate remains in the shipped
// weights as a documented, tested capability.
// Anchor states must cover the velocities real play actually produces
// (terminal speed under a held key is ~0.19, plus sign flips at wall
// bounces). The first version sampled anchor velocities only in ±0.05, so
// at speed the velocity part of the correction was out-of-distribution:
// the model snapped position but dropped the anchor's motion. Drawing half
// the anchors from genuine physics rollouts fixes the velocity carryover.
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

function buildGroundingDataset(n) {
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
    const trueNext = physicsStep(trueState, action);

    // Simulate "the model thinks it's somewhere else" — a displaced own
    // state, independent of confidence, so confidence must be read from its
    // own explicit input rather than inferred from how displaced things look.
    // Velocity displacement spans the full realistic range so corrections
    // are learned for badly-wrong motion too, not just wrong position.
    const posNoise = () => (Math.random() * 2 - 1) * (Math.random() < 0.5 ? 0.15 : 0.7);
    const velNoise = () => (Math.random() * 2 - 1) * (Math.random() < 0.5 ? 0.03 : 0.2);
    const ownState = [
      trueState[0] + posNoise(), trueState[1] + posNoise(),
      trueState[2] + velNoise(), trueState[3] + velNoise(),
    ];
    // Continuous, so inference can pick any pull strength — but biased
    // heavily toward the low end. With plain uniform sampling the network
    // learned a binary gate (conf>0.5 snaps, conf<0.2 does nothing at all):
    // the low-conf targets differ from ordinary drift by so little that the
    // proportional response there never got enough gradient signal, and
    // low-conf is exactly the regime continuous soft guidance runs in.
    const confidence = Math.random() < 0.5 ? Math.random() * 0.3 : Math.random();

    const ownNext = physicsStep(ownState, action);
    const driftTarget = [
      ownNext[0] - ownState[0], ownNext[1] - ownState[1],
      ownNext[2] - ownState[2], ownNext[3] - ownState[3],
    ];
    const groundTarget = [
      trueNext[0] - ownState[0], trueNext[1] - ownState[1],
      trueNext[2] - ownState[2], trueNext[3] - ownState[3],
    ];
    const target = [0, 1, 2, 3].map((k) => (1 - confidence) * driftTarget[k] + confidence * groundTarget[k]);

    X.push([...ownState, ...onehot, ...trueState, confidence]);
    Y.push(target);
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

  console.log('Phase 3: periodic grounding (learning to trust an anchor when told to)');
  {
    const ground = buildGroundingDataset(40000);
    shuffleInPlace(baseIdxPool);
    const replay = baseIdxPool.slice(0, ground.X.length);
    const mixX = ground.X.concat(replay.map((j) => base.X[j]));
    const mixY = ground.Y.concat(replay.map((j) => base.Y[j]));
    console.log(`${ground.X.length} grounding examples + ${replay.length} replayed base examples`);
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

  // Sanity check 3: same idle scenario, exercising the LEARNED anchor gate —
  // a full-confidence pulse every GROUND_PERIOD steps. The browser demo
  // doesn't drive this input (see the honest-outcome note above; it uses a
  // state blend instead), but the gate ships in the weights, so verify it.
  const GROUND_PERIOD = 15;
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    let maxDrift = 0;
    for (let t = 0; t < 180; t++) {
      state = physicsStep(state, 0);
      const grounded = t % GROUND_PERIOD === GROUND_PERIOD - 1;
      const anchor = grounded ? [...state, 1] : NO_ANCHOR;
      const delta = forward(model, [...modelState, 1, 0, 0, 0, 0, ...anchor]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const dx = state[0] - modelState[0], dy = state[1] - modelState[1];
      maxDrift = Math.max(maxDrift, Math.sqrt(dx * dx + dy * dy));
    }
    console.log(`Idle max drift over 180 frames, WITH periodic grounding: ${maxDrift.toFixed(4)}`);
  }

  // Sanity check 4: sticky random play, with the same periodic grounding.
  {
    let state = [0, 0, 0, 0];
    let modelState = [0, 0, 0, 0];
    let action = 0, hold = 0;
    let totalDrift = 0, maxDrift = 0;
    for (let t = 0; t < 300; t++) {
      if (hold <= 0) { action = Math.floor(Math.random() * N_ACTIONS); hold = 1 + Math.floor(Math.random() * 15); }
      hold--;
      state = physicsStep(state, action);
      const onehot = new Array(N_ACTIONS).fill(0);
      onehot[action] = 1;
      const grounded = t % GROUND_PERIOD === GROUND_PERIOD - 1;
      const anchor = grounded ? [...state, 1] : NO_ANCHOR;
      const delta = forward(model, [...modelState, ...onehot, ...anchor]);
      modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
      const dx = state[0] - modelState[0], dy = state[1] - modelState[1];
      const d = Math.sqrt(dx * dx + dy * dy);
      totalDrift += d; maxDrift = Math.max(maxDrift, d);
    }
    console.log(`Sticky-random play, 300 steps, WITH periodic grounding: avg=${(totalDrift / 300).toFixed(4)} max=${maxDrift.toFixed(4)}`);
  }

  // Sanity check 5: what the browser demo actually ships — PI soft guidance
  // toward the true trajectory before each step (keep GUIDE_KP/GUIDE_KI/
  // GUIDE_I_CLAMP in sync with assets/js/worldmodel.js). The proportional
  // pull alone stalls at a fixed offset where it balances the model's bias
  // (P-controller steady-state error); the clamped integral term cancels
  // the bias, so idle should converge to ~zero drift while play drift stays
  // clearly nonzero and per-step movement never exceeds natural motion.
  {
    const GUIDE_KP = 0.08, GUIDE_KI = 0.004, GUIDE_I_CLAMP = 0.02;
    const guide = (state, modelState, guideAcc) => {
      for (let k = 0; k < 4; k++) {
        const e = state[k] - modelState[k];
        guideAcc[k] = Math.max(-GUIDE_I_CLAMP, Math.min(GUIDE_I_CLAMP, guideAcc[k] + GUIDE_KI * e));
        modelState[k] += GUIDE_KP * e + guideAcc[k];
      }
    };

    // idle: drift should converge to ~0 (integral cancels the bias)
    {
      let state = [0, 0, 0, 0], modelState = [0, 0, 0, 0], guideAcc = [0, 0, 0, 0];
      let lateDrift = 0, lateN = 0;
      for (let t = 0; t < 600; t++) {
        guide(state, modelState, guideAcc);
        state = physicsStep(state, 0);
        const delta = forward(model, [...modelState, 1, 0, 0, 0, 0, ...NO_ANCHOR]);
        modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
        if (t >= 500) { lateDrift += Math.hypot(state[0] - modelState[0], state[1] - modelState[1]); lateN++; }
      }
      console.log(`Idle, WITH PI guidance (browser demo): late-100-step avg drift=${(lateDrift / lateN).toFixed(4)}`);
    }

    // sticky-random play: drift bounded but clearly visible
    {
      let state = [0, 0, 0, 0], modelState = [0, 0, 0, 0], guideAcc = [0, 0, 0, 0];
      let action = 0, hold = 0, totalDrift = 0, maxDrift = 0;
      for (let t = 0; t < 300; t++) {
        if (hold <= 0) { action = Math.floor(Math.random() * N_ACTIONS); hold = 1 + Math.floor(Math.random() * 15); }
        hold--;
        guide(state, modelState, guideAcc);
        state = physicsStep(state, action);
        const onehot = new Array(N_ACTIONS).fill(0);
        onehot[action] = 1;
        const delta = forward(model, [...modelState, ...onehot, ...NO_ANCHOR]);
        modelState = [modelState[0] + delta[0], modelState[1] + delta[1], modelState[2] + delta[2], modelState[3] + delta[3]];
        const d = Math.hypot(state[0] - modelState[0], state[1] - modelState[1]);
        totalDrift += d; maxDrift = Math.max(maxDrift, d);
      }
      console.log(`Sticky-random play, 300 steps, WITH PI guidance (browser demo): avg=${(totalDrift / 300).toFixed(4)} max=${maxDrift.toFixed(4)}`);
    }
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
