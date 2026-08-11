// Trains the /arena/ AUTOPILOT: a tiny policy net that plays the game.
//
// The meta part: the agent never sees the ground-truth simulator. It is
// trained entirely INSIDE the learned world model (arena_weights.json) —
// rollouts, collisions, scoring, all predicted by the ball/crate nets the
// page ships. The policy just emits action tokens into that dream and gets
// rewarded for crates delivered to the zone (world-models / Dreamer style,
// scaled down to a few hundred parameters).
//
// Optimizer: cross-entropy method (a.k.a. evolution) — no backprop through
// the world model needed, robust for ~500 parameters. Hand-rolled, no ML
// libraries. Run with: node arena_agent_train.js
'use strict';
const fs = require('fs');

const W = JSON.parse(fs.readFileSync(__dirname + '/arena_weights.json', 'utf8'));
const N_ACTIONS = W.meta.actions;
const GATE_THRESHOLD = W.meta.gateThreshold;

// --- world model runtime (identical to arena.js) ---------------------------
function forward(net, x) {
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
function nearest(from, list, excludeIdx) {
  let best = null, bd = Infinity;
  for (let i = 0; i < list.length; i++) {
    if (i === excludeIdx) continue;
    const d = Math.hypot(list[i].x - from.x, list[i].y - from.y);
    if (d < bd) { bd = d; best = list[i]; }
  }
  return best;
}
const rel = (a, b) => [b.x - a.x, b.y - a.y, b.vx - a.vx, b.vy - a.vy];
function proximity(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const d = Math.hypot(dx, dy);
  const closing = d > 1e-6 ? -(dx * (b.vx - a.vx) + dy * (b.vy - a.vy)) / d : 0;
  return [d, closing];
}
function clampEntity(e) {
  e.x = Math.max(-1.05, Math.min(1.05, e.x));
  e.y = Math.max(-1.05, Math.min(1.05, e.y));
  e.vx = Math.max(-0.5, Math.min(0.5, e.vx));
  e.vy = Math.max(-0.5, Math.min(0.5, e.vy));
}
function applyGated(e, d) {
  if (d[4] < GATE_THRESHOLD) { e.vx = 0; e.vy = 0; return; }
  e.x += d[0]; e.y += d[1]; e.vx += d[2]; e.vy += d[3];
}
function modelStep(ball, crates, action) {
  const onehot = new Array(N_ACTIONS).fill(0);
  onehot[action] = 1;
  const nb = nearest(ball, crates, -1);
  const bd = forward(W.ball, [ball.x, ball.y, ball.vx, ball.vy, ...onehot, ...rel(ball, nb), ...proximity(ball, nb)]);
  const cds = crates.map((c, i) => {
    const other = nearest(c, crates, i);
    return forward(W.crate, [c.x, c.y, c.vx, c.vy, ...rel(c, ball), ...rel(c, other), ...proximity(c, ball)]);
  });
  applyGated(ball, bd);
  clampEntity(ball);
  crates.forEach((c, i) => { applyGated(c, cds[i]); clampEntity(c); });
}

// --- policy -----------------------------------------------------------------
// Features are computed relative to a TARGET crate (nearest to the zone,
// skipping corner-camped ones) — target selection is the one hand-wired
// bit, everything else the policy has to discover. The exact same feature
// code runs on the page (arena.js).
const ZONE = { x: 0.55, y: -0.55, r: 0.22 };
const cornered = (c) => Math.abs(c.x) > 0.8 && Math.abs(c.y) > 0.8;
function pickTarget(crates) {
  let t = null, bd = Infinity;
  for (const c of crates) {
    if (cornered(c)) continue;
    const d = Math.hypot(c.x - ZONE.x, c.y - ZONE.y);
    if (d < bd) { bd = d; t = c; }
  }
  return t || crates[0];
}
function policyFeatures(ball, crate) {
  const dzx = ZONE.x - crate.x, dzy = ZONE.y - crate.y;
  const dz = Math.hypot(dzx, dzy) || 1;
  const ux = dzx / dz, uy = dzy / dz;              // push direction
  const bx = crate.x - ball.x, by = crate.y - ball.y;
  const db = Math.hypot(bx, by) || 1;
  const behindX = crate.x - ux * 0.22, behindY = crate.y - uy * 0.22;
  return [
    ball.x, ball.y, ball.vx, ball.vy,
    bx, by, db,
    ux, uy, dz,
    behindX - ball.x, behindY - ball.y,
    (bx / db) * ux + (by / db) * uy,               // alignment for the push
  ];
}
const FEAT = 13, HID = 24;
const N_PARAMS = FEAT * HID + HID + HID * N_ACTIONS + N_ACTIONS; // 461
function policyLogits(theta, feat) {
  const h = new Float64Array(HID);
  for (let j = 0; j < HID; j++) {
    let acc = theta[FEAT * HID + j];
    for (let i = 0; i < FEAT; i++) acc += feat[i] * theta[i * HID + j];
    h[j] = acc > 0 ? acc : 0;
  }
  const p = FEAT * HID + HID;
  const logits = new Float64Array(N_ACTIONS);
  for (let k = 0; k < N_ACTIONS; k++) {
    let acc = theta[p + HID * N_ACTIONS + k];
    for (let j = 0; j < HID; j++) acc += h[j] * theta[p + j * N_ACTIONS + k];
    logits[k] = acc;
  }
  return logits;
}
function policyAct(theta, feat) {
  const logits = policyLogits(theta, feat);
  let best = 0;
  for (let k = 1; k < N_ACTIONS; k++) if (logits[k] > logits[best]) best = k;
  return best;
}

// --- scripted demonstrator (the same push heuristic used to playtest) ------
// CEM from a random start collapses into "touch nothing" (reward 0 beats
// accidentally shoving crates away from the zone), so the policy is first
// behavior-cloned from this demonstrator DRIVING THE WORLD MODEL, then
// evolution refines it. The expert is only ever a teacher signal — the
// rollouts it labels are dream rollouts.
function expertAction(ball, crates) {
  const target = pickTarget(crates);
  const dl = Math.hypot(ZONE.x - target.x, ZONE.y - target.y) || 1;
  const ux = (ZONE.x - target.x) / dl, uy = (ZONE.y - target.y) / dl;
  let bx = target.x - ux * 0.22, by = target.y - uy * 0.22;
  bx = Math.max(-0.9, Math.min(0.9, bx));
  by = Math.max(-0.9, Math.min(0.9, by));
  const cbx = target.x - ball.x, cby = target.y - ball.y;
  const cbl = Math.hypot(cbx, cby) || 1;
  const align = (cbx / cbl) * ux + (cby / cbl) * uy;
  const distBehind = Math.hypot(bx - ball.x, by - ball.y);
  let dx, dy;
  if ((align > 0.65 && cbl < 0.4) || distBehind < 0.12) {
    dx = ux; dy = uy;
  } else {
    dx = bx - ball.x; dy = by - ball.y;
    const dl2 = Math.hypot(dx, dy) || 1;
    if (cbl < 0.32 && distBehind > 0.3 && (cbx * dx + cby * dy) / (cbl * dl2) > 0.5) {
      const p1 = [-cby / cbl, cbx / cbl];
      const p2 = [cby / cbl, -cbx / cbl];
      const inArena = (p) => Math.abs(ball.x + p[0] * 0.2) < 0.95 && Math.abs(ball.y + p[1] * 0.2) < 0.95;
      let perp = (p1[0] * dx + p1[1] * dy >= p2[0] * dx + p2[1] * dy) ? p1 : p2;
      if (!inArena(perp)) perp = perp === p1 ? p2 : p1;
      dx = perp[0] - (cbx / cbl) * 0.5;
      dy = perp[1] - (cby / cbl) * 0.5;
    }
  }
  const al = Math.hypot(dx, dy) || 1;
  const ax = dx / al - ball.vx, ay = dy / al - ball.vy;
  if (Math.abs(ax) < 0.05 && Math.abs(ay) < 0.05) return 0;
  if (Math.abs(ax) > Math.abs(ay)) return ax > 0 ? 4 : 3;
  return ay > 0 ? 2 : 1;
}

// --- reward: rollout inside the world model --------------------------------
function respawn(c, ball, rand) {
  for (let tries = 0; tries < 20; tries++) {
    const x = (rand() * 2 - 1) * 0.75, y = (rand() * 2 - 1) * 0.75;
    if (Math.hypot(x - ZONE.x, y - ZONE.y) > 0.45 && Math.hypot(x - ball.x, y - ball.y) > 0.3) {
      c.x = x; c.y = y; c.vx = 0; c.vy = 0;
      return;
    }
  }
  c.x = 0; c.y = 0; c.vx = 0; c.vy = 0;
}
function makeRand(seed) {
  let s = seed >>> 0;
  return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
}
function episode(theta, seed, steps) {
  const rand = makeRand(seed);
  const ball = { x: (rand() * 2 - 1) * 0.7, y: (rand() * 2 - 1) * 0.7, vx: 0, vy: 0 };
  const crates = [];
  while (crates.length < 3) {
    const c = { x: (rand() * 2 - 1) * 0.75, y: (rand() * 2 - 1) * 0.75, vx: 0, vy: 0 };
    if (Math.hypot(c.x - ball.x, c.y - ball.y) > 0.3 && crates.every((o) => Math.hypot(o.x - c.x, o.y - c.y) > 0.3)) crates.push(c);
  }
  const cornerTicks = [0, 0, 0];
  let reward = 0, scored = 0;
  for (let t = 0; t < steps; t++) {
    const target = pickTarget(crates);
    const dCrateBefore = Math.hypot(target.x - ZONE.x, target.y - ZONE.y);
    const dBallBefore = Math.hypot(target.x - ball.x, target.y - ball.y);
    const action = policyAct(theta, policyFeatures(ball, target));
    modelStep(ball, crates, action);
    // shaped reward: target crate progress toward zone (dominant), ball
    // progress toward the crate (weak, gets the agent to engage at all),
    // and a time penalty so "touch nothing" is never a local optimum
    reward += (dCrateBefore - Math.hypot(target.x - ZONE.x, target.y - ZONE.y)) * 10;
    reward += (dBallBefore - Math.hypot(target.x - ball.x, target.y - ball.y)) * 1;
    reward -= 0.005;
    crates.forEach((c, i) => {
      cornerTicks[i] = cornered(c) ? cornerTicks[i] + 1 : 0;
      if (cornerTicks[i] > 120) { respawn(c, ball, rand); cornerTicks[i] = 0; }
      if (Math.hypot(c.x - ZONE.x, c.y - ZONE.y) < ZONE.r) {
        reward += 15; scored++;
        respawn(c, ball, rand);
      }
    });
  }
  return { reward, scored };
}
function evaluate(theta, gen, nEps, steps) {
  let r = 0, s = 0;
  for (let e = 0; e < nEps; e++) {
    const out = episode(theta, 1000 + gen * 97 + e * 13, steps);
    r += out.reward; s += out.scored;
  }
  return { reward: r / nEps, scored: s / nEps };
}

// --- CEM loop ---------------------------------------------------------------
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
// --- phase 1: behavior cloning of the demonstrator (in the dream) ----------
function behaviorClone() {
  // collect (features, expert action) from dream rollouts
  const X = [], Y = [];
  for (let e = 0; e < 80; e++) {
    const rand = makeRand(5000 + e);
    const ball = { x: (rand() * 2 - 1) * 0.7, y: (rand() * 2 - 1) * 0.7, vx: 0, vy: 0 };
    const crates = [];
    while (crates.length < 3) {
      const c = { x: (rand() * 2 - 1) * 0.75, y: (rand() * 2 - 1) * 0.75, vx: 0, vy: 0 };
      if (Math.hypot(c.x - ball.x, c.y - ball.y) > 0.3 && crates.every((o) => Math.hypot(o.x - c.x, o.y - c.y) > 0.3)) crates.push(c);
    }
    const cornerTicks = [0, 0, 0];
    for (let t = 0; t < 400; t++) {
      const target = pickTarget(crates);
      const a = expertAction(ball, crates);
      X.push(policyFeatures(ball, target));
      Y.push(a);
      modelStep(ball, crates, a);
      crates.forEach((c, i) => {
        cornerTicks[i] = cornered(c) ? cornerTicks[i] + 1 : 0;
        if (cornerTicks[i] > 120 || Math.hypot(c.x - ZONE.x, c.y - ZONE.y) < ZONE.r) {
          respawn(c, ball, rand); cornerTicks[i] = 0;
        }
      });
    }
  }
  console.log(`BC dataset: ${X.length} (features, expert action) pairs from dream rollouts`);

  // softmax cross-entropy on the policy architecture, hand-rolled Adam
  const theta = new Float64Array(N_PARAMS);
  for (let i = 0; i < FEAT * HID; i++) theta[i] = randn() * Math.sqrt(2 / FEAT);
  const m = new Float64Array(N_PARAMS), v = new Float64Array(N_PARAMS);
  const B = 64, lr = 0.003, b1 = 0.9, b2 = 0.999;
  let step = 0;
  const order = X.map((_, i) => i);
  for (let ep = 0; ep < 8; ep++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let correct = 0, seen = 0;
    for (let s = 0; s + B <= order.length; s += B) {
      const grad = new Float64Array(N_PARAMS);
      for (let bi = 0; bi < B; bi++) {
        const idx = order[s + bi];
        const feat = X[idx], label = Y[idx];
        // forward, caching hidden
        const h = new Float64Array(HID), z = new Float64Array(HID);
        for (let j = 0; j < HID; j++) {
          let acc = theta[FEAT * HID + j];
          for (let i = 0; i < FEAT; i++) acc += feat[i] * theta[i * HID + j];
          z[j] = acc; h[j] = acc > 0 ? acc : 0;
        }
        const p0 = FEAT * HID + HID;
        const logits = new Float64Array(N_ACTIONS);
        let maxL = -Infinity;
        for (let k = 0; k < N_ACTIONS; k++) {
          let acc = theta[p0 + HID * N_ACTIONS + k];
          for (let j = 0; j < HID; j++) acc += h[j] * theta[p0 + j * N_ACTIONS + k];
          logits[k] = acc;
          if (acc > maxL) maxL = acc;
        }
        let sum = 0;
        const soft = new Float64Array(N_ACTIONS);
        for (let k = 0; k < N_ACTIONS; k++) { soft[k] = Math.exp(logits[k] - maxL); sum += soft[k]; }
        let am = 0;
        for (let k = 0; k < N_ACTIONS; k++) { soft[k] /= sum; if (logits[k] > logits[am]) am = k; }
        if (am === label) correct++;
        seen++;
        // backward
        const dh = new Float64Array(HID);
        for (let k = 0; k < N_ACTIONS; k++) {
          const dk = (soft[k] - (k === label ? 1 : 0)) / B;
          grad[p0 + HID * N_ACTIONS + k] += dk;
          for (let j = 0; j < HID; j++) {
            grad[p0 + j * N_ACTIONS + k] += h[j] * dk;
            dh[j] += theta[p0 + j * N_ACTIONS + k] * dk;
          }
        }
        for (let j = 0; j < HID; j++) {
          if (z[j] <= 0) continue;
          grad[FEAT * HID + j] += dh[j];
          for (let i = 0; i < FEAT; i++) grad[i * HID + j] += feat[i] * dh[j];
        }
      }
      step++;
      for (let i = 0; i < N_PARAMS; i++) {
        m[i] = b1 * m[i] + (1 - b1) * grad[i];
        v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
        theta[i] -= lr * (m[i] / (1 - b1 ** step)) / (Math.sqrt(v[i] / (1 - b2 ** step)) + 1e-8);
      }
    }
    console.log(`  BC epoch ${ep + 1}/8 argmax-match ${(100 * correct / seen).toFixed(1)}%`);
  }
  return theta;
}

function main() {
  console.log('Phase 1: behavior cloning from the scripted demonstrator (dream rollouts)...');
  let mean = behaviorClone();
  let bcEval = evaluate(mean, 999, 10, 600);
  console.log(`BC policy, 10x600-step dream episodes: reward ${bcEval.reward.toFixed(1)}, scored/ep ${bcEval.scored.toFixed(2)}`);

  console.log('Phase 2: CEM refinement in the dream...');
  const POP = 40, ELITE = 10, GENS = 40, EPS = 3, STEPS = 300;
  let sigma = new Float64Array(N_PARAMS).fill(0.08);
  let bestTheta = Float64Array.from(mean), bestScore = bcEval.reward;
  for (let g = 0; g < GENS; g++) {
    const pop = [{ theta: Float64Array.from(mean), ...evaluate(mean, g, EPS, STEPS) }];
    for (let k = 1; k < POP; k++) {
      const theta = new Float64Array(N_PARAMS);
      for (let i = 0; i < N_PARAMS; i++) theta[i] = mean[i] + sigma[i] * randn();
      pop.push({ theta, ...evaluate(theta, g, EPS, STEPS) });
    }
    pop.sort((a, b) => b.reward - a.reward);
    const elite = pop.slice(0, ELITE);
    for (let i = 0; i < N_PARAMS; i++) {
      let mu = 0;
      for (const e of elite) mu += e.theta[i];
      mu /= ELITE;
      let va = 0;
      for (const e of elite) va += (e.theta[i] - mu) ** 2;
      mean[i] = mu;
      sigma[i] = Math.min(0.15, Math.sqrt(va / ELITE) + 0.01);
    }
    const check = evaluate(mean, 999, 10, 600); // fixed seeds: comparable across gens
    if (check.reward > bestScore) { bestScore = check.reward; bestTheta = Float64Array.from(mean); }
    if (g % 5 === 0 || g === GENS - 1) {
      console.log(` gen ${String(g).padStart(2)}: elite ${elite[0].reward.toFixed(1)}/${elite[ELITE - 1].reward.toFixed(1)}  mean-policy fixed-eval reward ${check.reward.toFixed(1)} scored/ep ${check.scored.toFixed(2)}`);
    }
  }

  const final = evaluate(bestTheta, 424242, 20, 600);
  console.log(`Best policy, 20x600-step dream episodes: reward ${final.reward.toFixed(1)}, scored/ep ${final.scored.toFixed(2)} (${(final.scored * 2).toFixed(1)} per minute)`);
  fs.writeFileSync(
    __dirname + '/arena_policy.json',
    JSON.stringify({ meta: { feat: FEAT, hidden: HID, actions: N_ACTIONS }, theta: Array.from(bestTheta) })
  );
  console.log('Saved arena_policy.json,', fs.statSync(__dirname + '/arena_policy.json').size, 'bytes');
}
main();
