// Runs a tiny neural "world model" (~1,800 params, trained offline — see
// /assets/worldmodel/train.js) entirely client-side. No server, no CDN
// dependency: fetches its own weights.json and does the forward pass by hand.
// Only loaded on /play/, so it has zero footprint on the rest of the site.
(function () {
  'use strict';

  const canvas = document.getElementById('wm-canvas');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');

  // Must match assets/worldmodel/train.js exactly.
  const ACC = 0.012;
  const FRICTION = 0.94;
  const BOUNCE = 0.8;
  const ACTION_DIR = [
    [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
  ];
  const N_ACTIONS = ACTION_DIR.length;

  function physicsStep(state, action) {
    let [x, y, vx, vy] = state;
    const [ax, ay] = ACTION_DIR[action];
    vx = (vx + ax * ACC) * FRICTION;
    vy = (vy + ay * ACC) * FRICTION;
    x += vx; y += vy;
    if (x < -1) { x = -1; vx = -vx * BOUNCE; }
    if (x > 1) { x = 1; vx = -vx * BOUNCE; }
    if (y < -1) { y = -1; vy = -vy * BOUNCE; }
    if (y > 1) { y = 1; vy = -vy * BOUNCE; }
    return [x, y, vx, vy];
  }

  let model = null;
  function forward(x) {
    const { W1, b1, W2, b2, meta } = model;
    const { hidden, out, in: inDim } = meta;
    const a1 = new Float64Array(hidden);
    for (let h = 0; h < hidden; h++) {
      let acc = b1[h];
      for (let i = 0; i < inDim; i++) acc += x[i] * W1[i * hidden + h];
      a1[h] = acc > 0 ? acc : 0;
    }
    const o = new Float64Array(out);
    for (let k = 0; k < out; k++) {
      let acc = b2[k];
      for (let h = 0; h < hidden; h++) acc += a1[h] * W2[h * out + k];
      o[k] = acc;
    }
    return o;
  }

  // The input carries an optional ground-truth token: the true [x,y,vx,vy]
  // plus a presence flag. Trained with conditioning dropout (train.js
  // Phase 3), the network learned to steer its own prediction gently toward
  // the token when present — position and velocity, at a fixed rate baked
  // in during training — and to behave as a plain world model when the
  // token is zeroed. So in guided mode the on-screen correction is the
  // network's own output, not post-processing.
  const NO_TOKEN = [0, 0, 0, 0, 0];
  // Returns the next state AND the behavior prior: the network's predicted
  // distribution over the NEXT action (outputs 4..8, trained by one-hot
  // regression — see train.js). Sampling from it lets a ball play itself.
  function modelStep(state, action, token) {
    const onehot = new Array(N_ACTIONS).fill(0);
    onehot[action] = 1;
    const tokenInput = token ? [...token, 1] : NO_TOKEN;
    const o = forward([...state, ...onehot, ...tokenInput]);
    return {
      state: [state[0] + o[0], state[1] + o[1], state[2] + o[2], state[3] + o[3]],
      actionOut: o.slice(4, 4 + N_ACTIONS),
    };
  }

  // Clamp negatives, renormalize, sample — scale-invariant, so the tiny
  // training-target scale cancels out here. Two inference-time shaping
  // knobs for the wanderers: exclude "none" (they never just coast — the
  // raw prior under-thrusts and the ring contracts under the pull) and
  // square the probabilities (sharpen toward the mode for longer runs).
  function sampleAction(actionOut, noIdle, sharpen) {
    let z = 0;
    const p = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      let v = Math.max(0, actionOut[a]);
      if (noIdle && a === 0) v = 0;
      if (sharpen) v = v * v;
      p.push(v); z += v;
    }
    if (z <= 1e-9) return noIdle ? 1 + Math.floor(Math.random() * (N_ACTIONS - 1)) : Math.floor(Math.random() * N_ACTIONS);
    let r = Math.random() * z;
    for (let a = 0; a < N_ACTIONS; a++) { r -= p[a]; if (r <= 0) return a; }
    return N_ACTIONS - 1;
  }

  // --- input ---------------------------------------------------------------
  const KEYMAP = { ArrowUp: 1, KeyW: 1, ArrowDown: 2, KeyS: 2, ArrowLeft: 3, KeyA: 3, ArrowRight: 4, KeyD: 4 };
  let currentAction = 0;
  const held = new Set();
  function recomputeAction() {
    // Most-recently-pressed wins so opposite keys don't cancel confusingly.
    currentAction = held.size ? [...held][held.size - 1] : 0;
  }
  window.addEventListener('keydown', (e) => {
    const a = KEYMAP[e.code];
    if (a) { held.add(a); recomputeAction(); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => {
    const a = KEYMAP[e.code];
    if (a) { held.delete(a); recomputeAction(); }
  });

  document.querySelectorAll('.wm-btn[data-action]').forEach((btn) => {
    const a = parseInt(btn.dataset.action, 10);
    const press = (e) => { e.preventDefault(); held.add(a); recomputeAction(); };
    const release = (e) => { e.preventDefault(); held.delete(a); recomputeAction(); };
    btn.addEventListener('mousedown', press);
    btn.addEventListener('touchstart', press, { passive: false });
    btn.addEventListener('mouseup', release);
    btn.addEventListener('mouseleave', release);
    btn.addEventListener('touchend', release);
  });

  // --- state -----------------------------------------------------------------
  let modelState = [0, 0, 0, 0];
  let truthState = [0, 0, 0, 0];
  let showTruth = false;
  let running = true;
  // Guidance on = pass the true state as the network's GT token every step.
  // The learned pull is proportional-style control, so a small steady-state
  // offset against the model's own bias remains — a memoryless network
  // can't learn integral action (that would need internal state).
  let guidance = true;

  // Cycle mode: three copies of the model in a ring, each following the
  // NEXT ball's current state. Pure following would collapse the ring onto
  // one point, so the two non-player balls also live their own lives with
  // sticky random actions — their exploration keeps the ring spread out
  // while the pull keeps it loosely together. Ball 0 takes the player's
  // actions. There is no ground truth here at all — every ball is a model
  // rollout, coupled only through the learned token input.
  //
  // FOLLOW_STRENGTH softens the chase without retraining: the learned pull
  // is (verified) roughly linear in (token - state), so feeding a token
  // interpolated partway toward the target scales the pull by the same
  // fraction — "follow slowly" for free.
  let mode = 'solo';
  const FOLLOW_STRENGTH = 0.5;
  const TRAIL_LEN = 45;
  const CYCLE_STARTS = [[-0.5, 0, 0, 0], [0.5, 0.3, 0, 0], [0, -0.5, 0, 0]];
  let balls = [], trails = [];
  // Every ball's action comes from the model's own behavior prior — the
  // wanderers always, and the player's ball whenever no key is held
  // (autopilot: grab it any time to take over). Two implementation notes,
  // both found empirically:
  //  - The policy is read from a second, UNCONDITIONED forward pass: the
  //    flag=1 action head was trained largely on uniform targets (the
  //    guidance dataset has no real next-actions), so the conditioned
  //    policy is jitter; the sticky learned behavior lives at flag=0.
  //  - A sampled action is held for ACTION_HOLD steps: the per-step head
  //    can't carry temporal persistence (memoryless), and without holds
  //    the ring contracts into a clump under the follow pull.
  const ACTION_HOLD = 8;
  const cycleActs = [0, 0, 0], cycleCool = [0, 0, 0];
  function resetCycle() {
    balls = CYCLE_STARTS.map((s) => s.slice());
    trails = [[], [], []];
    for (let i = 0; i < 3; i++) { cycleActs[i] = 0; cycleCool[i] = 0; }
  }
  function clampState(s) {
    // safety net so a coupling surprise can't fling a ball to infinity
    return [
      Math.max(-1.5, Math.min(1.5, s[0])), Math.max(-1.5, Math.min(1.5, s[1])),
      Math.max(-0.5, Math.min(0.5, s[2])), Math.max(-0.5, Math.min(0.5, s[3])),
    ];
  }
  resetCycle();

  const toggleBtn = document.getElementById('wm-toggle-truth');
  const groundBtn = document.getElementById('wm-toggle-ground');
  const modeBtn = document.getElementById('wm-toggle-mode');
  const resetBtn = document.getElementById('wm-reset');
  const driftEl = document.getElementById('wm-drift');
  const hudEl = document.querySelector('.wm-hud');
  const statusEl = document.getElementById('wm-status');

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      showTruth = !showTruth;
      toggleBtn.textContent = showTruth ? 'Hide ground truth' : 'Show ground truth';
      toggleBtn.setAttribute('aria-pressed', String(showTruth));
    });
  }
  if (groundBtn) {
    groundBtn.addEventListener('click', () => {
      guidance = !guidance;
      groundBtn.textContent = guidance ? 'Guidance: on' : 'Guidance: off';
      groundBtn.setAttribute('aria-pressed', String(guidance));
    });
  }
  if (modeBtn) {
    modeBtn.addEventListener('click', () => {
      mode = mode === 'solo' ? 'cycle' : 'solo';
      modeBtn.textContent = mode === 'solo' ? 'Mode: solo' : 'Mode: cycle';
      // Ground truth and guidance only mean something in solo mode.
      const soloOnly = mode === 'solo' ? '' : 'none';
      if (toggleBtn) toggleBtn.style.display = soloOnly;
      if (groundBtn) groundBtn.style.display = soloOnly;
      if (hudEl) hudEl.style.display = soloOnly;
      modelState = [0, 0, 0, 0];
      truthState = [0, 0, 0, 0];
      resetCycle();
    });
  }
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      modelState = [0, 0, 0, 0];
      truthState = [0, 0, 0, 0];
      resetCycle();
    });
  }

  // --- render ----------------------------------------------------------------
  function accent() {
    return getComputedStyle(document.documentElement).getPropertyValue('--global-theme-color').trim() || '#4f46e5';
  }
  function toPixel(nx, ny, w, h) {
    return [((nx + 1) / 2) * w, ((ny + 1) / 2) * h];
  }

  const CYCLE_COLORS = [null, '#f59e0b', '#14b8a6']; // null = accent (player)

  function drawSolo(w, h) {
    if (showTruth) {
      const [tx, ty] = toPixel(truthState[0], truthState[1], w, h);
      ctx.strokeStyle = '#9ca3af';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(tx, ty, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const [mx, my] = toPixel(modelState[0], modelState[1], w, h);
    ctx.fillStyle = accent();
    ctx.beginPath();
    ctx.arc(mx, my, 10, 0, Math.PI * 2);
    ctx.fill();

    if (driftEl) {
      const dx = truthState[0] - modelState[0], dy = truthState[1] - modelState[1];
      driftEl.textContent = Math.sqrt(dx * dx + dy * dy).toFixed(3);
    }
  }

  function drawCycle(w, h) {
    for (let i = 0; i < 3; i++) {
      const color = CYCLE_COLORS[i] || accent();
      const trail = trails[i];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (let t = 1; t < trail.length; t++) {
        ctx.globalAlpha = (t / trail.length) * 0.35;
        const [x0, y0] = toPixel(trail[t - 1][0], trail[t - 1][1], w, h);
        const [x1, y1] = toPixel(trail[t][0], trail[t][1], w, h);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const [bx, by] = toPixel(balls[i][0], balls[i][1], w, h);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(bx, by, i === 0 ? 10 : 8, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function draw() {
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);
    if (mode === 'solo') drawSolo(w, h); else drawCycle(w, h);
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // The model steps at a fixed ~20Hz (rendering still happens every animation
  // frame for smoothness). This isn't just performance throttling: since the
  // model is autoregressive, every physics step compounds a little numerical
  // error — fewer steps per second means the "drift from reality" the demo
  // is about builds up gradually over real play, instead of the tiny model
  // visibly running away within the first second just from being idle.
  const STEP_INTERVAL_MS = 50;
  let lastStepTime = 0;

  function stepSolo() {
    // The token must be the true state at the SAME timestep as modelState —
    // i.e. captured before either advances — matching how the network was
    // trained.
    const token = guidance ? truthState : null;
    truthState = physicsStep(truthState, currentAction);
    modelState = modelStep(modelState, currentAction, token).state;
  }

  function stepCycle() {
    // Token for ball i = partway toward ball i+1's current state.
    const tokens = balls.map((b, i) => {
      const target = balls[(i + 1) % 3];
      return [
        b[0] + FOLLOW_STRENGTH * (target[0] - b[0]),
        b[1] + FOLLOW_STRENGTH * (target[1] - b[1]),
        b[2] + FOLLOW_STRENGTH * (target[2] - b[2]),
        b[3] + FOLLOW_STRENGTH * (target[3] - b[3]),
      ];
    });
    // Player keys override ball 0; otherwise every ball drives itself with
    // the model's predicted next action (sampled from the unconditioned
    // pass, held for ACTION_HOLD steps — see notes above).
    const playerDriving = held.size > 0;
    balls = balls.map((b, i) => {
      const action = (i === 0 && playerDriving) ? currentAction : cycleActs[i];
      const r = modelStep(b, action, tokens[i]);
      if (--cycleCool[i] <= 0) {
        const prior = modelStep(b, action, null);
        cycleActs[i] = sampleAction(prior.actionOut, true, true);
        cycleCool[i] = ACTION_HOLD;
      }
      return clampState(r.state);
    });
    for (let i = 0; i < 3; i++) {
      trails[i].push([balls[i][0], balls[i][1]]);
      if (trails[i].length > TRAIL_LEN) trails[i].shift();
    }
  }

  function tick(now) {
    if (running) {
      if (now - lastStepTime >= STEP_INTERVAL_MS) {
        lastStepTime = now;
        if (mode === 'solo') stepSolo(); else stepCycle();
      }
      draw();
    }
    requestAnimationFrame(tick);
  }

  let wmVisible = true, wmOnscreen = true;
  const wmUpdateRunning = () => { running = wmVisible && wmOnscreen; };
  document.addEventListener('visibilitychange', () => { wmVisible = !document.hidden; wmUpdateRunning(); });
  // several games share one page — sleep whenever scrolled out of view
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((es) => {
      for (const en of es) wmOnscreen = en.isIntersecting;
      wmUpdateRunning();
    }, { threshold: 0.02 }).observe(canvas);
  }

  window.addEventListener('resize', resize);

  fetch('/assets/worldmodel/weights.json')
    .then((r) => r.json())
    .then((w) => {
      model = w;
      if (statusEl) statusEl.textContent = `one net, ${(w.W1.length + w.b1.length + w.W2.length + w.b2.length).toLocaleString()} parameters`;
      resize();
      requestAnimationFrame(tick);
    })
    .catch(() => {
      if (statusEl) statusEl.textContent = 'could not load model weights';
    });
})();
