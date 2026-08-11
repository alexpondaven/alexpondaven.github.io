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

  // anchor: null for "no correction", or a real [x,y,vx,vy] state the model
  // was trained to trust when paired with confidence=1. See train.js's
  // buildGroundingDataset for how that trust was taught.
  const NO_ANCHOR = [0, 0, 0, 0, 0];
  function modelStep(state, action, anchor) {
    const onehot = new Array(N_ACTIONS).fill(0);
    onehot[action] = 1;
    const anchorInput = anchor ? [...anchor, 1] : NO_ANCHOR;
    const delta = forward([...state, ...onehot, ...anchorInput]);
    return [state[0] + delta[0], state[1] + delta[1], state[2] + delta[2], state[3] + delta[3]];
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
  let grounding = true; // periodic correction pulses, on by default
  let stepsSinceGround = 0;
  let flashUntil = 0; // ms timestamp; draw() shows a brief ring while flashing

  // Every GROUND_PERIOD model steps, feed the real physics state back in as
  // a trusted "anchor" — the network was explicitly trained to snap toward
  // it (see train.js). Between pulses it's a normal, uncorrected rollout, so
  // real drift still happens; this just stops it from running away forever.
  const GROUND_PERIOD = 15;

  const toggleBtn = document.getElementById('wm-toggle-truth');
  const groundBtn = document.getElementById('wm-toggle-ground');
  const resetBtn = document.getElementById('wm-reset');
  const driftEl = document.getElementById('wm-drift');
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
      grounding = !grounding;
      groundBtn.textContent = grounding ? 'Grounding: on' : 'Grounding: off';
      groundBtn.setAttribute('aria-pressed', String(grounding));
    });
  }
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      modelState = [0, 0, 0, 0];
      truthState = [0, 0, 0, 0];
      stepsSinceGround = 0;
    });
  }

  // --- render ----------------------------------------------------------------
  function accent() {
    return getComputedStyle(document.documentElement).getPropertyValue('--global-theme-color').trim() || '#4f46e5';
  }
  function toPixel(nx, ny, w, h) {
    return [((nx + 1) / 2) * w, ((ny + 1) / 2) * h];
  }

  function draw() {
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

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
    if (performance.now() < flashUntil) {
      ctx.strokeStyle = accent();
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(mx, my, 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = accent();
    ctx.beginPath();
    ctx.arc(mx, my, 10, 0, Math.PI * 2);
    ctx.fill();

    if (driftEl) {
      const dx = truthState[0] - modelState[0], dy = truthState[1] - modelState[1];
      const drift = Math.sqrt(dx * dx + dy * dy);
      driftEl.textContent = drift.toFixed(3);
    }
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

  function tick(now) {
    if (running) {
      if (now - lastStepTime >= STEP_INTERVAL_MS) {
        lastStepTime = now;

        // Anchor must be the true state at the SAME timestep as modelState
        // (before this step's transition) — that's what the network was
        // trained on. Capture it before advancing truthState.
        stepsSinceGround++;
        const pulse = grounding && stepsSinceGround >= GROUND_PERIOD;
        const anchor = pulse ? truthState : null;
        if (pulse) {
          stepsSinceGround = 0;
          flashUntil = now + 220;
        }

        truthState = physicsStep(truthState, currentAction);
        modelState = modelStep(modelState, currentAction, anchor);
      }
      draw();
    }
    requestAnimationFrame(tick);
  }

  document.addEventListener('visibilitychange', () => { running = !document.hidden; });

  window.addEventListener('resize', resize);

  fetch('/assets/worldmodel/weights.json')
    .then((r) => r.json())
    .then((w) => {
      model = w;
      if (statusEl) statusEl.textContent = `loaded — ${w.meta.hidden}-unit hidden layer, ~${(w.W1.length + w.b1.length + w.W2.length + w.b2.length).toLocaleString()} parameters`;
      resize();
      requestAnimationFrame(tick);
    })
    .catch(() => {
      if (statusEl) statusEl.textContent = 'could not load model weights';
    });
})();
