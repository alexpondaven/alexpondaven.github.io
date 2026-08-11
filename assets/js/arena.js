// The /arena/ mini-game. three.js does the DRAWING only — every movement
// (ball, crates, collisions, wall bounces) is the output of two tiny MLPs
// trained offline (see /assets/worldmodel/arena_train.js). There is no
// physics engine in this file, just forward passes and rendering.
import * as THREE from 'three';

const canvas = document.getElementById('arena-canvas');
const scoreEl = document.getElementById('arena-score');
const statusEl = document.getElementById('arena-status');
if (canvas) init();

async function init() {
  let weights, policy = null;
  try {
    const r = await fetch('/assets/worldmodel/arena_weights.json');
    weights = await r.json();
  } catch (e) {
    if (statusEl) statusEl.textContent = 'could not load model weights';
    return;
  }
  try {
    const r = await fetch('/assets/worldmodel/arena_policy.json');
    if (r.ok) policy = await r.json();
  } catch (e) { /* autopilot simply unavailable */ }

  // --- model runtime -------------------------------------------------------
  const N_ACTIONS = weights.meta.actions;
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

  // --- game state ----------------------------------------------------------
  const N_CRATES = 3;
  let ball, crates, score = 0;
  let zone = { x: 0.55, y: -0.55, r: 0.22 };
  function randPos(margin) {
    return { x: (Math.random() * 2 - 1) * (1 - margin), y: (Math.random() * 2 - 1) * (1 - margin), vx: 0, vy: 0 };
  }
  function respawnCrate(c) {
    // away from the zone and the ball
    for (let tries = 0; tries < 20; tries++) {
      const p = randPos(0.25);
      if (Math.hypot(p.x - zone.x, p.y - zone.y) > 0.45 && Math.hypot(p.x - ball.x, p.y - ball.y) > 0.3) {
        c.x = p.x; c.y = p.y; c.vx = 0; c.vy = 0;
        return;
      }
    }
    c.x = 0; c.y = 0; c.vx = 0; c.vy = 0;
  }
  function resetGame() {
    ball = { x: -0.6, y: 0.6, vx: 0, vy: 0 };
    crates = [{ x: -0.1, y: -0.1, vx: 0, vy: 0 }, { x: 0.35, y: 0.3, vx: 0, vy: 0 }, { x: -0.45, y: -0.35, vx: 0, vy: 0 }];
    score = 0;
    if (scoreEl) scoreEl.textContent = '0';
  }
  resetGame();

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
  // distance + closing speed of b toward a, precomputed as inputs (a tiny
  // one-layer net can't build this product of relative direction and
  // relative velocity on its own — the motion gate reads it off linearly)
  function proximity(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.hypot(dx, dy);
    const closing = d > 1e-6 ? -(dx * (b.vx - a.vx) + dy * (b.vy - a.vy)) / d : 0;
    return [d, closing];
  }
  // 5th output is the net's own motion gate: below threshold means "this
  // entity is at rest" — hold it still instead of integrating the tiny
  // residual deltas (which would otherwise compound into drift)
  const GATE_THRESHOLD = weights.meta.gateThreshold ?? -Infinity;
  function applyGated(e, d) {
    if (d[4] < GATE_THRESHOLD) { e.vx = 0; e.vy = 0; return; }
    e.x += d[0]; e.y += d[1]; e.vx += d[2]; e.vy += d[3];
  }
  function clampEntity(e) {
    // pure safety net — the nets learned the walls; this only catches a
    // runaway before it could leave the arena for good
    e.x = Math.max(-1.05, Math.min(1.05, e.x));
    e.y = Math.max(-1.05, Math.min(1.05, e.y));
    e.vx = Math.max(-0.5, Math.min(0.5, e.vx));
    e.vy = Math.max(-0.5, Math.min(0.5, e.vy));
  }

  // --- input ---------------------------------------------------------------
  const KEYMAP = { ArrowUp: 1, KeyW: 1, ArrowDown: 2, KeyS: 2, ArrowLeft: 3, KeyA: 3, ArrowRight: 4, KeyD: 4 };
  let currentAction = 0;
  const held = new Set();
  const recompute = () => { currentAction = held.size ? [...held][held.size - 1] : 0; };
  window.addEventListener('keydown', (e) => {
    const a = KEYMAP[e.code];
    if (a) { held.add(a); recompute(); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => {
    const a = KEYMAP[e.code];
    if (a) { held.delete(a); recompute(); }
  });
  document.querySelectorAll('.arena-btn[data-action]').forEach((btn) => {
    const a = parseInt(btn.dataset.action, 10);
    const press = (e) => { e.preventDefault(); held.add(a); recompute(); };
    const release = (e) => { e.preventDefault(); held.delete(a); recompute(); };
    btn.addEventListener('mousedown', press);
    btn.addEventListener('touchstart', press, { passive: false });
    btn.addEventListener('mouseup', release);
    btn.addEventListener('mouseleave', release);
    btn.addEventListener('touchend', release);
  });
  const resetBtn = document.getElementById('arena-reset');
  if (resetBtn) resetBtn.addEventListener('click', resetGame);

  // --- autopilot: a ~460-param policy net trained by evolution ENTIRELY
  // inside this world model (it has never seen the real simulator).
  // Feature layout must match arena_agent_train.js exactly.
  const cornered = (c) => Math.abs(c.x) > 0.8 && Math.abs(c.y) > 0.8;
  function pickTarget() {
    let t = null, bd = Infinity;
    for (const c of crates) {
      if (cornered(c)) continue;
      const d = Math.hypot(c.x - zone.x, c.y - zone.y);
      if (d < bd) { bd = d; t = c; }
    }
    return t || crates[0];
  }
  function policyAction() {
    const crate = pickTarget();
    const dzx = zone.x - crate.x, dzy = zone.y - crate.y;
    const dz = Math.hypot(dzx, dzy) || 1;
    const ux = dzx / dz, uy = dzy / dz;
    const bx = crate.x - ball.x, by = crate.y - ball.y;
    const db = Math.hypot(bx, by) || 1;
    const feat = [
      ball.x, ball.y, ball.vx, ball.vy,
      bx, by, db,
      ux, uy, dz,
      crate.x - ux * 0.22 - ball.x, crate.y - uy * 0.22 - ball.y,
      (bx / db) * ux + (by / db) * uy,
    ];
    const { feat: F, hidden: H, actions: A } = policy.meta;
    const th = policy.theta;
    const h = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let acc = th[F * H + j];
      for (let i = 0; i < F; i++) acc += feat[i] * th[i * H + j];
      h[j] = acc > 0 ? acc : 0;
    }
    const p = F * H + H;
    let best = 0, bestV = -Infinity;
    for (let k = 0; k < A; k++) {
      let acc = th[p + H * A + k];
      for (let j = 0; j < H; j++) acc += h[j] * th[p + j * A + k];
      if (acc > bestV) { bestV = acc; best = k; }
    }
    return best;
  }
  let autopilot = false;
  const autoBtn = document.getElementById('arena-auto');
  if (autoBtn && policy) {
    autoBtn.addEventListener('click', () => {
      autopilot = !autopilot;
      autoBtn.textContent = `Autopilot: ${autopilot ? 'on' : 'off'}`;
      autoBtn.classList.toggle('is-on', autopilot);
    });
  } else if (autoBtn) {
    autoBtn.disabled = true;
  }

  // --- model step (20Hz) ---------------------------------------------------
  let zoneFlash = 0;
  const cornerTicks = [0, 0, 0];
  function stepGame() {
    // player input always wins; the agent only drives idle moments
    const action = (autopilot && held.size === 0 && policy) ? policyAction() : currentAction;
    const onehot = new Array(N_ACTIONS).fill(0);
    onehot[action] = 1;
    const nb = nearest(ball, crates, -1);
    const bd = forward(weights.ball, [ball.x, ball.y, ball.vx, ball.vy, ...onehot, ...rel(ball, nb), ...proximity(ball, nb)]);
    const cds = crates.map((c, i) => {
      const other = nearest(c, crates, i);
      return forward(weights.crate, [c.x, c.y, c.vx, c.vy, ...rel(c, ball), ...rel(c, other), ...proximity(c, ball)]);
    });
    applyGated(ball, bd);
    clampEntity(ball);
    crates.forEach((c, i) => {
      applyGated(c, cds[i]);
      clampEntity(c);
    });

    crates.forEach((c, i) => {
      // corner pockets are absorbing states (hard to push out of even with
      // real physics) — a crate camping one for 6s respawns into open field
      cornerTicks[i] = Math.abs(c.x) > 0.8 && Math.abs(c.y) > 0.8 ? cornerTicks[i] + 1 : 0;
      if (cornerTicks[i] > 120) { respawnCrate(c); cornerTicks[i] = 0; }
    });

    for (const c of crates) {
      if (Math.hypot(c.x - zone.x, c.y - zone.y) < zone.r) {
        score += 1;
        if (scoreEl) scoreEl.textContent = String(score);
        zoneFlash = 1;
        respawnCrate(c);
      }
    }
  }

  // --- three.js scene ------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1020);
  scene.fog = new THREE.Fog(0x0e1020, 4.5, 8);

  const camera = new THREE.PerspectiveCamera(46, 16 / 11, 0.1, 20);
  camera.position.set(0, 2.45, 2.6);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xbfd0ff, 0x30231a, 1.0));
  const sun = new THREE.DirectionalLight(0xfff2df, 2.8);
  sun.position.set(2.2, 3.5, 1.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = -1.8; sun.shadow.camera.right = 1.8;
  sun.shadow.camera.top = 1.8; sun.shadow.camera.bottom = -1.8;
  scene.add(sun);

  // floor with a subtle generated grid texture
  const gridCanvas = document.createElement('canvas');
  gridCanvas.width = gridCanvas.height = 512;
  const g = gridCanvas.getContext('2d');
  g.fillStyle = '#232742'; g.fillRect(0, 0, 512, 512);
  g.strokeStyle = '#2f3560'; g.lineWidth = 2;
  for (let i = 0; i <= 8; i++) {
    const p = (i / 8) * 512;
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, 512); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(512, p); g.stroke();
  }
  const gridTex = new THREE.CanvasTexture(gridCanvas);
  gridTex.colorSpace = THREE.SRGBColorSpace;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(2.4, 2.4),
    new THREE.MeshStandardMaterial({ map: gridTex, roughness: 0.85, metalness: 0.05 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // arena walls
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x3a4170, roughness: 0.5, metalness: 0.2, transparent: true, opacity: 0.55 });
  const wallGeoH = new THREE.BoxGeometry(2.4, 0.14, 0.05);
  const wallGeoV = new THREE.BoxGeometry(0.05, 0.14, 2.4);
  for (const [geo, x, z] of [[wallGeoH, 0, -1.175], [wallGeoH, 0, 1.175], [wallGeoV, -1.175, 0], [wallGeoV, 1.175, 0]]) {
    const w = new THREE.Mesh(geo, wallMat);
    w.position.set(x, 0.07, z);
    w.castShadow = true;
    scene.add(w);
  }

  // player ball
  const ballMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.085, 40, 24),
    new THREE.MeshStandardMaterial({ color: 0x6366f1, roughness: 0.22, metalness: 0.55 })
  );
  ballMesh.castShadow = true;
  scene.add(ballMesh);

  // crates
  const crateMat = new THREE.MeshStandardMaterial({ color: 0xb4622a, roughness: 0.75, metalness: 0.08 });
  const crateEdge = new THREE.MeshStandardMaterial({ color: 0x7c3f16, roughness: 0.8 });
  const crateMeshes = crates.map(() => {
    const grp = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.13, 0.17), crateMat);
    box.castShadow = true;
    grp.add(box);
    const top = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.02, 0.18), crateEdge);
    top.position.y = 0.065;
    grp.add(top);
    scene.add(grp);
    return grp;
  });

  // target zone ring
  const zoneMesh = new THREE.Mesh(
    new THREE.RingGeometry(0.16, 0.22, 48),
    new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.55, side: THREE.DoubleSide })
  );
  zoneMesh.rotation.x = -Math.PI / 2;
  zoneMesh.position.set(zone.x, 0.005, zone.y);
  scene.add(zoneMesh);

  // world mapping: model (x, y) -> three.js (x, z), plane y up
  function place(mesh, e, h) { mesh.position.set(e.x, h, e.y); }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setSize(rect.width * dpr, rect.height * dpr, false);
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  if (statusEl) {
    const p = (n) => n.W1.length + n.b1.length + n.W2.length + n.b2.length;
    statusEl.textContent = `loaded — ball net ${p(weights.ball).toLocaleString()} params, crate net ${p(weights.crate).toLocaleString()} params, no physics engine`;
  }

  // --- loop: model at 20Hz, render at rAF ---------------------------------
  const STEP_MS = 50;
  let last = 0, running = true;
  document.addEventListener('visibilitychange', () => { running = !document.hidden; });

  function frame(now) {
    if (running) {
      if (now - last >= STEP_MS) {
        last = now;
        stepGame();
      }
      place(ballMesh, ball, 0.085);
      ballMesh.rotation.z -= ball.vx * 2.2;
      ballMesh.rotation.x += ball.vy * 2.2;
      crates.forEach((c, i) => place(crateMeshes[i], c, 0.065));
      zoneFlash = Math.max(0, zoneFlash - 0.04);
      zoneMesh.material.opacity = 0.55 + zoneFlash * 0.45;
      const s = 1 + zoneFlash * 0.25;
      zoneMesh.scale.set(s, s, 1);
      // gentle camera parallax toward the ball
      camera.position.x += (ball.x * 0.25 - camera.position.x) * 0.03;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
