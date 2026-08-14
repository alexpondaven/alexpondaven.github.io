// The /chase/ game. three.js draws; two tiny MLPs (assets/worldmodel/
// chase_train.js) ARE the physics — and the hunter's mind. The hunter has
// no scripted AI: every 8 model steps it rolls nine candidate futures
// through the same nets that move the world, scores them by how close they
// get to the PREDICTED player, and takes the best. The faint threads on the
// floor are the futures it considered; the bright one is its plan. Hold
// Shift for your own premonition through the same model.
import * as THREE from 'three';

const canvas = document.getElementById('chase-canvas');
const scoreEl = document.getElementById('chase-score');
const bestEl = document.getElementById('chase-best');
const mindEl = document.getElementById('chase-mind');
const statusEl = document.getElementById('chase-status');
const flashEl = document.getElementById('chase-flash');
if (canvas) init();

async function init() {
  let weights;
  try {
    const r = await fetch('/assets/worldmodel/chase_weights.json');
    weights = await r.json();
  } catch (e) {
    if (statusEl) statusEl.textContent = 'could not load model weights';
    return;
  }
  const N_P = weights.meta.playerActions, N_H = weights.meta.hunterActions;
  weights.player.gateT = weights.meta.playerGateT;
  weights.hunter.gateT = weights.meta.hunterGateT;

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
  function stepEntity(net, e, action, nActions) {
    const x = [e.x, e.y, e.vx, e.vy];
    for (let i = 0; i < nActions; i++) x.push(i === action ? 1 : 0);
    const d = forward(net, x);
    if (d[4] < net.gateT) { e.vx = 0; e.vy = 0; } else {
      e.x += d[0]; e.y += d[1]; e.vx += d[2]; e.vy += d[3];
    }
    e.x = Math.max(-1.05, Math.min(1.05, e.x));
    e.y = Math.max(-1.05, Math.min(1.05, e.y));
    e.vx = Math.max(-0.5, Math.min(0.5, e.vx));
    e.vy = Math.max(-0.5, Math.min(0.5, e.vy));
  }

  // --- game state -----------------------------------------------------------
  const N_ORBS = 5;
  let player, hunter, orbs, score = 0, invuln = 0;
  let best = 0;
  try { best = parseInt(localStorage.getItem('chase-best') || '0', 10) || 0; } catch (e) { /* private mode */ }
  if (bestEl) bestEl.textContent = String(best);

  function randPos(margin) {
    return { x: (Math.random() * 2 - 1) * (1 - margin), y: (Math.random() * 2 - 1) * (1 - margin) };
  }
  function respawnOrb(o) {
    for (let t = 0; t < 30; t++) {
      const p = randPos(0.18);
      if (Math.hypot(p.x - player.x, p.y - player.y) > 0.4 && Math.hypot(p.x - hunter.x, p.y - hunter.y) > 0.4) {
        o.x = p.x; o.y = p.y;
        return;
      }
    }
    Object.assign(o, randPos(0.18));
  }
  function resetGame(caught) {
    if (caught) {
      if (score > best) {
        best = score;
        try { localStorage.setItem('chase-best', String(best)); } catch (e) { /* ok */ }
        if (bestEl) bestEl.textContent = String(best);
      }
      score = 0;
      if (scoreEl) scoreEl.textContent = '0';
      if (mindEl) mindEl.textContent = '33%';
    }
    player = { x: -0.6, y: 0.55, vx: 0, vy: 0 };
    hunter = { x: 0.7, y: -0.65, vx: 0, vy: 0 };
    invuln = 40;
    if (!orbs) {
      orbs = [];
      for (let i = 0; i < N_ORBS; i++) orbs.push({ x: 0, y: 0 });
    }
    for (const o of orbs) respawnOrb(o);
  }
  resetGame(false);

  // --- input ----------------------------------------------------------------
  const KEYMAP = { ArrowUp: 1, KeyW: 1, ArrowDown: 2, KeyS: 2, ArrowLeft: 3, KeyA: 3, ArrowRight: 4, KeyD: 4 };
  let currentAction = 0, premo = false;
  const held = new Set();
  const recompute = () => { currentAction = held.size ? [...held][held.size - 1] : 0; };
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') { premo = true; return; }
    const a = KEYMAP[e.code];
    if (a) { held.add(a); recompute(); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') { premo = false; return; }
    const a = KEYMAP[e.code];
    if (a) { held.delete(a); recompute(); }
  });
  document.querySelectorAll('.chase-btn[data-action]').forEach((btn) => {
    const a = parseInt(btn.dataset.action, 10);
    const press = (e) => { e.preventDefault(); held.add(a); recompute(); };
    const release = (e) => { e.preventDefault(); held.delete(a); recompute(); };
    btn.addEventListener('mousedown', press);
    btn.addEventListener('touchstart', press, { passive: false });
    btn.addEventListener('mouseup', release);
    btn.addEventListener('mouseleave', release);
    btn.addEventListener('touchend', release);
  });
  const premoBtn = document.getElementById('chase-premo');
  if (premoBtn) {
    const on = (e) => { e.preventDefault(); premo = true; premoBtn.classList.add('is-on'); };
    const off = () => { premo = false; premoBtn.classList.remove('is-on'); };
    premoBtn.addEventListener('mousedown', on);
    premoBtn.addEventListener('touchstart', on, { passive: false });
    premoBtn.addEventListener('mouseup', off);
    premoBtn.addEventListener('mouseleave', off);
    premoBtn.addEventListener('touchend', off);
  }

  // --- the hunter's imagination: MPC through the world model ------------------
  // Its planning budget IS the difficulty: every orb you take lets it dream
  // deeper and replan faster, from sluggish (horizon 8, replan every 20) to
  // lethal (horizon 16, replan every 8).
  const MAX_HORIZON = 16;
  const ramp = () => Math.min(1, score / 12);
  const horizonNow = () => Math.round(8 + ramp() * 8);
  const replanEvery = () => Math.round(20 - ramp() * 12);
  let plannedAction = 0, planTick = 0;
  const futures = Array.from({ length: N_H }, () => ({ pts: new Array(MAX_HORIZON + 1), score: 0, len: 0 }));
  let chosen = 0;
  function replan() {
    const HOR = horizonNow();
    for (let a = 0; a < N_H; a++) {
      const h = { ...hunter };
      const p = { ...player };
      const f = futures[a];
      f.pts[0] = { x: h.x, y: h.y };
      f.len = HOR;
      let sc = Infinity;
      for (let t = 0; t < HOR; t++) {
        stepEntity(weights.hunter, h, a, N_H);
        stepEntity(weights.player, p, currentAction, N_P);   // its model of YOU
        f.pts[t + 1] = { x: h.x, y: h.y };
        const d = Math.hypot(h.x - p.x, h.y - p.y);
        sc = Math.min(sc, d + t * 0.002);                    // sooner is better
      }
      const dEnd = Math.hypot(h.x - player.x, h.y - player.y);
      f.score = sc + 0.3 * dEnd;
    }
    chosen = 0;
    for (let a = 1; a < N_H; a++) if (futures[a].score < futures[chosen].score) chosen = a;
    plannedAction = chosen;
  }

  // your premonition through the same model
  const PREMO_STEPS = 40;
  const premoPts = new Array(PREMO_STEPS + 1);
  function runPremonition() {
    const p = { ...player };
    premoPts[0] = { x: p.x, y: p.y };
    for (let t = 0; t < PREMO_STEPS; t++) {
      stepEntity(weights.player, p, currentAction, N_P);
      premoPts[t + 1] = { x: p.x, y: p.y };
    }
  }

  // --- model step (20Hz) ------------------------------------------------------
  let flashT = 0;
  function stepGame() {
    if (--planTick <= 0) { replan(); planTick = replanEvery(); }
    stepEntity(weights.player, player, currentAction, N_P);
    stepEntity(weights.hunter, hunter, plannedAction, N_H);
    if (premo) runPremonition();
    if (invuln > 0) invuln--;

    for (const o of orbs) {
      if (Math.hypot(o.x - player.x, o.y - player.y) < 0.12) {
        score++;
        if (scoreEl) scoreEl.textContent = String(score);
        if (mindEl) mindEl.textContent = Math.round(33 + ramp() * 67) + '%';
        respawnOrb(o);
      }
    }
    if (invuln <= 0 && Math.hypot(hunter.x - player.x, hunter.y - player.y) < 0.11) {
      flashT = 20;
      if (flashEl) flashEl.style.opacity = '1';
      resetGame(true);
    }
    if (flashT > 0 && --flashT === 0 && flashEl) flashEl.style.opacity = '0';
  }

  // --- three.js scene ----------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07070f);
  scene.fog = new THREE.Fog(0x07070f, 4.5, 8.5);

  const camera = new THREE.PerspectiveCamera(46, 16 / 11, 0.1, 20);
  camera.position.set(0, 2.5, 2.55);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0x8fa3ff, 0x1a1030, 0.75));
  const sun = new THREE.DirectionalLight(0xdfe8ff, 1.6);
  sun.position.set(2, 3.5, 1.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = -1.8; sun.shadow.camera.right = 1.8;
  sun.shadow.camera.top = 1.8; sun.shadow.camera.bottom = -1.8;
  scene.add(sun);

  // neon grid floor
  const gridCanvas = document.createElement('canvas');
  gridCanvas.width = gridCanvas.height = 512;
  const g = gridCanvas.getContext('2d');
  g.fillStyle = '#0b0c1c'; g.fillRect(0, 0, 512, 512);
  g.strokeStyle = '#1c2350'; g.lineWidth = 2;
  for (let i = 0; i <= 10; i++) {
    const p = (i / 10) * 512;
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, 512); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(512, p); g.stroke();
  }
  const gridTex = new THREE.CanvasTexture(gridCanvas);
  gridTex.colorSpace = THREE.SRGBColorSpace;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(2.4, 2.4),
    new THREE.MeshStandardMaterial({ map: gridTex, roughness: 0.9, metalness: 0.1 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const wallMat = new THREE.MeshStandardMaterial({ color: 0x232a5c, roughness: 0.5, metalness: 0.3, transparent: true, opacity: 0.5 });
  for (const [w, h, x, z] of [[2.4, 0.05, 0, -1.175], [2.4, 0.05, 0, 1.175], [0.05, 2.4, -1.175, 0], [0.05, 2.4, 1.175, 0]]) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(w, 0.12, h), wallMat);
    wall.position.set(x, 0.06, z);
    scene.add(wall);
  }

  const playerMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.075, 40, 24),
    new THREE.MeshStandardMaterial({ color: 0x6366f1, roughness: 0.25, metalness: 0.5, emissive: 0x1e1b7a, emissiveIntensity: 0.6 })
  );
  playerMesh.castShadow = true;
  scene.add(playerMesh);

  const hunterMesh = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.095),
    new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.3, metalness: 0.6, emissive: 0x7f1d1d, emissiveIntensity: 1.2 })
  );
  hunterMesh.castShadow = true;
  scene.add(hunterMesh);
  const hunterLight = new THREE.PointLight(0xef4444, 1.2, 1.6);
  scene.add(hunterLight);

  const orbMeshes = orbs.map(() => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 20, 14),
      new THREE.MeshStandardMaterial({ color: 0x2dd4bf, emissive: 0x0d9488, emissiveIntensity: 1.6, roughness: 0.4 })
    );
    m.castShadow = true;
    scene.add(m);
    return m;
  });

  // the hunter's imagination: one line per candidate future
  const futureLines = futures.map(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((MAX_HORIZON + 1) * 3), 3));
    const mat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.14 });
    const line = new THREE.Line(geo, mat);
    scene.add(line);
    return line;
  });
  // your premonition
  const premoGeo = new THREE.BufferGeometry();
  premoGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((PREMO_STEPS + 1) * 3), 3));
  const premoLine = new THREE.Line(premoGeo, new THREE.LineBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.8 }));
  premoLine.visible = false;
  scene.add(premoLine);
  const ghost = new THREE.Mesh(
    new THREE.SphereGeometry(0.075, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.35 })
  );
  ghost.visible = false;
  scene.add(ghost);

  function updateLines() {
    for (let a = 0; a < futures.length; a++) {
      const line = futureLines[a];
      const pos = line.geometry.attributes.position.array;
      const f = futures[a];
      if (!f.pts[0] || !f.len) { line.visible = false; continue; }
      line.visible = true;
      for (let t = 0; t <= MAX_HORIZON; t++) {
        const p = f.pts[Math.min(t, f.len)];
        pos[t * 3] = p.x;
        pos[t * 3 + 1] = 0.02;
        pos[t * 3 + 2] = p.y;
      }
      line.geometry.attributes.position.needsUpdate = true;
      line.material.opacity = a === chosen ? 0.9 : 0.13;
      line.material.color.setHex(a === chosen ? 0xff8f8f : 0xef4444);
    }
    premoLine.visible = premo;
    ghost.visible = premo;
    if (premo && premoPts[0]) {
      const pos = premoGeo.attributes.position.array;
      for (let t = 0; t <= PREMO_STEPS; t++) {
        pos[t * 3] = premoPts[t].x;
        pos[t * 3 + 1] = 0.02;
        pos[t * 3 + 2] = premoPts[t].y;
      }
      premoGeo.attributes.position.needsUpdate = true;
      const last = premoPts[PREMO_STEPS];
      ghost.position.set(last.x, 0.075, last.y);
    }
  }

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
    statusEl.textContent = `loaded — player net ${p(weights.player).toLocaleString()} params, hunter net ${p(weights.hunter).toLocaleString()} params · no physics engine, no scripted AI: the hunter searches these nets`;
  }

  // --- loop --------------------------------------------------------------------
  const STEP_MS = 50;
  let last = 0, running = true;
  document.addEventListener('visibilitychange', () => { running = !document.hidden; });
  function frame(now) {
    if (running) {
      if (now - last >= STEP_MS) { last = now; stepGame(); }
      playerMesh.position.set(player.x, 0.075, player.y);
      playerMesh.rotation.z -= player.vx * 2.2;
      playerMesh.rotation.x += player.vy * 2.2;
      hunterMesh.position.set(hunter.x, 0.1 + Math.sin(now / 300) * 0.015, hunter.y);
      hunterMesh.rotation.y += 0.02;
      hunterLight.position.set(hunter.x, 0.35, hunter.y);
      orbs.forEach((o, i) => {
        orbMeshes[i].position.set(o.x, 0.06 + Math.sin(now / 400 + i * 1.7) * 0.02, o.y);
        orbMeshes[i].material.emissiveIntensity = 1.3 + Math.sin(now / 250 + i) * 0.5;
      });
      updateLines();
      camera.position.x += (player.x * 0.25 - camera.position.x) * 0.03;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
