// Backup dancers: kinematic clones of the render rig that replay the real
// duck's policy-generated motion at whole-bar delays. No extra physics, no
// extra inference — the crowd's motion is the same trained policy, echoed.
//
// Each dancer stands on a fixed spot in formation and replays the lead's
// pose from k bars ago, RE-BASED: the lead's slow positional/heading drift
// (an honest limitation of the policy — station keeping wanders ~30 cm/20 s)
// is subtracted with a slow EMA, so the gestures (bob, sway, steps, head
// swings) come through while every dancer holds its spot. Whole-bar delays
// keep everyone on the beat together while the gestures ripple through the
// crew, and each dancer can only appear once the buffer holds its delay —
// so the crowd joins in naturally, one bar at a time.

import * as THREE from "three";

const MAX_DANCERS = 12;
const BUF_SECONDS = 30; // longest delay is 12 bars @100 BPM = 14.4 s
const EMA_ALPHA = 0.006; // per 50 Hz step: ~3.3 s time constant

// Formation: staggered rows behind the spawn area (MJCF coords, z-up),
// facing the same way the lead spawns (yaw 0 toward +x, camera default).
const SPOTS = [];
for (let row = 0; row < 3; row++) {
  for (let col = 0; col < 4; col++) {
    SPOTS.push({
      x: -0.55 - row * 0.42,
      y: (col - 1.5) * 0.45 + (row % 2 ? 0.12 : -0.12),
      yaw: 0,
    });
  }
}

export function createCrowd({
  scene, getRig, cloneRig, setJoint, setJawOpen, applyVariant,
  jointNames, variantNames, ctrlDt,
}) {
  const dancers = []; // { rig, trunk, spot, variant, visible }
  let want = 0;
  let active = false;

  // Snapshot ring: { t, x, y, z, q:[w,x,y,z], j:Float32Array, w }
  const buf = [];
  let simT = 0;
  // Lead's drift EMA: position + heading (heading as a vector so the wrap
  // at ±π never kicks the average).
  let ex = null, ey = null, ecos = 1, esin = 0;

  const _q = new THREE.Quaternion();
  const _qYaw = new THREE.Quaternion();
  const _qa = new THREE.Quaternion();
  const _qb = new THREE.Quaternion();
  const Z = new THREE.Vector3(0, 0, 1);

  function ensureDancers() {
    const base = getRig();
    while (dancers.length < Math.min(want, MAX_DANCERS)) {
      const i = dancers.length;
      const rig = cloneRig(base);
      // Cycle the colourways so the crew reads as a crew, not copies.
      const variant = variantNames[(i + 1) % variantNames.length];
      try { applyVariant(rig, variant); } catch { /* keep base look */ }
      rig.placer.visible = false;
      scene.add(rig.placer);
      dancers.push({
        rig, variant,
        trunk: rig.bodies.get("trunk_base"),
        spot: SPOTS[i % SPOTS.length],
        visible: false,
      });
    }
    for (let i = 0; i < dancers.length; i++) {
      if (i >= want) { dancers[i].rig.placer.visible = false; dancers[i].visible = false; }
    }
  }

  // Called once per control step (50 Hz) with the lead's raw MJCF state.
  function record(qpos, qposAdr, jaw) {
    if (!active) return;
    simT += ctrlDt;
    const yaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    if (ex === null) { ex = qpos[0]; ey = qpos[1]; ecos = Math.cos(yaw); esin = Math.sin(yaw); }
    ex += EMA_ALPHA * (qpos[0] - ex);
    ey += EMA_ALPHA * (qpos[1] - ey);
    ecos += EMA_ALPHA * (Math.cos(yaw) - ecos);
    esin += EMA_ALPHA * (Math.sin(yaw) - esin);
    const eyaw = Math.atan2(esin, ecos);
    // Store the pose already re-based into the lead's drift frame: position
    // relative to the EMA base, orientation with the EMA heading removed.
    const dx = qpos[0] - ex, dy = qpos[1] - ey;
    const c = Math.cos(-eyaw), s = Math.sin(-eyaw);
    _q.set(qpos[4], qpos[5], qpos[6], qpos[3]); // THREE order
    _qYaw.setFromAxisAngle(Z, -eyaw);
    _q.premultiply(_qYaw);
    const j = new Float32Array(jointNames.length);
    for (let i = 0; i < jointNames.length; i++) j[i] = qpos[qposAdr[i]];
    buf.push({
      t: simT,
      x: c * dx - s * dy, y: s * dx + c * dy, z: qpos[2],
      q: [_q.w, _q.x, _q.y, _q.z],
      j, w: jaw,
    });
    while (buf.length > 2 && buf[0].t < simT - BUF_SECONDS) buf.shift();
  }

  // Called every render frame. barPeriod in seconds (120 / driveBpm).
  function update(barPeriod) {
    if (!active || !buf.length || !barPeriod) return;
    ensureDancers();
    for (let i = 0; i < Math.min(want, dancers.length); i++) {
      const d = dancers[i];
      const delay = (i + 1) * barPeriod;
      const tWant = simT - delay;
      if (tWant < buf[0].t) {
        // Not enough history yet — this dancer hasn't joined in.
        if (d.visible) { d.rig.placer.visible = false; d.visible = false; }
        continue;
      }
      // Binary search the bracketing pair.
      let lo = 0, hi = buf.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (buf[mid].t <= tWant) lo = mid; else hi = mid;
      }
      const a = buf[lo], b = buf[hi];
      const span = b.t - a.t;
      const u = span > 0 ? Math.min(1, Math.max(0, (tWant - a.t) / span)) : 1;

      const spot = d.spot;
      const c = Math.cos(spot.yaw), s = Math.sin(spot.yaw);
      const rx = a.x + (b.x - a.x) * u;
      const ry = a.y + (b.y - a.y) * u;
      d.trunk.position.set(
        spot.x + c * rx - s * ry,
        spot.y + s * rx + c * ry,
        a.z + (b.z - a.z) * u,
      );
      _qa.set(a.q[1], a.q[2], a.q[3], a.q[0]);
      _qb.set(b.q[1], b.q[2], b.q[3], b.q[0]);
      _qa.slerp(_qb, u);
      _qYaw.setFromAxisAngle(Z, spot.yaw);
      d.trunk.quaternion.copy(_qYaw.multiply(_qa));
      for (let k = 0; k < jointNames.length; k++) {
        setJoint(d.rig, jointNames[k], a.j[k] + (b.j[k] - a.j[k]) * u);
      }
      setJawOpen(d.rig, a.w + (b.w - a.w) * u);
      if (!d.visible) { d.rig.placer.visible = true; d.visible = true; }
    }
  }

  function setActive(on) {
    if (active === on) return;
    active = on;
    if (!on) {
      buf.length = 0;
      ex = null;
      for (const d of dancers) { d.rig.placer.visible = false; d.visible = false; }
    }
  }

  function setCount(n) {
    want = Math.max(0, Math.min(MAX_DANCERS, n | 0));
    if (active) ensureDancers();
  }

  return {
    record, update, setActive, setCount,
    get count() { return want; },
    get visible() { return dancers.reduce((n, d) => n + (d.visible ? 1 : 0), 0); },
    get debug() {
      return { active, simT, buf: buf.length, dancers: dancers.length,
        oldest: buf.length ? buf[0].t : null };
    },
  };
}
