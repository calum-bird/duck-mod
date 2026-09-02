// Beat-synced dance support: analyze an uploaded track, play it through
// WebAudio, and expose the beat clock the dance policy is commanded with.
//
// The analysis is a direct port of duck-mod/scripts/beat_track.py:
// half-wave-rectified spectral-flux onset envelope (512-sample frames,
// 256 hop), autocorrelation over the BPM window for tempo, then the onset
// energy folded at the beat period for the first-downbeat offset.
//
// The clock is derived from AudioContext.currentTime, not an accumulator:
// audio hardware time IS the truth the ear judges sync against, so phase
// drawn from it can never drift from what's playing.

// Matches microduck_dance.beat_clock: tempo rides twist[2] normalized
// over the full training envelope [60, 160].
export const BPM_MIN = 60.0;
export const BPM_MAX = 160.0;
// Where the policy actually practised (eval: 0% falls). Outside it the
// tempo gets folded by powers of two, or driven raw with a warning.
export const TRAINED_BAND = [100.0, 140.0];
// The shipped demo amplitudes (bob m, sway m, yaw rad) — 0% falls in 20 s.
export const DANCE_BOB = 0.012;
export const DANCE_SWAY = 0.006;
export const DANCE_YAW = 0.05;

const WIN = 512;
const HOP = 256;

export const tempoToNorm = (bpm) =>
  Math.max(-1, Math.min(1, (2 * (bpm - BPM_MIN)) / (BPM_MAX - BPM_MIN) - 1));

// ── Small in-place radix-2 FFT (real input, magnitude out) ─────────────
function makeFft(n) {
  const levels = Math.log2(n) | 0;
  const cos = new Float32Array(n / 2);
  const sin = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < levels; b++) r = (r << 1) | ((i >>> b) & 1);
    rev[i] = r;
  }
  return (re, im) => {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tre = re[l] * cos[k] + im[l] * sin[k];
          const tim = im[l] * cos[k] - re[l] * sin[k];
          re[l] = re[j] - tre; im[l] = im[j] - tim;
          re[j] += tre; im[j] += tim;
        }
      }
    }
  };
}

// ── beat_track.py port ─────────────────────────────────────────────────
export function analyzeBeat(audioBuffer, bpmMin = 60, bpmMax = 170) {
  // Mono mixdown.
  const n0 = audioBuffer.length;
  const y = new Float32Array(n0);
  for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
    const ch = audioBuffer.getChannelData(c);
    for (let i = 0; i < n0; i++) y[i] += ch[i];
  }
  const inv = 1 / audioBuffer.numberOfChannels;
  for (let i = 0; i < n0; i++) y[i] *= inv;

  // Spectral-flux onset envelope at sr/HOP Hz.
  const sr = audioBuffer.sampleRate;
  const fps = sr / HOP;
  const nFrames = Math.floor((n0 - WIN) / HOP);
  if (nFrames < fps * 4) throw new Error("track too short to beat-track");
  const fft = makeFft(WIN);
  const hann = new Float32Array(WIN);
  for (let i = 0; i < WIN; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / WIN);
  const re = new Float32Array(WIN);
  const im = new Float32Array(WIN);
  const prevMag = new Float32Array(WIN / 2 + 1);
  const mag = new Float32Array(WIN / 2 + 1);
  const env = new Float32Array(Math.max(nFrames - 1, 1));
  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    for (let i = 0; i < WIN; i++) { re[i] = y[off + i] * hann[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= WIN / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
    if (f > 0) {
      let flux = 0;
      for (let k = 0; k <= WIN / 2; k++) {
        const d = mag[k] - prevMag[k];
        if (d > 0) flux += d;
      }
      env[f - 1] = flux;
    }
    prevMag.set(mag);
  }
  let mean = 0;
  for (let i = 0; i < env.length; i++) mean += env[i];
  mean /= env.length;
  for (let i = 0; i < env.length; i++) env[i] -= mean;

  // Autocorrelation over the BPM window (direct sums: the lag range is
  // narrow, ~200 lags at 86 fps, so O(n·lags) beats an FFT correlate).
  const lagMin = Math.max(1, Math.floor((60 * fps) / bpmMax));
  const lagMax = Math.min(env.length - 1, Math.ceil((60 * fps) / bpmMin));
  let bestLag = lagMin, bestVal = -Infinity, sumAbs = 0, count = 0;
  for (let lag = lagMin; lag <= lagMax; lag++) {
    let s = 0;
    for (let i = lag; i < env.length; i++) s += env[i] * env[i - lag];
    if (s > bestVal) { bestVal = s; bestLag = lag; }
    sumAbs += Math.abs(s);
    count++;
  }
  const bpm = (60 * fps) / bestLag;
  const steadiness = bestVal / (sumAbs / count + 1e-9);

  // Phase: fold onset energy at the beat period, strongest bin = beat 1.
  const period = (60 * fps) / bpm;
  const BINS = 32;
  const phaseEnergy = new Float64Array(BINS);
  for (let i = 0; i < env.length; i++) {
    const b = Math.floor(((i % period) / period) * BINS) % BINS;
    if (env[i] > 0) phaseEnergy[b] += env[i];
  }
  let bestBin = 0;
  for (let b = 1; b < BINS; b++) if (phaseEnergy[b] > phaseEnergy[bestBin]) bestBin = b;
  const firstBeatS = (((bestBin + 0.5) / BINS) * period) / fps;

  return { bpm, steadiness, firstBeatS };
}

// Fold an out-of-band tempo into the trained band by powers of two — the
// only ratios that keep the duck on the song's grid.
export function chooseDriveBpm(songBpm, band = TRAINED_BAND) {
  const [lo, hi] = band;
  if (songBpm >= lo && songBpm <= hi) return { bpm: songBpm, note: null };
  const words = { 1: "double-time", "-1": "half-time", 2: "quadruple-time", "-2": "quarter-time" };
  for (const k of [1, -1, 2, -2]) {
    const cand = songBpm * 2 ** k;
    if (cand >= lo && cand <= hi) {
      return { bpm: cand, note: `song is ${songBpm.toFixed(1)} BPM — dancing ${words[k]} at ${cand.toFixed(1)}` };
    }
  }
  return {
    bpm: songBpm,
    note: `song is ${songBpm.toFixed(1)} BPM, outside the trained ${lo}–${hi} band — expect sloppier footwork`,
  };
}

// ── Player: decode, analyze, play, and BE the beat clock ───────────────
export function createDancePlayer() {
  let ctx = null;
  let source = null;
  let current = null; // { bpm, driveBpm, note, steadiness, firstBeatS, t0, title, duration }
  let onEnded = null;

  async function load(file) {
    ctx ??= new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") await ctx.resume();
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const { bpm, steadiness, firstBeatS } = analyzeBeat(buf);
    const { bpm: driveBpm, note } = chooseDriveBpm(bpm);
    return { buf, bpm, driveBpm, note, steadiness, firstBeatS, title: file.name };
  }

  function play(prep) {
    stop();
    source = ctx.createBufferSource();
    source.buffer = prep.buf;
    source.connect(ctx.destination);
    const t0 = ctx.currentTime + 0.1; // tiny scheduling margin
    source.start(t0);
    current = { ...prep, t0, duration: prep.buf.duration };
    source.onended = () => {
      if (source) { current = null; source = null; onEnded?.(); }
    };
  }

  function stop() {
    if (source) {
      const s = source;
      source = null; // silence the onended callback path
      current = null;
      try { s.onended = null; s.stop(); } catch { /* already stopped */ }
    }
  }

  // Bar phase in [0,1): one bar = two beats, zero on the track's first
  // downbeat — exactly the clock the policy was trained against. Before
  // the downbeat (lead-in, scheduling margin) it counts up from negative
  // time, which the modulo folds; the duck simply starts dancing early.
  function barPhase() {
    if (!current) return 0;
    const t = ctx.currentTime - current.t0 - current.firstBeatS;
    const bars = (t * current.driveBpm) / 120;
    return ((bars % 1) + 1) % 1;
  }

  return {
    load, play, stop, barPhase,
    tempoNorm: () => (current ? tempoToNorm(current.driveBpm) : 0),
    get playing() { return current !== null; },
    get info() { return current; },
    set onEnded(fn) { onEnded = fn; },
  };
}
