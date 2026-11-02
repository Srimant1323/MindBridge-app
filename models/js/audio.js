/**
 * audio.js — Vocal Pattern Analysis Module
 * Extracts F0 (autocorrelation), energy, frame-level jitter and shimmer, speech rate and pause ratio
 * using the WebAudio API. Experimental research features; not clinically validated.
 * Also uses the Web Speech API for speech-to-text transcription.
 */
const audio = (function () {

  // ── State ────────────────────────────────────────────────────────────────
  let isRecording    = false;
  let audioContext   = null;
  let analyserNode   = null;
  let mediaStream    = null;
  let animFrameId    = null;
  let recognition    = null;

  // Collected frame data
  let pitchHistory   = [];   // F0 (Hz) per voiced frame, in time order
  let energyHistory  = [];   // RMS per voiced frame
  let ampHistory     = [];   // peak amplitude per voiced frame
  let periodHistory  = [];   // pitch period (s) per voiced frame; null marks an unvoiced gap
  let totalFrames    = 0;
  let silentFrames   = 0;
  let envelope       = [];   // RMS of every frame (for speech-rate / syllable counting)
  let featureTimer   = null;
  let startedAt      = 0;
  const FRAME_MS     = 50;   // analysis hop (window ~43 ms at 48 kHz => almost no overlap)
  let transcript     = '';
  let finalMetrics   = null;

  // ── UI refs ───────────────────────────────────────────────────────────────
  const micBtnEl      = () => document.getElementById('mic-btn');
  const visualizerEl  = () => document.getElementById('voice-visualizer');
  const statusEl      = () => document.getElementById('voice-status');
  const metricsEl     = () => document.getElementById('voice-metrics-row');
  const transcriptBox = () => document.getElementById('voice-transcript-box');
  const transcriptEl  = () => document.getElementById('voice-transcript-text');
  const canvasEl      = () => document.getElementById('waveform-canvas');
  const pitchEl       = () => document.getElementById('metric-pitch');
  const energyEl      = () => document.getElementById('metric-energy');
  const monotonyEl    = () => document.getElementById('metric-monotony');

  // ── Public: Toggle recording ─────────────────────────────────────────────
  async function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  }

  // ── Start recording ───────────────────────────────────────────────────────
  async function startRecording() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (err) {
      app.toast('Microphone access denied. Please allow mic access in your browser.', 'warn');
      return;
    }

    // Reset state
    pitchHistory  = [];
    energyHistory = [];
    ampHistory = []; periodHistory = []; envelope = [];
    totalFrames = 0; silentFrames = 0; startedAt = performance.now();
    transcript    = '';
    finalMetrics  = null;

    // Create Audio Context
    audioContext  = new (window.AudioContext || window.webkitAudioContext)();
    analyserNode  = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;

    const source = audioContext.createMediaStreamSource(mediaStream);
    source.connect(analyserNode);

    isRecording = true;
    updateUI(true);
    startSpeechRecognition();
    drawLoop();
    featureTimer = setInterval(extractFrame, FRAME_MS);

    statusEl().textContent = 'Listening… speak naturally';
  }

  // ── Stop recording ────────────────────────────────────────────────────────
  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    if (animFrameId) cancelAnimationFrame(animFrameId);
    if (featureTimer) { clearInterval(featureTimer); featureTimer = null; }
    if (recognition)  { try { recognition.stop(); } catch(e) {} }
    if (mediaStream)  { mediaStream.getTracks().forEach(t => t.stop()); }
    if (audioContext) { audioContext.close(); }

    updateUI(false);
    finalMetrics = computeMetrics();
    displayMetrics(finalMetrics);

    // Mirror transcript to text area if empty
    if (transcript.trim()) {
      const ta = document.getElementById('feeling-textarea');
      if (ta && !ta.value.trim()) {
        ta.value = transcript.trim();
        document.getElementById('char-count').textContent = ta.value.length;
      }
      transcriptEl().textContent = transcript;
      transcriptBox().style.display = 'block';
    }

    statusEl().textContent = 'Recording complete ✓';
  }

  // ── Speech Recognition (Web Speech API) ──────────────────────────────────
  function startSpeechRecognition() {
    const SpeechRecog = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecog) return; // silently skip if not supported

    recognition = new SpeechRecog();
    recognition.continuous      = true;
    recognition.interimResults  = true;
    recognition.lang            = app.currentLang === 'hi' ? 'hi-IN' : 'en-IN';

    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          transcript += result[0].transcript + ' ';
        } else {
          interim += result[0].transcript;
        }
      }
      // Show live transcript
      const ta = document.getElementById('feeling-textarea');
      if (ta) {
        ta.value = transcript + interim;
        document.getElementById('char-count').textContent = ta.value.length;
      }
    };

    recognition.onerror = () => {};
    try { recognition.start(); } catch(e) {}
  }

  // ── Draw Loop (Waveform + Feature Extraction) ─────────────────────────────
  function drawLoop() {
    if (!isRecording) return;
    animFrameId = requestAnimationFrame(drawLoop);
    const timeArr = new Uint8Array(analyserNode.fftSize);
    analyserNode.getByteTimeDomainData(timeArr);
    drawWaveform(timeArr);
  }

  // Runs every FRAME_MS: real F0 (autocorrelation), RMS, peak amplitude, silence detection
  function extractFrame() {
    if (!isRecording || !analyserNode) return;
    const buf = new Float32Array(analyserNode.fftSize);
    analyserNode.getFloatTimeDomainData(buf);
    const f = analyzeFrame(buf, audioContext.sampleRate);
    totalFrames++;
    envelope.push(f.rms);
    if (!f.voiced) {
      if (f.rms < SILENCE_RMS) silentFrames++;
      periodHistory.push(null);
    } else {
      pitchHistory.push(f.f0);
      energyHistory.push(f.rms);
      ampHistory.push(f.peak);
      periodHistory.push(1 / f.f0);
    }
    if (pitchEl())  pitchEl().textContent  = f.voiced ? hz2label(f.f0) : '—';
    if (energyEl()) energyEl().textContent = f.rms > SILENCE_RMS ? energyLabel(f.rms) : '—';
    const mono = monotonyScore(pitchHistory);
    if (monotonyEl()) monotonyEl().textContent = mono !== null ? monoLabel(mono) : '—';
  }

  // ── DSP Helpers ───────────────────────────────────────────────────────────

  const SILENCE_RMS = 0.01;   // below this a frame counts as a pause
  const F0_MIN = 70, F0_MAX = 400;

  /**
   * Analyse one frame of float samples [-1,1]:
   * RMS, peak, and F0 by normalised autocorrelation (voiced only if the peak is clear).
   */
  function analyzeFrame(x, sr) {
    const n = x.length;
    let sum = 0, peak = 0;
    for (let i = 0; i < n; i++) { const v = x[i]; sum += v * v; const a = Math.abs(v); if (a > peak) peak = a; }
    const rms = Math.sqrt(sum / n);
    if (rms < SILENCE_RMS) return { rms, peak, voiced: false, f0: 0 };
    const minLag = Math.floor(sr / F0_MAX), maxLag = Math.min(Math.floor(sr / F0_MIN), n >> 1);
    let r0 = 0; for (let i = 0; i < n - maxLag; i++) r0 += x[i] * x[i];
    if (r0 <= 0) return { rms, peak, voiced: false, f0: 0 };
    let best = 0, bestLag = 0;
    const corr = new Float32Array(maxLag + 2);
    for (let lag = minLag; lag <= maxLag + 1 && lag < n; lag++) {
      let c = 0; for (let i = 0; i < n - maxLag - 1; i++) c += x[i] * x[i + lag];
      corr[lag] = c / r0;
    }
    for (let lag = minLag + 1; lag <= maxLag; lag++) {
      // local maximum, take the first strong one (avoids octave errors)
      if (corr[lag] > corr[lag - 1] && corr[lag] >= corr[lag + 1] && corr[lag] > best) {
        best = corr[lag]; bestLag = lag;
        if (best > 0.9) break;
      }
    }
    if (best < 0.45 || !bestLag) return { rms, peak, voiced: false, f0: 0 };
    // parabolic interpolation for sub-sample lag
    const y1 = corr[bestLag - 1], y2 = corr[bestLag], y3 = corr[bestLag + 1];
    const d = (y1 - 2 * y2 + y3);
    const lag = d !== 0 ? bestLag + 0.5 * (y1 - y3) / d : bestLag;
    return { rms, peak, voiced: true, f0: sr / lag };
  }

  /** Mean absolute difference of consecutive values / mean (jitter & shimmer definition), voiced neighbours only */
  function relPerturbation(values) {
    // values may contain null (gaps); only consecutive non-null pairs are used
    let diffSum = 0, pairs = 0, valSum = 0, valN = 0;
    for (let i = 0; i < values.length; i++) {
      if (values[i] == null) continue;
      valSum += values[i]; valN++;
      if (i > 0 && values[i - 1] != null) { diffSum += Math.abs(values[i] - values[i - 1]); pairs++; }
    }
    if (pairs < 3 || valN === 0) return null;
    return (diffSum / pairs) / (valSum / valN);
  }

  /** Syllable-nuclei style speech rate: count RMS envelope peaks per second of non-silent time */
  function speechRate(env, frameMs) {
    if (env.length < 6) return null;
    const sm = env.map((v, i) => (v + (env[i - 1] ?? v) + (env[i + 1] ?? v)) / 3);
    const speakingFrames = sm.filter(v => v >= SILENCE_RMS).length;
    if (speakingFrames < 5) return null;
    const thr = Math.max(SILENCE_RMS * 1.5, 0.25 * Math.max(...sm));
    let peaks = 0;
    for (let i = 1; i < sm.length - 1; i++) if (sm[i] > thr && sm[i] > sm[i - 1] && sm[i] >= sm[i + 1]) peaks++;
    return peaks / (speakingFrames * frameMs / 1000);   // syllable-like peaks per second
  }

  /** Spectral centroid — proxy for perceived pitch (brightness) — kept for reference only */
  function spectralCentroid(freqData, sampleRate) {
    let numerator = 0, denominator = 0;
    const binHz = sampleRate / (freqData.length * 2);
    for (let i = 0; i < freqData.length; i++) {
      const mag = freqData[i];
      numerator   += mag * (i * binHz);
      denominator += mag;
    }
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /** RMS energy of time-domain signal */
  function rmsEnergy(timeData) {
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) {
      const norm = (timeData[i] - 128) / 128;
      sum += norm * norm;
    }
    return Math.sqrt(sum / timeData.length);
  }

  /** Coefficient of variation of pitch array → monotony [0, 1] */
  function monotonyScore(pitches) {
    if (pitches.length < 5) return null;
    const mean = pitches.reduce((a, b) => a + b, 0) / pitches.length;
    const variance = pitches.reduce((acc, p) => acc + (p - mean) ** 2, 0) / pitches.length;
    const cv = Math.sqrt(variance) / (mean || 1);
    // High CV = varied pitch (less monotone), low CV = flat (monotone)
    // Typical conversational F0 CV is ~0.10-0.25; CV below ~0.05 is flat, above ~0.25 is very varied.
    return Math.max(0, Math.min(1, 1 - cv / 0.25));
  }

  // ── Compute Final Summary Metrics ─────────────────────────────────────────
  function computeMetrics() {
    if (pitchHistory.length < 5) return null;   // not enough voiced speech to measure

    const mean = arr => arr.reduce((p, c) => p + c, 0) / arr.length;
    const std  = arr => { const m = mean(arr); return Math.sqrt(arr.reduce((p, c) => p + (c - m) ** 2, 0) / arr.length); };

    const pitchMean  = mean(pitchHistory);              // real F0 (Hz), voiced frames
    const pitchStd   = std(pitchHistory);
    const pitchCV    = pitchStd / (pitchMean || 1);
    const energyMean = mean(energyHistory);
    const monoScore  = monotonyScore(pitchHistory);     // 0 = varied, 1 = flat
    // Frame-to-frame jitter (period) and shimmer (peak amplitude) between consecutive voiced frames.
    // Not Praat-grade cycle-to-cycle values, but defined the same way (mean |diff| / mean).
    const jitter     = relPerturbation(periodHistory) ?? 0;
    const shimmer    = relPerturbation(ampHistory.map((v, i) => periodHistory[i] == null ? null : v)) ?? 0;
    const pauseRatio = totalFrames ? silentFrames / totalFrames : 0;
    const rate       = speechRate(envelope, FRAME_MS);

    // Hand-chosen heuristic (weights are NOT fitted to data): flat pitch, low energy, long pauses, instability
    const vocalRiskBase =
      (monoScore ?? 0.5) * 40 +
      (1 - Math.min(energyMean * 10, 1)) * 30 +
      Math.min(pauseRatio / 0.5, 1) * 15 +
      Math.min(jitter / 0.05, 1) * 15;
    const vocalScore = Math.round(Math.min(Math.max(vocalRiskBase, 0), 100));

    return {
      pitchMean:  Math.round(pitchMean),
      pitchCV:    parseFloat(pitchCV.toFixed(3)),
      energyMean: parseFloat(energyMean.toFixed(4)),
      monoScore:  monoScore !== null ? parseFloat(monoScore.toFixed(2)) : null,
      jitter:     parseFloat(jitter.toFixed(3)),
      shimmer:    parseFloat(shimmer.toFixed(3)),
      speechRate: rate !== null ? parseFloat(rate.toFixed(2)) : null,
      pauseRatio: parseFloat(pauseRatio.toFixed(2)),
      vocalScore,
      frameCount: pitchHistory.length
    };
  }

  // ── Display Metrics ───────────────────────────────────────────────────────
  function displayMetrics(m) {
    if (!m) return;
    if (metricsEl()) metricsEl().style.display = 'flex';
    if (canvasEl())  canvasEl().style.display  = 'block';
    if (pitchEl())   pitchEl().textContent      = hz2label(m.pitchMean);
    if (energyEl())  energyEl().textContent     = energyLabel(m.energyMean);
    if (monotonyEl()) monotonyEl().textContent  = m.monoScore !== null ? monoLabel(m.monoScore) : '—';
  }

  // ── Waveform Canvas ────────────────────────────────────────────────────────
  function drawWaveform(timeArr) {
    const canvas = canvasEl();
    if (!canvas) return;
    canvas.style.display = 'block';
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(0,0,0,0.0)';
    ctx.fillRect(0, 0, W, H);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#FF7F51';
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#FF7F51';
    ctx.beginPath();

    const sliceW = W / timeArr.length;
    let x = 0;
    for (let i = 0; i < timeArr.length; i++) {
      const v = timeArr[i] / 128.0;
      const y = v * H / 2;
      if (i === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
  }

  // ── Label helpers ─────────────────────────────────────────────────────────
  function hz2label(hz) {
    if (hz < 150) return 'Low';
    if (hz < 250) return 'Med';
    return 'High';
  }
  function energyLabel(e) {
    if (e < 0.02) return 'Soft';
    if (e < 0.08) return 'Med';
    return 'Loud';
  }
  function monoLabel(m) {
    if (m < 0.3) return 'Varied';
    if (m < 0.6) return 'Medium';
    return 'Flat';
  }

  // ── UI State ──────────────────────────────────────────────────────────────
  function updateUI(recording) {
    const btn = micBtnEl();
    const viz = visualizerEl();
    if (btn) btn.classList.toggle('recording', recording);
    if (viz) viz.classList.toggle('recording', recording);
    if (metricsEl()) metricsEl().style.display = recording ? 'flex' : 'none';
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return {
    toggleRecording,
    stopRecording,
    getMetrics: () => finalMetrics,
    _dsp: { analyzeFrame, relPerturbation, speechRate, monotonyScore },
    getTranscript: () => transcript,
    isRecording: () => isRecording
  };

})();
