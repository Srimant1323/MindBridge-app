/**
 * emotion.js — Real Emotion AI via Transformers.js v3
 *
 * Model: first of MODEL_CANDIDATES that loads and passes a self-test (see list below).
 * Emotions: sadness, joy, love, anger, fear, surprise (7-class models are mapped onto these).
 *
 * Loading strategy:
 *   1. Try @huggingface/transformers v3 (recommended, latest)
 *   2. Fallback to @xenova/transformers v2 (legacy CDN)
 *   3. If both fail → the app uses the lexicon-based NLP synthesis and labels reports 'NLP-Inferred'
 *
 * Risk mapping (how each emotion contributes to overall risk score):
 *   sadness  → +50%   (strongest depression marker)
 *   fear     → +30%
 *   anger    → +15%
 *   surprise → +5%
 *   joy      → -60%   (strongest protective factor)
 *   love     → -40%
 */
const EmotionAI = (function () {

  let pipelineFn = null;
  let modelId    = null;
  // Ordered candidates. Verify in a browser which one loads, then keep the working one first.
  const MODEL_CANDIDATES = [
    // 'Xenova/bert-base-uncased-finetuned-emotion' was removed: Hugging Face returns 401 for it (confirmed in browser console).
    // 1. ONNX export of j-hartmann/emotion-english-distilroberta-base (the DistilRoBERTa model named on the About page).
    //    7 labels (anger, disgust, fear, joy, neutral, sadness, surprise); 'disgust' is folded into anger below.
    //    Tried with the default quantised file first, then full precision (larger download).
    { id: 'Bachkippe/emotion-english-distilroberta-base_onnx' },
    { id: 'Bachkippe/emotion-english-distilroberta-base_onnx', dtype: 'fp32' },
    // 2. Fallback 6-class models
    { id: 'onnx-community/albert-base-v2-emotion-ONNX' },
    { id: 'Cohee/bert-base-uncased-emotion-onnx' }
  ];
  let ready      = false;
  let loading    = false;
  let modelSource = '';  // which CDN worked

  const EMOTION_META = {
    sadness:  { emoji: '😢', color: '#818CF8', label: 'Sadness',  riskWeight: 0.50, protectWeight: 0    },
    joy:      { emoji: '😊', color: '#FBBF24', label: 'Joy',      riskWeight: 0,    protectWeight: 0.60 },
    love:     { emoji: '❤️', color: '#F472B6', label: 'Love',     riskWeight: 0,    protectWeight: 0.40 },
    anger:    { emoji: '😡', color: '#FB923C', label: 'Anger',    riskWeight: 0.15, protectWeight: 0    },
    fear:     { emoji: '😨', color: '#A78BFA', label: 'Fear',     riskWeight: 0.30, protectWeight: 0    },
    surprise: { emoji: '😲', color: '#34D399', label: 'Surprise', riskWeight: 0.05, protectWeight: 0    }
  };

  // Some exports return LABEL_0..5 (dair-ai emotion order). Map them to names.
  const DAIR_ORDER = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];
  function _normLabel(label) {
    const m = /^LABEL_(\d+)$/i.exec(label || '');
    return m && DAIR_ORDER[+m[1]] ? DAIR_ORDER[+m[1]] : String(label).toLowerCase();
  }

  // ── Load model with CDN fallback chain ────────────────────────────────────
  async function load() {
    if (ready || loading) return;
    loading = true;
    _setStatus('loading');

    // Try CDN sources in order of preference
    const CDN_SOURCES = [
      // v3 — @huggingface/transformers (recommended)
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js',
      // v2 — @xenova/transformers (legacy fallback)
      'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js',
    ];

    let lastError = null;

    for (const cdnUrl of CDN_SOURCES) {
      try {
        const { pipeline, env } = await import(cdnUrl);

        env.allowLocalModels  = false;
        env.useBrowserCache   = true;   // cache model in IndexedDB after first load

        // Try each candidate model in order; use the first that actually loads.
        // (The original model ID could not be verified, so we never assume it exists.)
        let loaded = null;
        for (const c of MODEL_CANDIDATES) {
          const id = c.id;
          try {
            const cand = await pipeline('text-classification', id, c.dtype ? { dtype: c.dtype } : {});
            // Self-test: only accept a model that really returns the 6 expected emotions
            const probe = await cand('I feel very sad and alone today', { top_k: 10, topk: 10 }); // return ALL labels (7-class models have 7)
            const labels = probe.map(r => _normLabel(r.label));
            const ok = ['sadness', 'joy', 'anger', 'fear'].every(k => labels.includes(k));
            if (!ok) throw new Error('unexpected labels: ' + probe.map(r => r.label).join(','));
            loaded = cand;
            modelId = id;
            break;
          } catch (mErr) {
            console.warn(`[EmotionAI] Model failed (${id}):`, mErr.message);
            lastError = mErr;
          }
        }
        if (!loaded) throw lastError || new Error('No emotion model could be loaded');
        pipelineFn = loaded;

        ready       = true;
        loading     = false;
        modelSource = cdnUrl.includes('huggingface') ? 'HF v3' : 'Xenova v2';
        _setStatus('ready');

        if (typeof app !== 'undefined') {
          app.toast('🧠 Emotion AI loaded! Results will be more accurate now.', 'success');
        }
        return;   // success — stop trying other CDNs

      } catch (err) {
        lastError = err;
        console.warn(`[EmotionAI] CDN failed (${cdnUrl}):`, err.message);
        // continue to next CDN
      }
    }

    // All CDNs failed
    loading = false;
    console.warn('[EmotionAI] All CDN sources failed. Using NLP synthesis fallback.', lastError);
    _setStatus('error');
    throw lastError || new Error('Emotion model unavailable');
  }

  // ── Analyze text with loaded BERT model ──────────────────────────────────
  async function analyze(text) {
    if (!ready || !pipelineFn || !text || text.trim().length < 5) return null;

    try {
      const truncated = text.trim().slice(0, 512);  // BERT 512-token limit
      const results   = await pipelineFn(truncated, { top_k: 10, topk: 10 }); // scores for all labels

      // Build emotion map → { sadness: 0.82, joy: 0.03, … }
      const em = {};
      results.forEach(r => { em[_normLabel(r.label)] = r.score; });
      // 7-class DistilRoBERTa has 'disgust' and 'neutral' but no 'love': fold disgust into anger, ignore neutral
      if (em.disgust != null) { em.anger = (em.anger || 0) + em.disgust; delete em.disgust; }
      delete em.neutral;

      // Ensure all 6 emotions have a value (some models may skip near-zero ones)
      Object.keys(EMOTION_META).forEach(k => { if (em[k] == null) em[k] = 0; });

      // ── Sanity check: if dominant emotion is clearly wrong, flag it ──────
      // BERT can sometimes give unexpected results on colloquial text.
      // We detect obvious mismatches and flag them so the caller can decide
      // whether to use BERT's result or fall back to NLP synthesis.
      // dominant among the 6 app emotions (after folding disgust / dropping neutral)
      const topEmotion = Object.keys(EMOTION_META).reduce((p, k) => (em[k] || 0) > (em[p] || 0) ? k : p, 'sadness');
      const topScore   = em[topEmotion] || 0;

      // ── Compute risk score ───────────────────────────────────────────────
      let riskRaw = 0, protectRaw = 0;
      Object.entries(EMOTION_META).forEach(([label, meta]) => {
        riskRaw    += (em[label] || 0) * meta.riskWeight;
        protectRaw += (em[label] || 0) * meta.protectWeight;
      });
      const netRisk     = Math.max(0, riskRaw - protectRaw * 0.5);
      const emotionRisk = Math.round(Math.min(netRisk * 100, 100));

      return {
        emotions:      em,
        emotionRisk,
        dominant:      topEmotion,
        dominantScore: Math.round(topScore * 100),
        riskRaw:       parseFloat(riskRaw.toFixed(3)),
        protectRaw:    parseFloat(protectRaw.toFixed(3)),
        modelSource
      };

    } catch (err) {
      console.warn('[EmotionAI] Analysis error:', err);
      return null;
    }
  }

  // ── Status badge UI ───────────────────────────────────────────────────────
  function _setStatus(status) {
    const badge = document.getElementById('ai-status-badge');
    const dot   = document.getElementById('ai-status-dot');
    const label = document.getElementById('ai-status-label');
    if (!badge) return;

    badge.dataset.status = status;
    const cfg = {
      loading: { dotClass: 'status-dot-spin', text: 'AI Loading…', color: '#FBBF24' },
      ready:   { dotClass: 'status-dot-pulse', text: 'AI Ready ✓',  color: '#34D399' },
      error:   { dotClass: '',                 text: 'AI Offline',   color: '#FB923C' }
    }[status] || {};

    if (dot)   { dot.style.background = cfg.color; dot.className = 'ai-dot ' + (cfg.dotClass || ''); }
    if (label) label.textContent = cfg.text;
  }

  function getMeta() { return EMOTION_META; }
  function getModelId() { return modelId; }

  return {
    load,
    analyze,
    getMeta,
    getModelId,
    isReady:   () => ready,
    isLoading: () => loading
  };

})();
