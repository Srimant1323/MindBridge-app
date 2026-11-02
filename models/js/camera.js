/**
 * camera.js — Camera Capture + OCR Module for MindBridge
 *
 * Features:
 *   1. Live camera viewfinder (rear camera preferred on mobile)
 *   2. Snap to canvas → extract still image
 *   3. Gallery upload fallback (file picker, works on all devices)
 *   4. Tesseract.js v4 OCR — reads text from captured image
 *   5. Extracted text → fills feeling-textarea → ready for NLP analysis
 *   6. Lazy-loads Tesseract.js from CDN only when first used (no startup cost)
 *
 * How BERT/NLP assesses the captured content:
 *   - After OCR, the extracted text is placed into #feeling-textarea
 *   - User clicks "Analyze My Check-in" → app.analyze() runs as normal
 *   - NLP.analyze() + EmotionAI.analyze() process the captured text
 *   - Full emotion profile + risk score + recommendations shown
 */
const cameraOCR = (function () {

  // ── State ───────────────────────────────────────────────────────────────────
  let stream       = null;   // MediaStream from getUserMedia
  let tesseractLoaded = false;
  let ocrWorker    = null;   // Tesseract worker instance
  let lastImageData = null;  // last captured blob URL (for retake)
  let currentMode  = 'ocr';  // 'ocr' | 'facial'
  let capturedFacialMetrics = null;

  // ── DOM refs ─────────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  // ── Lazy-load Tesseract.js from CDN ──────────────────────────────────────────
  async function loadTesseract() {
    if (tesseractLoaded) return true;
    try {
      await _loadScript('https://cdn.jsdelivr.net/npm/tesseract.js@4.1.1/dist/tesseract.min.js');
      tesseractLoaded = true;
      return true;
    } catch (e) {
      console.error('[CameraOCR] Tesseract.js load failed:', e);
      _showToast('⚠️ OCR library failed to load. Check your connection.', 'error');
      return false;
    }
  }

  function _loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
      const s = document.createElement('script');
      s.src = src; s.onload = resolve; s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  // ── Mode Switching ─────────────────────────────────────────────────────────
  function setMode(mode) {
    currentMode = mode;
    const isFacial = mode === 'facial';
    
    // Toggle active UI tabs if present
    const ocrTab = $('cam-tab-ocr');
    const facialTab = $('cam-tab-facial');
    if (ocrTab) ocrTab.classList.toggle('active', !isFacial);
    if (facialTab) facialTab.classList.toggle('active', isFacial);

    const desc = $('cam-mode-desc');
    if (desc) {
      desc.textContent = isFacial
        ? 'Real-time Facial Affective Computing: Tracks AU1/AU4 brow furrow, AU12 smile, eye aspect ratio & blink rate'
        : 'Photo your journal, diary, or notes — AI reads & assesses it via OCR';
    }

    const snapBtn = $('camera-snap-btn');
    if (snapBtn) {
      snapBtn.innerHTML = isFacial
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg> Capture Affect'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/></svg> Snap &amp; Read';
    }

    // Restart camera if live
    if (stream || (typeof facialAffect !== 'undefined' && facialAffect.isRunning)) {
      stop();
      openCamera();
    }
  }

  // ── Open camera ──────────────────────────────────────────────────────────────
  async function openCamera() {
    _setStatus('opening');

    if (currentMode === 'facial' && typeof facialAffect !== 'undefined') {
      const ok = await facialAffect.start($('camera-video'), $('camera-canvas'));
      if (ok) {
        _showEl('camera-video');
        _showEl('camera-canvas');
        _hideEl('camera-idle');
        _hideEl('camera-preview');
        _hideEl('ocr-result-box');
        _showEl('camera-snap-btn');
        _showEl('camera-stop-btn');
        _hideEl('camera-open-btn');
        _hideEl('camera-retry-btn');
        _setStatus('live');
        _showToast('🙂 Facial Affective Computing live: Tracking Action Units (AU1/4, AU12) and blinks', 'info');
        return;
      }
    }

    // Fallback or Document OCR mode
    try {
      const facing = currentMode === 'facial' ? 'user' : { ideal: 'environment' };
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facing,
          width:  { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      const video = $('camera-video');
      video.srcObject = stream;
      await video.play();

      _showEl('camera-video');
      _hideEl('camera-idle');
      _hideEl('camera-preview');
      _hideEl('ocr-result-box');

      _showEl('camera-snap-btn');
      _showEl('camera-stop-btn');
      _hideEl('camera-open-btn');
      _hideEl('camera-retry-btn');

      _setStatus('live');

    } catch (err) {
      console.warn('[CameraOCR] getUserMedia error:', err.name, err.message);

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        _showToast('📷 Camera permission denied. Use "Upload Image" instead.', 'error');
      } else if (err.name === 'NotFoundError') {
        _showToast('📷 No camera found on this device. Use "Upload Image".', 'error');
      } else {
        _showToast('📷 Could not access camera. Try "Upload Image".', 'error');
      }
      _setStatus('idle');
    }
  }

  // ── Snap photo or Capture Facial Affect ─────────────────────────────────────
  async function snap() {
    if (currentMode === 'facial' && typeof facialAffect !== 'undefined') {
      const metrics = facialAffect.getMetrics();
      if (metrics) {
        capturedFacialMetrics = metrics;
        _showToast(`✅ Captured Facial Affect! Score: ${metrics.facialScore}/100 (${metrics.dominantExpression})`, 'success');
        
        // Show captured badge in viewfinder
        const resultBox = $('ocr-result-box');
        if (resultBox) {
          resultBox.style.display = 'block';
          const title = resultBox.querySelector('.ocr-result-badge');
          if (title) title.innerHTML = '🙂 Facial Biomarkers Captured';
          const textEl = $('ocr-result-text');
          if (textEl) {
            textEl.innerHTML = `
              <div style="font-size:0.85rem;line-height:1.6">
                <strong>Dominant State:</strong> ${metrics.dominantExpression}<br/>
                <strong>Facial Risk Index:</strong> ${metrics.facialScore} / 100<br/>
                <strong>AU1/4 Brow Furrow (Distress):</strong> ${metrics.au1_4}%<br/>
                <strong>AU12 Lip Corner Pull (Smile):</strong> ${metrics.au12}%<br/>
                <strong>Eye Aspect Ratio (EAR):</strong> ${metrics.ear} (${metrics.blinkRate} blinks/min)
              </div>
            `;
          }
          const useBtn = resultBox.querySelector('.ocr-use-btn');
          if (useBtn) {
            useBtn.textContent = 'Armed for Analysis ✓';
            useBtn.onclick = () => {
              _showToast('✅ Facial affect locked for Tri-Modal Fusion!', 'info');
              _hideEl('ocr-result-box');
            };
          }
        }
        return;
      }
    }

    const video  = $('camera-video');
    const canvas = $('camera-canvas');
    if (!video || !canvas) return;

    // Draw current video frame onto canvas
    canvas.width  = video.videoWidth  || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Show preview image
    const dataURL = canvas.toDataURL('image/jpeg', 0.92);
    lastImageData = dataURL;
    const preview = $('camera-preview');
    preview.src = dataURL;

    // Stop camera stream to save battery
    stop(true); // true = keep preview visible

    _hideEl('camera-video');
    _showEl('camera-preview');
    _showEl('camera-retry-btn');
    _hideEl('camera-snap-btn');
    _hideEl('camera-stop-btn');
    _hideEl('camera-open-btn');

    // Run OCR on captured frame
    await _runOCR(canvas);
  }

  // ── Upload from gallery ───────────────────────────────────────────────────────
  async function fromFile(input) {
    const file = input.files[0];
    if (!file) return;

    // Show image preview
    const reader = new FileReader();
    reader.onload = async (e) => {
      const preview = $('camera-preview');
      preview.src = e.target.result;
      lastImageData = e.target.result;
      _showEl('camera-preview');
      _hideEl('camera-idle');
      _hideEl('camera-video');
      _showEl('camera-retry-btn');
      _hideEl('camera-snap-btn');
      _hideEl('camera-stop-btn');
      _hideEl('camera-open-btn');

      // Draw onto canvas for OCR
      const img = new Image();
      img.onload = async () => {
        const canvas = $('camera-canvas');
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext('2d').drawImage(img, 0, 0);
        await _runOCR(canvas);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // ── OCR Engine ─────────────────────────────────────────────────────────────
  async function _runOCR(canvas) {
    _showEl('ocr-overlay');
    _hideEl('ocr-result-box');
    _setOCRStatus('⏳ Loading OCR engine…');

    // Load Tesseract lazily
    const ok = await loadTesseract();
    if (!ok) { _hideEl('ocr-overlay'); return; }

    _setOCRStatus('📖 Reading text from image…');

    try {
      // Create/reuse worker
      if (!ocrWorker) {
        ocrWorker = await Tesseract.createWorker('eng', 1, {
          logger: m => {
            if (m.status === 'recognizing text') {
              const pct = Math.round((m.progress || 0) * 100);
              _setOCRStatus(`📖 Reading text… ${pct}%`);
            }
          }
        });
      }

      const { data } = await ocrWorker.recognize(canvas);
      const text = (data.text || '').trim();

      _hideEl('ocr-overlay');

      if (!text || text.length < 5) {
        _showToast('🔍 No readable text found. Try a clearer photo with more light.', 'error');
        _showEl('ocr-overlay');
        _setOCRStatus('❌ No text detected — try better lighting');
        setTimeout(() => _hideEl('ocr-overlay'), 3000);
        return;
      }

      // Show extracted text preview
      _showExtractedText(text, data.confidence);

      // ── AUTO-INJECT: immediately push text into textarea for analysis ──
      // User no longer needs to click 'Use Text' — it's automatic
      _autoInjectToTextarea(text);

    } catch (err) {
      console.error('[CameraOCR] OCR error:', err);
      _hideEl('ocr-overlay');
      _showToast('❌ OCR failed. Please try again or type manually.', 'error');
    }
  }

  // ── Show extracted text to user ───────────────────────────────────────────────
  function _showExtractedText(text, confidence) {
    const resultBox  = $('ocr-result-box');
    const resultText = $('ocr-result-text');
    if (!resultBox || !resultText) return;

    // Clean up common OCR artifacts
    const cleaned = text
      .replace(/\f/g, '\n')          // form feed → newline
      .replace(/\r\n/g, '\n')        // normalize line endings
      .replace(/[ \t]+\n/g, '\n')    // trailing spaces on lines
      .replace(/\n{3,}/g, '\n\n')    // max 2 consecutive blank lines
      .trim();

    resultText.textContent = cleaned;
    resultText.dataset.rawText = cleaned;

    // Show confidence badge
    const conf = Math.round(confidence || 0);
    const confColor = conf >= 70 ? '#34D399' : conf >= 40 ? '#FBBF24' : '#FB923C';
    const badge = resultBox.querySelector('.ocr-result-badge');
    if (badge) {
      badge.innerHTML = `📝 Extracted Text <span style="font-size:0.7rem;color:${confColor};margin-left:6px">Confidence: ${conf}%</span>`;
    }

    _showEl('ocr-result-box');

    // Scroll into view smoothly
    setTimeout(() => resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
  }

  // ── Auto-inject OCR text into textarea (no button click needed) ───────────────
  function _autoInjectToTextarea(text) {
    const textarea  = $('feeling-textarea');
    const charCount = $('char-count');
    if (!textarea || !text) return;

    // Clean OCR artifacts
    const cleaned = text
      .replace(/\f/g, '\n')
      .replace(/\r\n/g, '\n')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
      .slice(0, 1000);

    // If textarea already has content, append with separator
    const existing = (textarea.value || '').trim();
    textarea.value = existing
      ? existing + '\n\n[From image]\n' + cleaned
      : cleaned;

    if (charCount) charCount.textContent = textarea.value.length;

    // Trigger input event so live-waveform / char-counter update
    textarea.dispatchEvent(new Event('input', { bubbles: true }));

    // Visual highlight on the textarea
    textarea.style.transition = 'box-shadow 0.4s';
    textarea.style.boxShadow  = '0 0 0 3px rgba(52,211,153,0.5)';
    setTimeout(() => { textarea.style.boxShadow = ''; }, 2000);

    // Navigate to checkin view if not already there (text needs the textarea)
    if (typeof app !== 'undefined' && app.navigate) {
      const checkinSection = document.getElementById('view-checkin');
      if (checkinSection && !checkinSection.classList.contains('active')) {
        app.navigate('checkin');
      }
    }

    // Scroll textarea into view and show CTA toast
    setTimeout(() => {
      textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
      _showToast('✅ Image text loaded! Click "Analyze My Check-in" to score it.', 'success');
    }, 600);
  }

  // ── Use extracted text → fill textarea ────────────────────────────────────────
  function useText() {
    const resultText = $('ocr-result-text');
    const textarea   = $('feeling-textarea');
    const charCount  = $('char-count');
    if (!resultText || !textarea) return;

    const text = (resultText.dataset.rawText || resultText.textContent || '').trim();
    if (!text) return;

    // Cap to textarea maxlength (1000 chars)
    const capped = text.slice(0, 1000);
    textarea.value = capped;

    // Update char counter
    if (charCount) charCount.textContent = capped.length;

    // Trigger input event so any live-analysis listeners fire
    textarea.dispatchEvent(new Event('input', { bubbles: true }));

    // Scroll to textarea and highlight it briefly
    textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    textarea.style.transition = 'box-shadow 0.4s';
    textarea.style.boxShadow  = '0 0 0 3px rgba(99, 102, 241, 0.6)';
    setTimeout(() => { textarea.style.boxShadow = ''; }, 1800);

    _showToast('✅ Text loaded! Click "Analyze My Check-in" to assess it.', 'success');

    // Collapse result box
    _hideEl('ocr-result-box');
  }

  // ── Stop camera ───────────────────────────────────────────────────────────────
  function stop(keepPreview = false) {
    if (typeof facialAffect !== 'undefined' && facialAffect.isRunning) {
      facialAffect.stop();
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    const video = $('camera-video');
    if (video) { video.srcObject = null; video.style.display = 'none'; }
    const canvas = $('camera-canvas');
    if (canvas && currentMode !== 'facial') { canvas.style.display = 'none'; }

    if (!keepPreview) {
      _hideEl('camera-preview');
      _showEl('camera-idle');
      _hideEl('ocr-result-box');
      _hideEl('ocr-overlay');
      _showEl('camera-open-btn');
      _hideEl('camera-snap-btn');
      _hideEl('camera-stop-btn');
      _hideEl('camera-retry-btn');
    }
    _setStatus('idle');
  }

  // ── Retake ────────────────────────────────────────────────────────────────────
  function retry() {
    _hideEl('camera-preview');
    _hideEl('ocr-result-box');
    _hideEl('ocr-overlay');
    _hideEl('camera-retry-btn');
    openCamera();
  }

  // ── Cleanup on view change ────────────────────────────────────────────────────
  function cleanup() {
    stop();
    if (ocrWorker) {
      ocrWorker.terminate();
      ocrWorker = null;
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────
  function _showEl(id) { const e = $(id); if (e) e.style.display = ''; }
  function _hideEl(id) { const e = $(id); if (e) e.style.display = 'none'; }

  function _setOCRStatus(msg) {
    const el = $('ocr-status-text');
    if (el) el.textContent = msg;
  }

  function _setStatus(state) {
    const vf = $('camera-viewfinder');
    if (vf) vf.dataset.state = state;
  }

  function _showToast(msg, type = 'info') {
    if (typeof app !== 'undefined' && app.toast) {
      app.toast(msg, type);
      return;
    }
    // Fallback simple toast
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = `position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
      background:${type==='error'?'#ef4444':'#10b981'};color:#fff;padding:12px 20px;
      border-radius:10px;z-index:9999;font-size:0.9rem;box-shadow:0 4px 20px rgba(0,0,0,0.3)`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  }

  // ── Auto-cleanup when navigating away from check-in ──────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    // Listen for view changes — stop camera if user navigates away
    const observer = new MutationObserver(() => {
      const checkinView = document.getElementById('view-checkin');
      if (checkinView && checkinView.style.display === 'none' && stream) {
        stop();
      }
    });
    const container = document.getElementById('view-checkin');
    if (container) {
      observer.observe(container, { attributes: true, attributeFilter: ['style', 'class'] });
    }
  });

  function getFacialMetrics() {
    return capturedFacialMetrics || (typeof facialAffect !== 'undefined' ? facialAffect.getMetrics() : null);
  }

  function getMode() {
    return currentMode;
  }

  return {
    openCamera,
    snap,
    stop,
    retry,
    fromFile,
    useText,
    cleanup,
    setMode,
    getMode,
    getFacialMetrics
  };

})();
