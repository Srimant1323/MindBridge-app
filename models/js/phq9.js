/**
 * phq9.js — MindBridge PHQ-9 + GAD-7 Clinical Screening Module
 * PHQ-9: Patient Health Questionnaire (depression screening)
 * GAD-7: Generalized Anxiety Disorder Scale
 * Both are widely used, peer-reviewed screening instruments.
 * Scores are saved to localStorage and logged to community dashboard.
 */
const phq9 = (function () {

  const STORAGE_KEY = 'mindbridge_phq9_results';

  // ── PHQ-9 Questions ────────────────────────────────────────────────────────
  const PHQ9_QUESTIONS = [
    { id: 'p1', text: 'Little interest or pleasure in doing things' },
    { id: 'p2', text: 'Feeling down, depressed, or hopeless' },
    { id: 'p3', text: 'Trouble falling or staying asleep, or sleeping too much' },
    { id: 'p4', text: 'Feeling tired or having little energy' },
    { id: 'p5', text: 'Poor appetite or overeating' },
    { id: 'p6', text: 'Feeling bad about yourself — or that you are a failure or have let yourself or your family down' },
    { id: 'p7', text: 'Trouble concentrating on things, such as reading or watching television' },
    { id: 'p8', text: 'Moving or speaking so slowly that others noticed — or being so fidgety/restless that you moved around more than usual' },
    { id: 'p9', text: 'Thoughts that you would be better off dead, or thoughts of hurting yourself in some way' }
  ];

  // ── GAD-7 Questions ─────────────────────────────────────────────────────────
  const GAD7_QUESTIONS = [
    { id: 'g1', text: 'Feeling nervous, anxious, or on edge' },
    { id: 'g2', text: 'Not being able to stop or control worrying' },
    { id: 'g3', text: 'Worrying too much about different things' },
    { id: 'g4', text: 'Trouble relaxing' },
    { id: 'g5', text: 'Being so restless that it\'s hard to sit still' },
    { id: 'g6', text: 'Becoming easily annoyed or irritable' },
    { id: 'g7', text: 'Feeling afraid, as if something awful might happen' }
  ];

  // Each answer has a colour on a green → red scale and an emoji face.
  // The numeric value (0–3) is still used for scoring but is no longer shown as a number.
  const OPTIONS = [
    { label: 'Not at all',              value: 0, color: '#22c55e', face: '😊' },
    { label: 'Several days',            value: 1, color: '#eab308', face: '🙂' },
    { label: 'More than half the days', value: 2, color: '#f97316', face: '😟' },
    { label: 'Nearly every day',        value: 3, color: '#ef4444', face: '😣' }
  ];
  const SCALE_GRADIENT = 'linear-gradient(90deg,#22c55e 0%,#eab308 38%,#f97316 68%,#ef4444 100%)';

  // Styles live here so the colour feature works without touching the CSS files.
  (function injectStyles() {
    if (document.getElementById('phq-color-styles')) return;
    const st = document.createElement('style');
    st.id = 'phq-color-styles';
    st.textContent = `
      .phq-question-card{border-left:4px solid var(--phq-c,rgba(255,255,255,.12));transition:border-color .25s, background .25s}
      .phq-question-card.answered{background:linear-gradient(90deg,color-mix(in srgb,var(--phq-c) 10%,transparent),transparent 60%)}
      .phq-options{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
      @media(max-width:640px){.phq-options{grid-template-columns:repeat(2,1fr)}}
      .phq-option-label input{position:absolute;opacity:0;pointer-events:none}
      .phq-option-label .phq-option-box{display:flex;flex-direction:column;align-items:center;gap:6px;padding:12px 8px;border-radius:14px;
        border:2px solid color-mix(in srgb,var(--c) 45%,transparent);background:color-mix(in srgb,var(--c) 8%,transparent);
        cursor:pointer;transition:transform .15s, box-shadow .2s, background .2s;text-align:center;min-height:86px;justify-content:center}
      .phq-option-label:hover .phq-option-box{transform:translateY(-2px);background:color-mix(in srgb,var(--c) 16%,transparent)}
      .phq-option-label input:focus-visible + .phq-option-box{outline:3px solid var(--c);outline-offset:2px}
      .phq-option-label.selected .phq-option-box{background:var(--c);border-color:var(--c);color:#111;box-shadow:0 6px 22px color-mix(in srgb,var(--c) 45%,transparent);transform:scale(1.04)}
      .phq-option-label.selected .phq-option-text{color:#111;font-weight:800}
      .phq-face{font-size:1.7rem;line-height:1}
      .phq-option-text{font-size:.74rem;font-weight:700;line-height:1.2}
      .phq-live-meter{margin:6px 0 18px}
      .phq-live-track{position:relative;height:14px;border-radius:99px;background:${SCALE_GRADIENT};opacity:.9}
      .phq-live-marker{position:absolute;top:-5px;width:6px;height:24px;border-radius:4px;background:#fff;box-shadow:0 0 0 2px rgba(0,0,0,.5);left:0;transition:left .5s cubic-bezier(.22,1,.36,1)}
      .phq-live-labels{display:flex;justify-content:space-between;font-size:.68rem;color:var(--text-muted);margin-top:6px}
      .phq-dots{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
      .phq-dot{width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,.1);border:2px solid rgba(255,255,255,.15);transition:background .3s}
      .phq-result-gradient{position:relative;height:16px;border-radius:99px;background:${SCALE_GRADIENT};margin:12px 0 6px}
      .phq-result-pin{position:absolute;top:-6px;width:8px;height:28px;border-radius:5px;background:#fff;box-shadow:0 0 0 2px #000a;transform:translateX(-50%)}
      .phq-zone-labels{display:flex;justify-content:space-between;font-size:.64rem;color:var(--text-muted)}
    `;
    document.head.appendChild(st);
  })();

  // ── Scoring Thresholds ────────────────────────────────────────────────────
  const PHQ9_LEVELS = [
    { max: 4,  level: 'none',     label: 'Minimal / No Depression',      color: '#22c55e', short: 'Minimal', lo: 0,  rec: 'No significant depression detected. Keep up positive habits and regular self-check-ins.' },
    { max: 9,  level: 'mild',     label: 'Mild Depression',              color: '#84cc16', short: 'Mild', lo: 5,    rec: 'Mild symptoms present. Lifestyle changes (sleep, exercise, social connection) can help significantly.' },
    { max: 14, level: 'moderate', label: 'Moderate Depression',          color: '#eab308', short: 'Moderate', lo: 10, rec: 'Moderate symptoms. Consider speaking with a counselor or mental health professional.' },
    { max: 19, level: 'mod-sev',  label: 'Moderately Severe Depression', color: '#f97316', short: 'Mod-severe', lo: 15, rec: 'Significant symptoms. Active treatment is recommended. Please reach out to a mental health professional.' },
    { max: 27, level: 'severe',   label: 'Severe Depression',            color: '#ef4444', short: 'Severe', lo: 20, rec: 'Severe symptoms. Please contact a mental health professional or helpline immediately.' }
  ];

  const GAD7_LEVELS = [
    { max: 4,  level: 'none',     label: 'Minimal Anxiety',    color: '#22c55e', short: 'Minimal',  lo: 0 },
    { max: 9,  level: 'mild',     label: 'Mild Anxiety',       color: '#eab308', short: 'Mild',     lo: 5 },
    { max: 14, level: 'moderate', label: 'Moderate Anxiety',   color: '#f97316', short: 'Moderate', lo: 10 },
    { max: 21, level: 'severe',   label: 'Severe Anxiety',     color: '#ef4444', short: 'Severe',   lo: 15 }
  ];

  // Gradient bar whose colour stops and labels sit exactly on the real score bands.
  function zoneBar(score, max, levels) {
    const centres = levels.map((l, i) => {
      const hi = (i + 1 < levels.length) ? levels[i + 1].lo : max + 1;   // band covers lo .. hi-1
      return { l, c: ((l.lo + hi) / 2) / (max + 1) * 100 };
    });
    const grad = 'linear-gradient(90deg,' + centres.map(x => `${x.l.color} ${x.c.toFixed(1)}%`).join(',') + ')';
    const pos = Math.min(Math.max((score + 0.5) / (max + 1), 0), 1) * 100;   // same scale as the band labels
    return `<div class="phq-result-gradient" style="background:${grad}"><div class="phq-result-pin" style="left:${pos}%"></div></div>
      <div style="position:relative;height:14px">${centres.map(x => `<span style="position:absolute;left:${x.c.toFixed(1)}%;transform:translateX(-50%);font-size:.62rem;color:var(--text-muted);white-space:nowrap">${x.l.short}</span>`).join('')}</div>`;
  }

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // ── Storage ─────────────────────────────────────────────────────────────
  function loadResults() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch { return []; }
  }
  function saveResult(result) {
    const results = loadResults();
    results.unshift(result);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(results.slice(0, 30)));
  }

  // ── Render PHQ-9 Form ────────────────────────────────────────────────────
  function renderForms() {
    renderQuestionGroup('phq9-questions', PHQ9_QUESTIONS, 'phq9');
    renderQuestionGroup('gad7-questions', GAD7_QUESTIONS, 'gad7');
    ensureLiveMeter('phq9-questions', 'phq9', PHQ9_QUESTIONS);
    ensureLiveMeter('gad7-questions', 'gad7', GAD7_QUESTIONS);
  }

  function renderQuestionGroup(containerId, questions, prefix) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    questions.forEach((q, idx) => {
      const card = document.createElement('div');
      card.className = 'phq-question-card';
      card.innerHTML = `
        <div class="phq-q-header">
          <span class="phq-q-num">${idx + 1}</span>
          <span class="phq-q-text">${q.text}</span>
          ${prefix === 'phq9' && idx === 8 ? '<span class="phq-critical-badge">⚠ Critical</span>' : ''}
        </div>
        <div class="phq-options" role="radiogroup" aria-label="${q.text}">
          ${OPTIONS.map(opt => `
            <label class="phq-option-label" style="--c:${opt.color}" id="${prefix}-${q.id}-opt-${opt.value}">
              <input type="radio" name="${prefix}-${q.id}" value="${opt.value}" onchange="phq9.onAnswer('${prefix}', '${q.id}', ${opt.value})" />
              <span class="phq-option-box">
                <span class="phq-face" aria-hidden="true">${opt.face}</span>
                <span class="phq-option-text">${opt.label}</span>
              </span>
            </label>
          `).join('')}
        </div>
      `;
      container.appendChild(card);
    });
  }

  // ── Track answers live ─────────────────────────────────────────────────
  const answers = {};

  function onAnswer(prefix, qId, value) {
    const key = `${prefix}-${qId}`;
    answers[key] = parseInt(value);

    // Visual feedback on the selected option
    document.querySelectorAll(`[name="${prefix}-${qId}"]`).forEach(r => {
      const lbl = r.closest('.phq-option-label');
      lbl.classList.toggle('selected', parseInt(r.value) === parseInt(value));
    });

    // Colour the question card and the live meter
    const opt = OPTIONS.find(o => o.value === parseInt(value));
    const card = document.querySelector(`[name="${prefix}-${qId}"]`)?.closest('.phq-question-card');
    if (card && opt) { card.style.setProperty('--phq-c', opt.color); card.classList.add('answered'); }
    const dot = document.getElementById(`${prefix}-dot-${qId}`);
    if (dot && opt) { dot.style.background = opt.color; dot.style.borderColor = opt.color; }

    // Check if section is complete and update progress
    updateProgress(prefix);

    // Crisis check — PHQ-9 Question 9 (suicidal ideation)
    if (prefix === 'phq9' && qId === 'p9' && value >= 1) {
      showCrisisAlert();
    }
  }

  function updateProgress(prefix) {
    const questions = prefix === 'phq9' ? PHQ9_QUESTIONS : GAD7_QUESTIONS;
    const answered  = questions.filter(q => answers[`${prefix}-${q.id}`] !== undefined).length;
    const pct       = Math.round((answered / questions.length) * 100);

    const bar   = document.getElementById(`${prefix}-progress-bar`);
    const label = document.getElementById(`${prefix}-progress-label`);
    if (bar)   bar.style.width = pct + '%';
    if (label) label.textContent = `${answered} / ${questions.length} answered`;
    updateLiveMeter(prefix, questions);
  }

  // Running green→red marker based on the answers so far (shows the average severity of what you have answered)
  function updateLiveMeter(prefix, questions) {
    const vals = questions.map(q => answers[`${prefix}-${q.id}`]).filter(v => v !== undefined);
    const marker = document.getElementById(`${prefix}-live-marker`);
    if (!marker) return;
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;   // 0..3
    marker.style.left = `calc(${(avg / 3) * 100}% - 3px)`;
  }

  function ensureLiveMeter(containerId, prefix, questions) {
    const container = document.getElementById(containerId);
    if (!container || document.getElementById(`${prefix}-live-meter`)) return;
    const wrap = document.createElement('div');
    wrap.className = 'phq-live-meter';
    wrap.id = `${prefix}-live-meter`;
    wrap.innerHTML = `
      <div class="phq-live-track"><div class="phq-live-marker" id="${prefix}-live-marker"></div></div>
      <div class="phq-live-labels"><span>🟢 Doing well</span><span>🟡 Some strain</span><span>🔴 Heavy burden</span></div>
      <div class="phq-dots">${questions.map(q => `<span class="phq-dot" id="${prefix}-dot-${q.id}" title="Question ${q.id.slice(1)}"></span>`).join('')}</div>`;
    container.parentNode.insertBefore(wrap, container);
  }

  function showCrisisAlert() {
    const banner = document.getElementById('phq9-crisis-banner');
    if (banner) {
      banner.style.display = 'flex';
      banner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    app.toast('⚠️ Question 9 selected — please see the support message below.', 'warn');
  }

  // ── Submit + Score ─────────────────────────────────────────────────────
  function submit() {
    // Get respondent name
    const nameEl = document.getElementById('phq9-respondent-name');
    const name   = nameEl ? nameEl.value.trim() || 'Anonymous' : 'Anonymous';

    // Validate PHQ-9
    const phq9Missing = PHQ9_QUESTIONS.filter(q => answers[`phq9-${q.id}`] === undefined);
    // Validate GAD-7
    const gad7Missing = GAD7_QUESTIONS.filter(q => answers[`gad7-${q.id}`] === undefined);

    if (phq9Missing.length > 0) {
      app.toast(`Please answer all PHQ-9 questions (${phq9Missing.length} remaining).`, 'warn');
      document.getElementById('phq9-questions').scrollIntoView({ behavior: 'smooth' });
      return;
    }
    if (gad7Missing.length > 0) {
      app.toast(`Please answer all GAD-7 questions (${gad7Missing.length} remaining).`, 'warn');
      document.getElementById('gad7-questions').scrollIntoView({ behavior: 'smooth' });
      return;
    }

    // Compute PHQ-9 score
    const phq9Score = PHQ9_QUESTIONS.reduce((sum, q) => sum + (answers[`phq9-${q.id}`] || 0), 0);
    const gad7Score = GAD7_QUESTIONS.reduce((sum, q) => sum + (answers[`gad7-${q.id}`] || 0), 0);

    const phq9Level = PHQ9_LEVELS.find(l => phq9Score <= l.max) || PHQ9_LEVELS[4];
    const gad7Level = GAD7_LEVELS.find(l => gad7Score <= l.max) || GAD7_LEVELS[3];

    const result = {
      name,
      phq9Score,
      gad7Score,
      phq9Level:  phq9Level.level,
      gad7Level:  gad7Level.level,
      q9Flag:     (answers['phq9-p9'] || 0) >= 1,
      date:       new Date().toISOString(),
      ts:         Date.now()
    };

    saveResult(result);

    // Also log to main community dashboard
    const compositeRisk = computeCompositeRiskLevel(phq9Score, gad7Score);
    dashboard.logSession(compositeRisk.level, compositeRisk.score, 'PHQ-9/GAD-7', name, `PHQ-9: ${phq9Score} · GAD-7: ${gad7Score}`);

    renderResult(result, phq9Level, gad7Level);
    app.toast('✅ Screening complete! Your results are shown below.', 'success');
  }

  function computeCompositeRiskLevel(phq9Score, gad7Score) {
    const combined = phq9Score + gad7Score;
    const maxScore = 27 + 21;
    const score    = Math.round((combined / maxScore) * 100);
    let   level    = 'low';
    if (phq9Score >= 20 || gad7Score >= 15) level = 'severe';
    else if (phq9Score >= 15 || gad7Score >= 10) level = 'high';
    else if (phq9Score >= 10 || gad7Score >= 5)  level = 'moderate';
    return { level, score };
  }

  // ── Render Result ──────────────────────────────────────────────────────
  function renderResult(result, phq9Level, gad7Level) {
    const panel = document.getElementById('phq9-result-panel');
    if (!panel) return;

    const q9Warning = result.q9Flag ? `
      <div class="phq9-crisis-result">
        🆘 <strong>Important:</strong> You indicated thoughts of self-harm (Question 9). Please reach out to a counsellor or helpline immediately.
        <a href="tel:9152987821" class="crisis-call-link">Call iCall: 9152987821</a>
        <a href="tel:14416" class="crisis-call-link">Tele-MANAS: 14416 (24/7)</a>
      </div>
    ` : '';

    panel.innerHTML = `
      <div class="phq9-result-header">
        <h3 class="phq9-result-title">Screening Results</h3>
        <span style="font-size:0.78rem;color:var(--text-muted)">${esc(result.name)} · ${new Date(result.date).toLocaleDateString('en-IN', { dateStyle: 'medium' })}</span>
      </div>
      ${q9Warning}
      <div class="phq9-scores-grid">
        <div class="phq9-score-card">
          <div class="phq9-score-instrument">PHQ-9</div>
          <div class="phq9-score-value" style="color:${phq9Level.color}">${result.phq9Score}<span>/27</span></div>
          <div class="phq9-score-label" style="color:${phq9Level.color}">${phq9Level.label}</div>
          ${zoneBar(result.phq9Score, 27, PHQ9_LEVELS)}
        </div>
        <div class="phq9-score-card">
          <div class="phq9-score-instrument">GAD-7</div>
          <div class="phq9-score-value" style="color:${gad7Level.color}">${result.gad7Score}<span>/21</span></div>
          <div class="phq9-score-label" style="color:${gad7Level.color}">${gad7Level.label}</div>
          ${zoneBar(result.gad7Score, 21, GAD7_LEVELS)}
        </div>
      </div>
      <div class="phq9-recommendation">
        💡 <strong>Clinical Note:</strong> ${phq9Level.rec}
      </div>
      <div class="phq9-disclaimer">
        ⚠️ <em>This screening is not a substitute for professional diagnosis. PHQ-9 and GAD-7 are validated screening tools; a positive screen should be followed up with a qualified mental health professional.</em>
      </div>
      <div class="phq9-result-actions">
        <button class="btn btn-primary" onclick="window.print()">🖨 Print / Save PDF</button>
        <button class="btn btn-ghost" onclick="phq9.reset()">↩ Retake Screening</button>
      </div>
      <div class="phq9-citations">
        📚 <strong>Research basis:</strong>
        Kroenke K, Spitzer RL. "The PHQ-9: A New Depression Scale for Primary Care." <em>Psychosomatics</em>, 2002 · 
        Spitzer RL, et al. "A Brief Measure for Assessing Generalized Anxiety Disorder." <em>Archives of Internal Medicine</em>, 2006.
      </div>
    `;

    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Reset form ─────────────────────────────────────────────────────────
  function reset() {
    Object.keys(answers).forEach(k => delete answers[k]);
    document.querySelectorAll('.phq-option-label').forEach(l => l.classList.remove('selected'));
    document.querySelectorAll('.phq-option-label input[type="radio"]').forEach(r => r.checked = false);
    document.querySelectorAll('.phq-question-card').forEach(c => { c.classList.remove('answered'); c.style.removeProperty('--phq-c'); });
    document.querySelectorAll('.phq-dot').forEach(d => { d.style.background = ''; d.style.borderColor = ''; });
    ['phq9', 'gad7'].forEach(p => { updateProgress(p); });
    const panel = document.getElementById('phq9-result-panel');
    if (panel) { panel.style.display = 'none'; panel.innerHTML = ''; }
    const banner = document.getElementById('phq9-crisis-banner');
    if (banner) banner.style.display = 'none';
    const nameEl = document.getElementById('phq9-respondent-name');
    if (nameEl) nameEl.value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    app.toast('Screening reset. You can start again.', 'info');
  }

  // ── History ────────────────────────────────────────────────────────────
  function renderHistory() {
    const container = document.getElementById('phq9-history');
    if (!container) return;
    const results = loadResults();
    if (!results.length) {
      container.innerHTML = '<p class="empty-state">No previous screenings saved.</p>';
      return;
    }
    container.innerHTML = results.slice(0, 5).map(r => `
      <div class="phq9-history-item">
        <div class="phq9-history-left">
          <span class="phq9-history-name">${esc(r.name)}</span>
          <span class="phq9-history-date">${new Date(r.date).toLocaleDateString('en-IN', { dateStyle: 'medium' })}</span>
        </div>
        <div class="phq9-history-scores">
          <span>PHQ-9: <strong>${r.phq9Score}</strong></span>
          <span>GAD-7: <strong>${r.gad7Score}</strong></span>
          ${r.q9Flag ? '<span class="phq9-q9-flag">⚠ Q9</span>' : ''}
        </div>
      </div>
    `).join('');
  }

  // ── Init ──────────────────────────────────────────────────────────────
  function init() {
    renderForms();
    renderHistory();
  }

  return { init, onAnswer, submit, reset };

})();
