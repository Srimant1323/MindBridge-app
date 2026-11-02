/**
 * tracker.js — MindBridge v3 Lifestyle Tracker Module
 * localStorage-backed daily log for sleep, activity, stress, and mood notes.
 * Renders sparkline charts + 30-day mood heatmap.
 */
const tracker = (function () {

  const STORAGE_KEY = 'mindbridge_logs';
  let selectedActivity = 'light';

  // ── Storage helpers ───────────────────────────────────────────────────────
  function loadLogs() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch { return []; }
  }

  function saveLogs(logs) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(logs));
  }

  function dayKey(d) {
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  }
  function todayKey() {
    return dayKey(new Date());   // local day (toISOString is UTC: wrong day for IST users)
  }
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // ── Streak ────────────────────────────────────────────────────────────────
  function computeStreak(logs) {
    if (!logs.length) return 0;
    const days = [...new Set(logs.map(l => l.date))].sort().reverse();
    let streak = 0;
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    for (let i = 0; i < days.length; i++) {
      const d = new Date(days[i] + 'T00:00:00');
      const diff = Math.round((today - d) / 86400000);
      if (diff === streak) streak++;
      else break;
    }
    return streak;
  }

  // ── Public: Update slider display labels ──────────────────────────────────
  function updateDisplay(type, value) {
    if (type === 'sleep') {
      const el = document.getElementById('sleep-display');
      if (el) el.textContent = parseFloat(value) + ' hrs';
    } else if (type === 'stress') {
      const el = document.getElementById('stress-display');
      if (el) el.textContent = value + ' / 10';
    }
  }

  // ── Public: Set activity level ─────────────────────────────────────────────
  function setActivity(val) {
    selectedActivity = val;
    document.querySelectorAll('.activity-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.val === val);
    });
  }

  // ── Public: Save today's log ───────────────────────────────────────────────
  function saveLog() {
    const sleep    = parseFloat(document.getElementById('sleep-slider')?.value  ?? 7);
    const stress   = parseInt(document.getElementById('stress-slider')?.value   ?? 5);
    const moodNote = document.getElementById('mood-note')?.value.trim() || '';

    const logs   = loadLogs();
    const today  = todayKey();
    const exists = logs.findIndex(l => l.date === today);

    const entry = { date: today, sleep, stress, activity: selectedActivity, note: moodNote, ts: Date.now() };

    if (exists >= 0) { logs[exists] = entry; }
    else             { logs.unshift(entry); }

    const trimmed = logs.slice(0, 30);
    saveLogs(trimmed);

    renderLogList();
    renderCharts();
    renderHeatmap();
    renderStreakBadge();
    app.toast('Today\'s log saved! 📋', 'success');
  }

  // ── Public: Clear all logs ────────────────────────────────────────────────
  function clearLogs() {
    if (!confirm('Clear all lifestyle logs? This cannot be undone.')) return;
    saveLogs([]);
    renderLogList();
    renderCharts();
    renderHeatmap();
    renderStreakBadge();
    app.toast('All logs cleared.');
  }

  // ── Streak badge ──────────────────────────────────────────────────────────
  function renderStreakBadge() {
    const numEl = document.getElementById('tracker-streak-num');
    const streak = computeStreak(loadLogs());
    if (numEl) numEl.textContent = streak;
  }

  // ── 30-Day Mood Heatmap ───────────────────────────────────────────────────
  function renderHeatmap() {
    const container = document.getElementById('mood-heatmap');
    if (!container) return;

    const logs = loadLogs();
    const logMap = {};
    logs.forEach(l => { logMap[l.date] = l; });

    container.innerHTML = '';

    // Generate last 30 days
    const today = new Date();
    for (let i = 29; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      const key = dayKey(d);
      const log = logMap[key];

      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';

      if (log) {
        const stress = log.stress || 0;
        const sleep  = log.sleep  || 0;
        // Wellness score: lower stress + better sleep = greener
        const wellnessScore = Math.max(0, 10 - stress) + Math.min(sleep, 9);
        const normalized = wellnessScore / 19; // 0–1, 1 = best

        // Color scale
        let bg;
        if (normalized >= 0.75) bg = `rgba(16,185,129,${0.3 + normalized * 0.5})`;
        else if (normalized >= 0.5) bg = `rgba(245,158,11,0.5)`;
        else if (normalized >= 0.3) bg = `rgba(249,115,22,0.55)`;
        else bg = `rgba(239,68,68,${0.4 + (1-normalized)*0.4})`;

        cell.style.background = bg;
        cell.title = `${formatDate(key)}: Sleep ${log.sleep}h · Stress ${log.stress}/10${log.note ? ' · "' + log.note + '"' : ''}`;
      } else {
        cell.style.background = 'rgba(255,255,255,0.04)';
        cell.title = `${formatDate(key)}: No data`;
      }

      container.appendChild(cell);
    }
  }

  // ── Render log list ────────────────────────────────────────────────────────
  function renderLogList() {
    const container = document.getElementById('log-list');
    if (!container) return;

    const logs = loadLogs();
    if (logs.length === 0) {
      container.innerHTML = '<p class="empty-state">No logs yet. Save today\'s log to get started!</p>';
      return;
    }

    container.innerHTML = '';
    logs.slice(0, 10).forEach(log => {
      const date = formatDate(log.date);
      const activityLabel = { none: '🚫 None', light: '🚶 Light', moderate: '🏃 Moderate', intense: '💪 Intense' }[log.activity] || log.activity;
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.innerHTML = `
        <span class="log-date">${date}</span>
        <div class="log-summary">
          ${log.note ? `<em>"${esc(log.note)}"</em>` : '<span style="color:var(--text-muted)">No mood note</span>'}
        </div>
        <div class="log-meta">
          <span class="log-chip chip-sleep">😴 ${log.sleep}h</span>
          <span class="log-chip chip-stress">😤 ${log.stress}/10</span>
          <span class="log-chip chip-activity">${activityLabel}</span>
        </div>
      `;
      container.appendChild(entry);
    });
  }

  // ── Render 7-day sparkline charts ─────────────────────────────────────────
  function renderCharts() {
    const logs       = loadLogs().slice(0, 7).reverse();
    if (logs.length === 0) return;

    const sleepData  = logs.map(l => l.sleep);
    const stressData = logs.map(l => l.stress);
    const labels     = logs.map(l => l.date.slice(5));

    drawSparkline('sleep-chart',  sleepData,  { min: 0, max: 12, color: '#10b981', fill: 'rgba(16,185,129,0.15)' }, labels);
    drawSparkline('stress-chart', stressData, { min: 0, max: 10, color: '#f43f5e', fill: 'rgba(244,63,94,0.15)'   }, labels);

    const sleepAvg  = (sleepData.reduce((a, b) => a + b, 0) / sleepData.length).toFixed(1);
    const stressAvg = (stressData.reduce((a, b) => a + b, 0) / stressData.length).toFixed(1);
    const sleepEl  = document.getElementById('sleep-avg');
    const stressEl = document.getElementById('stress-avg');
    if (sleepEl)  sleepEl.textContent  = `7-day avg: ${sleepAvg} hrs`;
    if (stressEl) stressEl.textContent = `7-day avg: ${stressAvg} / 10`;
  }

  // ── Sparkline renderer ─────────────────────────────────────────────────────
  function drawSparkline(canvasId, data, opts, labels) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top: 10, bottom: 22, left: 6, right: 6 };
    const chartW = W - pad.left - pad.right;
    const chartH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);
    if (data.length < 2) return;

    const range = opts.max - opts.min || 1;
    const xStep = chartW / (data.length - 1);

    const points = data.map((v, i) => ({
      x: pad.left + i * xStep,
      y: pad.top + chartH - ((v - opts.min) / range) * chartH
    }));

    // Fill gradient
    const fillGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
    fillGrad.addColorStop(0,   opts.fill);
    fillGrad.addColorStop(1,   'rgba(0,0,0,0)');

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const cpx = (points[i - 1].x + points[i].x) / 2;
      ctx.bezierCurveTo(cpx, points[i - 1].y, cpx, points[i].y, points[i].x, points[i].y);
    }
    ctx.lineTo(points[points.length - 1].x, pad.top + chartH);
    ctx.lineTo(points[0].x, pad.top + chartH);
    ctx.closePath();
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const cpx = (points[i - 1].x + points[i].x) / 2;
      ctx.bezierCurveTo(cpx, points[i - 1].y, cpx, points[i].y, points[i].x, points[i].y);
    }
    ctx.strokeStyle = opts.color;
    ctx.lineWidth   = 2.5;
    ctx.shadowBlur  = 8;
    ctx.shadowColor = opts.color;
    ctx.stroke();
    ctx.shadowBlur  = 0;

    // Dots
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = opts.color;
      ctx.shadowBlur = 6;
      ctx.shadowColor = opts.color;
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // Labels
    ctx.fillStyle = 'rgba(100,116,139,0.8)';
    ctx.font      = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    if (labels) {
      labels.forEach((lbl, i) => {
        ctx.fillText(lbl, points[i].x, H - 4);
      });
    }
  }

  // ── Date formatter ─────────────────────────────────────────────────────────
  function formatDate(dateStr) {
    try {
      const d = new Date(dateStr + 'T00:00:00');
      return d.toLocaleDateString('en-IN', { weekday: 'short', month: 'short', day: 'numeric' });
    } catch { return dateStr; }
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  function init() {
    renderLogList();
    renderCharts();
    renderHeatmap();
    renderStreakBadge();
  }

  return {
    updateDisplay,
    setActivity,
    saveLog,
    clearLogs,
    init,
    renderCharts,
    renderLogList
  };

})();
