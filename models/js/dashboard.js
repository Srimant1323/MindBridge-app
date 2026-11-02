/**
 * dashboard.js — MindBridge Dashboard Module
 * Live stats, session history, donut chart, sparklines, mood tips.
 */
const dashboard = (function () {

  const SESSIONS_KEY = 'mindbridge_sessions';
  let tipIndex = 0;

  const TIPS = [
    '💧 Drink a glass of water right now — hydration directly impacts mood and cognitive clarity.',
    '🌿 5 minutes of sunlight can boost serotonin. Step outside if you can.',
    '📵 Try a 30-min "phone-free" window. Your nervous system will thank you.',
    '😴 Going to bed and waking at the same time each day is the single most powerful mood regulator.',
    '🎵 Music with 60 BPM matches your resting heart rate and induces calm. Try ambient or lo-fi.',
    '🧘 Box breathing (4-4-4-4) activates the vagus nerve and reduces cortisol in under 2 minutes.',
    '📝 Writing 3 things you noticed today (not "grateful for") builds mindful awareness without pressure.',
    '🤝 Loneliness amplifies perceived stress. Even a 5-minute conversation with someone helps.',
    '🏃 A 10-minute walk increases BDNF — the brain\'s growth factor — more effectively than meditation.',
    '🌙 The "worry window" technique: dedicate 15 min/day to worries, then gently defer anxious thoughts outside it.',
  ];

  const RISK_COLORS = {
    low:      '#10b981',
    moderate: '#f59e0b',
    high:     '#f97316',
    severe:   '#ef4444'
  };

  // ── Session Storage ──────────────────────────────────────────────────────
  // Escape text before putting it into innerHTML (stored names/snippets are user input)
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  // Local calendar day (YYYY-MM-DD) — avoids the UTC off-by-one for IST users
  function localDay(d) {
    const x = (d instanceof Date) ? d : new Date(d);
    if (isNaN(x)) return '';
    return x.getFullYear() + '-' + String(x.getMonth() + 1).padStart(2, '0') + '-' + String(x.getDate()).padStart(2, '0');
  }

  function loadSessions() {
    try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || '[]'); }
    catch { return []; }
  }

  function saveSessions(sessions) {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.slice(0, 50)));
  }

  function logSession(riskLevel, score, dominantEmotion, name, snippet) {
    const sessions = loadSessions();
    const entry = {
      id:        Date.now(),
      riskLevel: riskLevel || 'low',
      score:     score || 0,
      emotion:   dominantEmotion || '—',
      name:      name || 'Anonymous',
      snippet:   snippet || '',
      date:      new Date().toISOString(),
      ts:        Date.now()
    };
    sessions.unshift(entry);
    saveSessions(sessions);
    refresh();
    updateNavBadge(sessions.length);
  }

  // ── Greeting ────────────────────────────────────────────────────────────
  function setGreeting() {
    const h = new Date().getHours();
    const greet = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
    const el = document.getElementById('dashboard-greeting');
    if (el) el.textContent = greet + ' 👋';

    const dateEl = document.getElementById('dashboard-date');
    if (dateEl) {
      const now = new Date();
      dateEl.textContent = now.toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    }
  }

  // ── Nav badge ────────────────────────────────────────────────────────────
  function updateNavBadge(count) {
    const badge = document.getElementById('nav-sessions-badge');
    if (!badge) return;
    if (count > 0) {
      badge.textContent = count > 99 ? '99+' : count;
      badge.classList.add('visible');
    } else {
      badge.classList.remove('visible');
    }
  }

  // ── Stat Cards ──────────────────────────────────────────────────────────
  function updateStats(sessions) {
    const total = sessions.length;

    // Total
    const totalEl = document.getElementById('stat-total-val');
    if (totalEl) animateNumber(totalEl, total);
    const deltaEl = document.getElementById('stat-delta-total');
    if (deltaEl) {
      const today = localDay(new Date());
      const todayCount = sessions.filter(s => s && s.date && localDay(s.date) === today).length;
      deltaEl.textContent = '+' + todayCount + ' today';
    }

    // Streak
    const streak = computeStreak(sessions);
    const streakEl = document.getElementById('stat-streak-val');
    if (streakEl) animateNumber(streakEl, streak);
    renderFlames(streak);

    // Avg risk
    if (total > 0) {
      const avg = Math.round(sessions.reduce((sum, s) => sum + (s.score || 0), 0) / total);
      const riskEl = document.getElementById('stat-risk-val');
      if (riskEl) animateNumber(riskEl, avg);
      const miniBar = document.getElementById('risk-mini-bar');
      setTimeout(() => { if (miniBar) miniBar.style.width = avg + '%'; }, 200);

      const riskDelta = document.getElementById('stat-delta-risk');
      if (riskDelta) {
        const lastRisk = sessions[0]?.riskLevel || 'low';
        riskDelta.textContent = lastRisk.charAt(0).toUpperCase() + lastRisk.slice(1);
        riskDelta.style.background = RISK_COLORS[lastRisk] + '22';
        riskDelta.style.color = RISK_COLORS[lastRisk];
      }
    }

    // Reports
    const reportsEl = document.getElementById('stat-reports-val');
    if (reportsEl) animateNumber(reportsEl, total);

    // Sparkline
    drawStatSparkline(sessions);
  }

  // ── Streak Computation ───────────────────────────────────────────────────
  function computeStreak(sessions) {
    if (!sessions.length) return 0;
    const days = [...new Set(sessions.map(s => localDay(s.date)).filter(Boolean))].sort().reverse();
    if (!days.length) return 0;

    let streak = 0;
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    for (let i = 0; i < days.length; i++) {
      const d = new Date(days[i] + 'T00:00:00');
      const diff = Math.round((today - d) / 86400000);
      if (diff === streak) {
        streak++;
      } else {
        break;
      }
    }
    return streak;
  }

  // ── Flame indicators ────────────────────────────────────────────────────
  function renderFlames(streak) {
    const el = document.getElementById('streak-flames');
    if (!el) return;
    const max = Math.min(streak, 7);
    el.innerHTML = '';
    for (let i = 0; i < 7; i++) {
      const dot = document.createElement('div');
      dot.className = 'flame-dot' + (i >= max ? ' empty' : '');
      if (i < max) dot.style.animationDelay = (i * 0.15) + 's';
      el.appendChild(dot);
    }
  }

  // ── Sessions Table ────────────────────────────────────────────────────
  function renderSessionsTable(sessions) {
    const tbody = document.getElementById('sessions-tbody');
    if (!tbody) return;

    if (!sessions.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-sessions">No sessions yet. Start your first check-in!</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    sessions.slice(0, 8).forEach((s, idx) => {
      const date = formatRelDate(s.date);
      const rl = s.riskLevel || 'low';
      const riskDot = { low: '🟢', moderate: '🟡', high: '🟠', severe: '🔴' }[rl] || '⚪';
      const name = s.name && s.name !== 'Anonymous' ? `<span class="session-name-tag">${esc(s.name)}</span>` : '<span style="color:var(--text-muted);font-size:0.72rem">Anonymous</span>';

      const tr = document.createElement('tr');
      tr.style.animationDelay = (idx * 0.05) + 's';
      tr.innerHTML = `
        <td>
          ${name}
          <div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.snippet ? '"' + esc(s.snippet.slice(0,60)) + '…"' : 'Session #' + (sessions.length - idx)}</div>
        </td>
        <td><span class="risk-badge risk-badge-${rl}">${riskDot} ${rl.charAt(0).toUpperCase() + rl.slice(1)}</span></td>
        <td><span class="score-cell">${s.score}</span></td>
        <td style="color:var(--text-muted);font-size:0.78rem">${date}</td>
        <td><button class="session-report-btn" onclick="dashboard.viewSession(${idx})">View</button></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // ── Community Log (grouped by person) ────────────────────────────────
  function renderCommunityLog(sessions) {
    const container = document.getElementById('community-log-body');
    if (!container) return;

    if (!sessions.length) {
      container.innerHTML = '<p class="empty-state">No community entries yet. Share the link with your circle!</p>';
      return;
    }

    // Group by name
    const groups = {};
    sessions.forEach(s => {
      const key = s.name || 'Anonymous';
      if (!groups[key]) groups[key] = [];
      groups[key].push(s);
    });

    container.innerHTML = '';
    Object.entries(groups).forEach(([name, entries]) => {
      const avgScore = Math.round(entries.reduce((a, s) => a + s.score, 0) / entries.length);
      const lastRisk = entries[0].riskLevel;
      const initials = name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
      const riskColor = { low: '#10b981', moderate: '#f59e0b', high: '#f97316', severe: '#ef4444' }[lastRisk] || '#94a3b8';
      const riskDot   = { low: '🟢', moderate: '🟡', high: '🟠', severe: '🔴' }[lastRisk] || '⚪';

      const row = document.createElement('tr');
      row.innerHTML = `
        <td>
          <div class="person-cell">
            <div class="person-avatar" style="background:${riskColor}22;border-color:${riskColor}44;color:${riskColor}">${initials}</div>
            <div>
              <div class="person-name">${esc(name)}</div>
              <div class="person-count">${entries.length} session${entries.length > 1 ? 's' : ''}</div>
            </div>
          </div>
        </td>
        <td><span class="risk-badge risk-badge-${lastRisk}">${riskDot} ${lastRisk.charAt(0).toUpperCase() + lastRisk.slice(1)}</span></td>
        <td><span class="score-cell">${avgScore}</span></td>
        <td style="color:var(--text-muted);font-size:0.78rem">${formatRelDate(entries[0].date)}</td>
        <td>
          <button class="session-report-btn" data-name="${esc(name)}" onclick="dashboard.expandPerson(this.dataset.name)">Details</button>
        </td>
      `;
      container.appendChild(row);
    });

    // Update people count
    const countEl = document.getElementById('community-people-count');
    if (countEl) countEl.textContent = Object.keys(groups).length;
  }

  function expandPerson(name) {
    const sessions = loadSessions().filter(s => (s.name || 'Anonymous') === name);
    const detail = document.getElementById('person-detail-panel');
    if (!detail) return;

    const riskColor = { low: '#10b981', moderate: '#f59e0b', high: '#f97316', severe: '#ef4444' };
    const entries = sessions.map(s => `
      <div class="person-session-row">
        <span class="risk-badge risk-badge-${s.riskLevel}" style="flex-shrink:0">${s.riskLevel}</span>
        <span style="font-size:0.78rem;color:var(--text-secondary);flex:1">${s.snippet ? '"' + esc(s.snippet.slice(0,80)) + '"' : '—'}</span>
        <span style="font-size:0.72rem;color:var(--text-muted);white-space:nowrap">${formatRelDate(s.date)}</span>
        <span class="score-cell" style="font-size:0.95rem">${s.score}</span>
      </div>
    `).join('');

    detail.innerHTML = `
      <div class="person-detail-header">
        <strong style="color:var(--text-primary)">${esc(name)}</strong>
        <span style="font-size:0.78rem;color:var(--text-muted)">${sessions.length} sessions</span>
        <button class="btn-text-link" onclick="document.getElementById('person-detail-panel').innerHTML=''" style="margin-left:auto">× Close</button>
      </div>
      ${entries}
    `;
    detail.scrollIntoView({ behavior: 'smooth' });
  }

  // ── Donut Chart ──────────────────────────────────────────────────────────
  function drawDonut(sessions) {
    const canvas = document.getElementById('donut-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2, r = W * 0.42, inner = r * 0.6;

    const counts = { low: 0, moderate: 0, high: 0, severe: 0 };
    sessions.forEach(s => { if (counts[s.riskLevel] !== undefined) counts[s.riskLevel]++; });
    const total = sessions.length || 1;

    const segments = [
      { key: 'low',      color: '#10b981', val: counts.low },
      { key: 'moderate', color: '#f59e0b', val: counts.moderate },
      { key: 'high',     color: '#f97316', val: counts.high },
      { key: 'severe',   color: '#ef4444', val: counts.severe },
    ].filter(s => s.val > 0);

    ctx.clearRect(0, 0, W, H);

    if (!sessions.length) {
      // Empty state ring
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = r - inner;
      ctx.stroke();
    } else {
      let startAngle = -Math.PI / 2;
      segments.forEach(seg => {
        const angle = (seg.val / total) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, startAngle, startAngle + angle);
        ctx.closePath();
        ctx.fillStyle = seg.color;
        ctx.fill();
        startAngle += angle;
      });

      // cutout
      ctx.beginPath();
      ctx.arc(cx, cy, inner, 0, Math.PI * 2);
      ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--bg-card') || '#111827';
      // Use the actual bg color
      ctx.fillStyle = '#111827';
      ctx.fill();

      // Gap lines (optional polish)
      startAngle = -Math.PI / 2;
      segments.forEach(seg => {
        const angle = (seg.val / total) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(startAngle) * inner, cy + Math.sin(startAngle) * inner);
        ctx.lineTo(cx + Math.cos(startAngle) * r, cy + Math.sin(startAngle) * r);
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = 2;
        ctx.stroke();
        startAngle += angle;
      });
    }

    // Update center
    const totalEl = document.getElementById('donut-total');
    if (totalEl) totalEl.textContent = sessions.length;

    // Legend values
    const pct = v => total > 0 ? Math.round((v / total) * 100) : 0;
    const legLow  = document.getElementById('leg-low');
    const legMod  = document.getElementById('leg-mod');
    const legHigh = document.getElementById('leg-high');
    const legSev  = document.getElementById('leg-sev');
    if (legLow)  legLow.textContent  = `${counts.low} (${pct(counts.low)}%)`;
    if (legMod)  legMod.textContent  = `${counts.moderate} (${pct(counts.moderate)}%)`;
    if (legHigh) legHigh.textContent = `${counts.high} (${pct(counts.high)}%)`;
    if (legSev)  legSev.textContent  = `${counts.severe} (${pct(counts.severe)}%)`;
  }

  // ── Stat Sparkline ────────────────────────────────────────────────────────
  function drawStatSparkline(sessions) {
    const canvas = document.getElementById('sparkline-total');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (sessions.length < 2) return;

    // Last 10 sessions' scores
    const data = sessions.slice(0, 10).reverse().map(s => s.score || 0);
    const maxVal = Math.max(...data, 10);
    const points = data.map((v, i) => ({
      x: (i / (data.length - 1)) * (W - 4) + 2,
      y: H - 2 - ((v / maxVal) * (H - 6))
    }));

    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#10b981');
    grad.addColorStop(1, '#8b5cf6');

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const cp = (points[i - 1].x + points[i].x) / 2;
      ctx.bezierCurveTo(cp, points[i-1].y, cp, points[i].y, points[i].x, points[i].y);
    }
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 6;
    ctx.shadowColor = '#10b981';
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // ── Daily Tip ─────────────────────────────────────────────────────────────
  function renderTip() {
    const el = document.getElementById('daily-tip');
    if (!el) return;
    el.style.opacity = 0;
    setTimeout(() => {
      el.textContent = TIPS[tipIndex % TIPS.length];
      el.style.transition = 'opacity 0.4s';
      el.style.opacity = 1;
    }, 200);
  }

  function nextTip() {
    tipIndex++;
    renderTip();
  }

  // ── View a specific session ───────────────────────────────────────────────
  function viewSession(idx) {
    app.navigate('report');
    app.toast('Showing your most recent report.', 'info');
  }

  // ── Relative date formatter ───────────────────────────────────────────────
  function formatRelDate(isoStr) {
    try {
      const d = new Date(isoStr);
      const now = new Date();
      const diffMs = now - d;
      const diffMin = Math.floor(diffMs / 60000);
      if (diffMin < 2)   return 'Just now';
      if (diffMin < 60)  return diffMin + ' min ago';
      const diffH = Math.floor(diffMin / 60);
      if (diffH < 24)    return diffH + 'h ago';
      const diffD = Math.floor(diffH / 24);
      if (diffD === 1)   return 'Yesterday';
      if (diffD < 7)     return diffD + ' days ago';
      return d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
    } catch { return '—'; }
  }

  // ── Animate number ────────────────────────────────────────────────────────
  function animateNumber(el, target) {
    const start = parseInt(el.textContent) || 0;
    if (start === target) return;
    const step = Math.max(Math.abs(target - start) / 30, 1);
    let cur = start;
    const timer = setInterval(() => {
      cur = cur < target ? Math.min(cur + step, target) : Math.max(cur - step, target);
      el.textContent = Math.round(cur);
      if (Math.round(cur) === target) clearInterval(timer);
    }, 25);
  }

  // ── Refresh all dashboard data ────────────────────────────────────────────
  function refresh() {
    const sessions = loadSessions();
    setGreeting();
    updateStats(sessions);
    renderSessionsTable(sessions);
    renderCommunityLog(sessions);
    drawDonut(sessions);
    renderTip();
    updateNavBadge(sessions.length);
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  function init() {
    // Set up daily tip rotation
    tipIndex = Math.floor(Math.random() * TIPS.length);
    refresh();
  }

  // ── Export all sessions as CSV ─────────────────────────────────────────────
  function exportCSV() {
    const sessions = loadSessions();
    if (!sessions.length) { app.toast('No sessions to export yet.', 'warn'); return; }

    const headers = ['Name', 'Risk Level', 'Score', 'Dominant Emotion', 'Snippet', 'Date'];
    const rows = sessions.map(s => [
      `"${(s.name || 'Anonymous').replace(/"/g, '""')}"`,
      s.riskLevel,
      s.score,
      s.emotion || '—',
      `"${(s.snippet || '').replace(/"/g, '""')}"`,
      new Date(s.date).toLocaleString('en-IN')
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `mindbridge-community-log-${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    app.toast('✅ Community log exported as CSV!', 'success');
  }

  return {
    init,
    refresh,
    logSession,
    nextTip,
    viewSession,
    expandPerson,
    exportCSV,
    loadSessions
  };

})();
