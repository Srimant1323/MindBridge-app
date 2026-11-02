/**
 * streak.js — Wellness Streak System + Daily Micro Check-in v1
 *
 * Features:
 *   - 5-emoji quick mood logging (one tap, no full analysis needed)
 *   - Consecutive-day streak counter (Bronze/Silver/Gold/Platinum badges)
 *   - "Already checked in today" detection with nice state
 *   - Integrates with existing 'mindbridge_sessions' localStorage key
 *   - Refreshes dashboard streak counter after logging
 *
 * Session schema matches dashboard.js: { id, riskLevel, score, emotion, name, snippet, date, ts }
 */
const StreakSystem = (function () {

  const SESSIONS_KEY = 'mindbridge_sessions';

  // 5 moods: emoji, label, score (0-100), emotion (for NLP compat), riskLevel
  const MOODS = [
    { emoji:'😊', label:'Calm',    score:12, emotion:'Joy',     risk:'low' },
    { emoji:'🙂', label:'Okay',    score:32, emotion:'Neutral', risk:'low' },
    { emoji:'😔', label:'Sad',     score:58, emotion:'Sadness', risk:'moderate' },
    { emoji:'😰', label:'Anxious', score:70, emotion:'Fear',    risk:'high' },
    { emoji:'😤', label:'Angry',   score:68, emotion:'Anger',   risk:'moderate' },
  ];

  const BADGES = [
    { min:30, icon:'🏆', label:'Platinum · 30+ days', cls:'sb-plat' },
    { min:21, icon:'🥇', label:'Gold · 21+ days',     cls:'sb-gold' },
    { min:7,  icon:'🥈', label:'Silver · 7+ days',    cls:'sb-silv' },
    { min:3,  icon:'🥉', label:'Bronze · 3+ days',    cls:'sb-brnz' },
    { min:1,  icon:'🌱', label:'Getting started',     cls:'sb-strt' },
    { min:0,  icon:'💫', label:'Begin today',         cls:'sb-none' },
  ];

  // Local calendar day (YYYY-MM-DD); toISOString() is UTC and is off by a day for IST users
  function localDay(d) {
    const x = (d instanceof Date) ? d : new Date(d);
    if (isNaN(x)) return '';
    return x.getFullYear() + '-' + String(x.getMonth() + 1).padStart(2, '0') + '-' + String(x.getDate()).padStart(2, '0');
  }

  // ── Init (called on DOMContentLoaded) ──────────────────────────────────────
  function init() {
    _renderMicro();
    _renderBadge();
  }

  // ── Storage helpers ────────────────────────────────────────────────────────
  function _load() {
    try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || '[]'); } catch { return []; }
  }
  function _save(sessions) {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.slice(0, 100)));
  }

  // ── Streak computation ─────────────────────────────────────────────────────
  function computeStreak(sessions) {
    if (!sessions.length) return 0;
    const days = [...new Set(sessions.map(s => s.date ? localDay(s.date) : '').filter(Boolean))].sort().reverse();
    const today = localDay(new Date());
    const yest  = localDay(new Date(Date.now()-864e5));
    if (days[0] !== today && days[0] !== yest) return 0;
    let streak = 1;
    for (let i = 1; i < days.length; i++) {
      if (Math.round((new Date(days[i-1]) - new Date(days[i])) / 864e5) === 1) streak++;
      else break;
    }
    return streak;
  }

  function _badge(n) {
    return BADGES.find(b => n >= b.min) || BADGES[BADGES.length-1];
  }

  function _todayLogged(sessions) {
    const today = localDay(new Date());
    return sessions.some(s => s.date && localDay(s.date) === today);
  }

  // ── Log a quick mood ───────────────────────────────────────────────────────
  function logMood(index) {
    const mood = MOODS[index];
    if (!mood) return;

    // Button tap animation
    const btn = document.querySelector(`.micro-btn[data-i="${index}"]`);
    if (btn) { btn.classList.add('micro-tapped'); setTimeout(() => btn.classList.remove('micro-tapped'), 700); }

    const sessions = _load();
    const entry = {
      id: Date.now(), riskLevel: mood.risk, score: mood.score, emotion: mood.emotion,
      name: 'Me', snippet: `Quick mood: ${mood.label} ${mood.emoji}`,
      date: new Date().toISOString(), ts: Date.now(), source: 'micro'
    };
    sessions.unshift(entry);
    _save(sessions);

    const streak = computeStreak(sessions);

    // Show success state
    const widget = document.getElementById('micro-widget');
    if (widget) {
      widget.innerHTML = `
        <div class="micro-success">
          <div class="micro-success-e">${mood.emoji}</div>
          <div>
            <p class="micro-success-txt">Logged as <strong>${mood.label}</strong></p>
            <p class="micro-success-streak">🔥 ${streak} day${streak !== 1 ? 's' : ''} in a row</p>
          </div>
        </div>`;
      // Restore after 2.5s
      setTimeout(() => { _renderMicro(); _renderBadge(); _syncDashboard(); }, 2500);
    }

    _renderBadge();
    _syncDashboard();

    // Show toast for milestone streaks
    if (streak === 3)  _toast('🥉 Bronze badge unlocked — 3 day streak!');
    if (streak === 7)  _toast('🥈 Silver badge unlocked — 7 day streak!');
    if (streak === 21) _toast('🥇 Gold badge unlocked — 21 day streak!');
    if (streak === 30) _toast('🏆 Platinum badge unlocked — 30 day streak!');
  }

  // ── Sync existing dashboard streak elements ────────────────────────────────
  function _syncDashboard() {
    const sessions = _load();
    const streak = computeStreak(sessions);

    // Top stats bar
    const v1 = document.getElementById('stat-streak-val');
    if (v1) v1.textContent = streak;

    // Stats card
    const v2 = document.getElementById('stat-streak-v2');
    if (v2) v2.textContent = streak;

    const lbl = document.getElementById('stat-streak-label');
    if (lbl) lbl.textContent = streak > 0 ? `🔥 ${streak} days` : '🔥 days';

    // Trigger dashboard refresh if available
    if (typeof dashboard !== 'undefined' && typeof dashboard.refresh === 'function') {
      dashboard.refresh();
    }
  }

  // ── Render micro check-in widget ───────────────────────────────────────────
  function _renderMicro() {
    const el = document.getElementById('micro-widget');
    if (!el) return;

    const sessions = _load();

    if (_todayLogged(sessions)) {
      const streak = computeStreak(sessions);
      el.innerHTML = `
        <div class="micro-done">
          <span class="micro-done-tick">✓</span>
          <div class="micro-done-text">
            <p class="micro-done-title">Checked in today</p>
            <p class="micro-done-sub">Come back tomorrow to keep your streak alive!</p>
          </div>
          <div class="micro-streak-pill">🔥 ${streak}</div>
        </div>`;
    } else {
      el.innerHTML = `
        <p class="micro-q">How are you feeling right now?</p>
        <div class="micro-row">
          ${MOODS.map((m, i) => `
            <button class="micro-btn" data-i="${i}" onclick="StreakSystem.logMood(${i})" title="${m.label}">
              <span class="micro-e">${m.emoji}</span>
              <span class="micro-lbl">${m.label}</span>
            </button>`).join('')}
        </div>`;
    }
  }

  // ── Render streak badge widget ─────────────────────────────────────────────
  function _renderBadge() {
    const el = document.getElementById('streak-badge-widget');
    if (!el) return;
    const sessions = _load();
    const streak = computeStreak(sessions);
    const b = _badge(streak);
    el.innerHTML = `
      <div class="sbw-row">
        <span class="sbw-fire ${streak > 0 ? 'sbw-fire-on' : ''}">🔥</span>
        <div>
          <span class="sbw-num">${streak}</span>
          <span class="sbw-days">day${streak !== 1 ? 's' : ''}</span>
        </div>
      </div>
      <div class="sbw-badge ${b.cls}">${b.icon} ${b.label}</div>`;
  }

  // ── Toast ──────────────────────────────────────────────────────────────────
  function _toast(msg) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = `position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
      background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;padding:12px 22px;
      border-radius:12px;z-index:9999;font-size:.88rem;font-weight:700;
      box-shadow:0 8px 32px rgba(99,102,241,.5);max-width:90vw;text-align:center;
      font-family:'Nunito',sans-serif;letter-spacing:.02em;`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  }

  // Public API
  return { init, logMood, computeStreak, refresh: () => { _renderMicro(); _renderBadge(); } };

})();

// Auto-init
document.addEventListener('DOMContentLoaded', () => StreakSystem.init());
