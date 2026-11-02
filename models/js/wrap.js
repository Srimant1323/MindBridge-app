/**
 * wrap.js — Monthly Emotional Wrap (Spotify-style) v1
 *
 * Reads this month's sessions from localStorage and presents a cinematic
 * 8-slide animated story: check-in count → dominant emotion → calmest day
 * → hardest day → top words → streak → affirmation.
 *
 * Zero external dependencies. Injects its own CSS. Creates its own DOM overlay.
 * Session schema: { id, riskLevel, score, emotion, name, snippet, date, ts }
 */
const WrapEngine = (function () {

  const SESSIONS_KEY = 'mindbridge_sessions';
  const STOP = new Set(['i','the','a','and','is','was','my','me','to','it','of','in',
    'that','this','but','so','for','on','are','with','have','been','not','do','be',
    'had','he','she','they','we','you','your','his','her','its','our','their','just',
    'am','an','at','by','if','as','or','from','up','about','into','can','will',
    'more','also','than','then','very','even','no','yes','all','out','when','what',
    'how','who','which','after','before','because','him','them','feel','felt','get',
    'got','like','know','think','want','need','going','would','could','should','day',
    'time','really','much','every','some','too','still','dont','cant','im','ive']);

  const EC = {
    Joy:      { e:'😊', g:['#FBBF24','#F59E0B'], label:'Joy' },
    Sadness:  { e:'😢', g:['#818CF8','#6366F1'], label:'Sadness' },
    Fear:     { e:'😰', g:['#A78BFA','#7C3AED'], label:'Fear' },
    Anger:    { e:'😤', g:['#FB923C','#EF4444'], label:'Anger' },
    Burnout:  { e:'🔥', g:['#F97316','#DC2626'], label:'Burnout' },
    Neutral:  { e:'😌', g:['#6366F1','#8B5CF6'], label:'Calm' },
    Love:     { e:'❤️', g:['#FB7185','#EC4899'], label:'Love' },
    Surprise: { e:'😮', g:['#34D399','#10B981'], label:'Surprise' }
  };

  const AFF = {
    Joy:      'Keep nurturing what brings you joy — it\'s the quiet foundation of resilience.',
    Sadness:  'Every wave of sadness passes. You showed up this month, and that takes real strength.',
    Fear:     'Naming your fears is the first step to understanding them. You\'re braver than you know.',
    Anger:    'Your feelings are valid. Being aware of them — rather than pushing them away — is how healing starts.',
    Burnout:  'Recognizing burnout before it breaks you is a kind of wisdom. Even one moment of rest matters.',
    Neutral:  'Steady and grounded — that quiet kind of stability is easy to overlook, but it\'s a real strength.',
    Love:     'The connections you nurture are your superpower. Keep showing up for the people you love.',
    Surprise: 'Life surprised you this month. You adapted, and you\'re still here. That\'s everything.'
  };

  let cssInjected = false;
  let current = 0;
  let slides = [];
  let overlay = null;
  let touchX = 0;

  // Local calendar day (YYYY-MM-DD); toISOString() is UTC and is off by a day for IST users
  function localDay(d) {
    const x = (d instanceof Date) ? d : new Date(d);
    if (isNaN(x)) return '';
    return x.getFullYear() + '-' + String(x.getMonth() + 1).padStart(2, '0') + '-' + String(x.getDate()).padStart(2, '0');
  }

  // ── Entry point ────────────────────────────────────────────────────────────
  function launch() {
    const all = _load();
    const now  = new Date();
    const monthSessions = all.filter(s => {
      const d = new Date(s.date);
      return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth();
    });

    if (monthSessions.length < 2) {
      _toast('✨ Check in at least 2 times this month to unlock your Emotional Wrap!');
      return;
    }

    _injectCSS();
    const data = _analyze(monthSessions, all, now);
    slides = _buildSlides(data, now);
    _mount();
    _show(0);
  }

  // ── Data analysis ──────────────────────────────────────────────────────────
  function _load() {
    try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || '[]'); } catch { return []; }
  }

  function _analyze(ms, all, now) {
    // Dominant emotion
    const eCount = {};
    ms.forEach(s => { const e = s.emotion || 'Neutral'; eCount[e] = (eCount[e] || 0) + 1; });
    const dominant = Object.entries(eCount).sort((a,b) => b[1]-a[1])[0]?.[0] || 'Neutral';

    // Best / hardest day
    const sorted = [...ms].sort((a,b) => (a.score||0) - (b.score||0));

    // Top words from snippets
    const freq = {};
    ms.forEach(s => {
      (s.snippet || '').toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/)
        .filter(w => w.length > 3 && !STOP.has(w))
        .forEach(w => { freq[w] = (freq[w]||0) + 1; });
    });
    const topWords = Object.entries(freq).sort((a,b) => b[1]-a[1]).slice(0,3).map(e => e[0]);

    // Streak
    const streak = _streak(all);
    const avgScore = Math.round(ms.reduce((s,e) => s+(e.score||0), 0) / ms.length);

    return { count: ms.length, dominant, eCount, calmest: sorted[0],
             hardest: sorted[sorted.length-1], topWords, streak, avgScore,
             affirmation: AFF[dominant] || AFF.Neutral };
  }

  function _streak(sessions) {
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

  // ── Build slide configs ────────────────────────────────────────────────────
  function _buildSlides(d, now) {
    const ec = EC[d.dominant] || EC.Neutral;
    const mn = now.toLocaleString('default', { month:'long' });
    const fmt = s => s?.date ? new Date(s.date).toLocaleDateString('en-IN', { weekday:'long', day:'numeric', month:'short' }) : '—';
    const snip = s => s?.snippet ? `"${s.snippet.replace(/Quick mood:/,'').trim().slice(0,70)}…"` : 'A memorable moment.';
    const badgeLabel = n => n>=30?'🏆 Platinum':n>=21?'🥇 Gold':n>=7?'🥈 Silver':n>=3?'🥉 Bronze':n>=1?'🌱 Getting started':'💫 Start today';
    const badgeCls   = n => n>=30?'wb-plat':n>=21?'wb-gold':n>=7?'wb-silv':n>=3?'wb-brnz':'wb-start';

    return [
      // Slide 0 — Title
      { type:'title', bg:'multi', html:`
        <div id="wparts"></div>
        <div class="wy-eyebrow">✨ Your Monthly Wrap</div>
        <h1 class="wy-title">${mn}</h1>
        <p class="wy-sub">A story only you can tell</p>
        <p class="wy-hint">Tap anywhere to begin →</p>
      `},
      // Slide 1 — Check-ins
      { type:'count', bg:ec.g, html:`
        <div class="wy-icon">📋</div>
        <p class="wy-label">You showed up</p>
        <div class="wy-big" id="wy-counter" data-t="${d.count}">0</div>
        <p class="wy-sub">times in ${mn}</p>
        <p class="wy-note">Every check-in is an act of self-awareness.</p>
      `},
      // Slide 2 — Dominant emotion
      { type:'emotion', bg:ec.g, html:`
        <p class="wy-label">Your most felt emotion was</p>
        <div class="wy-emotion-e" id="wy-emj">${ec.e}</div>
        <div class="wy-emotion-name">${ec.label}</div>
        <p class="wy-note">${d.eCount[d.dominant]} out of ${d.count} sessions</p>
      `},
      // Slide 3 — Calmest
      { type:'day', bg:['#34D399','#059669'], html:`
        <div class="wy-icon">🌿</div>
        <p class="wy-label">Your calmest moment was</p>
        <div class="wy-day">${fmt(d.calmest)}</div>
        <div class="wy-day-score">Wellbeing score: ${d.calmest?.score ?? '—'}</div>
        <p class="wy-note">${snip(d.calmest)}</p>
      `},
      // Slide 4 — Hardest
      { type:'day', bg:['#FB923C','#DC2626'], html:`
        <div class="wy-icon">⚡</div>
        <p class="wy-label">Your most intense moment was</p>
        <div class="wy-day">${fmt(d.hardest)}</div>
        <div class="wy-day-score">Wellbeing score: ${d.hardest?.score ?? '—'}</div>
        <p class="wy-note">You faced it. That's what matters.</p>
      `},
      // Slide 5 — Top words
      { type:'words', bg:ec.g, html:`
        <div class="wy-icon">💬</div>
        <p class="wy-label">Words that defined your ${mn}</p>
        <div class="wy-words">
          ${(d.topWords.length ? d.topWords : ['your story','this month','matters'])
            .map(w => `<div class="wy-chip">${String(w).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}</div>`).join('')}
        </div>
      `},
      // Slide 6 — Streak
      { type:'streak', bg:['#FBBF24','#F59E0B'], html:`
        <p class="wy-label">Your consistency streak</p>
        <div class="wy-fire">🔥</div>
        <div class="wy-big wy-dark" id="wy-streak" data-t="${d.streak}">0</div>
        <p class="wy-sub wy-dark">${d.streak === 1 ? 'day in a row' : 'days in a row'}</p>
        <div class="wy-badge ${badgeCls(d.streak)}">${badgeLabel(d.streak)}</div>
      `},
      // Slide 7 — Affirmation + close
      { type:'final', bg:ec.g, html:`
        <div class="wy-fin-e">${ec.e}</div>
        <p class="wy-label">A message for you</p>
        <p class="wy-aff">"${d.affirmation}"</p>
        <div class="wy-fin-btns">
          <button class="wy-share-btn" onclick="WrapEngine._share()">📸 Screenshot &amp; Share</button>
          <button class="wy-done-btn"  onclick="WrapEngine.close()">Done ✕</button>
        </div>
      `}
    ];
  }

  // ── Mount overlay ──────────────────────────────────────────────────────────
  function _mount() {
    overlay?.remove();
    overlay = document.createElement('div');
    overlay.className = 'wy-overlay';
    overlay.innerHTML = `
      <div class="wy-prog" id="wy-prog"></div>
      <button class="wy-x" onclick="WrapEngine.close()">✕</button>
      <div class="wy-stage" id="wy-stage"></div>
      <div class="wy-dots" id="wy-dots">
        ${slides.map((_,i)=>`<div class="wy-dot" id="wy-d${i}"></div>`).join('')}
      </div>`;
    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden';

    overlay.addEventListener('click', e => {
      if (!e.target.closest('button')) _next();
    });
    document.addEventListener('keydown', _onKey);
    overlay.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive:true });
    overlay.addEventListener('touchend', e => {
      const dx = e.changedTouches[0].clientX - touchX;
      if (dx < -40) _next(); else if (dx > 40) _prev();
    });
    requestAnimationFrame(() => overlay.classList.add('wy-in'));
  }

  function _onKey(e) {
    if (!overlay) return;
    if (e.key === 'Escape') { close(); return; }
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); _next(); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); _prev(); }
  }

  // ── Show slide ─────────────────────────────────────────────────────────────
  function _show(idx) {
    current = Math.max(0, Math.min(slides.length-1, idx));
    const sl = slides[current];
    const stage = document.getElementById('wy-stage');
    if (!stage) return;

    stage.innerHTML = `<div class="wy-slide wy-slide-enter" style="background:${_grad(sl.bg)}">
      <div class="wy-content">${sl.html}</div></div>`;

    requestAnimationFrame(() => {
      const s = stage.querySelector('.wy-slide');
      if (s) { s.classList.remove('wy-slide-enter'); s.classList.add('wy-slide-on'); }
    });

    const prog = document.getElementById('wy-prog');
    if (prog) prog.style.width = `${((current+1)/slides.length)*100}%`;

    document.querySelectorAll('.wy-dot').forEach((d,i) => d.classList.toggle('wy-dot-on', i===current));

    // Slide animations
    if (sl.type === 'title')  setTimeout(_particles, 50);
    if (sl.type === 'count')  setTimeout(() => _counter('wy-counter'), 350);
    if (sl.type === 'streak') setTimeout(() => _counter('wy-streak'), 350);
    if (sl.type === 'emotion') setTimeout(() => document.getElementById('wy-emj')?.classList.add('wy-pulse'), 300);
    if (sl.type === 'words')  setTimeout(() => document.querySelectorAll('.wy-chip').forEach(c => c.classList.add('wy-chip-in')), 200);
  }

  function _grad(bg) {
    if (bg === 'multi') return 'linear-gradient(135deg,#1a1033 0%,#2d1b69 30%,#0d2240 70%,#1a1033 100%)';
    return `linear-gradient(135deg,${bg[0]} 0%,${bg[1]} 100%)`;
  }

  function _next() { if (current < slides.length-1) _show(current+1); }
  function _prev() { if (current > 0) _show(current-1); }

  // ── Animated counter ───────────────────────────────────────────────────────
  function _counter(id) {
    const el = document.getElementById(id);
    if (!el) return;
    const target = parseInt(el.dataset.t || '0');
    if (!target) { el.textContent = '0'; return; }
    const start = performance.now();
    const dur = 1200;
    const raf = now => {
      const p = Math.min((now-start)/dur, 1);
      el.textContent = Math.round((1-Math.pow(1-p,3)) * target);
      if (p < 1) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
  }

  // ── Floating particles on title slide ─────────────────────────────────────
  function _particles() {
    const c = document.getElementById('wparts');
    if (!c) return;
    for (let i = 0; i < 28; i++) {
      const p = document.createElement('div');
      p.className = 'wy-par';
      p.style.cssText = `left:${Math.random()*100}%;top:${Math.random()*100}%;
        width:${2+Math.random()*5}px;height:${2+Math.random()*5}px;
        animation-delay:${Math.random()*4}s;animation-duration:${3+Math.random()*4}s;`;
      c.appendChild(p);
    }
  }

  function _share() { _toast('📸 Take a screenshot to share your Emotional Wrap!'); }

  function close() {
    if (!overlay) return;
    overlay.classList.remove('wy-in');
    document.body.style.overflow = '';
    document.removeEventListener('keydown', _onKey);
    setTimeout(() => { overlay?.remove(); overlay = null; }, 400);
  }

  function _toast(msg) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = `position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
      background:#6366f1;color:#fff;padding:12px 22px;border-radius:12px;z-index:99999;
      font-size:.88rem;font-weight:600;box-shadow:0 8px 32px rgba(0,0,0,.4);
      max-width:90vw;text-align:center;font-family:'Nunito',sans-serif;`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3500);
  }

  // ── Inject CSS once ────────────────────────────────────────────────────────
  function _injectCSS() {
    if (cssInjected) return; cssInjected = true;
    const s = document.createElement('style');
    s.textContent = `
.wy-overlay{position:fixed;inset:0;z-index:9990;display:flex;flex-direction:column;
  align-items:center;justify-content:center;opacity:0;transition:opacity .4s ease;
  font-family:'Nunito','Calibri',sans-serif;}
.wy-in{opacity:1;}
.wy-prog{position:absolute;top:0;left:0;height:3px;background:rgba(255,255,255,.75);
  transition:width .5s ease;z-index:2;border-radius:0 2px 2px 0;}
.wy-x{position:absolute;top:16px;right:18px;z-index:3;background:rgba(255,255,255,.15);
  border:1px solid rgba(255,255,255,.25);color:#fff;width:36px;height:36px;border-radius:50%;
  font-size:.9rem;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .2s;}
.wy-x:hover{background:rgba(255,255,255,.3);}
.wy-stage{width:100%;height:100%;position:relative;overflow:hidden;}
.wy-slide{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  opacity:0;transform:translateX(60px) scale(.97);transition:opacity .45s ease,transform .45s ease;}
.wy-slide-enter{opacity:0;transform:translateX(60px) scale(.97);}
.wy-slide-on{opacity:1;transform:translateX(0) scale(1);}
.wy-content{text-align:center;color:#fff;padding:40px 24px;max-width:500px;width:100%;position:relative;z-index:1;}
.wy-eyebrow{font-size:.78rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;opacity:.7;margin-bottom:12px;}
.wy-title{font-size:clamp(3rem,12vw,5.5rem);font-weight:900;line-height:1.05;margin:0 0 12px;
  background:linear-gradient(135deg,#fff 0%,rgba(255,255,255,.7) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.wy-sub{font-size:1.05rem;opacity:.75;margin:0 0 32px;}
.wy-hint{font-size:.8rem;opacity:.45;margin-top:40px;letter-spacing:.05em;}
.wy-label{font-size:.98rem;opacity:.82;margin-bottom:12px;font-weight:600;}
.wy-note{font-size:.8rem;opacity:.6;margin-top:16px;line-height:1.6;}
.wy-icon{font-size:3rem;margin-bottom:14px;display:block;}
.wy-big{font-size:clamp(4rem,18vw,7rem);font-weight:900;line-height:1;text-shadow:0 4px 24px rgba(0,0,0,.2);}
.wy-dark{color:#1a1033;}
.wy-emotion-e{font-size:clamp(4.5rem,20vw,7rem);display:block;margin:12px auto 16px;transition:transform .6s ease;}
.wy-emotion-e.wy-pulse{animation:wy-pulse 1.5s ease-in-out infinite;}
@keyframes wy-pulse{0%,100%{transform:scale(1);}50%{transform:scale(1.12);}}
.wy-emotion-name{font-size:clamp(2rem,8vw,3.5rem);font-weight:900;letter-spacing:-.02em;}
.wy-day{font-size:clamp(1.5rem,5vw,2.2rem);font-weight:900;margin:8px 0 4px;}
.wy-day-score{font-size:.82rem;opacity:.65;margin-bottom:8px;}
.wy-words{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin-top:20px;}
.wy-chip{background:rgba(255,255,255,.15);border:1.5px solid rgba(255,255,255,.3);
  border-radius:50px;padding:10px 22px;font-size:clamp(1rem,3vw,1.3rem);font-weight:800;
  opacity:0;transform:translateY(16px);transition:opacity .4s ease,transform .4s ease;}
.wy-chip.wy-chip-in{opacity:1;transform:translateY(0);}
.wy-chip:nth-child(1){transition-delay:.1s;}.wy-chip:nth-child(2){transition-delay:.3s;}.wy-chip:nth-child(3){transition-delay:.55s;}
.wy-fire{font-size:3.5rem;display:block;margin-bottom:8px;animation:wy-fire .8s ease-in-out infinite alternate;}
@keyframes wy-fire{from{transform:scale(1) rotate(-3deg);}to{transform:scale(1.1) rotate(3deg);}}
.wy-badge{display:inline-block;padding:8px 20px;border-radius:50px;font-size:.8rem;font-weight:800;margin-top:14px;letter-spacing:.04em;}
.wb-plat{background:linear-gradient(135deg,#e2e8f0,#94a3b8);color:#1e293b;}
.wb-gold{background:linear-gradient(135deg,#fbbf24,#f59e0b);color:#1a1033;}
.wb-silv{background:linear-gradient(135deg,#e2e8f0,#cbd5e1);color:#1e293b;}
.wb-brnz{background:linear-gradient(135deg,#c97d4e,#a0522d);color:#fff;}
.wb-start{background:rgba(255,255,255,.2);color:#fff;}
.wy-fin-e{font-size:3.5rem;display:block;margin-bottom:8px;animation:wy-float 3s ease-in-out infinite;}
@keyframes wy-float{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}
.wy-aff{font-size:clamp(.95rem,3vw,1.15rem);line-height:1.75;font-style:italic;opacity:.95;
  background:rgba(255,255,255,.12);border-radius:12px;padding:18px 20px;margin:12px 0 24px;
  border-left:3px solid rgba(255,255,255,.4);text-align:left;}
.wy-fin-btns{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;}
.wy-share-btn,.wy-done-btn{padding:12px 22px;border-radius:50px;border:none;
  font-size:.88rem;font-weight:700;cursor:pointer;transition:transform .15s,box-shadow .15s;font-family:inherit;}
.wy-share-btn{background:rgba(255,255,255,.95);color:#1a1033;}
.wy-done-btn{background:rgba(255,255,255,.15);color:#fff;border:1.5px solid rgba(255,255,255,.3);}
.wy-share-btn:hover,.wy-done-btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.2);}
.wy-dots{position:absolute;bottom:22px;left:50%;transform:translateX(-50%);display:flex;gap:8px;z-index:2;}
.wy-dot{width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.3);transition:all .3s ease;}
.wy-dot-on{background:#fff;transform:scale(1.35);}
.wy-par{position:absolute;border-radius:50%;background:rgba(255,255,255,.5);animation:wy-par linear infinite;}
#wparts{position:absolute;inset:0;overflow:hidden;pointer-events:none;}
@keyframes wy-par{0%{opacity:.6;transform:translateY(0) scale(1);}60%{opacity:.9;}100%{opacity:0;transform:translateY(-130px) scale(.5);}}
@media(max-width:480px){
  .wy-content{padding:32px 18px;}
  .wy-fin-btns{flex-direction:column;}
  .wy-share-btn,.wy-done-btn{width:100%;}
}`;
    document.head.appendChild(s);
  }

  return { launch, close, _share };
})();
