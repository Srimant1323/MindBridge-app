/**
 * reflection.js — AI Conversation Reflection Agent v1
 *
 * NOT therapy. Guided self-reflection that:
 *   1. Reads the user's initial text from #feeling-textarea
 *   2. Quick-detects dominant emotion (no NLP overhead)
 *   3. Asks 2 empathetic follow-up questions (tailored to emotion)
 *   4. Shows an affirmation with a live visual action animation
 *   5. Concatenates all user answers into the textarea for full NLP analysis
 *
 * Architecture: Zero external dependencies. Pure vanilla JS + CSS animations.
 * All question banks and affirmations are fully offline.
 */
const ReflectionAgent = (function () {

  // ── State ─────────────────────────────────────────────────────────────────
  let _history       = [];   // [{role:'user'|'ai', text}]
  let _emotion       = 'general';
  let _qIndex        = 0;    // which follow-up question we're on
  let _active        = false;
  let _cardOpen      = true;
  let _breatheTimer  = null;

  // ── Question banks (emotion → ordered follow-up questions) ─────────────────
  // Max 2 follow-up questions per emotion for conversational brevity.
  const Q = {
    sadness: [
      "How long have you been carrying this feeling — days, weeks, or longer?",
      "Is there a specific moment or change that you think triggered this?"
    ],
    fear: [
      "What feels most uncertain or scary right now — is it something specific?",
      "When this worry comes up, does it feel overwhelming or more like background noise?"
    ],
    anger: [
      "What happened that brought this feeling up for you?",
      "What do you need most right now — space to breathe, to feel heard, or something else?"
    ],
    burnout: [
      "When did you last remember feeling rested or genuinely energized?",
      "What's been draining the most from you lately — work, relationships, or something internal?"
    ],
    low: [
      "How long have you been feeling this low — is it recent or has it been building?",
      "Is there a particular area of life — work, health, connection — that feels heaviest?"
    ],
    joy: [
      "What's been bringing you this energy today? Tell me more.",
      "Is there something you want to protect or build on from this feeling?"
    ],
    anxious: [
      "What's the thought that keeps coming back to you the most?",
      "Does this anxiety feel tied to something specific, or more like a general sense of unease?"
    ],
    general: [
      "How long have you been experiencing this?",
      "Is there a specific area of your life that feels most affected right now?"
    ]
  };

  // ── Affirmations (emotion → message + visual action type) ──────────────────
  const AFF = {
    sadness:  { text: "It takes real courage to name what you're feeling. What you're experiencing is valid — and you don't have to carry it alone.", action: 'breathe', emoji: '🌧️', color: '#818CF8' },
    fear:     { text: "Naming your fears is the first step to understanding them. You've already shown strength by being honest with yourself.", action: 'ground',  emoji: '🌱', color: '#A78BFA' },
    anger:    { text: "Your feelings are telling you something important. Being aware of them — instead of pushing them away — is how healing starts.", action: 'breathe', emoji: '🔥', color: '#FB923C' },
    burnout:  { text: "Recognizing burnout before it breaks you is wisdom. Even one small act of rest today matters more than you know.", action: 'rest',    emoji: '🌊', color: '#34D399' },
    low:      { text: "Just showing up here today — being honest about how you feel — that's a quiet kind of strength. One step at a time.", action: 'breathe', emoji: '🌙', color: '#60A5FA' },
    joy:      { text: "This positive energy is real and it belongs to you. Noticing it — and naming it — helps it last longer.", action: 'celebrate', emoji: '✨', color: '#FBBF24' },
    anxious:  { text: "You've done something important by putting words to this anxiety. Awareness is the very first step toward calm.", action: 'ground',  emoji: '🌿', color: '#6EE7B7' },
    general:  { text: "Thank you for reflecting honestly with yourself. Self-awareness is the quiet foundation that all well-being is built on.", action: 'breathe', emoji: '💫', color: '#6366F1' }
  };

  // ── Quick emotion detector (runs on textarea text, no full NLP needed) ──────
  function _detectEmotion(text) {
    const t = text.toLowerCase();
    if (/burnout|exhausted|drained|no energy|overwork|depleted|worn out|used up/.test(t)) return 'burnout';
    if (/angry|anger|furious|rage|hate|irritat|resentment|bitter|frustrated/.test(t)) return 'anger';
    if (/panic|attack|racing thought|intrusive|terrif|horror|overwhelmed/.test(t)) return 'anxious';
    if (/anxious|anxiety|scared|fear|worried|nervous|dread|apprehens/.test(t)) return 'fear';
    if (/happy|grateful|excit|joy|wonderful|great|amazing|content|blessed/.test(t)) return 'joy';
    if (/sad|depress|cry|grief|lonely|hurt|miss|empty|numb|hollow/.test(t)) return 'sadness';
    if (/tired|low|down|bored|lost|stuck|unmotivat|no motivation|letharg/.test(t)) return 'low';
    return 'general';
  }

  // ── Card toggle (expand/collapse) ─────────────────────────────────────────
  function toggleCard() {
    _cardOpen = !_cardOpen;
    const hintRow  = _el('reflection-hint-row');
    const body     = _el('reflection-body');
    const chevron  = _el('reflection-chevron');
    if (hintRow) hintRow.style.display  = _cardOpen ? '' : 'none';
    if (body && _active) body.style.display = _cardOpen ? '' : 'none';
    if (chevron) chevron.style.transform = _cardOpen ? 'rotate(0deg)' : 'rotate(-90deg)';
  }

  // ── Start reflection ───────────────────────────────────────────────────────
  function start() {
    const ta = _el('feeling-textarea');
    const initialText = (ta ? ta.value.trim() : '');

    if (!initialText || initialText.length < 10) {
      _toast('✍️ Please write a few words about how you feel first, then start reflection.', 'info');
      ta && ta.focus();
      return;
    }

    if (_active) return;
    _active = true;
    _qIndex = 0;
    _history = [{ role: 'user', text: initialText }];
    _emotion = _detectEmotion(initialText);

    // Show conversation body
    const body    = _el('reflection-body');
    const hintRow = _el('reflection-hint-row');
    const resetBtn = _el('reflection-reset-btn');
    if (body)    { body.style.display = ''; }
    if (hintRow) { hintRow.style.display = 'none'; }
    if (resetBtn){ resetBtn.style.display = ''; }

    // Clear previous messages
    const msgs = _el('reflection-messages');
    if (msgs) msgs.innerHTML = '';

    // Show user's initial text as first bubble
    _addUserBubble(initialText, false); // false = no input row needed

    // Greeting + first question after a short pause
    _showTyping();
    setTimeout(() => {
      _hideTyping();
      const greeting = _getGreeting(_emotion);
      _addAIBubble(greeting, false);      // greeting (no input yet)
      setTimeout(() => {
        _showTyping();
        setTimeout(() => {
          _hideTyping();
          _addAIBubble(Q[_emotion][0], true); // first question WITH input row
          _qIndex = 1;
        }, 900);
      }, 600);
    }, 1300);
  }

  // ── Contextual greeting by emotion ────────────────────────────────────────
  function _getGreeting(emotion) {
    const greetings = {
      sadness:  "Thank you for sharing that. I hear you — that sounds heavy.",
      fear:     "I appreciate you opening up. What you're feeling makes complete sense.",
      anger:    "That's a lot to carry. Thank you for putting it into words.",
      burnout:  "It sounds like you've been running on empty for a while. I'm glad you're here.",
      low:      "I hear you. Sometimes these feelings are hard to even put into words.",
      joy:      "That's wonderful to hear! Let's explore that a little.",
      anxious:  "That sounds really overwhelming. Thank you for sharing that.",
      general:  "Thank you for being honest about how you feel. Let me ask you something."
    };
    return greetings[emotion] || greetings.general;
  }

  // ── Handle user submitting a reflection answer ─────────────────────────────
  function _handleAnswer(text) {
    if (!text || text.trim().length < 2) return;
    const clean = text.trim();
    _history.push({ role: 'user', text: clean });

    const questions = Q[_emotion] || Q.general;

    if (_qIndex < questions.length) {
      // Ask next question
      _showTyping();
      setTimeout(() => {
        _hideTyping();
        _addAIBubble(questions[_qIndex], true);
        _qIndex++;
      }, 900 + Math.random() * 500);
    } else {
      // All questions answered → show affirmation
      _showTyping();
      setTimeout(() => {
        _hideTyping();
        _addBridgeMessage();
        setTimeout(() => _showAffirmation(), 700);
        // Combine all user answers into the textarea
        _mergeIntoTextarea();
      }, 1100);
    }
  }

  // ── Bridge message before affirmation ─────────────────────────────────────
  function _addBridgeMessage() {
    const bridges = {
      sadness:  "Thank you for opening up. What you shared matters.",
      fear:     "You've been very reflective. That takes courage.",
      anger:    "You've articulated something important here.",
      burnout:  "Recognizing all of this is a meaningful first step.",
      low:      "Thank you for being this honest with yourself.",
      joy:      "That's beautiful. Hold onto this.",
      anxious:  "You've done something powerful by naming all of this.",
      general:  "Thank you for this honest reflection."
    };
    _addAIBubble(bridges[_emotion] || bridges.general, false);
  }

  // ── Merge all user answers into the feeling textarea ──────────────────────
  function _mergeIntoTextarea() {
    const combined = _history
      .filter(m => m.role === 'user')
      .map(m => m.text)
      .join('. ');
    const ta = _el('feeling-textarea');
    const cc = _el('char-count');
    if (ta) {
      ta.value = combined.slice(0, 1000);
      if (cc) cc.textContent = ta.value.length;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  // ── Show affirmation card with visual action + CBT Socratic Reframing ────
  function _showAffirmation() {
    const msgs = _el('reflection-messages');
    if (!msgs) return;

    // Check for Beckian Cognitive Distortions in user input
    const combinedUserText = _history.filter(m => m.role === 'user').map(m => m.text).join('. ');
    if (typeof cbtAgent !== 'undefined') {
      const cbtAnalysis = cbtAgent.analyzeText(combinedUserText);
      if (cbtAnalysis.hasDistortions) {
        const cbtCard = document.createElement('div');
        cbtCard.className = 'ref-cbt-container';
        cbtCard.style.margin = '14px 0';
        cbtAgent.renderDistortionCards(cbtAnalysis, cbtCard);
        msgs.appendChild(cbtCard);
      }
    }

    const aff = AFF[_emotion] || AFF.general;

    const card = document.createElement('div');
    card.className = 'ref-affirmation-card';
    // Apply emotion color directly as inline styles (CSS var() unreliable in border/bg shorthand)
    card.style.border      = `1.5px solid ${aff.color}40`;  // hex + 25% opacity
    card.style.background  = `linear-gradient(135deg, ${aff.color}12 0%, rgba(139,92,246,0.08) 100%)`;
    card.innerHTML = `
      <div class="aff-emoji-row">${aff.emoji}</div>
      <div class="aff-action-wrap" id="aff-action-wrap">
        ${_buildActionHTML(aff.action, aff.color)}
      </div>
      <p class="aff-msg-text">${aff.text}</p>
      <div class="aff-footer">
        <span class="aff-check-icon">✓</span>
        <p class="aff-cta">Your reflection and cognitive restructuring have been added to the check-in.<br>Click <strong>Analyze My Check-in</strong> below for your full assessment.</p>
      </div>
    `;
    msgs.appendChild(card);
    setTimeout(() => card.classList.add('aff-in'), 20);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Start breathing animation if action=breathe
    if (aff.action === 'breathe') _startBreatheCycle();
    if (aff.action === 'ground')  _animateGroundSteps();
  }

  // ── Build action HTML ──────────────────────────────────────────────────────
  function _buildActionHTML(action, color) {
    switch (action) {
      case 'breathe':
        return `
          <div class="breathe-wrap">
            <div class="breathe-ring b-outer"></div>
            <div class="breathe-ring b-middle"></div>
            <div class="breathe-ring b-inner"></div>
            <div class="breathe-core">🫁</div>
          </div>
          <p class="breathe-guide" id="breathe-guide">Breathe in…</p>`;
      case 'ground':
        return `
          <div class="ground-list" id="ground-list">
            ${['👁️ 5 things you can <b>see</b>',
               '🤚 4 things you can <b>touch</b>',
               '👂 3 things you can <b>hear</b>',
               '👃 2 things you can <b>smell</b>',
               '👅 1 thing you can <b>taste</b>']
              .map((s,i) => `<div class="ground-step" style="animation-delay:${i*0.35}s">${s}</div>`)
              .join('')}
          </div>`;
      case 'rest':
        return `
          <div class="rest-wrap">
            <div class="rest-wave w1"></div>
            <div class="rest-wave w2"></div>
            <div class="rest-wave w3"></div>
          </div>
          <p class="rest-label">🌊 Let your thoughts settle like waves…</p>`;
      case 'celebrate':
        return `
          <div class="celebrate-wrap" id="celebrate-wrap">
            <div class="spark s1">✨</div><div class="spark s2">⭐</div>
            <div class="spark s3">💫</div><div class="spark s4">🌟</div>
            <div class="spark s5">✨</div><div class="spark s6">⭐</div>
          </div>`;
      default:
        return `<div class="breathe-wrap"><div class="breathe-ring b-outer"></div><div class="breathe-ring b-middle"></div><div class="breathe-ring b-inner"></div><div class="breathe-core">🌿</div></div>`;
    }
  }

  // ── Breathing cycle animation (4-4-6 pattern) ─────────────────────────────
  function _startBreatheCycle() {
    const guide = _el('breathe-guide');
    if (!guide) return;
    const phases = [
      { text: 'Breathe in…',  dur: 4000 },
      { text: 'Hold…',        dur: 4000 },
      { text: 'Breathe out…', dur: 6000 },
    ];
    let phase = 0;
    function tick() {
      if (!_el('breathe-guide')) return; // element removed (reset)
      guide.textContent = phases[phase].text;
      guide.className   = `breathe-guide phase-${phase}`;
      _breatheTimer = setTimeout(() => {
        phase = (phase + 1) % phases.length;
        tick();
      }, phases[phase].dur);
    }
    tick();
  }

  // ── Grounding 5-4-3-2-1 animation ────────────────────────────────────────
  function _animateGroundSteps() {
    // Use requestAnimationFrame to ensure elements are in the DOM
    requestAnimationFrame(() => {
      document.querySelectorAll('.ground-step').forEach((el, i) => {
        // Set CSS custom property --d so the CSS animation-delay var() picks it up
        el.style.setProperty('--d', `${i * 0.35}s`);
        el.classList.add('ground-step-animate');
      });
    });
  }

  // ── Reset reflection ───────────────────────────────────────────────────────
  function reset() {
    if (_breatheTimer) { clearTimeout(_breatheTimer); _breatheTimer = null; }
    _active = false;
    _history = [];
    _qIndex = 0;
    _emotion = 'general';

    const msgs    = _el('reflection-messages');
    const body    = _el('reflection-body');
    const hintRow = _el('reflection-hint-row');
    const resetBtn = _el('reflection-reset-btn');
    if (msgs)     msgs.innerHTML = '';
    if (body)     body.style.display = 'none';
    if (hintRow)  hintRow.style.display = '';
    if (resetBtn) resetBtn.style.display = 'none';
  }

  // ── UI: add AI chat bubble ─────────────────────────────────────────────────
  function _addAIBubble(text, withInput) {
    const msgs = _el('reflection-messages');
    if (!msgs) return;

    const div = document.createElement('div');
    div.className = 'ref-bubble ref-ai';
    div.innerHTML = `
      <div class="ref-avatar">🌿</div>
      <div class="ref-bubble-body">
        <p class="ref-bubble-text">${_esc(text)}</p>
        ${withInput ? `
        <div class="ref-input-row">
          <textarea class="ref-textarea" placeholder="Share your thoughts…" rows="2"
            onkeydown="ReflectionAgent._keyDown(event,this)"></textarea>
          <button class="ref-send-btn" onclick="ReflectionAgent._submit(this)" title="Send">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>` : ''}
      </div>`;
    msgs.appendChild(div);
    requestAnimationFrame(() => div.classList.add('ref-in'));
    div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    if (withInput) {
      setTimeout(() => {
        const ta = div.querySelector('.ref-textarea');
        if (ta) ta.focus();
      }, 200);
    }
  }

  // ── UI: add user chat bubble ───────────────────────────────────────────────
  function _addUserBubble(text, animate = true) {
    const msgs = _el('reflection-messages');
    if (!msgs) return;
    const div = document.createElement('div');
    div.className = 'ref-bubble ref-user';
    div.innerHTML = `<div class="ref-bubble-body user-body"><p class="ref-bubble-text">${_esc(text)}</p></div>`;
    msgs.appendChild(div);
    if (animate) requestAnimationFrame(() => div.classList.add('ref-in'));
    else div.classList.add('ref-in');
    div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ── UI: typing indicator ───────────────────────────────────────────────────
  function _showTyping() {
    const msgs = _el('reflection-messages');
    if (!msgs) return;
    const div = document.createElement('div');
    div.className = 'ref-bubble ref-ai ref-typing';
    div.id = 'ref-typing';
    div.innerHTML = `
      <div class="ref-avatar">🌿</div>
      <div class="ref-bubble-body">
        <div class="ref-dots"><span></span><span></span><span></span></div>
      </div>`;
    msgs.appendChild(div);
    requestAnimationFrame(() => div.classList.add('ref-in'));
    div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function _hideTyping() {
    const el = _el('ref-typing');
    if (el) el.remove();
  }

  // ── Handle key/submit from input row ──────────────────────────────────────
  function _keyDown(event, textarea) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      _submit(textarea.nextElementSibling);
    }
  }

  function _submit(btn) {
    const row  = btn ? btn.closest('.ref-input-row') : null;
    const ta   = row ? row.querySelector('.ref-textarea') : null;
    if (!ta || !ta.value.trim()) return;
    const text = ta.value.trim();
    // Disable input row so user can't re-submit
    row.style.display = 'none';
    _addUserBubble(text);
    _handleAnswer(text);
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function _el(id) { return document.getElementById(id); }
  function _esc(t) { return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  function _toast(msg, type = 'info') {
    if (typeof app !== 'undefined' && app.toast) { app.toast(msg, type); return; }
    const t = document.createElement('div');
    t.textContent = msg;
    const bg = type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#6366f1';
    t.style.cssText = `position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:${bg};color:#fff;padding:12px 20px;border-radius:10px;z-index:9999;font-size:0.88rem;box-shadow:0 4px 20px rgba(0,0,0,0.3);max-width:90vw;text-align:center`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3500);
  }

  // Public API — ALL functions called from inline onclick HTML must be exported
  return {
    start,
    reset,
    toggleCard,
    _keyDown,   // called from onkeydown in dynamically generated bubble HTML
    _submit,    // called from onclick in dynamically generated bubble HTML
    isActive: () => _active
  };

})();
