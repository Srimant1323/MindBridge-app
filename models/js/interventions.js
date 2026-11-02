/**
 * interventions.js — MindBridge v3 Support Tools Module
 * Breathing exercises, 5-4-3-2-1 grounding, Indian helplines, affirmations.
 * Enhanced with SVG progress ring and grounding step tracker.
 */
const interventions = (function () {

  // ── Indian Helplines ──────────────────────────────────────────────────────
  const HELPLINES = [
    { name: 'Tele-MANAS (Govt. of India)', desc: 'Free national tele-mental-health service', number: '14416', hours: '24/7' },
    { name: 'iCall',               desc: 'TISS — Counselling & psychological support',   number: '9152987821',  hours: 'Mon–Sat, 8am–10pm' },
    { name: 'Vandrevala Foundation', desc: 'Free 24/7 mental health helpline',            number: '1860-2662-345', hours: '24/7' },
    { name: 'NIMHANS',             desc: 'National Institute of Mental Health, Bengaluru', number: '080-46110007', hours: 'Mon–Sat, 9am–5pm' },
    { name: 'Sneha Foundation (Chennai)', desc: 'Emotional support & suicide prevention',         number: '044-24640050', hours: '24/7' },
    { name: 'Fortis Stress Helpline', desc: 'Psychological support in Hindi & English',   number: '8376804102',  hours: '24/7' }
  ];

  // ── Affirmations ──────────────────────────────────────────────────────────
  const AFFIRMATIONS = [
    '"I am worthy of love, support, and healing."',
    '"This feeling is temporary. I have survived difficult moments before."',
    '"I do not need to have everything figured out today. One step is enough."',
    '"My struggles do not define me. My resilience does."',
    '"It is okay to ask for help. Reaching out is an act of courage."',
    '"I am allowed to take up space and express my needs."',
    '"My emotions are valid, even when they are difficult."',
    '"I am more than my worst days. Growth is possible."',
    '"Every breath I take is a new beginning."',
    '"I am not alone. Others care about my wellbeing."',
    '"Healing is not linear. Setbacks are part of the journey."',
    '"I choose to be gentle with myself today."',
    '"Small progress is still progress. I celebrate each step."',
    '"I deserve the same compassion I give to others."',
    '"My mind and body are working to protect me, even when it doesn\'t feel that way."'
  ];

  let affirmationIndex = Math.floor(Math.random() * AFFIRMATIONS.length);

  // ── 4-7-8 Breathing ───────────────────────────────────────────────────────
  const BREATH_PHASES = [
    { name: 'Inhale',  seconds: 4, instruction: 'Breathe in slowly through your nose',  cssClass: 'inhale' },
    { name: 'Hold',    seconds: 7, instruction: 'Hold your breath gently',               cssClass: 'hold'   },
    { name: 'Exhale',  seconds: 8, instruction: 'Exhale fully through your mouth',       cssClass: 'exhale' }
  ];

  const RING_CIRCUMFERENCE = 2 * Math.PI * 54; // ≈ 339.3

  let breathInterval  = null;
  let breathPhaseIdx  = 0;
  let breathProgress  = 0;
  let breathTickMs    = 100;

  function startBreathing() {
    const stage    = document.getElementById('breathing-stage');
    const startBtn = document.getElementById('start-breathing-btn');
    if (!stage || !startBtn) return;

    stage.style.display = 'flex';
    startBtn.style.display = 'none';

    breathPhaseIdx = 0;
    breathProgress = 0;
    renderBreathPhase();

    breathInterval = setInterval(breathTick, breathTickMs);
    app.toast('Starting 4-7-8 breathing cycle…');
  }

  function breathTick() {
    const phase = BREATH_PHASES[breathPhaseIdx];
    breathProgress += breathTickMs / 1000;

    const pct    = Math.min(breathProgress / phase.seconds, 1);
    const offset = RING_CIRCUMFERENCE * (1 - pct);

    const ring = document.getElementById('breath-ring-circle');
    if (ring) ring.style.strokeDashoffset = offset;

    const remaining = Math.ceil(phase.seconds - breathProgress);
    const countEl   = document.getElementById('breath-count');
    if (countEl) countEl.textContent = Math.max(remaining, 0);

    if (breathProgress >= phase.seconds) {
      breathPhaseIdx = (breathPhaseIdx + 1) % BREATH_PHASES.length;
      breathProgress = 0;
      renderBreathPhase();
    }
  }

  function renderBreathPhase() {
    const phase   = BREATH_PHASES[breathPhaseIdx];
    const circle  = document.getElementById('breath-circle');
    const phaseEl = document.getElementById('breath-phase');
    const countEl = document.getElementById('breath-count');
    const instrEl = document.getElementById('breath-instruction');
    const ring    = document.getElementById('breath-ring-circle');

    if (circle)  circle.className  = 'breath-circle ' + phase.cssClass;
    if (phaseEl) phaseEl.textContent = phase.name;
    if (countEl) countEl.textContent = phase.seconds;
    if (instrEl) instrEl.textContent = phase.instruction;
    if (ring)    ring.style.strokeDashoffset = RING_CIRCUMFERENCE;

    // Ring color per phase
    if (ring) {
      const colors = { inhale: '#10b981', hold: '#8b5cf6', exhale: '#f59e0b' };
      ring.style.stroke = colors[phase.cssClass] || '#10b981';
    }
  }

  function stopBreathing() {
    clearInterval(breathInterval);
    breathInterval = null;

    const stage    = document.getElementById('breathing-stage');
    const startBtn = document.getElementById('start-breathing-btn');
    if (stage)    stage.style.display    = 'none';
    if (startBtn) startBtn.style.display = 'block';

    const circle = document.getElementById('breath-circle');
    if (circle) circle.className = 'breath-circle';

    const ring = document.getElementById('breath-ring-circle');
    if (ring) { ring.style.strokeDashoffset = RING_CIRCUMFERENCE; ring.style.stroke = 'url(#bringGrad)'; }

    app.toast('Great job! Remember: you can do this anytime you feel overwhelmed. 💚', 'success');
  }

  // ── 5-4-3-2-1 Grounding ───────────────────────────────────────────────────
  const GROUNDING_STEPS = [
    { n: 5, sense: 'see',   emoji: '👀', prompt: 'Name 5 things you can SEE right now' },
    { n: 4, sense: 'hear',  emoji: '👂', prompt: 'Name 4 things you can HEAR right now' },
    { n: 3, sense: 'touch', emoji: '✋', prompt: 'Name 3 things you can physically FEEL (touch, temperature, texture)' },
    { n: 2, sense: 'smell', emoji: '👃', prompt: 'Name 2 things you can SMELL (or remember smelling recently)' },
    { n: 1, sense: 'taste', emoji: '👅', prompt: 'Name 1 thing you can TASTE right now' }
  ];

  let groundingStepIdx   = 0;
  let groundingItemCount = 0;
  let groundingItems     = [];

  function startGrounding() {
    const stage    = document.getElementById('grounding-stage');
    const startBtn = document.getElementById('start-grounding-btn');
    if (!stage || !startBtn) return;

    stage.style.display = 'flex';
    startBtn.style.display = 'none';

    groundingStepIdx   = 0;
    groundingItemCount = 0;
    groundingItems     = [];

    updateGroundingSteps(0);
    renderGroundingStep();
    document.getElementById('grounding-items').innerHTML = '';
    app.toast('Focus on what\'s around you right now…');
  }

  function updateGroundingSteps(activeIdx) {
    document.querySelectorAll('.g-step').forEach((el, i) => {
      el.classList.remove('active', 'done');
      if (i < activeIdx)  el.classList.add('done');
      if (i === activeIdx) el.classList.add('active');
    });
  }

  function renderGroundingStep() {
    const step    = GROUNDING_STEPS[groundingStepIdx];
    const promptEl = document.getElementById('grounding-prompt');
    const inputEl  = document.getElementById('grounding-input');
    if (promptEl) promptEl.innerHTML = `${step.emoji} <strong>${step.n}</strong> — ${step.prompt}`;
    if (inputEl)  { inputEl.value = ''; inputEl.focus(); }
    groundingItemCount = 0;
  }

  function groundingNext() {
    const inputEl = document.getElementById('grounding-input');
    const itemsEl = document.getElementById('grounding-items');
    const step    = GROUNDING_STEPS[groundingStepIdx];

    const value = inputEl ? inputEl.value.trim() : '';
    if (!value) { app.toast('Please type something before continuing.', 'warn'); return; }

    groundingItems.push({ sense: step.sense, emoji: step.emoji, text: value });
    groundingItemCount++;

    const item = document.createElement('div');
    item.className = 'grounding-item';
    item.textContent = `${step.emoji} ${value}`;
    if (itemsEl) itemsEl.appendChild(item);

    if (inputEl) inputEl.value = '';

    if (groundingItemCount >= step.n) {
      const nextStep = groundingStepIdx + 1;
      if (nextStep >= GROUNDING_STEPS.length) {
        // Completed
        updateGroundingSteps(999); // all done
        const promptEl = document.getElementById('grounding-prompt');
        if (promptEl) promptEl.innerHTML = '🌿 <strong>Well done!</strong> You\'ve completed the grounding exercise. Take a deep breath and notice how you feel.';
        const nextBtn = document.getElementById('grounding-next-btn');
        if (nextBtn)  nextBtn.style.display = 'none';
        inputEl && (inputEl.style.display = 'none');
        app.toast('Grounding complete! You did great 💚', 'success');
      } else {
        groundingStepIdx = nextStep;
        updateGroundingSteps(nextStep);
        renderGroundingStep();
      }
    }
  }

  // ── Helplines Render ──────────────────────────────────────────────────────
  function renderHelplines() {
    const container = document.getElementById('helplines-list');
    if (!container) return;

    container.innerHTML = '';
    HELPLINES.forEach(h => {
      const item = document.createElement('div');
      item.className = 'helpline-item';
      item.innerHTML = `
        <div class="helpline-info">
          <div class="helpline-name">${h.name}</div>
          <div class="helpline-desc">${h.desc} · ${h.hours}</div>
        </div>
        <a href="tel:${h.number.replace(/[^0-9]/g, '')}" class="helpline-number">${h.number}</a>
      `;
      container.appendChild(item);
    });
  }

  // ── Affirmation ────────────────────────────────────────────────────────────
  function renderAffirmation() {
    const box = document.getElementById('affirmation-box');
    if (!box) return;

    // Fade out → swap → fade in
    box.style.opacity = '0';
    box.style.transform = 'scale(0.95)';
    setTimeout(() => {
      box.textContent = AFFIRMATIONS[affirmationIndex % AFFIRMATIONS.length];
      box.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      box.style.opacity = '1';
      box.style.transform = 'scale(1)';
    }, 280);
  }

  function newAffirmation() {
    affirmationIndex = (affirmationIndex + 1) % AFFIRMATIONS.length;
    renderAffirmation();
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  function init() {
    renderHelplines();
    renderAffirmation();
  }

  return {
    startBreathing,
    stopBreathing,
    startGrounding,
    groundingNext,
    newAffirmation,
    init
  };

})();
