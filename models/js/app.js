/**
 * app.js — MindBridge v3 Main Controller
 * Orchestrates NLP + Emotion AI + Vocal analysis + Dashboard.
 * Composite score: 40% emotion, 40% lexicon, 20% vocal.
 */
const app = (function () {

  let currentView   = 'dashboard';
  let currentLang   = 'en';
  let menuOpen      = false;
  const BACKEND_URL = ''; // optional: set to your deployed backend URL; empty = browser-only mode

  // ── Navigation ─────────────────────────────────────────────────────────────
  function navigate(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const target = document.getElementById('view-' + viewName);
    if (target) target.classList.add('active');

    document.querySelectorAll('.sidebar-link').forEach(l => {
      l.classList.toggle('active', l.dataset.view === viewName);
    });

    currentView = viewName;
    closeMenu();

    // Scroll to top of main content
    const main = document.getElementById('main-content');
    if (main) main.scrollTo({ top: 0, behavior: 'smooth' });

    // Defer view-specific init until DOM is fully laid out
    // (requestAnimationFrame ensures display:block has been processed by the browser)
    requestAnimationFrame(() => {
      try {
        if (viewName === 'tracker')       tracker.init();
        if (viewName === 'interventions') interventions.init();
        if (viewName === 'dashboard')     dashboard.refresh();
        if (viewName === 'phq9')          phq9.init();
        if (viewName === 'journal')       journal.init();
      } catch (e) {
        console.warn('View init error for', viewName, e);
      }
    });
  }

  // ── Menu (mobile sidebar) ──────────────────────────────────────────────────
  function toggleMenu() {
    menuOpen = !menuOpen;
    document.getElementById('sidebar')?.classList.toggle('open', menuOpen);
    document.getElementById('nav-overlay')?.classList.toggle('open', menuOpen);
  }
  function closeMenu() {
    menuOpen = false;
    document.getElementById('sidebar')?.classList.remove('open');
    document.getElementById('nav-overlay')?.classList.remove('open');
  }

  // ── Language ─────────────────────────────────────────────────────────────────────
  const CHECKIN_STRINGS = {
    en: { placeholder: "I've been feeling…",       title: 'Type your feelings',       desc: 'Write freely — there are no wrong answers' },
    hi: { placeholder: 'मैं महसूस कर रहा/रही हूं…',   title: 'अपनी भावनाएं लिखें',     desc: 'स्वतंत्र रूप से लिखें — आपके शब्द निजी हैं' },
    es: { placeholder: 'Me he sentido…',               title: 'Escribe tus sentimientos',  desc: 'Escribe libremente — tus palabras son privadas' },
    fr: { placeholder: 'Je me suis senti(e)…',         title: 'Écrivez vos émotions',       desc: 'Écrivez librement — vos mots restent privés' },
    te: { placeholder: 'నేను అనుభవిస్తున్నాను…',         title: 'మీ భావాలు రాయండి',        desc: 'స్వేచ్ఛగా రాయండి — మీ మాటలు ప్రైవేట్' },
    bn: { placeholder: 'আমি অনুভব করছি…',         title: 'আপনার অনুভূতি লিখুন', desc: 'স্বাধীনভাবে লিখুন — আপনার কথা গোপন' }
  };

  function setLanguage(lang) {
    currentLang = lang;
    // Toggle active on ALL lang buttons
    document.querySelectorAll('.lang-btn').forEach(b => {
      b.classList.toggle('active', b.id === 'lang-' + lang);
    });
    const S = CHECKIN_STRINGS[lang] || CHECKIN_STRINGS.en;
    const ta    = document.getElementById('feeling-textarea');
    const title = document.getElementById('text-card-title');
    const desc  = document.getElementById('text-card-desc');
    if (ta)    ta.placeholder   = S.placeholder;
    if (title) title.textContent = S.title;
    if (desc)  desc.textContent  = S.desc;
  }

  // ── Main Analyze ───────────────────────────────────────────────────────────
  async function analyze() {
    const ta        = document.getElementById('feeling-textarea');
    const textInput = ta ? ta.value.trim() : '';
    const transcript = audio.getTranscript().trim();
    const inputText  = textInput || transcript;

    if (!inputText) {
      toast('Please type how you feel or use the microphone first.', 'warn');
      return;
    }

    let btn, btnText, spinner;
    try {

    // Get respondent name (optional)
    const nameInput = document.getElementById('respondent-name');
    const respondentName = nameInput ? nameInput.value.trim() || 'Anonymous' : 'Anonymous';

    // Get demographic profile (optional — collected from chips)
    const demoProfile     = (typeof getDemoProfile      === 'function') ? getDemoProfile()      : {};
    const lifestyleProfile= (typeof getLifestyleProfile === 'function') ? getLifestyleProfile() : {};

    // Loading state
    btn     = document.getElementById('analyze-btn');
    btnText = document.getElementById('analyze-btn-text');
    spinner = document.getElementById('btn-spinner');
    if (btn)     btn.disabled = true;
    if (btnText) btnText.textContent = 'Analyzing…';
    if (spinner) spinner.classList.remove('hidden');

    // Show beautiful loading overlay
    const overlay = document.getElementById('mb-loading-overlay');
    if (overlay) overlay.classList.add('active');
    const _step = (id, state) => {
      const el = document.getElementById(id);
      if (el) { el.classList.remove('active','done'); el.classList.add(state); }
    };
    _step('mb-step-nlp', 'active');

    await delay(400);


    // ── Run analyses in parallel ─────────────────────────────────────────────
    const vocalMetrics = audio.getMetrics();
    const facialMetrics = (typeof cameraOCR !== 'undefined' && cameraOCR.getFacialMetrics)
      ? cameraOCR.getFacialMetrics()
      : ((typeof facialAffect !== 'undefined') ? facialAffect.getMetrics() : null);

    // 1. NLP (always runs, browser-based)
    let nlpResult = null;
    try {
      const backendResult = await tryBackend(inputText);
      nlpResult = backendResult;
    } catch {
      nlpResult = NLP.analyze(inputText, currentLang);
    }
    _step('mb-step-nlp', 'done'); _step('mb-step-emotion', 'active');

    // 2. Emotion AI (async — may not be ready yet)
    let emotionResult = null;
    if (EmotionAI.isReady() && currentLang === 'en') {
      try {
        emotionResult = await EmotionAI.analyze(inputText);
      } catch (e) {
        console.warn('Emotion analysis failed:', e);
      }
    }
    _step('mb-step-emotion', 'done'); _step('mb-step-vocal', 'active');

    // 3. If BERT not ready, synthesize emotion profile from NLP score
    if (!emotionResult && nlpResult) {
      emotionResult = synthesizeEmotionFromNLP(nlpResult, inputText, demoProfile);
    }
    _step('mb-step-vocal', 'done'); _step('mb-step-composite', 'active');

    // 4. CLINICAL SAFETY OVERRIDE ─────────────────────────────────────────────
    const nlpFlaggedSevere = !!(nlpResult && (
      nlpResult.hasSevereKeyword === true ||
      nlpResult.riskLevel === 'severe'    ||
      (typeof nlpResult.score === 'number' && nlpResult.score >= 80)
    ));

    if (emotionResult && (hasCrisisLanguage(inputText) || nlpFlaggedSevere)) {
      emotionResult = applyCrisisOverride(emotionResult);
    }

    // ── Tri-Modal Composite Score ────────────────────────────────────────────
    const nlpScore     = nlpResult     ? nlpResult.score          : 0;
    const emotionScore = emotionResult ? emotionResult.emotionRisk : null;
    const vocalScore   = vocalMetrics  ? vocalMetrics.vocalScore  : null;
    const facialScore  = facialMetrics ? facialMetrics.facialScore : null;

    const composite = computeComposite(nlpScore, emotionScore, vocalScore, facialScore);
    const riskLevel = nlpResult?.riskLevel || 'low';

    // Restore button + hide overlay
    _step('mb-step-composite', 'done');
    await delay(300);
    if (overlay) overlay.classList.remove('active');
    if (btn)     btn.disabled       = false;
    if (btnText) btnText.textContent = 'Analyze My Check-in';
    if (spinner) spinner.classList.add('hidden');

    // Log session to dashboard (includes respondent name)
    const dominantEmotion = emotionResult?.dominant || null;
    dashboard.logSession(riskLevel, composite, dominantEmotion, respondentName, inputText.slice(0, 120));

    navigate('report');
    renderReport(nlpResult, emotionResult, vocalMetrics, composite, inputText, respondentName, demoProfile, lifestyleProfile, facialMetrics);
    } catch (err) {
      console.error('Analysis error:', err);
      const overlayEl = document.getElementById('mb-loading-overlay');
      if (overlayEl) overlayEl.classList.remove('active');
      if (btn)     btn.disabled = false;
      if (btnText) btnText.textContent = 'Analyze My Check-in';
      if (spinner) spinner.classList.add('hidden');
      app.toast('Something went wrong during analysis. Please try again.', 'error');
    }
  }

  // ── Tri-Modal Composite Scoring Formula ────────────────────────────────────
  // Integrates Text (40%) + Voice (30%) + Facial Affect (30%)
  function computeComposite(nlpScore, emotionScore, vocalScore, facialScore) {
    const textScore = (emotionScore !== null)
      ? Math.round(0.50 * nlpScore + 0.50 * emotionScore)
      : nlpScore;

    // Full Tri-Modal Fusion: Text (40%) + Voice (30%) + Face (30%)
    if (vocalScore !== null && facialScore !== null) {
      return Math.round(0.40 * textScore + 0.30 * vocalScore + 0.30 * facialScore);
    }
    // Dual-Modality: Text + Face (no voice)
    if (facialScore !== null) {
      return Math.round(0.60 * textScore + 0.40 * facialScore);
    }
    // Dual-Modality: Text + Voice (no face)
    if (vocalScore !== null) {
      return Math.round(0.60 * textScore + 0.40 * vocalScore);
    }
    // Uni-modal: Text only
    return textScore;
  }

  // ── Crisis Language Detector ───────────────────────────────────────────────
  // Returns true if the text contains suicide/self-harm/crisis terminology.
  // This is a safety-critical function — err on the side of caution.
  //
  // IMPORTANT: Uses REGEX patterns with wildcards (.*?) instead of rigid exact
  // phrases. Plain .includes('end my life') misses real-world variants like
  // "ending my life", "I've been thinking about ending it", "nothing left to
  // live for" (extra words in the middle break a literal substring match).
  // Regex patterns close that gap by matching the underlying idea, not one
  // exact wording.
  // Safe replacement for eval(): only a fixed set of actions is allowed.
  function runAction(action) {
    let m;
    if ((m = /^app\.navigate\('([a-z0-9_-]+)'\)$/i.exec(action))) { navigate(m[1]); return; }
    if ((m = /^window\.open\('tel:([0-9+]+)'\)$/.exec(action))) { window.location.href = 'tel:' + m[1]; return; }
    console.warn('[MindBridge] Blocked unknown action:', action);
  }

  function hasCrisisLanguage(text) {
    // Normalize: lowercase, strip punctuation noise, collapse extra whitespace
    const t = text.toLowerCase()
      .replace(/[’‘]/g, "'")
      // \p{M} keeps Unicode combining marks (e.g. Hindi matras) intact —
      // without it, Devanagari vowel signs get stripped and words like
      // "मुझे" corrupt into "म झ", breaking Hindi crisis-term matching.
      .replace(/[^\p{L}\p{M}\p{N}\s']/gu, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    const CRISIS_PATTERNS = [
      // Direct suicidal ideation (stems catch kill/killing/killed, end/ending/ended, etc.)
      /\bsuicid\w*/,
      /\bkill(ing|ed)?\s+(myself|me)\b/,
      /\bend(ing|ed)?\s+(my\s+)?(life|it\s+all|things?|everything)\b/,
      /\btake\s+(my|his|her|their)\s+(own\s+)?life\b/,
      /\bwant(ed)?\s+to\s+die\b/,
      /\bwish(ed)?\s+i\s+(was|were)\s+dead\b/,
      /\bbetter\s+off\s+dead\b/,
      /\b(don'?t|do\s+not)\s+want\s+to\s+(live|be\s+(here|alive))\b/,
      /\bno\s+(reason|point)\s+(to|in)\s+(live|living|life)\b/,
      /\bno\s+point\s+(anymore|left)\b/,
      /\bnot\s+worth\s+living\b/,
      /\blife\s+(is\s+)?not\s+worth\b/,
      /\bnothing\s+(left\s+)?to\s+live\s+for\b/,
      /\b(don'?t|do\s+not)\s+want\s+to\s+exist\b/,
      /\bgoodbye\s+forever\b/,
      /\bfinal\s+goodbye\b/,
      /\bwish(ed)?\s+i\s+(wasn'?t|weren'?t|was\s+not|were\s+not)\s+(here|born|alive)\b/,
      /\bi\s*'?m\s+done\s+(with\s+(life|everything|it\s+all))?\b.{0,20}(die|end|goodbye)/,
      /\bwant(ed)?\s+(it|everything)(\s+\w+){0,2}\s+to\s+(stop|end)\b/,
      /\bno\s+longer\s+want\s+to\s+(live|be\s+here|exist)\b/,
      /\b(plan|planning|thinking\s+about)\s+.{0,15}(kill|end|suicide|hurt)\w*/,

      // Self-harm
      /\bself[\s-]?harm\w*/,
      /\b(cut|cutting|hurt|hurting|harm|harming)\s+(myself|me)\b/,

      // Crisis / hopelessness states
      /\bcan'?t\s+(go\s+on|take\s+it\s+anymore|do\s+this\s+anymore|handle\s+this\s+anymore)\b/,
      /\bcannot\s+(go\s+on|take\s+it\s+anymore)\b/,
      /\bgiv(e|en|ing)\s+up\s+on\s+(life|everything)\b/,
      /\bgiv(e|en|ing)\s+up\b.{0,15}\b(life|living|everything)\b/,
      /\bhopeless\b.{0,20}\b(die|end|give\s*up|worthless)\b/,
      /\b(better|happier|easier)\b.{0,10}\bwithout\s+me\b|\bwithout\s+me\b.{0,25}\b(better|happier|easier)\b/,
      /\bwhen\s+i\s*'?m\s+gone\b|\bwhen\s+i\s+am\s+gone\b/,
      /\bafter\s+i'?m\s+gone\b/,

      // Euphemisms for death/dying — VERY commonly used instead of literal
      // words, especially by people who feel ashamed to say "suicide" or
      // "die" directly. These must be treated with equal weight to direct terms.
      /\bleave\s+this\s+world\b/,
      /\bgo(ing)?\s+to\s+heaven\b/,
      /\bgo(ing)?\s+to\s+a\s+better\s+place\b/,
      /\bbe\s+with\s+(god|the\s+angels)\b/,
      /\bjoin\s+.{0,20}\bin\s+heaven\b/,
      /\bmeet\s+my\s+maker\b/,
      /\bno\s+longer\s+be\s+(a\s+)?burden\b|\b(don'?t|do\s+not|won'?t)\s+want\s+to\s+be\s+a\s+burden\b|\bbe\s+a\s+burden\s+anymore\b/,
      /\bset\s+(myself\s+)?free\b.{0,20}(pain|world|life|end)/,
      /\bstop\s+(the|this)\s+pain\s+forever\b/,
      /\b(long|eternal|permanent)\s+sleep\b/,
      /\b(don'?t|do\s+not|won'?t)\s+(want\s+to\s+)?wake\s+up\s+tomorrow\b|\bnot\s+(be\s+here|wake\s+up)\s+tomorrow\b/,

      // Social isolation + unwantedness expressing desire to disappear/leave
      // These are SEPARATE independent patterns — each fires on its own.
      // "nobody likes me" alone may not be crisis, but "nobody likes me" +
      // any leave/disappear/heaven phrase in the SAME message is a crisis signal.
      // We detect both parts individually with a combined-sentence check below.
      /\bno\s*body\s+(in\s+the\s+world\s+)?(likes?|wants?|loves?|cares?\s+(about|for))\s+me\b/,
      /\bnobody\s+(in\s+the\s+world\s+)?(likes?|wants?|loves?|cares?\s+(about|for))\s+me\b/,
      /\bno\s+one\s+(likes?|wants?|loves?|cares?\s+(about|for))\s+me\b/,
      /\beveryone\s+(hates?|ignores?|avoids?)\s+me\b/,
      /\bi\s+(am|'m)\s+(completely\s+)?(alone|unwanted|unloved|worthless|invisible)\b/,
      /\bneed\s+to\s+(leave|disappear|escape|vanish|go away)\b.{0,40}(world|life|here|everyone|pain)/,
      /\bi\s+should\s+(just\s+)?(disappear|vanish|not\s+exist|cease\s+to\s+exist)\b/,
      /\bwould\s+be\s+better\s+(off\s+)?(without\s+me|if\s+i\s+(was|were|wasn'?t|weren'?t)\s+here)\b/,
      /\bno\s+one\s+would\s+(miss|care|notice)\s+(me|if\s+i\s+(was|were)\s+gone)\b/,

      // Hindi crisis terms
      /मरना|मर\s*जाना|जीना\s*नहीं|आत्महत्या|खुद\s*को\s*मारना|जीने\s*का\s*मन\s*नहीं/,
    ];

    // ── Combined sentence check ───────────────────────────────────────────────
    // Some signals are weak alone but critical together. If BOTH a social
    // isolation phrase AND a leave/disappear/afterlife phrase appear in the
    // same message, treat it as a crisis — even if neither alone matches a
    // single pattern. This catches: "nobody likes me ... I need to go to heaven"
    const hasIsolation = /\b(nobody|no\s+one|no\s*body)\b.{0,80}\b(likes?|wants?|cares?|loves?)\s+me\b/.test(t) ||
      /\b(unwanted|unloved|alone|worthless|invisible|rejected|abandoned)\b/.test(t);
    const hasLeaving = /\b(leave|leaving|go|going|escape|disappear|vanish)\b.{0,30}\b(world|life|here|everything|everyone|heaven|better\s+place)\b/.test(t) ||
      /\b(heaven|better\s+place|afterlife|angels|god)\b/.test(t) ||
      /\b(end|stop|over|done|finished)\b.{0,20}\b(pain|suffering|everything|it\s+all)\b/.test(t);
    if (hasIsolation && hasLeaving) return true;

    return CRISIS_PATTERNS.some(rx => rx.test(t));
  }

  // ── Crisis Override ────────────────────────────────────────────────────────
  // Forcibly corrects an emotion result when crisis language is detected.
  // Sets dominant to 'sadness', zeros joy/love, and raises emotionRisk.
  function applyCrisisOverride(emotionResult) {
    const corrected = { ...emotionResult };
    const emotions  = { ...(emotionResult.emotions || {}) };

    // Zero out positive emotions — they must not dominate crisis text
    emotions.joy   = 0;
    emotions.love  = Math.min(emotions.love || 0, 0.05);

    // Boost sadness and fear to reflect true distress
    emotions.sadness = Math.max(emotions.sadness || 0, 0.55);
    emotions.fear    = Math.max(emotions.fear    || 0, 0.25);

    // Re-normalize so emotions sum to 1
    const total = Object.values(emotions).reduce((a, b) => a + b, 0) || 1;
    Object.keys(emotions).forEach(k => { emotions[k] = emotions[k] / total; });

    corrected.emotions      = emotions;
    corrected.dominant      = 'sadness';   // always sadness for crisis text
    corrected.dominantScore = Math.round((emotions.sadness || 0.55) * 100);
    corrected.emotionRisk   = Math.max(emotionResult.emotionRisk || 0, 80); // floor at 80
    corrected.crisisOverride = true; // flag for UI

    return corrected;
  }

  // ── Synthesize emotion profile from NLP when BERT not ready ───────────────
  // Maps NLP score + trigger/protective word hits to realistic emotion profile.
  // IMPORTANT: Uses NLP's actual word hit results — not just raw text matching —
  // so words the lexicon already scored (e.g. joyful, energetic) are properly used.
  function synthesizeEmotionFromNLP(nlpResult, text, demoProfile) {
    const s             = nlpResult.score || 0;
    const wordHits      = nlpResult.wordHits || {};
    const allHitWords   = Object.keys(wordHits);
    const normText      = text.toLowerCase();
    const occupation    = demoProfile?.occupation || null;

    // ── CRISIS SAFETY CHECK ──────────────────────────────────────────────────
    if (hasCrisisLanguage(text)) {
      const emotions = { sadness: 0.60, fear: 0.25, anger: 0.08, surprise: 0.04, joy: 0, love: 0.03 };
      return { emotions, emotionRisk: Math.max(s, 80), dominant: 'sadness', dominantScore: 60, synthesized: true, crisisOverride: true };
    }

    // ── NEGATION DETECTION ──────────────────────────────────────────────────
    // Catches: "not feeling good", "don't feel well", "not okay", "feeling bad"
    // The NLP lexicon treats "good/well/happy" as protective but ignores "not" —
    // this block corrects that by flagging negated positives as negative.
    const NEGATION_PATTERNS = [
      /\b(not|n't|dont|don't|didn't|didnt|never|no longer)\s+(feel|feeling|doing|been|am|was|get|getting)\s+(good|well|okay|ok|great|fine|happy|alright|better)/i,
      /\b(not|n't|dont|don't|never)\s+(good|well|okay|ok|great|fine|happy|alright|better)\b/i,
      /\b(feel|feeling|felt)\s+(bad|terrible|awful|horrible|miserable|low|down|dull|flat|off|sick|worse|worst)/i,
      /\bi\s*(am|'m)?\s*not\s+(okay|ok|fine|good|well|great|happy|alright)\b/i,
      /\bnot\s+(doing\s+)?(well|good|okay|ok|great|fine)\b/i,
    ];
    const hasNegation = NEGATION_PATTERNS.some(rx => rx.test(normText));

    // ── Step 1: Determine which emotions are signalled ───────────────────────
    // PRIMARY source: NLP's own wordHits (words the lexicon already scored).
    // Words like 'joyful' score -1.5 (protective) in the lexicon — use that.
    // SECONDARY source: NLP score range and raw text patterns.

    // Count protective vs risk hits from NLP.
    // If negation is present, zero out protective sum — "not good" should NOT
    // count "good" as a positive signal. Add a risk boost to effectiveScore.
    const protectHits     = allHitWords.filter(w => (wordHits[w]?.weight || 0) < 0);
    const riskHits        = allHitWords.filter(w => (wordHits[w]?.weight || 0) > 0);
    const protectSum      = hasNegation ? 0 : protectHits.reduce((acc, w) => acc + Math.abs(wordHits[w].weight), 0);
    const riskSum         = riskHits.reduce((acc, w) => acc + wordHits[w].weight, 0);
    const effectiveScore  = hasNegation ? Math.max(s, 30) + 20 : s; // negation bumps effective risk
    const isPositive      = !hasNegation && protectSum > riskSum && s < 35;

    // Comprehensive emotion keyword sets (raw text scan as secondary check)
    const JOY_WORDS     = ['happy','happiness','joyful','joy','excited','excitement','wonderful',
                           'great','amazing','fantastic','awesome','excellent','brilliant',
                           'cheerful','elated','ecstatic','delighted','pleased','thrilled',
                           'glad','nice','positive','upbeat','lively','vibrant',
                           'energetic','energised','energized','enthusiastic','motivated',
                           'uplifted','uplifting','refreshed','rejuvenated','alive',
                           'content','contented','satisfied','fulfilled','blissful',
                           'optimistic','hopeful','confident','proud','accomplished',
                           'laughing','smiling','smiles','grateful','thankful','blessed',
                           // Achievement & success emotions
                           'achieved','achievement','success','successful','winning','won',
                           'rewarded','recognised','recognized','celebrated','thriving',
                           'flourishing','growing','progressing','improving','recovered',
                           'healed','strong','stronger','resilient','capable','able',
                           // Social joy
                           'loved','cherished','surrounded','together','bonded','belonging',
                           'connected','included','welcomed','accepted','celebrated',
                           // Spiritual & peace
                           'peaceful','peace','calm','serene','tranquil','centred','centered',
                           'mindful','present','grounded','balanced','harmonious',
                           // Indian & everyday positive
                           'feeling nice','feeling great','very happy','so happy','too good',
                           'feeling light','feeling free','relieved','relief','better now',
                           'good news','great news','excited about','looking forward',
                           'can\'t wait','pumped','on top','doing well','all good'];

    const LOVE_WORDS    = [
      'love','loved','loving','adore','cherish','care','caring',
      'affection','warm','warmth','connected','connection','belong',
      'belonging','family','friend','friendship','together','support',
      'supported','appreciated','valued','understood','accepted',
      'compassion','kind','kindness','tender','gentle','heart',
      // Relationship depth
      'bond','bonded','attachment','devoted','devoted to','miss you',
      'thinking of you','close','closeness','intimate','intimacy',
      'partner','spouse','husband','wife','girlfriend','boyfriend',
      'parents','mother','father','sibling','brother','sister','child',
      'unconditional','grateful for you','thankful for','lucky to have',
      // Social belonging
      'community','team','crew','squad','group','circle','tribe',
      'reunion','together again','home','homecoming','welcome back',
      'feel safe','feel secure','feel seen','feel heard','feel understood'
    ];

    const SADNESS_WORDS = [
      // Core sadness
      'sad','sadness','unhappy','depressed','depression','grief',
      'sorrow','cry','crying','tears','tearful','miserable','gloomy',
      'hopeless','helpless','empty','hollow','void','numb','lonely',
      'alone','lost','broken','shattered','devastated','hurt',
      'pain','ache','aching','missing','miss','mourn','mourning',
      'regret','guilt','ashamed','shame','worthless','useless',
      'failure','defeated','abandoned','rejected','unwanted',
      // Low energy / fatigue states
      'lethargy','lethargic','sluggish','listless','fatigued','fatigue',
      'exhausted','exhaustion','drained','no energy','low energy',
      'tired all the time','no motivation','lost motivation','unmotivated',
      'demotivated','apathetic','apathy','joyless','disinterested',
      'can\'t get up','no drive','flat','dull','blah','lifeless',
      'no interest','lost interest','nothing excites','not excited anymore',
      // Burnout & work stress
      'burned out','burnt out','burnout','burn out',
      'overworked','over worked','work stress','job stress',
      'work pressure','workplace stress','career stress',
      'no work-life balance','always working','can\'t switch off',
      'mentally drained','emotionally drained','emotionally exhausted',
      // Negative wellbeing expressions
      'terrible','awful','horrible','dreadful','wretched','pathetic',
      'not good','not okay','not well','not fine','not great',
      'not happy','not alright','feeling bad','feel bad',
      'feeling down','feeling low','feeling terrible','feeling awful',
      'feeling horrible','feeling sad','feeling empty','feel empty',
      'feel nothing','feel numb','feel lost','feel broken',
      'feel alone','feel like crying','want to cry','need to cry',
      // Grief and loss
      'lost someone','someone died','death','passed away','gone forever',
      'heartbreak','heartbroken','breakup','broke up','divorced','separation',
      'relationship ended','left me','left alone','left behind',
      // Existential / identity
      'meaningless','pointless','purposeless','no purpose','no meaning',
      'what\'s the point','why bother','giving up','want to give up',
      'feel like giving up','no reason to go on','no future',
      'feel invisible','feel ignored','feel overlooked','feel unheard',
      // Somatic sadness signals (body feels)
      'heavy heart','heavy feeling','chest feels heavy','weight on chest',
      'can\'t breathe','hard to breathe','tight chest','sinking feeling',
      'knot in stomach','can\'t eat','no appetite','not eating','eating too much',
      'can\'t sleep','not sleeping','sleepless','insomnia','sleep deprived',
      'sleeping too much','don\'t want to get out of bed','staying in bed',
      'headache from stress','stress headache','crying myself to sleep',
      // Indian English expressions
      'very bad feeling','not at all okay','totally down','completely lost',
      'mind is not working','brain not working','can\'t think straight',
      'no peace of mind','too much tension','mind full of tension',
      'heart is hurting','feeling very bad','feeling very sad',
      'no happiness','happiness gone','lost all hope','completely hopeless',
      'don\'t want to do anything','don\'t feel like doing anything',
      'no strength','lost all strength','no will to live','not interested in life',
      // Isolation & social pain
      'no one understands','nobody cares','no one listens','nobody there',
      'all alone','completely alone','feeling isolated','socially isolated',
      'don\'t fit in','can\'t connect','disconnected','cut off',
      'no friends','friendship ended','trust broken','betrayed','cheated'
    ];

    // Occupational stress / burnout — maps to FEAR (anxiety, dread)
    const BURNOUT_WORDS = [
      // Workaholism
      'workaholic','work addict','can\'t stop working','always at work',
      'obsessed with work','work is my life','no time for myself',
      'working non-stop','overcommitted','over-committed','all work no play',
      // Performance pressure
      'deadline pressure','constant pressure','performance pressure',
      'performance anxiety','target pressure','kpi pressure','appraisal stress',
      'under pressure','too much on plate','can\'t keep up','falling behind',
      // Job insecurity & identity
      'job insecurity','fear of failure','imposter syndrome','self doubt',
      'not good enough','never good enough','fraud','feel like a fraud',
      'people pleasing','can\'t say no','saying yes to everything',
      'can\'t disappoint','scared to fail','scared of judgement',
      // Chronic stress patterns
      'no rest','no break','no holiday','no vacation','always busy',
      'weekend also working','no personal time','sacrificing health',
      'neglecting family','missing events','missing out','fomo',
      'can\'t relax','can\'t unwind','mind always at work',
      'work from home stress','wfh stress','blurred boundaries'
    ];

    // Occupation-specific risk keywords — activated when occupation chip is selected
    const OCCUPATION_RISK = {
      student: [
        // Exams & academics
        'exam stress','exam pressure','exam anxiety','assignment','submission',
        'marks','grades','fail','failed','failed exam','result','results',
        'placement','placement stress','peer pressure','competitive','competition',
        'ragging','hostel','homesick','study pressure','academic pressure',
        'can\'t focus','can\'t study','distracted','study stress',
        // India-specific student stress
        'jee','neet','board exam','10th board','12th board','entrance exam',
        'coaching','coaching class','tuition pressure','rank','ranking',
        'scholarship','backlog','arrear','attendance','attendance shortage',
        'parent pressure','parents expectations','comparison','compared with siblings',
        'college stress','campus stress','semester','viva','project deadline',
        'internship pressure','cgpa','gpa','percentage','topper','not a topper',
        // Social
        'campus loneliness','new city','away from home','no friends in college',
        'roommate conflict','mess food','can\'t adjust'
      ],
      school_teacher: [
        'student behaviour','classroom management','lesson plan','syllabus pressure',
        'parent complaint','parent meeting','evaluation load','teaching load',
        'teacher burnout','no appreciation','underpaid','overloaded','staff politics',
        'school politics','principal pressure','extra duty','sports day','events',
        'correction work','paper correction','marking load','administrative work',
        'student discipline','abusive students','disrespect','no authority',
        'private school pressure','target admission','school management'
      ],
      college_faculty: [
        'research pressure','publish or perish','publication','journal','paper rejected',
        'grant','funding','project funding','academic politics','admin work',
        'student load','faculty burnout','phd pressure','thesis guide stress',
        'conference pressure','accreditation','naac','nirf ranking',
        'student evaluation','slow learners','remedial classes','heavy workload'
      ],
      healthcare: [
        'patient load','night shift','night duty','on call','24 hour duty',
        'emergency','emergency ward','patient death','lose a patient',
        'medical error','hospital politics','compassion fatigue',
        'healthcare burnout','doctor burnout','nurse burnout',
        'icu','ot stress','operation theatre','surgical stress',
        'long shift','double shift','no sleep after duty','standing all day',
        'covid fatigue','pandemic stress','trauma exposure','secondary trauma',
        'doctor patient conflict','patient relatives','aggressive patient',
        'insufficient resources','lack of medicines','overloaded wards',
        'intern stress','pg stress','residency stress','junior doctor'
      ],
      it_tech: [
        'sprint','scrum','agile stress','crunch','crunch time',
        'deployment','production issue','bug','critical bug','hotfix',
        'deadline','project deadline','client deadline','release pressure',
        'micromanagement','manager','toxic manager','bad manager',
        'layoff','job loss','pink slip','retrenchment','downsizing',
        'tech burnout','screen fatigue','remote work isolation','always online',
        'on-call','on-call duty','pagerduty','3am alert','server down',
        'code review stress','pr rejected','merge conflict','tech debt',
        'upskill pressure','learning pressure','certification pressure',
        'performance improvement plan','pip','put on pip'
      ],
      business: [
        'business loss','business failure','startup failed','company failed',
        'financial stress','loan','debt','emi','bank loan','overdue',
        'sales pressure','sales target','target not met','revenue pressure',
        'client pressure','client lost','customer complaint','bad review',
        'team management','team conflict','employee issue','staff problem',
        'cash flow','no cash','liquidity crisis','salary delay',
        'competition pressure','market stress','startup stress','investor pressure',
        'pitch failed','funding rejected','no growth','business stagnant'
      ],
      government: [
        'transfer','unexpected transfer','forced transfer','posting','remote posting',
        'bureaucracy','red tape','political pressure','government job stress',
        'departmental inquiry','vigilance','corruption pressure','bribe pressure',
        'workload','file work','pending files','audit stress','inspection'
      ],
      homemaker: [
        'no recognition','invisible work','household burden','family pressure',
        'ignored','feel ignored','overlooked','no appreciation at home',
        'caregiver stress','caring for elderly','caring for sick','childcare stress',
        'no support at home','husband not supportive','in-law pressure',
        'financial dependence','no income of own','identity crisis',
        'domestic violence','marital stress','isolation at home',
        'cooking stress','managing house','no time for self','no me-time'
      ],
      unemployed: [
        'no job','job search','job hunting','can\'t find job','unemployed',
        'rejection','interview failed','interview rejected','no call back',
        'no income','no salary','financial stress','money problem',
        'self-worth','identity','lost purpose','nothing to do','no routine',
        'not hired','overqualified','underqualified','gap in resume',
        'career gap','not selected','background check failed',
        'comparing with friends','friends are settled','everyone has a job',
        'dependency on family','taking money from parents','feel like burden'
      ]
    };

    // Check if any occupation-specific risk words appear in text
    const occupationRiskWords = occupation ? (OCCUPATION_RISK[occupation] || []) : [];
    const hasOccupationRisk   = occupationRiskWords.some(kw => normText.includes(kw));
    const hasBurnout          = BURNOUT_WORDS.some(kw => normText.includes(kw));

    const ANGER_WORDS = [
      // Core anger
      'angry','anger','furious','fury','rage','mad','livid','irate','enraged',
      'frustrated','frustration','irritated','irritable','irritation','annoyed',
      'hate','hatred','despise','resent','resentment','bitter','bitterness',
      'hostile','hostility','aggressive','aggression','outraged','disgusted','disgust',
      // Injustice & unfairness
      'unfair','not fair','injustice','discrimination','bias','biased',
      'treated badly','mistreated','disrespected','disrespect','humiliated',
      'bullied','bullying','harassed','harassment','targeted','victimised',
      'wrongly accused','blamed unfairly','scapegoated','taken for granted',
      // Passive frustration
      'fed up','sick of','tired of','had enough','can\'t take it anymore',
      'done with it','over it','can\'t stand','can\'t tolerate',
      'no patience','lost patience','patience is gone','enough is enough',
      // Relationship anger
      'cheated on','betrayed','lied to','deceived','manipulated',
      'toxic relationship','toxic person','gaslighting','narcissist',
      'fighting with','argument with','conflict with','disagreement',
      // Work anger
      'toxic workplace','hostile work','boss is bad','unfair treatment',
      'credit stolen','hard work not recognised','colleague problem'
    ];

    const FEAR_WORDS = [
      // Core fear
      'afraid','fear','scared','terrified','terror','horrified','horror',
      'anxious','anxiety','panic','panicking','panicked','panic attack',
      'worried','worry','worrying','nervous','dread','dreading','apprehensive',
      'uneasy','insecure','threatened','phobia','paranoid','paranoia',
      'stressed','stress','overwhelmed','tense','tension',
      // Future uncertainty
      'uncertain','uncertainty','don\'t know what will happen','scared of future',
      'worried about future','what will happen','everything going wrong',
      'nothing is stable','nothing is certain','unpredictable','no control',
      'out of control','losing control','can\'t control','helpless about',
      // Social anxiety
      'social anxiety','scared of people','scared of crowds','crowd anxiety',
      'public speaking fear','stage fright','scared to talk','scared to speak',
      'scared of judgement','scared of rejection','fear of rejection',
      'worried what people think','scared to go out','don\'t want to meet people',
      'can\'t face people','avoiding people','avoiding social','withdrawing',
      // Health anxiety
      'health anxiety','scared of illness','hypochondria','what if i am sick',
      'scared of dying','death anxiety','fear of death','afraid to die',
      'chest pain','heart racing','palpitation','palpitations','heart pounding',
      'trembling','shaking','dizziness','dizzy','nausea','nauseous',
      'shortness of breath','can\'t breathe properly','hyperventilating',
      // Financial anxiety
      'money anxiety','scared of debt','financial fear','bill stress',
      'rent anxiety','emi stress','can\'t pay bills','no money','broke',
      'financial insecurity','poverty fear','job loss fear','income anxiety',
      // Academic & career fear
      'scared of failing','scared of exams','exam fear','test anxiety',
      'scared of results','fear of results','scared of interview',
      'career anxiety','future anxiety','scared of competition',
      // Physical anxiety symptoms
      'can\'t sleep because of worry','mind racing','racing thoughts',
      'worst case scenario','catastrophising','overthinking','can\'t stop thinking',
      'intrusive thoughts','disturbing thoughts','negative thoughts'
    ];

    const SURPRISE_WORDS = [
      'shocked','surprised','surprise','astonished','amazed','astounded',
      'disbelief','unexpected','stunned','startled','baffled',
      'bewildered','confused','sudden','suddenly','out of nowhere',
      'didn\'t expect','never expected','caught off guard','blindsided',
      'jaw dropped','can\'t believe it','unbelievable','speechless'
    ];

    // Count hit frequency for each emotion category
    const countEmotionHits = (wordList) => {
      let count = 0;
      for (const kw of wordList) {
        if (normText.includes(kw)) count++;
      }
      for (const w of allHitWords) {
        if (wordList.some(kw => w.includes(kw))) count += 1.5;
      }
      return count;
    };

    const angerHits    = countEmotionHits(ANGER_WORDS);
    const sadnessHits  = countEmotionHits(SADNESS_WORDS);
    const fearHits     = countEmotionHits(FEAR_WORDS);
    const loveHits     = countEmotionHits(LOVE_WORDS);
    const surpriseHits = countEmotionHits(SURPRISE_WORDS);
    const joyHits      = countEmotionHits(JOY_WORDS);

    // Only infer sadness from score if sadness hits aren't outweighed by anger or fear
    const scoreIndicatesSadness = (effectiveScore > 55 && riskSum > protectSum && sadnessHits >= angerHits && sadnessHits >= fearHits);
    const hasSadness  = hasNegation || sadnessHits > 0 || scoreIndicatesSadness;
    const hasAnger    = angerHits > 0;
    const hasFear     = fearHits > 0 || (effectiveScore > 65 && riskSum > protectSum && fearHits >= sadnessHits);
    const hasLove     = loveHits > 0;
    const hasSurprise = surpriseHits > 0;

    // Joy: ONLY if joy words present AND no negative distress signal AND no negation
    const hasNegativeSignal = (hasSadness && (sadnessHits > 0 || hasNegation)) || hasAnger || hasFear;
    const hasJoy            = joyHits > 0 && !hasNegativeSignal && !hasNegation;

    // ── Step 2: Assign emotion values ─────────────────────────────────────
    let sadness, joy, love, anger, fear, surprise;

    if (isPositive && !hasNegativeSignal && !hasNegation) {
      // Clearly positive text — joy/love lead
      joy      = hasJoy  ? 0.55 + (protectSum * 0.05) + (Math.min(joyHits, 3) * 0.05) : (hasLove ? 0.20 : 0.40);
      love     = hasLove ? 0.28 + (Math.min(loveHits, 3) * 0.05) : 0.08;
      sadness  = 0.04;
      anger    = 0.03;
      fear     = 0.03;
      surprise = hasSurprise ? 0.12 : 0.04;
    } else if (hasJoy && !hasNegativeSignal && !hasNegation) {
      // Joy words present, no distress, no negation
      joy      = 0.55 + (Math.min(joyHits, 3) * 0.05);
      love     = hasLove ? 0.22 : 0.08;
      sadness  = 0.05;
      anger    = 0.04;
      fear     = 0.04;
      surprise = hasSurprise ? 0.12 : 0.04;
    } else if (hasSadness || hasFear || hasAnger) {
      // Negative emotion (includes negation, burnout, occupation risk)
      const negBoost  = hasNegation       ? 0.12 : 0;
      const burnBoost = hasBurnout        ? 0.08 : 0;
      const occBoost  = hasOccupationRisk ? 0.06 : 0;

      const sadnessBase = hasSadness ? 0.20 + (effectiveScore / 200) + (Math.min(sadnessHits, 4) * 0.08) + negBoost + burnBoost : 0.05;
      const fearBase    = hasFear    ? 0.18 + (effectiveScore / 220) + (Math.min(fearHits, 4) * 0.08) + occBoost : 0.04;
      const angerBase   = hasAnger   ? 0.20 + (effectiveScore / 200) + (Math.min(angerHits, 4) * 0.08) : 0.04;

      // Dynamic competition among negative emotions based on signal density
      if (angerHits > sadnessHits && angerHits > fearHits) {
        anger   = angerBase + 0.18;
        sadness = Math.max(0.04, sadnessBase * 0.4);
        fear    = Math.max(0.04, fearBase * 0.4);
      } else if (fearHits > sadnessHits && fearHits > angerHits) {
        fear    = fearBase + 0.18;
        sadness = Math.max(0.04, sadnessBase * 0.4);
        anger   = Math.max(0.04, angerBase * 0.4);
      } else if (sadnessHits > angerHits && sadnessHits > fearHits) {
        sadness = sadnessBase + 0.18;
        anger   = Math.max(0.04, angerBase * 0.4);
        fear    = Math.max(0.04, fearBase * 0.4);
      } else {
        anger   = angerBase;
        sadness = sadnessBase;
        fear    = fearBase;
      }

      joy      = 0.02;
      love     = hasLove ? 0.08 : 0.02;
      surprise = hasSurprise ? 0.10 : 0.04;
    } else {
      // Neutral / truly ambiguous
      joy      = effectiveScore < 25 ? 0.25 : 0.12;
      sadness  = effectiveScore > 35 ? 0.20 + (effectiveScore / 250) : 0.14;
      love     = 0.14;
      anger    = 0.10;
      fear     = 0.10;
      surprise = 0.08;
    }

    // ── Step 3: Normalize ────────────────────────────────────────────────
    const emotions = { sadness, joy, love, anger, fear, surprise };
    const total    = Object.values(emotions).reduce((a, b) => a + b, 0) || 1;
    Object.keys(emotions).forEach(k => { emotions[k] = Math.max(0, Math.min(1, emotions[k] / total)); });

    // ── Step 4: Dominant emotion ──────────────────────────────────────────
    const dominant      = Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0];
    const dominantScore = Math.round(emotions[dominant] * 100);

    // ── Step 5: Emotion risk score ─────────────────────────────────────────
    const RISK_W = { sadness: 0.50, fear: 0.30, anger: 0.15, surprise: 0.05, joy: 0, love: 0 };
    const PROT_W = { joy: 0.60,    love: 0.40,  sadness: 0,  fear: 0,       anger: 0, surprise: 0 };
    let riskRaw = 0, protectRaw = 0;
    Object.entries(emotions).forEach(([k, v]) => {
      riskRaw    += v * (RISK_W[k] || 0);
      protectRaw += v * (PROT_W[k] || 0);
    });
    const emotionRisk = Math.round(Math.min(Math.max(0, riskRaw - protectRaw * 0.5) * 100, 100));

    return { emotions, emotionRisk, dominant, dominantScore, synthesized: true };
  }

  // ── Try Flask backend ──────────────────────────────────────────────────────
  async function tryBackend(text) {
    if (!BACKEND_URL) throw new Error('no backend configured');
    const ctrl = new AbortController();
    const to   = setTimeout(() => ctrl.abort(), 1500);
    try {
      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: currentLang }),
        signal: ctrl.signal
      });
      clearTimeout(to);
      if (!res.ok) throw new Error('backend error');
      const data = await res.json();
      return {
        score:            data.score,
        riskLevel:        data.risk_level,
        wordHits:         data.word_hits        || {},
        triggerWords:     (data.top_words || []).filter(w => w.weight > 0),
        protectiveWords:  (data.top_words || []).filter(w => w.weight < 0),
        hasSevereKeyword: data.severe           || false,
        tokenCount:       data.token_count      || 0,
        fromBackend:      true
      };
    } finally { clearTimeout(to); }
  }

  // ── Render Report ─────────────────────────────────────────────────
  function renderReport(nlpResult, emotionResult, vocalMetrics, compositeScore, originalText, respondentName, demoProfile, lifestyleProfile, facialMetrics) {
    // Timestamp + respondent
    const tsEl = document.getElementById('report-timestamp');
    if (tsEl) {
      const who = respondentName && respondentName !== 'Anonymous' ? ` · ${respondentName}` : '';
      tsEl.textContent = `Generated on ${new Date().toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}${who}`;
    }

    const riskLevel = nlpResult?.riskLevel || 'low';

    // Gauge
    animateGauge(compositeScore, riskLevel);

    // Update gauge formula text to reflect Tri-Modal Architecture
    const formulaEl = document.querySelector('.gauge-formula');
    if (formulaEl) {
      formulaEl.innerHTML = 'Tri-Modal Fusion: <strong>40% Text + 30% Voice + 30% Facial Affect</strong> &nbsp;<span style="font-size:0.68rem;opacity:0.6">(DTU XAI Framework · IC3-2026)</span>';
    }

    // Risk bands
    document.querySelectorAll('.band').forEach(b => b.classList.remove('active'));
    const bandMap = { low: 'band-low', moderate: 'band-moderate', high: 'band-high', severe: 'band-severe' };
    document.querySelector('.' + bandMap[riskLevel])?.classList.add('active');

    // LIME summary
    const summaryEl = document.getElementById('explainer-summary');
    if (summaryEl) {
      summaryEl.textContent = NLP.summary(nlpResult) ||
        'Analysis complete. No significant risk indicators detected.';
    }

    // Highlighted text + word chart
    if (nlpResult) {
      explainer.highlightText(originalText, nlpResult.wordHits || {});
      explainer.renderWordChart(nlpResult.triggerWords || [], nlpResult.protectiveWords || []);
    } else {
      const el = document.getElementById('highlighted-text');
      if (el) el.textContent = originalText;
    }

    // Vocal Biomarkers
    explainer.renderVocalResults(vocalMetrics);

    // Facial Affective Biomarkers (Track 1)
    renderFacialAffectCard(facialMetrics);

    // Emotion profile
    renderEmotionProfile(emotionResult);

    // Demographic risk context card
    renderDemographicCard(demoProfile);

    // SHAP-style multi-modal attribution card
    renderSHAPCard(nlpResult, vocalMetrics, lifestyleProfile, demoProfile, compositeScore, facialMetrics);

    // Recommendation
    renderRecommendation(riskLevel, compositeScore, nlpResult?.hasSevereKeyword);
  }

  // ── Render Facial Affective Biomarkers Card ────────────────────────────────
  function renderFacialAffectCard(metrics) {
    const container = document.getElementById('facial-results');
    if (!container) return;

    if (!metrics) {
      container.innerHTML = `
        <p class="no-data-msg">
          No facial affect session captured. Switch camera to <strong>Facial Affect Mode</strong> in Check-in to enable real-time FACS analysis.
        </p>
      `;
      return;
    }

    const au1_4Color = metrics.au1_4 > 50 ? '#ef4444' : metrics.au1_4 > 30 ? '#f59e0b' : '#10b981';
    const au12Color  = metrics.au12 > 35 ? '#10b981' : '#f59e0b';
    const riskBadgeColor = metrics.facialScore > 65 ? '#ef4444' : metrics.facialScore > 35 ? '#f59e0b' : '#10b981';

    container.innerHTML = `
      <div class="facial-metrics-summary" style="padding:10px 0">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
          <div>
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em">Dominant Affective State</div>
            <div style="font-size:1.15rem;font-weight:700;color:var(--text-primary);margin-top:2px">🙂 ${metrics.dominantExpression}</div>
          </div>
          <div style="text-align:right">
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase">Facial Risk Score</div>
            <div style="font-size:1.3rem;font-weight:800;color:${riskBadgeColor}">${metrics.facialScore} <span style="font-size:0.75rem;color:var(--text-muted)">/ 100</span></div>
          </div>
        </div>

        <div class="facial-chips-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px">
          <div class="vocal-stat-box" style="background:rgba(255,255,255,0.03);padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.07)">
            <span style="font-size:0.72rem;color:var(--text-muted);display:block">AU1/AU4 Brow Furrow</span>
            <strong style="font-size:1rem;color:${au1_4Color}">${metrics.au1_4}%</strong>
            <span style="font-size:0.65rem;color:var(--text-muted);display:block;margin-top:2px">Distress / Omega Sign</span>
          </div>

          <div class="vocal-stat-box" style="background:rgba(255,255,255,0.03);padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.07)">
            <span style="font-size:0.72rem;color:var(--text-muted);display:block">AU12 Zygomaticus</span>
            <strong style="font-size:1rem;color:${au12Color}">${metrics.au12}%</strong>
            <span style="font-size:0.65rem;color:var(--text-muted);display:block;margin-top:2px">Smile / Affect Blunting</span>
          </div>

          <div class="vocal-stat-box" style="background:rgba(255,255,255,0.03);padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.07)">
            <span style="font-size:0.72rem;color:var(--text-muted);display:block">Eye Aspect Ratio (EAR)</span>
            <strong style="font-size:1rem;color:#06d6a0">${metrics.ear}</strong>
            <span style="font-size:0.65rem;color:var(--text-muted);display:block;margin-top:2px">Aperture / Fatigue</span>
          </div>

          <div class="vocal-stat-box" style="background:rgba(255,255,255,0.03);padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,0.07)">
            <span style="font-size:0.72rem;color:var(--text-muted);display:block">Blink Frequency</span>
            <strong style="font-size:1rem;color:#38bdf8">${metrics.blinkRate} BPM</strong>
            <span style="font-size:0.65rem;color:var(--text-muted);display:block;margin-top:2px">Psychomotor index</span>
          </div>
        </div>
      </div>
    `;
  }

  // ── Animated Gauge ─────────────────────────────────────────────────────────
  function animateGauge(targetScore, riskLevel) {
    const arc     = document.getElementById('gauge-arc');
    const scoreEl = document.getElementById('gauge-score-text');
    const labelEl = document.getElementById('gauge-label-text');
    if (!arc) return;

    const ARC = 267;
    const riskLabels = { low: 'Low Risk', moderate: 'Moderate Risk', high: 'High Risk', severe: 'Severe Risk' };

    let current = 0;
    const step = Math.max(targetScore / 60, 0.5);
    const timer = setInterval(() => {
      current = Math.min(current + step, targetScore);
      arc.style.strokeDashoffset = ARC - (current / 100) * ARC;
      if (scoreEl) scoreEl.textContent = Math.round(current);
      if (current >= targetScore) clearInterval(timer);
    }, 16);

    if (labelEl) labelEl.textContent = riskLabels[riskLevel] || '';
  }

  // ── Emotion Profile Renderer ───────────────────────────────────────────────
  function renderEmotionProfile(result) {
    const container = document.getElementById('emotion-profile-content');
    if (!container) return;

    if (!result) {
      container.innerHTML = `<p class="no-data-msg">Unable to generate emotion profile for this input.</p>`;
      return;
    }

    const meta     = EmotionAI.getMeta();
    const emotions = result.emotions;
    const dominant = result.dominant;
    const domMeta  = meta[dominant] || {};
    const sortedEm = Object.entries(emotions).sort((a, b) => b[1] - a[1]);

    // ── Crisis override banner ─────────────────────────────────────────────
    const crisisBanner = result.crisisOverride ? `
      <div style="background:rgba(244,113,181,0.12);border:1px solid rgba(244,113,181,0.45);border-radius:12px;padding:14px 18px;margin-bottom:16px;display:flex;gap:12px;align-items:flex-start">
        <span style="font-size:1.5rem;flex-shrink:0">🆘</span>
        <div>
          <div style="font-weight:800;color:#f471b5;font-size:0.9rem;margin-bottom:4px">Crisis Language Detected</div>
          <div style="font-size:0.8rem;color:#a89cc8;line-height:1.6">
            Your text contains words associated with serious distress or suicidal ideation.
            <strong style="color:#ede9fe">You are not alone</strong> — please reach out to a helpline right now.
            The emotion profile below reflects your actual distress level.
          </div>
          <button onclick="crisisModal.open()" style="margin-top:10px;padding:7px 16px;background:rgba(244,113,181,0.2);border:1px solid rgba(244,113,181,0.4);border-radius:999px;color:#f471b5;font-size:0.78rem;font-weight:800;cursor:pointer;">
            🆘 Open Crisis Helplines
          </button>
        </div>
      </div>` : '';

    const sourceTag = result.crisisOverride
      ? '<span class="badge" style="background:rgba(244,113,181,0.15);color:#f471b5;border:1px solid rgba(244,113,181,0.3)">⚠ Crisis Override</span>'
      : result.synthesized
        ? '<span class="badge badge-lime" title="Generated from NLP lexicon — emotion model not loaded">NLP-Inferred</span>'
        : '<span class="badge badge-ai" title="Transformer emotion classifier running in your browser">' + ((EmotionAI.getModelId && EmotionAI.getModelId()) ? EmotionAI.getModelId().split('/').pop() : 'Model') + ' · Live</span>';

    container.innerHTML = `
      ${crisisBanner}
      <div class="dominant-emotion-row">
        <span class="emotion-emoji">${domMeta.emoji || '🎭'}</span>
        <div>
          <p class="emotion-dominant-label">Dominant Emotion ${sourceTag}</p>
          <p class="emotion-dominant-name">${domMeta.label || dominant}</p>
          <p class="emotion-dominant-pct">${result.dominantScore}% confidence · Emotion risk: ${result.emotionRisk}/100</p>
        </div>
      </div>
      <div class="emotion-bars-grid" id="emotion-bars-grid"></div>
    `;

    const grid = document.getElementById('emotion-bars-grid');
    sortedEm.forEach(([label, score]) => {
      const m   = meta[label] || {};
      const pct = Math.round(score * 100);
      const div = document.createElement('div');
      div.className = 'emotion-bar-item';
      div.innerHTML = `
        <span class="emotion-bar-emoji">${m.emoji || '❓'}</span>
        <span class="emotion-bar-label">${m.label || label}</span>
        <div class="emotion-bar-track">
          <div class="emotion-bar-fill" style="width:0%;background:${m.color || '#8b5cf6'}" data-pct="${pct}"></div>
        </div>
        <span class="emotion-bar-pct">${pct}%</span>
      `;
      grid.appendChild(div);
    });

    requestAnimationFrame(() => {
      grid.querySelectorAll('.emotion-bar-fill').forEach(el => {
        el.style.width = el.dataset.pct + '%';
      });
    });
  }


  // ── Recommendation Engine ──────────────────────────────────────────────────
  const RECS = {
    low: {
      icon: '💚',
      title: 'You\'re doing okay right now 🌸',
      subtitle: 'Keep up the healthy habits — here\'s how to maintain your glow',
      actions: [
        { label: '📅 Log your lifestyle',  action: "app.navigate('tracker')",       primary: true  },
        { label: '🫁 4-7-8 Breathing',     action: "app.navigate('interventions')", primary: false },
        { label: '✨ Daily affirmation',    action: "app.navigate('interventions')", primary: false }
      ]
    },
    moderate: {
      icon: '💛',
      title: 'Some emotional strain detected',
      subtitle: 'It\'s okay to not be okay. Consider speaking to a counsellor — early support makes a real difference 💕',
      actions: [
        { label: '🫁 4-7-8 Breathing',       action: "app.navigate('interventions')", primary: true  },
        { label: '🌿 5-4-3-2-1 Grounding',   action: "app.navigate('interventions')", primary: true  },
        { label: '📋 PHQ-9 Screening',        action: "app.navigate('phq9')",          primary: false },
        { label: '📅 Track sleep & stress',   action: "app.navigate('tracker')",       primary: false },
        { label: '📞 iCall: 9152987821',      action: "window.open('tel:9152987821')", primary: false }
      ]
    },
    high: {
      icon: '🧡',
      title: 'Significant distress detected',
      subtitle: 'Please speak to a mental health professional or counsellor. You don\'t have to carry this alone 🙏',
      actions: [
        { label: '📞 iCall (9152987821)',            action: "window.open('tel:9152987821')",  primary: true  },
        { label: '📞 Vandrevala: 1860-2662-345',     action: "window.open('tel:18602662345')", primary: true  },
        { label: '📋 Take PHQ-9 Assessment',         action: "app.navigate('phq9')",           primary: false },
        { label: '🫁 Breathing exercise',            action: "app.navigate('interventions')",  primary: false },
        { label: '📋 All helplines & resources',     action: "app.navigate('interventions')",  primary: false }
      ]
    },
    severe: {
      icon: '🆘',
      title: 'High risk — please reach out now',
      subtitle: 'You matter deeply. Please contact someone right now 💗',
      actions: [
        { label: '🚨 iCall NOW: 9152987821',        action: "window.open('tel:9152987821')",  primary: true },
        { label: '🚨 Vandrevala (24/7): 1860-2662-345', action: "window.open('tel:18602662345')", primary: true },
        { label: '🚨 NIMHANS: 080-46110007',         action: "window.open('tel:08046110007')", primary: true }
      ]
    }
  };

  // ── SHAP-Style Multi-Modal Contribution Bar ─────────────────────────────
  function renderSHAPCard(nlpResult, vocalMetrics, lifestyleProfile, demoProfile, compositeScore, facialMetrics) {
    // --- Compute raw weights for each modality ---
    const nlpScore   = nlpResult?.score || 0;
    const voiceScore = vocalMetrics?.overallRisk || vocalMetrics?.vocalScore || 0;
    const faceScore  = facialMetrics?.facialScore || 0;

    // Lifestyle risk: sleep deprivation (<6h or >9h), high stress, no activity, no support
    let lifestyleRisk = 0;
    if (lifestyleProfile) {
      const sleep   = lifestyleProfile.sleep   ?? 7;
      const stress  = lifestyleProfile.stress  ?? 5;
      const activityRisk = { none: 30, light: 10, moderate: 0, intense: 5 }[lifestyleProfile.activity] ?? 15;
      const supportRisk  = { none: 35, some: 15, good: 5, strong: 0 }[lifestyleProfile.support] ?? 15;
      const sleepRisk    = sleep < 6 ? 30 : sleep > 9 ? 10 : 0;
      const stressRisk   = Math.round((stress - 1) / 9 * 40);
      lifestyleRisk = Math.min(Math.round((sleepRisk + stressRisk + activityRisk + supportRisk) / 4), 100);
    }

    // Demographic risk (from demographics.js if available)
    let demoRisk = 0;
    if (demoProfile && Object.keys(demoProfile).length > 0 && typeof Demographics !== 'undefined') {
      const res = Demographics.analyze(demoProfile);
      if (res) demoRisk = Math.min(Math.round(res.prevalence * 1.2), 100);
    }

    // Decide which signals were actually used
    const hasVoice     = vocalMetrics && voiceScore > 0;
    const hasFace      = facialMetrics && faceScore > 0;
    const hasLifestyle = lifestyleProfile && Object.keys(lifestyleProfile).length > 0;
    const hasDemo      = demoProfile    && Object.keys(demoProfile).length > 0;

    // Weights per modality (Tri-modal: NLP, Voice, Face)
    let rawNLP  = nlpScore;
    let rawVoice= hasVoice     ? voiceScore       : 0;
    let rawFace = hasFace      ? faceScore        : 0;
    let rawLS   = hasLifestyle ? lifestyleRisk    : 0;
    let rawDemo = hasDemo      ? demoRisk         : 0;

    // Normalise to percentages that sum to 100%
    const total = (rawNLP || 1) + (rawVoice || 0.001) + (rawFace || 0.001) + (rawLS || 0.001) + (rawDemo || 0.001);
    const pctNLP   = Math.round(rawNLP   / total * 100);
    const pctVoice = Math.round(rawVoice / total * 100);
    const pctFace  = Math.round(rawFace  / total * 100);
    const pctLS    = Math.round(rawLS    / total * 100);
    const pctDemo  = Math.round(rawDemo  / total * 100);

    // Animate bars
    function animateBar(id, pctId, pct, color) {
      const el  = document.getElementById(id);
      const pEl = document.getElementById(pctId);
      if (!el || !pEl) return;
      el.style.background = color;
      pEl.textContent = pct + '%';
      setTimeout(() => { el.style.width = pct + '%'; }, 100);
    }

    animateBar('shap-nlp',       'shap-nlp-pct',       pctNLP,   'linear-gradient(90deg,#06d6a0,#22d3ee)');
    animateBar('shap-voice',     'shap-voice-pct',     pctVoice, 'linear-gradient(90deg,#c084fc,#f472b6)');
    animateBar('shap-face',      'shap-face-pct',      pctFace,  'linear-gradient(90deg,#f43f5e,#fb923c)');
    animateBar('shap-lifestyle', 'shap-lifestyle-pct', pctLS,    'linear-gradient(90deg,#fbbf24,#f59e0b)');
    animateBar('shap-demo',      'shap-demo-pct',      pctDemo,  'linear-gradient(90deg,#38bdf8,#06d6a0)');

    // If signals not available, dim those rows
    if (!hasVoice)     document.getElementById('shap-voice-pct')     && (document.getElementById('shap-voice-pct').textContent = 'No data');
    if (!hasFace)      document.getElementById('shap-face-pct')      && (document.getElementById('shap-face-pct').textContent = 'No data');
    if (!hasLifestyle) document.getElementById('shap-lifestyle-pct') && (document.getElementById('shap-lifestyle-pct').textContent = 'Not filled');
    if (!hasDemo)      document.getElementById('shap-demo-pct')      && (document.getElementById('shap-demo-pct').textContent = 'Not filled');
  }

  // ── Demographic Risk Context Card ────────────────────────────────────────
  function renderDemographicCard(profile) {
    const card    = document.getElementById('demo-report-card');
    const content = document.getElementById('demo-report-content');
    if (!card || !content) return;

    if (!profile || Object.keys(profile).length === 0) {
      card.style.display = 'none';
      return;
    }

    const result = Demographics.analyze(profile);
    if (!result) { card.style.display = 'none'; return; }

    card.style.display = 'block';

    // Build profile tags HTML
    const tagsHTML = result.tags.map(t =>
      `<span class="demo-report-tag" style="border-color:${t.color}30;color:${t.color}">${t.label}</span>`
    ).join('');

    // Prevalence bar
    const prevPct   = Math.min(result.prevalence, 100);
    const riskColor = result.atRiskColor;

    // Risk factors list
    const rfHTML = result.riskFactors.slice(0, 4).map(f =>
      `<li class="demo-rf-item"><span class="demo-rf-dot" style="background:${riskColor}"></span>${f}</li>`
    ).join('');

    // Protective factors
    const pfHTML = result.protective.slice(0, 3).map(f =>
      `<li class="demo-pf-item"><span class="demo-pf-dot"></span>${f}</li>`
    ).join('');

    content.innerHTML = `
      <!-- Profile tags -->
      <div class="demo-report-tags">${tagsHTML}</div>

      <!-- Risk label + prevalence -->
      <div class="demo-risk-banner" style="border-color:${riskColor}40;background:${riskColor}0d">
        <div class="demo-risk-banner-left">
          <span class="demo-risk-label" style="color:${riskColor}">${result.atRiskLabel}</span>
          <span class="demo-risk-prev">Est. prevalence in your group</span>
        </div>
        <div class="demo-risk-right">
          <span class="demo-risk-pct" style="color:${riskColor}">${result.prevalence}%</span>
        </div>
      </div>
      <div class="demo-prev-bar-wrap">
        <div class="demo-prev-bar-track">
          <div class="demo-prev-bar-fill" style="width:${prevPct}%;background:${riskColor}"></div>
          <div class="demo-prev-bar-national" title="National average ~14%"></div>
        </div>
        <div class="demo-prev-bar-labels">
          <span>0%</span>
          <span style="position:absolute;left:14%;transform:translateX(-50%);color:var(--text-muted);font-size:0.68rem">India avg 14%</span>
          <span>100%</span>
        </div>
      </div>

      <!-- Comparison insight -->
      <p class="demo-comparison">${result.comparisonText}</p>

      <!-- Two-column: risk factors + protective -->
      <div class="demo-factors-grid">
        <div class="demo-factors-col">
          <h4 class="demo-factors-title" style="color:${riskColor}">⚠ Risk Factors for Your Group</h4>
          <ul class="demo-rf-list">${rfHTML}</ul>
        </div>
        <div class="demo-factors-col">
          <h4 class="demo-factors-title" style="color:#06d6a0">🛡 Protective Factors</h4>
          <ul class="demo-pf-list">${pfHTML}</ul>
        </div>
      </div>

      ${result.insight ? `<p class="demo-insight-note">📖 ${result.insight}</p>` : ''}

      <p class="demo-source-note">Sources: NIMHANS NMHS 2016 · Lancet India · WHO South-East Asia · NCRB 2022</p>
    `;
  }

  function renderRecommendation(riskLevel, score, isSevere) {
    const level = isSevere ? 'severe' : (riskLevel || 'low');
    const rec   = RECS[level] || RECS.low;

    const iconEl    = document.getElementById('rec-icon');
    const titleEl   = document.getElementById('rec-title');
    const subEl     = document.getElementById('rec-subtitle');
    const actionsEl = document.getElementById('rec-actions');

    if (iconEl)    iconEl.textContent  = rec.icon;
    if (titleEl)   titleEl.textContent = rec.title;
    if (subEl)     subEl.textContent   = rec.subtitle;

    if (actionsEl) {
      actionsEl.innerHTML = '';
      rec.actions.forEach(a => {
        const btn = document.createElement('button');
        btn.className   = 'btn ' + (a.primary ? 'btn-primary' : 'btn-ghost');
        btn.textContent = a.label;
        btn.onclick     = () => runAction(a.action);
        actionsEl.appendChild(btn);
      });
    }
  }

  // ── Toast ──────────────────────────────────────────────────────────────────
  function toast(msg, type = 'info') {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.style.borderColor = type === 'warn'
      ? 'rgba(245,158,11,0.5)'
      : type === 'success'
        ? 'rgba(16,185,129,0.5)'
        : 'rgba(139,92,246,0.3)';
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 3800);
  }

  // ── Textarea live counter ──────────────────────────────────────────────────
  function setupTextarea() {
    const ta = document.getElementById('feeling-textarea');
    if (!ta) return;
    ta.addEventListener('input', () => {
      const el = document.getElementById('char-count');
      if (el) el.textContent = ta.value.length;
    });
  }

  // ── AI status bridge ───────────────────────────────────────────────────────
  function setAIStatus(status, label) {
    const badge = document.getElementById('ai-status-badge');
    const dot   = document.getElementById('ai-status-dot');
    const lbl   = document.getElementById('ai-status-label');
    if (badge) badge.dataset.status = status;
    if (lbl)   lbl.textContent = label;
  }

  // ── Utility ────────────────────────────────────────────────────────────────
  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Init ───────────────────────────────────────────────────────────────────
  function init() {
    setupTextarea();
    interventions.init();
    dashboard.init();
    navigate('dashboard');

    // First-run privacy notice
    if (!localStorage.getItem('mb_welcomed')) {
      localStorage.setItem('mb_welcomed', '1');
      setTimeout(() => toast('🔒 All your data stays on this device. Nothing is ever uploaded.', 'info'), 1200);
    }

    // Load emotion AI model in background
    EmotionAI.load()
      .then(() => setAIStatus('ready', 'AI Ready ✓'))
      .catch(() => setAIStatus('error', 'Emotion model unavailable — using lexicon only'));

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeMenu();
    });
  }

  // ── Data Export ────────────────────────────────────────────────────────────
  function exportData() {
    try {
      const sessions = JSON.parse(localStorage.getItem('mindbridge_sessions') || '[]');
      const journal  = JSON.parse(localStorage.getItem('mindbridge_journal')  || '[]');
      const tracker  = JSON.parse(localStorage.getItem('mindbridge_logs')  || '[]');
      const payload  = {
        exportedAt: new Date().toISOString(),
        app: 'MindBridge',
        totalSessions: sessions.length,
        totalJournalEntries: journal.length,
        sessions,
        journal,
        tracker
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `mindbridge-backup-${new Date().toISOString().slice(0,10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast('✅ Data exported successfully!', 'success');
    } catch (e) {
      toast('Export failed: ' + e.message, 'error');
    }
  }

  return {
    navigate, analyze, setLanguage,
    toggleMenu, closeMenu, toast, init, exportData,
    get currentLang() { return currentLang; }
  };

})();

// Bootstrap
document.addEventListener('DOMContentLoaded', () => app.init());
