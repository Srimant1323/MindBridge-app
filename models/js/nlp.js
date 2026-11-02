/**
 * nlp.js — Semantic NLP Scoring Engine v5
 *
 * Architecture:
 *   1. Single-word lexicon lookup   (with negation awareness)
 *   2. Multi-word phrase lookup     (substring match on LEXICON multi-word entries)
 *   3. LEXICON_PHRASES wiring       (was dead code in v4 — now live)
 *   4. Semantic phrase patterns     (regex — handles conjugations, euphemisms)
 *   5. Crisis injection             (severe phrase → guaranteed score ≥ 80)
 *   6. Sigmoid normalization        (raw score → 0–100)
 *
 * Bug fixes from audit (v4 → v5):
 *   BUG 1 FIX: Multi-word strings in LEXICON.en were never matched (token loop
 *              used words.includes(token) — single token can't equal 'end my life').
 *              Now a separate multi-word pass does normText.includes(phrase).
 *   BUG 2 FIX: LEXICON_PHRASES was declared in lexicon.js but never read here.
 *              Now wired into phrase matching with the same substring approach.
 *   BUG 3 FIX: Negation not handled in token scoring — "not happy" incorrectly
 *              reduced risk score (protective weight applied to negated joy word).
 *              Now a 2-token negation window flips/halves weight appropriately.
 *   BUG 4 FIX: PHRASE_PATTERNS dedup used entry.label + '_pattern' as check key
 *              but stored under matched-text displayKey — so the check was always
 *              false and patterns could double-count the same matched text.
 *              Now both check and store use displayKey consistently.
 *   BUG 5 FIX: hasCrisisLanguage() in app.js contained patterns absent from
 *              PHRASE_PATTERNS here (final goodbye, long sleep, goodbye forever,
 *              etc.). Added matching patterns to PHRASE_PATTERNS for consistent
 *              NLP scoring.
 *   BUG 6 FIX: NLP.composite() exported a different formula (0.70/0.30) from
 *              what app.js computeComposite() actually uses (0.40/0.40/0.20).
 *              Removed dead export to prevent confusion.
 *   BUG 7 NOTE: Score band comments in lexicon.js (<35/<60/<80) corrected to
 *              match actual thresholds here (<25/<50/<75/≥75).
 */
const NLP = (function () {

  // ── Negation vocabulary ────────────────────────────────────────────────────
  // Words that, when appearing 1–2 tokens before a lexicon hit, negate it:
  //   • Negated protective word  ("not happy")  → flip to moderate risk (+0.8)
  //   • Negated risk word        ("not sad")     → halve weight (× 0.5)
  const NEGATION_TOKENS = new Set([
    'not','no','never','neither','nor','without','barely','hardly','scarcely','rarely',
    'dont','doesnt','didnt','cant','cannot','wont','wouldnt','shouldnt','couldnt',
    'isnt','arent','wasnt','werent','havent','hasnt','hadnt',
    "don't","doesn't","didn't","can't","won't","wouldn't","shouldn't","couldn't",
    "isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't"
  ]);

  // ── Tokenizer ─────────────────────────────────────────────────────────────
  function tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s\u0900-\u097F]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 1);
  }

  // ── Normalize text for pattern matching ───────────────────────────────────
  function normalize(text) {
    return text
      .toLowerCase()
      .replace(/['']/g, "'")
      .replace(/[^\p{L}\p{M}\p{N}\s']/gu, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // ── Semantic Phrase Patterns ───────────────────────────────────────────────
  // Each entry: { pattern: RegExp, weight: number, level: string, label: string }
  // weight matches the lexicon scale: 3.0 = severe, 2.0 = high, 1.0 = moderate
  const PHRASE_PATTERNS = [

    // ── CRISIS / SUICIDAL IDEATION (weight 3.5–4.0) ────────────────────────
    { pattern: /\bsuicid\w*/,                                          weight: 4.0, level: 'severe', label: 'suicidal ideation' },
    { pattern: /\bkill(ing|ed)?\s+(my)?self\b/,                        weight: 4.0, level: 'severe', label: 'self-harm ideation' },
    { pattern: /\bend(ing|ed)?\s+(my\s+)?(life|it\s*all|everything)\b/,weight: 4.0, level: 'severe', label: 'ending life' },
    { pattern: /\btake\s+(my|his|her|their)\s+(own\s+)?life\b/,        weight: 4.0, level: 'severe', label: 'taking life' },
    { pattern: /\bwant(ed|s)?\s+to\s+die\b/,                          weight: 4.0, level: 'severe', label: 'wanting death' },
    { pattern: /\bwish(ed)?\s+i\s+(was|were|am)\s+dead\b/,            weight: 4.0, level: 'severe', label: 'wishing death' },
    { pattern: /\bbetter\s+off\s+dead\b/,                              weight: 4.0, level: 'severe', label: 'better off dead' },
    { pattern: /\bno\s+(will|reason|point)\s+to\s+live\b/,             weight: 4.0, level: 'severe', label: 'no reason to live' },
    { pattern: /\bnot\s+worth\s+living\b/,                             weight: 4.0, level: 'severe', label: 'not worth living' },
    { pattern: /\bnothing\s+(left\s+)?to\s+live\s+for\b/,             weight: 4.0, level: 'severe', label: 'nothing to live for' },
    { pattern: /\b(don'?t|do\s+not|won'?t)\s+want\s+to\s+(live|exist|be\s+(here|alive))\b/, weight: 4.0, level: 'severe', label: 'not wanting to live' },
    { pattern: /\bno\s+longer\s+want\s+to\s+(be|live|exist)\b/,       weight: 4.0, level: 'severe', label: 'no longer want to be' },

    // ── INDIRECT SUICIDAL EUPHEMISMS (weight 3.5) ──────────────────────────
    { pattern: /\bleave\s+this\s+world\b/,                             weight: 3.5, level: 'severe', label: 'leave this world' },
    { pattern: /\bgo(ing)?\s+to\s+heaven\b/,                          weight: 3.5, level: 'severe', label: 'going to heaven' },
    { pattern: /\bgo(ing)?\s+to\s+a\s+better\s+place\b/,              weight: 3.5, level: 'severe', label: 'better place' },
    { pattern: /\bbe\s+with\s+(god|the\s+angels|allah|bhagwan)\b/,     weight: 3.5, level: 'severe', label: 'be with god' },
    { pattern: /\bjoin\s+.{0,20}\bin\s+heaven\b/,                     weight: 3.5, level: 'severe', label: 'join in heaven' },
    { pattern: /\bmeet\s+my\s+maker\b/,                                weight: 3.5, level: 'severe', label: 'meet maker' },
    { pattern: /\bwhen\s+i('?m|\s+am|\s+was)\s+gone\b/,               weight: 3.5, level: 'severe', label: 'when im gone' },
    { pattern: /\bafter\s+i('?m|\s+am)\s+gone\b/,                     weight: 3.5, level: 'severe', label: 'after im gone' },
    { pattern: /\bworld\s+would\s+be\s+better\s+without\s+me\b/,      weight: 3.5, level: 'severe', label: 'world better without me' },
    { pattern: /\bbetter\s+(off\s+)?without\s+me\b/,                  weight: 3.5, level: 'severe', label: 'better without me' },
    { pattern: /\bno\s+one\s+would\s+(miss|care|notice)\s+me\b/,      weight: 3.5, level: 'severe', label: 'no one would miss me' },
    { pattern: /\bnobody\s+would\s+(miss|care|notice)\s+me\b/,        weight: 3.5, level: 'severe', label: 'nobody would miss me' },
    { pattern: /\bi\s+should\s+(just\s+)?(disappear|vanish|not\s+exist)\b/, weight: 3.5, level: 'severe', label: 'should disappear' },
    { pattern: /\bdon'?t\s+(want\s+to\s+)?wake\s+up\b/,               weight: 3.5, level: 'severe', label: 'dont want to wake up' },
    { pattern: /\bnever\s+wake\s+up\b/,                                weight: 3.5, level: 'severe', label: 'never wake up' },
    { pattern: /\b(not\s+be\s+here|no\s+longer\s+be\s+here)\s+tomorrow\b/, weight: 3.5, level: 'severe', label: 'not here tomorrow' },
    { pattern: /\bno\s+longer\s+be\s+a\s+burden\b/,                   weight: 3.5, level: 'severe', label: 'no longer a burden' },
    // BUG 5 FIX: these were in hasCrisisLanguage() but missing from PHRASE_PATTERNS
    { pattern: /\bfinal\s+goodbye\b/,                                  weight: 3.5, level: 'severe', label: 'final goodbye' },
    { pattern: /\bgoodbye\s+forever\b/,                                weight: 3.5, level: 'severe', label: 'goodbye forever' },
    { pattern: /\b(long|eternal|permanent)\s+sleep\b/,                weight: 3.0, level: 'severe', label: 'permanent sleep' },
    { pattern: /\bstop\s+(the|this)\s+pain\s+forever\b/,              weight: 3.5, level: 'severe', label: 'stop pain forever' },
    { pattern: /\bset\s+(myself\s+)?free\b.{0,20}(pain|world|life|end)/,weight: 3.0, level: 'severe', label: 'set free from pain' },

    // ── SELF-HARM (weight 3.5) ─────────────────────────────────────────────
    { pattern: /\bself[\s-]?harm\w*/,                                  weight: 3.5, level: 'severe', label: 'self-harm' },
    { pattern: /\b(cut|cutting|hurt|hurting|harm|harming)\s+(my)?self\b/, weight: 3.5, level: 'severe', label: 'self-harm act' },

    // ── SOCIAL REJECTION + ISOLATION (weight 2.5–3.0) ─────────────────────
    { pattern: /\bnobody\s+(in\s+the\s+world\s+)?(likes?|wants?|loves?|cares?\s+(about|for)|notices?|sees?)\s+me\b/, weight: 3.0, level: 'severe', label: 'nobody likes me' },
    { pattern: /\bno\s+one\s+(in\s+the\s+world\s+)?(likes?|wants?|loves?|cares?\s+(about|for)|notices?|sees?)\s+me\b/, weight: 3.0, level: 'severe', label: 'no one likes me' },
    { pattern: /\beveryone\s+(hates?|ignores?|avoids?|dislikes?)\s+me\b/, weight: 3.0, level: 'severe', label: 'everyone hates me' },
    { pattern: /\bno\s+one\s+(wants?\s+to\s+talk|talks?)\s+(to|with)\s+me\b/, weight: 3.0, level: 'severe', label: 'no one talks to me' },
    { pattern: /\bnobody\s+(wants?\s+to\s+talk|talks?)\s+(to|with)\s+me\b/, weight: 3.0, level: 'severe', label: 'nobody talks to me' },
    { pattern: /\bno\s+one\s+would\s+(miss|notice|care)\s+(about\s+)?me\b/, weight: 3.0, level: 'severe', label: 'no one would miss me' },
    { pattern: /\bfeel\s+(so\s+)?(completely\s+)?(unwanted|unloved|invisible|unnoticed|unheard|rejected|abandoned)\b/, weight: 2.5, level: 'high', label: 'social rejection feeling' },
    { pattern: /\bi\s+am\s+(completely\s+)?(alone|unwanted|unloved|worthless|invisible)\b/, weight: 2.5, level: 'high', label: 'alone unwanted' },
    { pattern: /\bi'?m\s+(completely\s+)?(alone|unwanted|unloved|worthless|invisible)\b/, weight: 2.5, level: 'high', label: 'im alone' },
    { pattern: /\balways\s+(feel\s+)?(alone|lonely|left\s+out|ignored|excluded)\b/, weight: 2.0, level: 'high', label: 'always alone' },
    { pattern: /\bno\s+one\s+(understands?|listens?|cares?)\b/,        weight: 2.0, level: 'high', label: 'no one understands' },
    { pattern: /\bnobody\s+(understands?|listens?|cares?)\b/,          weight: 2.0, level: 'high', label: 'nobody understands' },

    // ── CRISIS / HOPELESSNESS STATES (weight 2.5–3.0) ────────────────────
    { pattern: /\bcan'?t\s+(go\s+on|take\s+it\s+anymore|do\s+this\s+anymore|handle\s+this)\b/, weight: 3.0, level: 'severe', label: 'cant go on' },
    { pattern: /\bcannot\s+(go\s+on|take\s+it\s+anymore)\b/,           weight: 3.0, level: 'severe', label: 'cannot go on' },
    { pattern: /\bgiv(e|en|ing)\s+up\s+on\s+(life|everything|living)\b/, weight: 3.0, level: 'severe', label: 'giving up on life' },
    { pattern: /\bno\s+hope\s+(left|anymore|at\s+all)\b/,              weight: 2.5, level: 'severe', label: 'no hope left' },
    { pattern: /\bhopeless(ness)?\b/,                                  weight: 2.5, level: 'severe', label: 'hopelessness' },
    { pattern: /\blife\s+(is\s+)?(not|no\s+longer)\s+worth\b/,         weight: 3.0, level: 'severe', label: 'life not worth' },
    { pattern: /\bno\s+future\s+(for\s+me|left|anymore)\b/,            weight: 2.0, level: 'high',   label: 'no future' },
    { pattern: /\bnever\s+get\s+better\b/,                             weight: 2.0, level: 'high',   label: 'never get better' },
    { pattern: /\bnothing\s+(will\s+)?(ever\s+)?change\b/,             weight: 1.5, level: 'high',   label: 'nothing will change' },
    { pattern: /\bno\s+point\s+(in\s+)?(anything|living|trying|going\s+on|anymore)\b/, weight: 2.5, level: 'severe', label: 'no point' },
    { pattern: /\bnothing\s+matters?\b/,                               weight: 2.0, level: 'high',   label: 'nothing matters' },
    { pattern: /\bdead\s+inside\b/,                                    weight: 2.5, level: 'severe', label: 'dead inside' },
    { pattern: /\bfeel\s+nothing\b/,                                   weight: 2.0, level: 'high',   label: 'feel nothing' },

    // ── DEPRESSION / DISTRESS (weight 1.5–2.0) ────────────────────────────
    { pattern: /\b(deeply|severely|completely)\s+depressed\b/,         weight: 2.0, level: 'high',   label: 'severely depressed' },
    { pattern: /\bfall(ing)?\s+apart\b/,                               weight: 2.0, level: 'high',   label: 'falling apart' },
    { pattern: /\bbreak(ing)?\s+down\b/,                               weight: 2.0, level: 'high',   label: 'breaking down' },
    { pattern: /\bmental\s+breakdown\b/,                               weight: 2.0, level: 'high',   label: 'mental breakdown' },
    { pattern: /\bnervous\s+breakdown\b/,                              weight: 2.0, level: 'high',   label: 'nervous breakdown' },
    { pattern: /\bcan'?t\s+(cope|function|sleep|eat|breathe)\b/,       weight: 1.5, level: 'high',   label: 'cant function' },
    { pattern: /\bcannot\s+(cope|function|sleep|eat|breathe)\b/,       weight: 1.5, level: 'high',   label: 'cannot function' },
    { pattern: /\blosing\s+my\s+mind\b/,                               weight: 1.5, level: 'high',   label: 'losing my mind' },
    { pattern: /\bhate\s+(my(self)?|my\s+life|everything)\b/,          weight: 2.0, level: 'high',   label: 'self-hatred' },
    { pattern: /\bworthless\b/,                                        weight: 2.0, level: 'high',   label: 'worthless' },
    { pattern: /\buseless\b/,                                          weight: 1.5, level: 'high',   label: 'useless' },
    { pattern: /\bfeel(ing)?\s+(like\s+a\s+)?(burden|failure|fraud)\b/, weight: 2.0, level: 'high', label: 'feel like burden' },
    { pattern: /\bunbearable\s+(pain|sadness|loneliness|depression)\b/, weight: 2.0, level: 'high',  label: 'unbearable pain' },
    { pattern: /\bconstant\s+(pain|sadness|crying|suffering)\b/,       weight: 1.5, level: 'high',   label: 'constant suffering' },
    { pattern: /\bcompletely\s+(alone|empty|lost|broken|numb)\b/,      weight: 2.0, level: 'high',   label: 'completely broken' },

    // ── ANXIETY / PANIC (weight 1.5–2.0) ──────────────────────────────────
    { pattern: /\bpanic\s+attack\b/,                                   weight: 2.0, level: 'high',   label: 'panic attack' },
    { pattern: /\bracing\s+thoughts?\b/,                               weight: 2.0, level: 'high',   label: 'racing thoughts' },
    { pattern: /\bintrusive\s+thoughts?\b/,                            weight: 2.0, level: 'high',   label: 'intrusive thoughts' },
    { pattern: /\bcannot\s+stop\s+(overthinking|thinking|worrying)\b/, weight: 2.0, level: 'high',   label: 'cant stop overthinking' },
    { pattern: /\bcan'?t\s+stop\s+(overthinking|thinking|worrying)\b/, weight: 2.0, level: 'high',   label: 'cant stop thinking' },
    { pattern: /\bmind\s+(is\s+)?(always\s+)?(racing|spiralling|spiraling)\b/, weight: 2.0, level: 'high', label: 'mind racing' },
    { pattern: /\bspiralling\s+thoughts?\b/,                           weight: 2.0, level: 'high',   label: 'spiralling thoughts' },
    { pattern: /\bscared\s+of\s+(my\s+)?future\b/,                    weight: 2.0, level: 'high',   label: 'scared of future' },
    { pattern: /\beverything\s+(is\s+)?falling\s+apart\b/,            weight: 2.0, level: 'high',   label: 'everything falling apart' },
    { pattern: /\bworld\s+(is\s+)?collapsing\b/,                      weight: 2.0, level: 'high',   label: 'world collapsing' },
    { pattern: /\bparalys(ed|ed)\s+by\s+(fear|anxiety)\b/,            weight: 2.0, level: 'high',   label: 'paralysed by fear' },
    { pattern: /\bshortness\s+of\s+breath\b/,                         weight: 1.5, level: 'high',   label: 'shortness of breath' },
    { pattern: /\bheart\s+(is\s+)?(pounding|racing|hammering)\b/,     weight: 1.5, level: 'high',   label: 'heart pounding' },
    { pattern: /\bconstant\s+(worry|anxiety|fear|dread)\b/,            weight: 1.5, level: 'high',   label: 'constant anxiety' },
    { pattern: /\bcan'?t\s+stop\s+(worrying|crying|thinking)\b/,       weight: 1.5, level: 'high',   label: 'cant stop worrying' },
    { pattern: /\beverything\s+(scares?|terrifies?|frightens?)\s+me\b/, weight: 1.5, level: 'high',  label: 'everything scares me' },
    { pattern: /\boverwhelm(ed|ing)?\b/,                               weight: 1.0, level: 'moderate', label: 'overwhelmed' },

    // ── BURNOUT / EXHAUSTION (weight 1.5–2.0) ─────────────────────────────
    { pattern: /\bmentally\s+(exhausted|drained|burnt?\s+out)\b/,      weight: 2.0, level: 'high',   label: 'mentally exhausted' },
    { pattern: /\bemotionally\s+(exhausted|drained|destroyed)\b/,      weight: 2.0, level: 'high',   label: 'emotionally exhausted' },
    { pattern: /\bcompletely\s+(depleted|burnt?\s+out|burned\s+out)\b/, weight: 2.0, level: 'high',  label: 'completely burned out' },
    { pattern: /\brunning\s+on\s+empty\b/,                             weight: 1.5, level: 'high',   label: 'running on empty' },
    { pattern: /\bnothing\s+left\s+to\s+give\b/,                      weight: 1.5, level: 'high',   label: 'nothing left to give' },
    { pattern: /\bon\s+the\s+verge\s+of\s+(a\s+)?breakdown\b/,        weight: 2.0, level: 'high',   label: 'verge of breakdown' },

    // ── MODERATE DISTRESS (weight 0.8–1.2) ───────────────────────────────
    { pattern: /\bnot\s+okay\b/,                                       weight: 1.0, level: 'moderate', label: 'not okay' },
    { pattern: /\bstruggling\s+(a\s+lot|so\s+much|with\s+everything)\b/, weight: 1.0, level: 'moderate', label: 'struggling a lot' },
    { pattern: /\bhard\s+time\b/,                                      weight: 0.8, level: 'moderate', label: 'hard time' },
    { pattern: /\bdifficult\s+time\b/,                                 weight: 0.8, level: 'moderate', label: 'difficult time' },
    { pattern: /\blost\s+(all\s+)?(hope|motivation|interest|purpose)\b/, weight: 1.5, level: 'high', label: 'lost motivation' },
    { pattern: /\bno\s+(motivation|energy|appetite|interest)\b/,       weight: 1.0, level: 'moderate', label: 'no motivation' },
    { pattern: /\bcan'?t\s+(get\s+out\s+of\s+bed|stop\s+crying|focus|concentrate)\b/, weight: 1.2, level: 'moderate', label: 'cant get out of bed' },
    { pattern: /\bno\s+peace\s+of\s+mind\b/,                          weight: 1.5, level: 'high',   label: 'no peace of mind' },
    { pattern: /\btoo\s+much\s+tension\b/,                             weight: 1.0, level: 'moderate', label: 'too much tension' },
    { pattern: /\bmind\s+(is\s+)?full\s+of\s+tension\b/,              weight: 1.0, level: 'moderate', label: 'mind full of tension' },

    // ── PROTECTIVE / POSITIVE (weight -1.0 to -2.0) ──────────────────────
    { pattern: /(?<!\b(?:not|never|no|dont|don't|cant|can't|doesnt|doesn't|isnt|isn't|barely|hardly)\s+)\bfeel(ing)?\s+(better|good|great|wonderful|amazing)\b/, weight: -1.5, level: 'protective', label: 'feeling good' },
    { pattern: /(?<!\b(?:not|never|no|dont|don't|cant|can't|doesnt|doesn't|isnt|isn't|barely|hardly)\s+)\b(so\s+)?(happy|joyful|excited|grateful|thankful)\b/, weight: -1.5, level: 'protective', label: 'happy grateful' },
    { pattern: /\blooking\s+forward\s+to\b/,                           weight: -1.2, level: 'protective', label: 'looking forward' },
    { pattern: /\bgetting\s+better\b/,                                 weight: -1.5, level: 'protective', label: 'getting better' },
    { pattern: /\bfeel(ing)?\s+(loved|supported|cared\s+for|understood)\b/, weight: -1.5, level: 'protective', label: 'feeling supported' },
    { pattern: /\bthings\s+are\s+(getting|looking)\s+(better|up)\b/,  weight: -1.5, level: 'protective', label: 'things looking up' },

    // ── HINDI CRISIS PATTERNS ─────────────────────────────────────────────
    { pattern: /मरना|मर\s*जाना|जीना\s*नहीं|आत्महत्या/,               weight: 4.0, level: 'severe', label: 'Hindi crisis direct' },
    { pattern: /खुद\s*को\s*मारना|जीने\s*का\s*मन\s*नहीं/,              weight: 3.5, level: 'severe', label: 'Hindi crisis want' },
    { pattern: /इस\s*दुनिया\s*(को\s*)?छोड़ना|स्वर्ग\s*जाना/,          weight: 3.5, level: 'severe', label: 'Hindi suicidal euphemism' },
    { pattern: /कोई\s*नहीं\s*चाहता|कोई\s*पसंद\s*नहीं/,               weight: 3.0, level: 'severe', label: 'Hindi rejection' },
    { pattern: /उम्मीद\s*नहीं|निराश|बेकार|व्यर्थ/,                    weight: 2.5, level: 'high',   label: 'Hindi hopelessness' },
    { pattern: /पैनिक\s*अटैक|रेसिंग\s*थॉट्स|घबराहट\s*बहुत/,          weight: 2.0, level: 'high',   label: 'Hindi anxiety' },
  ];

  // ── Main Analysis Function ────────────────────────────────────────────────
  function analyze(text, lang = 'en') {
    if (!text || text.trim().length < 3) return null;

    const tokens   = tokenize(text);
    const normText = normalize(text);
    if (tokens.length === 0) return null;

    const wordHits    = {};
    let   rawScore    = 0;
    let   severityFlag = false;

    // ── 1. Single-word lexicon lookup (with negation awareness) ─────────────
    // BUG 3 FIX: Check 2-token negation window before each hit.
    // Negated protective word → flip to moderate risk (+0.8)
    // Negated risk word       → halve its weight (× 0.5)
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const prev1 = tokens[i - 1] || '';
      const prev2 = tokens[i - 2] || '';
      const isNegated = NEGATION_TOKENS.has(prev1) || NEGATION_TOKENS.has(prev2);

      for (const [level, data] of Object.entries(LEXICON)) {
        // BUG FIX: avoid duplicating entries when lang='en'
        // (data[lang] would equal data.en, making concat a doubled array)
        const words = lang !== 'en'
          ? [...(data[lang] || []), ...(data.en || [])].filter(e => !e.includes(' '))
          : (data.en || []).filter(e => !e.includes(' '));
        if (words.includes(token) && !wordHits[token]) {
          let effectiveWeight = data.weight;
          let effectiveLevel  = level;
          if (isNegated) {
            if (data.weight < 0) {
              // "not happy" → protective flipped to moderate risk
              effectiveWeight = 0.8;
              effectiveLevel  = 'moderate';
            } else {
              // "not sad" → risk word halved
              effectiveWeight = data.weight * 0.5;
            }
          }
          wordHits[token] = { weight: effectiveWeight, level: effectiveLevel, negated: isNegated };
          rawScore += effectiveWeight;
          if (level === 'severe' && !isNegated) severityFlag = true;
        }
      }
    }

    // ── 2. Multi-word LEXICON phrase lookup ──────────────────────────────────
    // BUG 1 FULL FIX: Multi-word strings in LEXICON (e.g. 'end my life',
    // 'racing thoughts', 'no peace of mind') were silently never matched
    // because the token loop above checked words.includes(singleToken) — a
    // single token can never equal a multi-word string. Now we do substring
    // matching on the full normalized text for any entry containing a space.
    for (const [level, data] of Object.entries(LEXICON)) {
      // BUG FIX: avoid duplicating entries when lang='en'
      const phrases = lang !== 'en'
        ? [...(data[lang] || []), ...(data.en || [])].filter(e => e.includes(' '))
        : (data.en || []).filter(e => e.includes(' '));
      for (const phrase of phrases) {
        if (normText.includes(phrase) && !wordHits[phrase]) {
          wordHits[phrase] = { weight: data.weight, level };
          rawScore += data.weight;
          if (level === 'severe') severityFlag = true;
        }
      }
    }

    // ── 3. LEXICON_PHRASES matching ──────────────────────────────────────────
    // BUG 2 FIX: LEXICON_PHRASES was declared in lexicon.js but never read here.
    // Now wired in with the same substring approach. LEXICON_PHRASES uses higher
    // weights (3.5 / 2.5) for multi-word clinical phrases.
    if (typeof LEXICON_PHRASES !== 'undefined') {
      for (const [level, data] of Object.entries(LEXICON_PHRASES)) {
        const phrases = (data[lang] || []).concat(data.en || []);
        for (const phrase of phrases) {
          if (normText.includes(phrase) && !wordHits[phrase]) {
            wordHits[phrase] = { weight: data.weight, level };
            rawScore += data.weight;
            if (level === 'severe') severityFlag = true;
          }
        }
      }
    }

    // ── 4. Semantic phrase pattern matching ─────────────────────────────────
    // Regex-based: catches trigrams, natural phrasing variants,
    // euphemisms, and conjugations that substring lookup misses.
    // BUG 4 FIX: Dedup now uses displayKey (matched text) for BOTH the
    // existence check AND storage — preventing same text being double-scored
    // while still allowing different matched phrases to score independently.
    for (const entry of PHRASE_PATTERNS) {
      if (entry.pattern.test(normText)) {
        const match      = normText.match(entry.pattern);
        const displayKey = match ? match[0].trim() : entry.label;
        if (!wordHits[displayKey]) {                  // BUG 4 FIX: check by matched text
          wordHits[displayKey] = { weight: entry.weight, level: entry.level };
          rawScore += entry.weight;
          if (entry.level === 'severe') severityFlag = true;
        }
      }
    }

    // ── 5. Crisis injection ─────────────────────────────────────────────────
    // If crisis patterns fire, GUARANTEE a severe NLP score regardless of
    // token count normalisation. Long sentences with indirect crisis language
    // are otherwise diluted to "low risk".
    const isCrisis = severityFlag && PHRASE_PATTERNS.some(
      e => e.level === 'severe' && e.pattern.test(normText)
    );

    // ── 6. Normalize score ──────────────────────────────────────────────────
    // Clamp denominator so long sentences aren't unfairly penalized.
    const tokenWeight = isCrisis
      ? Math.min(tokens.length, 6)   // cap at 6 for crisis — prevents dilution
      : Math.max(tokens.length, 4);

    const rawNorm = rawScore / tokenWeight;

    let score;
    if (rawNorm <= 0) {
      score = Math.max(0, 12 + rawNorm * 8);
    } else if (rawNorm < 0.3) {
      score = 25 + rawNorm * 50;
    } else if (rawNorm < 0.8) {
      score = 40 + (rawNorm - 0.3) * 60;
    } else if (rawNorm < 1.5) {
      score = 70 + (rawNorm - 0.8) * 20;
    } else {
      score = Math.min(100, 84 + (rawNorm - 1.5) * 10);
    }

    // Severe keyword / crisis phrase → always ≥ 80
    if (severityFlag && score < 80) score = Math.max(score, 80);
    score = Math.round(score);

    // ── 7. Risk level ────────────────────────────────────────────────────────
    // BUG 7 NOTE: These thresholds are authoritative. lexicon.js comment
    // previously said <35/<60/<80 — corrected there to match these.
    let riskLevel;
    if      (score < 25) riskLevel = 'low';
    else if (score < 50) riskLevel = 'moderate';
    else if (score < 75) riskLevel = 'high';
    else                 riskLevel = 'severe';

    // ── 8. Sort hits for display ─────────────────────────────────────────────
    const sortedHits = Object.entries(wordHits)
      .sort((a, b) => Math.abs(b[1].weight) - Math.abs(a[1].weight));

    const triggerWords    = sortedHits.filter(([, d]) => d.weight > 0).slice(0, 6);
    const protectiveWords = sortedHits.filter(([, d]) => d.weight < 0).slice(0, 4);

    return {
      score,
      riskLevel,
      rawScore,
      isCrisis,
      tokenCount:      tokens.length,
      wordHits,
      triggerWords:    triggerWords.map(([word, data])    => ({ word, ...data })),
      protectiveWords: protectiveWords.map(([word, data]) => ({ word, ...data })),
      hasSevereKeyword: severityFlag
    };
  }

  // ── Human-readable summary ────────────────────────────────────────────────
  function summary(result) {
    if (!result) return '';
    const { riskLevel, score, triggerWords, protectiveWords, hasSevereKeyword, isCrisis } = result;

    if (isCrisis || hasSevereKeyword) {
      return `Your language contains expressions associated with severe distress or crisis (score: ${score}/100). ` +
             `You don't have to face this alone — please reach out to a helpline right now. You matter.`;
    }

    const triggers = triggerWords.slice(0, 3).map(w => `"${w.word}"`).join(', ');
    const shields  = protectiveWords.slice(0, 2).map(w => `"${w.word}"`).join(', ');

    const levelText = {
      low:      `Your language suggests you're generally coping well`,
      moderate: `Your language indicates some emotional strain`,
      high:     `Your language shows significant signs of distress`,
      severe:   `Your language indicates a high level of emotional distress`
    }[riskLevel];

    let msg = `${levelText} (score: ${score}/100).`;
    if (triggers) msg += ` Expressions like ${triggers} reflect emotional pain.`;
    if (shields)  msg += ` Expressions like ${shields} show some resilience.`;

    return msg;
  }

  // BUG 6 FIX: Removed dead NLP.composite() export which used formula
  // 0.70×nlp + 0.30×vocal — different from app.js computeComposite()
  // which correctly uses 0.40×nlp + 0.40×emotion + 0.20×vocal.
  // app.js computeComposite() is the single source of truth.

  return { analyze, summary, tokenize };
})();
