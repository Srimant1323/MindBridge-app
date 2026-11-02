/**
 * lexicon.js — Clinical Sentiment Lexicon v4 (English + Hindi)
 *
 * Precision-trained on:
 *   - PHQ-9 · GAD-7 · BDI-II clinical instruments
 *   - NIMHANS India distress patterns
 *   - WHO ICD-11 depressive/anxiety descriptors
 *   - LIWC emotional lexicon categories
 *   - India-specific occupational & social distress language
 *   - Somatic complaint vocabulary
 *   - Indian English colloquial expressions
 *
 * Weight scale:
 *   severe     = 3.0   (crisis / suicidal ideation — immediate danger)
 *   high       = 2.0   (clinically significant distress — professional help needed)
 *   moderate   = 1.0   (mild–moderate emotional strain — support recommended)
 *   protective = -1.5  (resilience / positive affect — lowers risk score)
 *
 * Scoring logic (nlp.js):
 *   Each matched word/phrase contributes its weight × frequency to the raw score.
 *   Raw score is normalised to 0–100 against the word count of the text.
 *   Score bands: low <25 · moderate 25–49 · high 50–74 · severe ≥75
 */
const LEXICON = {

  // ══════════════════════════════════════════════════════════════════════════
  // TIER 1 — SEVERE (weight 3.0)
  // Crisis, suicidal ideation, direct and indirect self-harm language.
  // Any single hit from this tier pushes the score toward severe.
  // ══════════════════════════════════════════════════════════════════════════
  severe: {
    weight: 3.0,
    en: [
      // Direct suicidal ideation
      'suicide','suicidal','kill myself','end my life','take my life',
      'no reason to live','not worth living','want to die','wish i was dead',
      'wish i were dead','want to disappear forever','want to cease to exist',
      'wish i never existed','wish i was never born','better off dead',
      'no will to live','dead inside','end it all','cannot go on',
      'cant go on','life is over','life is meaningless','life is not worth',
      'nothing to live for','life isnt worth it','done with life',
      'done with everything','never wake up','never wake up again',
      // Indirect suicidal ideation (euphemisms — clinically equivalent)
      'leave this world','go to heaven','go to a better place','go to god',
      'be with god','be with the angels','meet my maker','be with my creator',
      'when i am gone','after i am gone','when im gone','before i go',
      'no longer be a burden','not want to be a burden','be a burden anymore',
      'i want to vanish','i should disappear','need to disappear',
      'need to leave','need to go away','need to escape everything',
      'escape permanently','permanent escape','permanent solution',
      // Absolute rejection and worthlessness
      'nobody wants me','nobody likes me','no one likes me','no one wants me',
      'no one loves me','nobody loves me','everybody hates me',
      'nobody cares about me','no one cares about me','no one cares',
      'world would be better without me','better without me here',
      'no one would miss me','nobody would miss me',
      'not want to exist','wish i wasnt here',
      // Self-harm
      'hurt myself','harm myself','cut myself','cutting myself','self harm',
      'self-harm','self injury','self-injury','burning myself',
      // Hopelessness (PHQ-9 item — clinically severe when combined)
      'completely hopeless','totally hopeless','absolutely hopeless',
      'no hope at all','lost all hope','no hope left','hope is gone',
      'nothing will ever change','will never get better','never be okay again'
    ],
    hi: [
      'आत्महत्या','मर जाना','मरना चाहता','जीने का कोई कारण नहीं',
      'जीना नहीं चाहता','उम्मीद नहीं है','सब छोड़ दो','मर जाऊं',
      'जीवन समाप्त करना','मौत चाहिए','मरना बेहतर है','जीवन व्यर्थ है',
      'खत्म कर दूं','सब बेकार है','आत्महत्या करना चाहता','जी नहीं है',
      'जीने की इच्छा नहीं','सहन नहीं होता','इस दुनिया से जाना',
      'कोई नहीं चाहता मुझे','कोई पसंद नहीं करता','सब नफरत करते हैं',
      'इस दुनिया को छोड़ना','स्वर्ग जाना','भगवान के पास जाना',
      'खुद को नुकसान पहुंचाना','खुद को तकलीफ देना',
      'बिल्कुल उम्मीद नहीं','कभी ठीक नहीं होगा','अब कुछ नहीं बदलेगा'
    ]
  },

  // ══════════════════════════════════════════════════════════════════════════
  // TIER 2 — HIGH (weight 2.0)
  // Clinically significant distress. Requires professional intervention.
  // Covers: severe anxiety, panic, clinical depression markers,
  // emotional breakdown, extreme isolation, burnout, somatic crisis.
  // ══════════════════════════════════════════════════════════════════════════
  high: {
    weight: 2.0,
    en: [
      // ── Core clinical depression markers ────────────────────────────────
      'empty','numb','trapped','burden','despair','broken','failure',
      'useless','abandoned','unloved','void','hollow','devastated',
      'miserable','ruined','destroyed','invisible','forgotten','unwanted',
      'rejected','isolated','drained','suffocating','drowning',
      'falling apart','losing my mind','cant function','cannot function',
      'cant cope','cannot cope','losing hope','feel nothing',
      'no point','shut down','given up','total failure','dont belong',
      'hate myself','self-hatred','loathing','pathetic','pitiful',
      'furious','rage','enraged','infuriated','livid','hostile','outraged',
      'stuck','no way out','trapped in my mind','constant sadness',
      'deep pain','unbearable','unlovable','unworthy','incompetent',
      'incapable','defeated','shattered','collapsed','broken inside',
      'disintegrating','crumbling','falling to pieces','completely broken',

      // ── Severe anxiety and panic ─────────────────────────────────────────
      'racing thoughts','intrusive thoughts','intrusive thought',
      'panic attack','panic attacks','having a panic attack',
      'cannot stop overthinking','cannot stop thinking','mind wont stop',
      'mind cant stop','thoughts wont stop','cant control my thoughts',
      'terrified','terror','horrified','absolute dread','extreme dread',
      'scared of my future','scared of the future','fear of the future',
      'completely helpless','completely overwhelmed','totally overwhelmed',
      'cannot breathe','cant breathe','hard to breathe',
      'shortness of breath','difficulty breathing',
      'heart is racing','heart racing','heart pounding','palpitations',
      'trembling','shaking uncontrollably','cold sweat',
      'hyperventilating','hyperventilation','feeling faint',
      'dizzy from anxiety','dizzy all the time',
      'scared of dying','fear of death','death anxiety','afraid to die',
      'losing grip','losing control','out of control feeling',
      'nothing is okay','everything is falling apart','world is collapsing',
      'paralysed by fear','paralysed by anxiety','frozen with fear',
      'mind is spiralling','spiralling thoughts','spiral of anxiety',
      'constant worry','worrying all the time','always worried',
      'never at peace','no peace of mind','mind never rests',
      'cannot relax','cannot calm down','cannot wind down',
      'anxious all the time','anxiety is constant','chronic anxiety',

      // ── Severe emotional exhaustion and burnout ──────────────────────────
      'mentally exhausted','mentally drained','emotionally exhausted',
      'emotionally drained','completely drained','total exhaustion',
      'mental breakdown','nervous breakdown','complete burnout',
      'total breakdown','on the verge of breakdown','about to break down',
      'breaking down inside','breaking down','collapsing inside',
      'no energy left','completely depleted','running on empty',
      'cannot go on like this','cannot keep going','have nothing left',
      'nothing left to give','gave everything','used up',

      // ── Somatic (body) distress signals ─────────────────────────────────
      'cannot sleep at all','not sleeping at all','sleepless for days',
      'sleep deprived for weeks','chronic insomnia','severe insomnia',
      'cannot eat','stopped eating','not eating at all','no appetite at all',
      'eating disorder','starving myself','bingeing','purging',
      'body aches all over','pain everywhere','physical pain from stress',
      'chest tightness','chest is tight','chest feels heavy',
      'weight on my chest','pressure on chest','chest pressure',
      'knot in my stomach','stomach in knots','sick to my stomach',
      'nausea from anxiety','vomiting from stress','shaking from fear',
      'sweating constantly','night sweats from anxiety',
      'migraine from stress','headache every day','daily headaches',

      // ── Social isolation and rejection ───────────────────────────────────
      'nobody understands','no one understands','no one listens',
      'always alone','completely alone','feel invisible','feel unwanted',
      'rejected by everyone','ignored by everyone','left out','left behind',
      'no one checks on me','no friends','lost all friends',
      'everyone leaves','people leave me','always get abandoned',
      'never belonged','dont fit in','never fit in','feel like a burden',
      'nobody notices me','nobody sees me','nobody hears me',
      'all by myself','totally isolated','cut off from everyone',
      'no one to talk to','no support','no one supports me',
      'pushed everyone away','cant connect with anyone',
      'feel disconnected from everyone','disconnected from reality',

      // ── Severe hopelessness / existential ────────────────────────────────
      'no future for me','no future','cannot see a future',
      'nothing will change','nothing ever changes','pointless existence',
      'meaningless life','no meaning in life','life means nothing',
      'no reason to keep going','no reason to try','why even bother',
      'what is the point','no point in trying','no point anymore',
      'given up on myself','given up on everything','stopped caring',
      'dont care about anything','nothing matters to me',

      // ── Severe relationship distress ──────────────────────────────────────
      'relationship destroyed','marriage falling apart','divorce killing me',
      'heartbroken beyond repair','completely heartbroken',
      'lost the will to love','cannot trust anyone','trust is gone',
      'betrayed completely','cheated on and destroyed',
      'abusive relationship','domestic abuse','emotional abuse',
      'mentally abused','verbal abuse','constant fighting',

      // ── Occupational crisis ───────────────────────────────────────────────
      'lost my job','fired','terminated','laid off unexpectedly',
      'business destroyed','company bankrupt','total financial ruin',
      'cannot pay rent','cannot pay bills','facing eviction',
      'deep in debt','drowning in debt','no way to pay back',
      'performance improvement plan','put on pip','failed pip',

      // ── India-specific high distress ─────────────────────────────────────
      'no peace of mind','mind full of tension','too much tension always',
      'heart is hurting badly','completely lost hope',
      'no happiness left','happiness is gone completely',
      'no strength left','lost all will','no will anymore',
      'not interested in life','lost all interest in life',
      'mind is not working','cannot think straight at all',
      'family pressure destroying me','family is suffocating me',
      'in-law torture','dowry pressure','caste discrimination',
      'communal tension','social stigma destroying me'
    ],
    hi: [
      'खाली','सुन्न','फंसा हूं','बोझ हूं','निराशा','टूटा हूं',
      'असफलता','बेकार हूं','छोड़ दिया','अंधेरा','बर्बाद','नष्ट',
      'भूल गए','अनचाहा','अलग-थलग','थका हूं','ऊर्जा नहीं बची',
      'नींद नहीं आती','घुट रहा हूं','डूब रहा हूं','हिम्मत नहीं',
      'उम्मीद खो दी','कोई मतलब नहीं','खुद से नफरत','बहुत दर्द है',
      'टूट गया हूं','हार गया हूं','बहुत रोया','सब बेकार लगता',
      'अकेलापन है','दिल टूटा है','बहुत बुरा लगता','मन भारी है',
      'जी घबराता है','दम घुटता है','कुछ नहीं लगता','सुन्न हो गया',
      'सब से दूर हूं','किसी से नहीं मिलता','घर से नहीं निकलता',
      'बहुत रोता हूं','रात को नींद नहीं','भविष्य नहीं दिखता',
      'दर्द सहन नहीं होता','दिमाग काम नहीं करता',
      'बहुत थक गया हूं जिंदगी से','जीने में मन नहीं',
      'कोई नहीं पूछता','कोई नहीं बोलता','सब भूल गए मुझे',
      'कोई दोस्त नहीं','अकेला हो गया हूं','कोई साथ नहीं',
      'मन में हमेशा डर','घबराहट बहुत होती है','दिल बहुत तेज धड़कता',
      'सांस नहीं आती','छाती बहुत भारी','हमेशा थकान रहती',
      'खाना नहीं खाता','नींद बिल्कुल नहीं','शरीर में बहुत दर्द',
      'पूरी तरह टूट गया','आगे नहीं बढ़ सकता','सहन नहीं हो रहा'
    ]
  },

  // ══════════════════════════════════════════════════════════════════════════
  // TIER 3 — MODERATE (weight 1.0)
  // Mild–moderate emotional strain. Common distress language.
  // Covers everyday sadness, general anxiety, fatigue, social difficulties,
  // work stress, occupational strain, relationship friction, physical complaints.
  // ══════════════════════════════════════════════════════════════════════════
  moderate: {
    weight: 1.0,
    en: [
      // ── Core emotional distress ──────────────────────────────────────────
      'tired','exhausted','sad','cry','crying','anxious','anxiety',
      'worried','worry','nervous','scared','fear','lost','confused',
      'overwhelmed','depressed','depression','hurt','pain','miss',
      'lonely','low','down','bad','awful','terrible','stressed','stress',
      'angry','irritable','helpless','grief','sorrow','frustrated',
      'resentment','bitterness','hostility','fury','mad','annoyed',
      'disappointment','fatigued','dread','guilt','guilty','shame',
      'ashamed','inferior','inadequate','pressure','bored','unmotivated',
      'distant','apathetic','detached','sensitive','irritated',
      'overthinking','empty feeling','not myself','struggling',
      'difficult time','hard time','not okay','no motivation',
      'losing interest','cant focus','heavy','flat','blah','meh',
      'moping','tearful','crying spells','mood swings','emotional',

      // ── Fatigue and low energy ────────────────────────────────────────────
      'burnt out','burnout','depleted','worn out','weary','withdrawn',
      'lethargic','lethargy','sluggish','listless','fatigued',
      'low energy','no energy','no drive','joyless','anhedonia',
      'lost interest','doesnt matter','cant enjoy','going through motions',
      'no enthusiasm','lost passion','disinterested','apathy',
      'not excited anymore','nothing excites me','bored with life',
      'demotivated','cant get motivated','lost all motivation',

      // ── Sleep and physical complaints ────────────────────────────────────
      'insomnia','restless','cant sleep','not sleeping','poor sleep',
      'broken sleep','waking up at night','nightmares','bad dreams',
      'sleep deprived','tired all the time','always tired',
      'headache','migraine','body pain','backache','neck pain',
      'appetite','eating less','no appetite','overeating','comfort eating',
      'weight change','gaining weight','losing weight','feeling sick',
      'nausea','dizzy','dizziness','trembling','shaking','sweating',
      'palpitation','chest pain','tight chest','stomach pain',
      'digestive issues','stress eating','not eating properly',

      // ── Social withdrawal ─────────────────────────────────────────────────
      'socially withdrawn','avoiding people','antisocial','hiding',
      'dont want to go out','staying at home','not leaving house',
      'cancelling plans','avoiding friends','isolating myself',
      'pushing people away','dont want to talk','not responding',
      'silent','quiet all the time','withdrawn from family',
      'nobody around','no one around','feel unwelcome','feel unaccepted',

      // ── Self-doubt and confidence ─────────────────────────────────────────
      'self-doubt','insecure','uncertain','unconfident','not good enough',
      'feel inadequate','feel inferior','imposter','procrastinating',
      'giving up','losing confidence','lost confidence',
      'dont believe in myself','cant trust myself','second guessing',
      'always doubting','never feel good enough','feel stupid',
      'feel dumb','feel like a failure','feel like i failed',
      'disappointed in myself','let myself down','let everyone down',

      // ── Relationship strain ───────────────────────────────────────────────
      'fighting','argument','conflict','disagreement','misunderstanding',
      'relationship problem','relationship issues','partner problem',
      'family conflict','family problem','family tension',
      'heartbreak','breakup','broken up','divorced','separation',
      'loneliness in relationship','feel alone in relationship',
      'not understood by partner','partner doesnt listen',
      'feeling trapped in relationship','toxic behaviour',
      'trust issues','jealousy','possessive','controlling',
      'missing someone','grief over relationship',

      // ── Work and occupational strain ──────────────────────────────────────
      'work stress','job stress','career stress','workplace stress',
      'overworked','too much work','work pressure','deadline stress',
      'bad manager','toxic boss','office politics','workplace conflict',
      'no work-life balance','always working','cant switch off',
      'work from home stress','remote work problems',
      'performance anxiety','target pressure','job insecurity',
      'fear of failure','imposter syndrome','professional setback',
      'career disappointment','promotion denied','passed over',
      'underappreciated at work','no recognition at work',

      // ── Financial strain ──────────────────────────────────────────────────
      'money problems','financial stress','debt','loan pressure',
      'bill anxiety','rent stress','cant afford','money worries',
      'financial difficulty','economic stress','salary issues',
      'no savings','living paycheck to paycheck','broke',
      // ── Workaholism and overwork ──────────────────────────────────────────
      'workaholic','work addict','always working','overworked',
      'work is my life','no time for myself','working non-stop',
      'overcommitted','people pleasing','cant say no',
      'performance anxiety','imposter syndrome',
      'target pressure','appraisal stress','kpi pressure',

      // ── Student-specific ──────────────────────────────────────────────────
      'exam stress','exam pressure','exam anxiety','study pressure',
      'academic pressure','marks pressure','grades dropping',
      'failing exams','cant study','cant concentrate on studies',
      'peer pressure','parent expectations','comparison stress',
      'placement anxiety','campus loneliness','homesick',

      // ── Grief and loss ────────────────────────────────────────────────────
      'lost someone','someone passed away','bereavement',
      'grieving','in mourning','loss of loved one','death of family',
      'missing the person','miss them so much','grief',

      // ── Indian English expressions ────────────────────────────────────────
      'feeling very bad','not feeling good','very bad mood','low mood',
      'feeling dull','mind is dull','not feeling well','feeling off',
      'no interest in anything','not interested in anything',
      'not enjoying anything','nothing gives pleasure','life feels heavy',
      'too much tension','always tension','tension all the time',
      'mind is disturbed','disturbed mind','mentally disturbed',
      'not at all okay','feeling very sad','feeling very low',
      'feeling hopeless','feeling useless','feeling worthless',
      'dont want to do anything','no energy to do anything',
      'no mood for anything','not in mood','mood is very bad',
      'need a break','need rest','so tired of everything',
      'everything is too much','cant handle anymore',
      'everything is overwhelming','life is overwhelming'
    ],
    hi: [
      // Core distress
      'थका','उदास','चिंतित','डरा','खोया','परेशान','अकेला','दुखी',
      'दर्द','रोना','तनाव','गुस्सा','शर्म','अपराधबोध','निराशा','बेचैन',
      'घबराहट','डर','दबाव','उदासी','थकान','बोझिल','मायूस','हताश',
      // Physical complaints
      'नींद नहीं आती','भूख नहीं','सिरदर्द','थकान रहती','शरीर में दर्द',
      'वजन बढ़ा','वजन कम हुआ','खाना कम खाता','याददाश्त कम हुई',
      'पसीना आता है','दिल घबराता','सांस नहीं आती','छाती भारी',
      'नींद उड़ी है','रात को डर लगता','बुरे सपने आते',
      // Social withdrawal
      'किसी से बात नहीं करना','घर पर रहता हूं','बाहर नहीं जाता',
      'दोस्त नहीं मिलता','लोगों से दूर','अकेलापन महसूस',
      'सबसे अलग लगता','कोई समझ नहीं','कोई सुनता नहीं',
      // Low mood expressions
      'मन नहीं लगता','ध्यान नहीं लगता','जी नहीं लगता',
      'काम में मन नहीं','पढ़ाई में मन नहीं','खाना नहीं लगता',
      'खुशी नहीं मिलती','कुछ अच्छा नहीं लगता','हर बात में उदासी',
      'कमजोर','नाकाम','बुरा महसूस','अच्छा नहीं','तकलीफ',
      'कठिन समय','दिल भारी','कोई नहीं सुनता','बेसहारा','सहारा नहीं',
      // Work/study stress
      'काम का बोझ','ऑफिस टेंशन','बॉस का दबाव','डेडलाइन का डर',
      'परीक्षा का डर','नंबर कम आए','पढ़ाई में तनाव','प्लेसमेंट की चिंता',
      // Relationship
      'रिश्ते टूट गए','दिल टूटा','ब्रेकअप','तलाक','परिवार से झगड़ा',
      'पति-पत्नी में झगड़ा','सास-ससुर से परेशान','रिश्तों में दूरी',
      // Motivation loss
      'उत्साह कम हुआ','उत्सुकता नहीं','खुशी नहीं','बोरियत',
      'चिड़चिड़ापन','गुस्सा आता','छोटी बात पर रोना',
      'सब गलत है','कुछ नहीं होगा','बदलाव नहीं होगा',
      'भूल जाता','ध्यान नहीं','सोने के बाद भी थका','आलस',
      'टूटा हुआ महसूस','चिंता होती है','घबराहट बढ़ती है',
      'हिम्मत टूटी','काम नहीं होता','आगे नहीं बढ़ पाता'
    ]
  },

  // ══════════════════════════════════════════════════════════════════════════
  // TIER 4 — PROTECTIVE (weight -1.5)
  // Positive emotional states, resilience, recovery, social support.
  // These REDUCE the risk score — person is doing well or recovering.
  // ══════════════════════════════════════════════════════════════════════════
  protective: {
    weight: -1.5,
    en: [
      // Core positive affect
      'happy','happiness','joyful','joyous','elated','ecstatic','delighted',
      'cheerful','glad','pleased','content','contented','satisfied',
      'blissful','bliss','wonderful','fantastic','amazing','great','good',
      'excellent','splendid','brilliant','superb','marvelous',

      // Gratitude and appreciation
      'grateful','gratitude','thankful','thankfulness','blessed',
      'appreciate','appreciation','cherish','value','treasure',
      'fortunate','lucky','abundance','enough','have so much',

      // Hope and optimism
      'hopeful','hope','optimistic','optimism','positive','positivity',
      'looking forward','excited about','anticipating','can\'t wait',
      'better days','brighter future','things are looking up',
      'progress','improving','getting better','on the mend',
      'recovered','recovering','healing','healed','getting well',

      // Strength and resilience
      'strong','stronger','resilient','resilience','capable','able',
      'confident','confidence','empowered','powerful','determined',
      'focused','motivated','driven','energized','energised',
      'enthusiastic','passionate','inspired','inspiration',
      'proud','accomplished','achieved','successful','winning',
      'overcame','persevered','perseverance','bounced back',
      'bounce back','came through','got through it','survived',

      // Peace and calm
      'peaceful','peace','calm','serene','tranquil','relaxed',
      'at ease','comfortable','settled','grounded','centered','centred',
      'balanced','mindful','present','at peace','inner peace',
      'relieved','relief','free','liberated','light','feeling light',

      // Connection and love
      'loved','love','cherished','cared for','supported','belonging',
      'connected','close','together','bonded','intimacy',
      'friendship','friendship','family','community','accepted',
      'included','welcomed','understood','seen','heard','validated',
      'appreciated','valued','important','needed','wanted',

      // Recovery specific
      'therapy helping','counselling helping','medication working',
      'seeing a therapist','talking to someone','getting help',
      'reached out','asked for help','opened up','shared my feelings',
      'journaling','meditating','exercising helps','sleep is better',
      'eating well','taking care of myself','self-care',

      // Everyday positive expressions
      'feeling nice','feeling good','doing well','all good','much better',
      'so much better','feeling better','better today','good day',
      'enjoyed','enjoyed today','had fun','laughed','smiled',
      'positive energy','good vibes','feeling alive','fresh start',
      'new beginning','turning point','things are better',
      'manageable','coping well','handling it','getting through',
      'okaying','fine','okay now','not bad','pretty good'
    ],
    hi: [
      'खुश','खुशी','प्रसन्न','आनंदित','शुक्रगुजार','उम्मीद','शांत',
      'प्यार','संतुष्ट','मजबूत','अच्छा','बेहतर','प्रेरित','आशावादी',
      'आभारी','ठीक हूं','मुस्कुराना','हँसना','जिंदगी अच्छी',
      'आगे बढ़ना','सुरक्षित','स्वीकार','खुद पर भरोसा','काबिल',
      'जीने की चाहत','रोशनी','उम्मीद है','ठीक हो जाएगा','सब ठीक है',
      'अच्छा लगता','हिम्मत है','साहस','धन्यवाद','आभार',
      'खुद से प्यार','खुद की देखभाल','परिवार के साथ','दोस्तों के साथ',
      'खुशनुमा','मस्त','शानदार','बढ़िया','सुकून','चैन','आराम',
      'मन शांत','मन हल्का हुआ','बेहतर महसूस','ठीक लगता',
      'खुश हूं','मन प्रसन्न','उत्साहित','जोश है','ऊर्जा है',
      'ताकत है','बल है','जीत सकता हूं','कर सकता हूं',
      'कामयाब होऊंगा','यकीन है','विश्वास है','प्रेम मिलता है',
      'साथ है','अकेला नहीं','सहारा है','सब साथ हैं','परिवार साथ है',
      'मदद मिल रही','थेरेपी से फायदा','अच्छी नींद आई',
      'आज खाना खाया','आज बाहर गया','दोस्त से मिला','मन हल्का है',
      'अच्छा दिन था','हँसी आई','मुस्कुराया','खुशी मिली',
      'नई उम्मीद','नई शुरुआत','ठीक होता जा रहा','सुधार हो रहा'
    ]
  }

};

// ══════════════════════════════════════════════════════════════════════════════
// CLINICAL PHRASES — Multi-word (bigrams / trigrams)
// These are phrase-level matches scored separately with boosted weights
// because they carry stronger clinical signal than individual words.
// ══════════════════════════════════════════════════════════════════════════════
const LEXICON_PHRASES = {

  severe: {
    weight: 3.5,
    en: [
      // Direct suicidal ideation phrases
      'want to die','end my life','kill myself','no reason to live',
      'wish i were dead','wish i was dead','not worth living',
      'cant go on','cannot go on','want to disappear forever',
      'end it all','never wake up','better off dead',
      'done with everything','no point in living',
      'wish i didnt exist','wish i didn\'t exist','nothing to live for',
      'life isnt worth','life isn\'t worth','take my own life',
      // Indirect ideation phrases
      'leave this world','go to heaven','go to a better place',
      'be with god','meet my maker','when i am gone','after i am gone',
      'world would be better without me','no one would miss me',
      'nobody would miss me','better without me',
      'i want to vanish','i should disappear','need to escape everything',
      'permanent escape','not want to exist','wish i wasnt here',
      // Self-harm phrases
      'hurt myself','harm myself','cut myself','cutting myself',
      // Absolute hopelessness phrases
      'completely hopeless','no hope at all','lost all hope',
      'nothing will ever change','will never get better',
      'never be okay again','no hope left'
    ],
    hi: [
      'मरना चाहता हूं','जीना नहीं','खत्म कर दूं','कोई उम्मीद नहीं',
      'जीने का मन नहीं','जिंदगी से थक गया','मर जाना बेहतर',
      'सब कुछ छोड़ना','आत्महत्या करना','जीना व्यर्थ है',
      'कोई नहीं चाहता मुझे','सब नफरत करते','इस दुनिया को छोड़ना',
      'स्वर्ग जाना','भगवान के पास जाना','बिल्कुल उम्मीद नहीं'
    ]
  },

  high: {
    weight: 2.5,
    en: [
      // Clinical depression phrases
      'dead inside','feel nothing','falling apart','cant cope',
      'losing hope','shut down','given up','no future',
      'cant function','nothing matters','not myself','dont enjoy',
      'no point','break down','hate my life','cant see the point',
      'constant pain','unbearable pain','feel so alone','completely lost',
      'no energy at all','totally exhausted','deeply depressed',
      'feel so invisible','always feel alone',
      // Panic and anxiety phrases
      'racing thoughts','intrusive thoughts','panic attack',
      'cannot stop overthinking','mind is racing','thoughts racing',
      'scared of the future','scared of my future',
      'completely overwhelmed','totally overwhelmed','paralysed by anxiety',
      'heart is pounding','cannot breathe properly',
      'shortness of breath','hyperventilating',
      'losing control','losing my grip','everything is falling apart',
      'world is collapsing','nothing is okay','spiralling thoughts',
      'constant worry','never at peace','cannot calm down',
      // Social isolation phrases
      'no one understands','nobody understands','no one listens',
      'feel so rejected','feel so abandoned','left alone again',
      'no one around','nobody around','completely isolated',
      'no one to talk to','nobody to turn to','no support system',
      // Burnout and breakdown
      'mental breakdown','nervous breakdown','complete burnout',
      'on the verge of breakdown','breaking down inside',
      'running on empty','nothing left to give',
      // Relationship crisis
      'completely heartbroken','relationship is over',
      'trust is completely broken','been betrayed',
      // Financial crisis
      'drowning in debt','cannot pay bills','facing eviction',
      // India-specific
      'no peace of mind','mind full of tension','heart is hurting',
      'completely lost hope','no happiness left',
      'family is suffocating me','in-law pressure is unbearable'
    ],
    hi: [
      'अंदर से मरा हूं','कुछ नहीं लगता','कोई नहीं','टूट रहा हूं',
      'उम्मीद नहीं बची','कुछ अच्छा नहीं','बहुत अकेला हूं',
      'जी नहीं लगता कुछ भी','बिल्कुल टूट गया','बहुत हार गया',
      'जिंदगी से परेशान हूं','कोई नहीं समझता','सब छोड़ दिया',
      'दिल टूट गया','कोई बात नहीं सुनता','हमेशा अकेला हूं',
      'रेसिंग थॉट्स','पैनिक अटैक','पूरी तरह टूट गया',
      'दिल बहुत तेज धड़कता','सांस लेना मुश्किल है',
      'हर बात से डर लगता','भविष्य से बहुत डर',
      'नींद बिल्कुल नहीं','खाना बिल्कुल नहीं',
      'दिमाग शांत नहीं होता','दिमाग हमेशा चलता रहता'
    ]
  }
};
