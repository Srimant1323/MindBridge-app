/**
 * journal.js — MindBridge Private Daily Journal Module v2
 * Fully private: all entries stored in localStorage only.
 * v2: Added 7-language support (en/hi/es/fr/te/bn/mr), improved init,
 *     animated prompts, staggered entry rendering.
 * Research ref: Pennebaker JW, "Writing to Heal", APA, 2004.
 */
const journal = (function () {

  const STORAGE_KEY = 'mindbridge_journal';
  let currentLang   = 'en';

  // ── Mood tags ──────────────────────────────────────────────────────────────
  const MOOD_TAGS = [
    { key: 'grateful',  emoji: '💚', label: 'Grateful'  },
    { key: 'happy',     emoji: '😊', label: 'Happy'     },
    { key: 'okay',      emoji: '😐', label: 'Okay'      },
    { key: 'sad',       emoji: '😢', label: 'Sad'       },
    { key: 'anxious',   emoji: '😰', label: 'Anxious'   },
    { key: 'angry',     emoji: '😤', label: 'Angry'     },
    { key: 'hopeful',   emoji: '🌱', label: 'Hopeful'   },
    { key: 'exhausted', emoji: '😴', label: 'Exhausted' }
  ];

  // ── Multilingual strings ───────────────────────────────────────────────────
  const LANG_STRINGS = {
    en: {
      eyebrow: "TODAY'S REFLECTION PROMPT",
      moodLabel: 'How are you feeling?',
      placeholder: "Write anything that's on your mind…",
      saveBtn: '💾 Save Entry',
      prompts: [
        'What is one thing that made you feel something today?',
        'Describe your energy level. What drained you? What charged you?',
        'What would you tell a close friend in your exact situation right now?',
        'Write about a worry without trying to fix it.',
        'What are you grateful for today, even if it\'s small?',
        'What did your body feel like today? Tension, calm, pain, lightness?',
        'Describe your mood as a weather pattern.',
        'What is one thing you are avoiding thinking about?',
        'Write a letter to your future self, one week from now.',
        'What went better than expected today?'
      ]
    },
    hi: {
      eyebrow: 'आज का विचार संकेत',
      moodLabel: 'आप कैसा महसूस कर रहे हैं?',
      placeholder: 'जो मन में आए, लिखें…',
      saveBtn: '💾 प्रविष्टि सहेजें',
      prompts: [
        'आज आपको सबसे ज़्यादा किस बात ने छू गई?',
        'आज का सबसे कठिन पल कौन सा था?',
        'अगर आपका कोई करीबी आपकी जगह होता, तो आप उसे क्या कहते?',
        'किसी एक चीज़ के बारे में लिखें जिसके लिए आप आज आभारी हैं।',
        'आपका शरीर आज कैसा महसूस कर रहा था — तनाव, शांति, थकान?',
        'आज का मौसम आपके मन जैसा होता तो कैसा होता?',
        'एक ऐसी बात जिसे आप टालना चाहते हैं, उसके बारे में लिखें।',
        'अपने भविष्य के आप को एक पत्र लिखें।',
        'आज क्या उम्मीद से बेहतर रहा?',
        'आज की सबसे बड़ी चुनौती क्या थी?'
      ]
    },
    es: {
      eyebrow: 'PROMPT DE REFLEXIÓN DEL DÍA',
      moodLabel: '¿Cómo te sientes hoy?',
      placeholder: 'Escribe lo que tengas en mente…',
      saveBtn: '💾 Guardar entrada',
      prompts: [
        '¿Qué fue lo que más te impactó emocionalmente hoy?',
        'Describe tu nivel de energía. ¿Qué te agotó? ¿Qué te recargó?',
        '¿Qué le dirías a un amigo cercano en tu situación actual?',
        'Escribe sobre una preocupación sin intentar resolverla.',
        '¿Por qué estás agradecido/a hoy, aunque sea algo pequeño?',
        '¿Cómo se sintió tu cuerpo hoy? ¿Tensión, calma, dolor, ligereza?',
        'Describe tu estado de ánimo como un patrón climático.',
        '¿Qué es algo que estás evitando pensar?',
        'Escribe una carta a tu yo del futuro, dentro de una semana.',
        '¿Qué salió mejor de lo esperado hoy?'
      ]
    },
    fr: {
      eyebrow: "INVITATION À LA RÉFLEXION DU JOUR",
      moodLabel: 'Comment vous sentez-vous ?',
      placeholder: 'Écrivez librement ce qui vous passe par la tête…',
      saveBtn: '💾 Sauvegarder',
      prompts: [
        "Qu'est-ce qui vous a le plus touché(e) aujourd'hui ?",
        "Décrivez votre niveau d'énergie. Qu'est-ce qui vous a épuisé(e) ?",
        "Que diriez-vous à un(e) ami(e) proche dans votre situation ?",
        'Écrivez sur une inquiétude sans chercher à la résoudre.',
        "Pour quoi êtes-vous reconnaissant(e) aujourd'hui ?",
        "Comment votre corps s'est-il senti aujourd'hui ?",
        "Décrivez votre humeur comme une météo.",
        'Quelle est une chose que vous évitez de penser ?',
        'Écrivez une lettre à votre futur moi, dans une semaine.',
        "Qu'est-ce qui s'est mieux passé que prévu ?"
      ]
    },
    te: {
      eyebrow: 'నేటి ప్రతిఫలన ప్రశ్న',
      moodLabel: 'మీరు ఎలా అనుభవిస్తున్నారు?',
      placeholder: 'మీ మనసులో ఉన్నది రాయండి…',
      saveBtn: '💾 నమోదు సేవ్ చేయండి',
      prompts: [
        'నేడు మిమ్మల్ని అత్యంత ప్రభావితం చేసింది ఏమిటి?',
        'మీ శక్తి స్థాయిని వివరించండి. ఏది మిమ్మల్ని అలసిపోయేలా చేసింది?',
        'మీ పరిస్థితిలో స్నేహితుడికి మీరు ఏమి చెప్తారు?',
        'ఒక ఆందోళన గురించి రాయండి, దాన్ని పరిష్కరించకుండా.',
        'నేడు మీరు ఏదైనా చిన్న విషయంలో కృతజ్ఞతగా ఉన్నారా?',
        'నేడు మీ శరీరం ఎలా అనిపించింది?',
        'మీ మనసు స్థితిని వాతావరణంగా వివరించండి.',
        'మీరు ఏ విషయాన్ని ఆలోచించడం మానేశారు?',
        'మీ భవిష్యత్ ఆత్మకు ఒక వారం తర్వాత లేఖ రాయండి.',
        'ఈరోజు ఏది అంచనా కంటే బాగా జరిగింది?'
      ]
    },
    bn: {
      eyebrow: 'আজকের প্রতিফলন প্রশ্ন',
      moodLabel: 'আপনি কেমন অনুভব করছেন?',
      placeholder: 'মনে যা আসছে তাই লিখুন…',
      saveBtn: '💾 এন্ট্রি সংরক্ষণ করুন',
      prompts: [
        'আজ কোন বিষয়টি আপনাকে সবচেয়ে বেশি প্রভাবিত করেছে?',
        'আপনার শক্তির মাত্রা বর্ণনা করুন। কী আপনাকে ক্লান্ত করেছে?',
        'আপনার পরিস্থিতিতে কোনো ঘনিষ্ঠ বন্ধুকে আপনি কী বলতেন?',
        'একটি উদ্বেগ নিয়ে লিখুন, তা সমাধান না করেই।',
        'আজ আপনি কোনো ছোট বিষয়ের জন্য কৃতজ্ঞ?',
        'আজ আপনার শরীর কেমন লেগেছিল?',
        'আপনার মেজাজকে একটি আবহাওয়ার মতো বর্ণনা করুন।',
        'কোন একটি বিষয় আপনি এড়িয়ে চলছেন?',
        'এক সপ্তাহ পরের ভবিষ্যৎ নিজের কাছে একটি চিঠি লিখুন।',
        'আজ কী প্রত্যাশার চেয়ে ভালো হয়েছে?'
      ]
    },
    mr: {
      eyebrow: 'आजचा प्रतिबिंब प्रश्न',
      moodLabel: 'तुम्हाला कसे वाटत आहे?',
      placeholder: 'मनात जे येईल ते लिहा…',
      saveBtn: '💾 नोंद जतन करा',
      prompts: [
        'आज कोणती गोष्ट तुम्हाला सर्वाधिक भावली?',
        'तुमची ऊर्जा पातळी सांगा. काय थकवले? काय उत्साहित केले?',
        'तुमच्या परिस्थितीत एखाद्या जवळच्या मित्राला तुम्ही काय सांगाल?',
        'एखाद्या काळजीबद्दल लिहा, ती सोडवण्याचा प्रयत्न न करता.',
        'आज तुम्ही कशाबद्दल कृतज्ञ आहात, अगदी लहान गोष्ट असली तरी?',
        'आज तुमचे शरीर कसे वाटले — ताण, शांती, वेदना?',
        'तुमचा मूड एखाद्या हवामानासारखा सांगा.',
        'एखादी गोष्ट जिचा विचार करणे तुम्ही टाळत आहात?',
        'एक आठवड्यानंतरच्या तुमच्यासाठी एक पत्र लिहा.',
        'आज काय अपेक्षेपेक्षा चांगले झाले?'
      ]
    }
  };

  let selectedMood = null;
  let searchQuery  = '';
  let filterMood   = null;

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // ── Storage ────────────────────────────────────────────────────────────────
  function loadEntries() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch { return []; }
  }

  function saveEntries(entries) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  }

  // ── Set language ───────────────────────────────────────────────────────────
  function setLang(langCode) {
    currentLang = langCode;
    const L = LANG_STRINGS[langCode] || LANG_STRINGS.en;

    // Eyebrow label
    const eyebrow = document.getElementById('journal-prompt-eyebrow');
    if (eyebrow) eyebrow.textContent = L.eyebrow;

    // Mood label
    const moodLabel = document.getElementById('journal-mood-label');
    if (moodLabel) moodLabel.textContent = L.moodLabel;

    // Textarea placeholder
    const ta = document.getElementById('journal-content');
    if (ta) ta.placeholder = L.placeholder;

    // Save button
    const saveBtn = document.querySelector('#view-journal .btn-primary');
    if (saveBtn) saveBtn.innerHTML = L.saveBtn;

    // Show a new prompt in the chosen language
    const promptEl = document.getElementById('journal-prompt');
    if (promptEl) {
      promptEl.style.opacity = 0;
      setTimeout(() => {
        const r = L.prompts[Math.floor(Math.random() * L.prompts.length)];
        promptEl.textContent = '💬 ' + r;
        promptEl.style.transition = 'opacity 0.4s';
        promptEl.style.opacity    = 1;
      }, 200);
    }

    // Highlight active pill
    document.querySelectorAll('.jlang-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.lang === langCode);
    });
  }

  // ── Mood picker ────────────────────────────────────────────────────────────
  function setMood(key) {
    selectedMood = key;
    document.querySelectorAll('.journal-mood-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mood === key);
    });
  }

  // ── Random prompt ──────────────────────────────────────────────────────────
  function newPrompt() {
    const el = document.getElementById('journal-prompt');
    if (!el) return;
    const L = LANG_STRINGS[currentLang] || LANG_STRINGS.en;
    el.style.opacity = 0;
    setTimeout(() => {
      const r = L.prompts[Math.floor(Math.random() * L.prompts.length)];
      el.textContent = '💬 ' + r;
      el.style.transition = 'opacity 0.35s';
      el.style.opacity    = 1;
    }, 200);
  }

  // ── Save entry ─────────────────────────────────────────────────────────────
  function saveEntry() {
    const titleEl   = document.getElementById('journal-title');
    const contentEl = document.getElementById('journal-content');
    const title   = titleEl   ? titleEl.value.trim()   : '';
    const content = contentEl ? contentEl.value.trim() : '';

    if (!content || content.length < 5) {
      if (typeof app !== 'undefined') app.toast('Please write at least a few words before saving.', 'warn');
      return;
    }

    const words = content.split(/\s+/).filter(Boolean).length;
    const entry = {
      id:      Date.now(),
      title:   title || null,
      content,
      mood:    selectedMood,
      lang:    currentLang,
      words,
      date:    new Date().toISOString()
    };

    const entries = loadEntries();
    entries.unshift(entry);
    saveEntries(entries);

    // Reset form
    if (titleEl)   titleEl.value   = '';
    if (contentEl) contentEl.value = '';
    setMood(null);
    updateWordCount(0);

    renderEntries();
    renderMoodSummary();
    if (typeof app !== 'undefined') app.toast(`Entry saved! ${words} words written 📔`, 'success');

    // Scroll to entries
    setTimeout(() => {
      document.getElementById('journal-entries')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
  }

  // ── Word count ─────────────────────────────────────────────────────────────
  function updateWordCount(count) {
    const el = document.getElementById('journal-word-count');
    if (el) el.textContent = count + ' words';
  }

  function onContentChange(textarea) {
    const words = textarea.value.split(/\s+/).filter(Boolean).length;
    updateWordCount(words);
  }

  // ── Delete entry ───────────────────────────────────────────────────────────
  function deleteEntry(id) {
    if (!confirm('Delete this journal entry? This cannot be undone.')) return;
    const entries = loadEntries().filter(e => e.id !== id);
    saveEntries(entries);
    renderEntries();
    renderMoodSummary();
    if (typeof app !== 'undefined') app.toast('Entry deleted.', 'info');
  }

  // ── Search ─────────────────────────────────────────────────────────────────
  function onSearch(query) {
    searchQuery = query.toLowerCase().trim();
    renderEntries();
  }

  function setFilterMood(key) {
    filterMood = filterMood === key ? null : key;
    document.querySelectorAll('.journal-filter-mood').forEach(b => {
      b.classList.toggle('active', b.dataset.mood === filterMood);
    });
    renderEntries();
  }

  // ── Render entry list ──────────────────────────────────────────────────────
  function renderEntries() {
    const container = document.getElementById('journal-entries');
    if (!container) return;

    let entries = loadEntries();

    if (filterMood)   entries = entries.filter(e => e.mood === filterMood);
    if (searchQuery) {
      entries = entries.filter(e =>
        (e.content || '').toLowerCase().includes(searchQuery) ||
        (e.title   || '').toLowerCase().includes(searchQuery)
      );
    }

    if (!entries.length) {
      const allEntries = loadEntries();
      container.innerHTML = `
        <div class="journal-empty-state">
          <div class="jes-icon">${allEntries.length ? '🔍' : '📔'}</div>
          <p class="jes-text">${allEntries.length
            ? 'No entries match your filter or search.'
            : 'No entries yet. Write your first one above!'}</p>
        </div>`;
      return;
    }

    // Group by date
    const groups = {};
    entries.forEach(e => {
      const key = new Date(e.date).toLocaleDateString('en-IN', {
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
      });
      if (!groups[key]) groups[key] = [];
      groups[key].push(e);
    });

    container.innerHTML = Object.entries(groups).map(([dateStr, dayEntries]) => `
      <div class="journal-date-group">
        <div class="journal-date-label">${dateStr}</div>
        ${dayEntries.map(e => {
          const moodMeta = MOOD_TAGS.find(m => m.key === e.mood);
          const preview  = e.content.length > 180 ? e.content.slice(0, 180) + '…' : e.content;
          const timeStr  = new Date(e.date).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
          const langFlag = { en:'🇬🇧', hi:'🇮🇳', es:'🇪🇸', fr:'🇫🇷', te:'🇮🇳', bn:'🇧🇩', mr:'🇮🇳' }[e.lang] || '';
          return `
            <div class="journal-entry-card" id="jentry-${e.id}">
              <div class="journal-entry-header">
                ${e.title ? `<span class="journal-entry-title">${esc(e.title)}</span>` : ''}
                <span class="journal-entry-time">${langFlag} ${timeStr}</span>
                ${moodMeta ? `<span class="journal-mood-tag">${moodMeta.emoji} ${moodMeta.label}</span>` : ''}
                <button class="journal-delete-btn" onclick="journal.deleteEntry(${e.id})" title="Delete entry">✕</button>
              </div>
              <div class="journal-entry-content" id="jcontent-${e.id}">${esc(preview)}</div>
              ${e.content.length > 180 ? `<button class="journal-read-more" onclick="journal.expandEntry(${e.id})">Read more ▼</button>` : ''}
              <div class="journal-entry-footer" style="display:flex;justify-content:space-between;align-items:center">
                <span>${e.words} words</span>
                <button class="btn-text-link j-cbt-btn" onclick="journal.analyzeDistortions(${e.id})" style="font-size:0.75rem;color:var(--purple,#c084fc);display:inline-flex;align-items:center;gap:4px;cursor:pointer;background:rgba(192,132,252,0.1);border:1px solid rgba(192,132,252,0.25);padding:3px 10px;border-radius:20px;font-weight:600">
                  🧠 CBT Reframe
                </button>
              </div>
              <div id="j-cbt-results-${e.id}" class="j-cbt-results" style="display:none;margin-top:12px"></div>
            </div>
          `;
        }).join('')}
      </div>
    `).join('');
  }

  function expandEntry(id) {
    const e  = loadEntries().find(e => e.id === id);
    if (!e) return;
    const el = document.getElementById(`jcontent-${id}`);
    if (el) {
      el.textContent = e.content;
      const btn = el.nextElementSibling;
      if (btn && btn.classList.contains('journal-read-more')) btn.remove();
    }
  }

  // ── Mood summary ───────────────────────────────────────────────────────────
  function renderMoodSummary() {
    const entries   = loadEntries();
    const container = document.getElementById('journal-mood-summary');
    if (!container) return;

    if (!entries.length) {
      container.innerHTML = `<p class="empty-state" style="font-size:0.8rem;padding:8px 0">Save entries to see mood trends 📊</p>`;
      return;
    }

    const counts = {};
    entries.forEach(e => { if (e.mood) counts[e.mood] = (counts[e.mood] || 0) + 1; });

    if (!Object.keys(counts).length) {
      container.innerHTML = `<p class="empty-state" style="font-size:0.8rem;padding:8px 0">Tag a mood next time you write!</p>`;
      return;
    }

    const total  = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5);

    container.innerHTML = `
      <div class="mood-summary-title">Your Most Common Moods</div>
      ${sorted.map(([key, count]) => {
        const m   = MOOD_TAGS.find(m => m.key === key) || { emoji: '❓', label: key };
        const pct = Math.round((count / total) * 100);
        return `
          <div class="mood-summary-row">
            <span class="mood-summary-emoji">${m.emoji}</span>
            <span class="mood-summary-label">${m.label}</span>
            <div class="mood-summary-track">
              <div class="mood-summary-fill" style="width:${pct}%"></div>
            </div>
            <span class="mood-summary-pct">${pct}%</span>
          </div>
        `;
      }).join('')}
      <div class="mood-summary-footer">${entries.length} entries · ${Object.keys(counts).length} moods recorded</div>
    `;
  }

  // ── Render mood picker ─────────────────────────────────────────────────────
  function renderMoodPicker() {
    const container = document.getElementById('journal-mood-picker');
    if (!container) return;
    container.innerHTML = MOOD_TAGS.map(m => `
      <button class="journal-mood-btn" data-mood="${m.key}" onclick="journal.setMood('${m.key}')" title="${m.label}">
        <span class="mood-emoji">${m.emoji}</span>
        <span class="mood-label">${m.label}</span>
      </button>
    `).join('');
  }

  // ── Render filter bar ──────────────────────────────────────────────────────
  function renderFilterBar() {
    const container = document.getElementById('journal-filter-bar');
    if (!container) return;
    container.innerHTML = MOOD_TAGS.map(m => `
      <button class="journal-filter-mood" data-mood="${m.key}" onclick="journal.setFilterMood('${m.key}')" title="Filter by ${m.label}">
        ${m.emoji}
      </button>
    `).join('');
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  function init() {
    renderMoodPicker();
    renderFilterBar();
    renderEntries();
    renderMoodSummary();
    setLang(currentLang);   // populates prompt + placeholder in current lang
  }

  // ── CBT Distortion Analysis Integration ─────────────────────────────────────
  function analyzeDraftDistortions() {
    const contentEl = document.getElementById('journal-content');
    const content = contentEl ? contentEl.value.trim() : '';
    if (!content || content.length < 8) {
      if (typeof app !== 'undefined') app.toast('Write a few thoughts in your journal draft first.', 'info');
      return;
    }
    const targetEl = document.getElementById('journal-draft-cbt-results');
    if (!targetEl) return;
    targetEl.style.display = 'block';
    if (typeof cbtAgent !== 'undefined') {
      const res = cbtAgent.analyzeText(content);
      cbtAgent.renderDistortionCards(res, targetEl);
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function analyzeDistortions(id) {
    const entry = loadEntries().find(e => e.id === id);
    if (!entry) return;
    const targetEl = document.getElementById(`j-cbt-results-${id}`);
    if (!targetEl) return;
    if (targetEl.style.display === 'block') {
      targetEl.style.display = 'none';
      return;
    }
    targetEl.style.display = 'block';
    if (typeof cbtAgent !== 'undefined') {
      const res = cbtAgent.analyzeText(entry.content);
      cbtAgent.renderDistortionCards(res, targetEl);
    }
  }

  return {
    init,
    setLang,
    setMood,
    saveEntry,
    deleteEntry,
    expandEntry,
    onSearch,
    onContentChange,
    setFilterMood,
    newPrompt,
    analyzeDraftDistortions,
    analyzeDistortions
  };

})();
