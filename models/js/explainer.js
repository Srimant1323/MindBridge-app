/**
 * explainer.js — LIME-Style Explainability Module
 * Highlights risk/protective words in the user's text
 * and renders a contribution bar chart.
 */
const explainer = (function () {

  /**
   * Render highlighted text into `#highlighted-text`.
   * Each recognized word is wrapped in a colored <span>.
   */
  function highlightText(originalText, wordHits) {
    const container = document.getElementById('highlighted-text');
    if (!container) return;

    if (!originalText || !wordHits || Object.keys(wordHits).length === 0) {
      container.textContent = originalText || '';
      return;
    }

    // Build a regex from all matched words (sorted longest-first to avoid partial matches)
    const words = Object.keys(wordHits).sort((a, b) => b.length - a.length);
    const escapedWords = words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const pattern = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'gi');

    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = pattern.exec(originalText)) !== null) {
      // Text before match
      if (match.index > lastIndex) {
        parts.push(document.createTextNode(originalText.slice(lastIndex, match.index)));
      }

      const word = match[0].toLowerCase();
      const data = wordHits[word];
      const span = document.createElement('span');
      span.textContent = match[0];

      if (data) {
        const cssClass = {
          severe:     'highlight-risk',
          high:       'highlight-high-risk',
          moderate:   'highlight-moderate',
          protective: 'highlight-protective'
        }[data.level] || '';
        if (cssClass) span.className = cssClass;
        span.title = `${data.level.toUpperCase()} | weight: ${data.weight > 0 ? '+' : ''}${data.weight}`;
      }

      parts.push(span);
      lastIndex = match.index + match[0].length;
    }

    // Remaining text
    if (lastIndex < originalText.length) {
      parts.push(document.createTextNode(originalText.slice(lastIndex)));
    }

    container.innerHTML = '';
    parts.forEach(p => container.appendChild(p));
  }

  /**
   * Render a horizontal bar chart of top contributing words.
   * Each bar is proportional to word weight relative to the max in the set.
   */
  function renderWordChart(triggerWords, protectiveWords) {
    const container = document.getElementById('word-chart');
    const wrap      = document.getElementById('word-chart-wrap');
    if (!container) return;
    container.innerHTML = '';

    const allWords = [...triggerWords, ...protectiveWords];
    if (allWords.length === 0) {
      wrap && (wrap.style.display = 'none');
      return;
    }

    wrap && (wrap.style.display = 'block');
    const maxWeight = Math.max(...allWords.map(w => Math.abs(w.weight)));

    allWords.forEach(({ word, weight, level }) => {
      const pct = Math.round((Math.abs(weight) / maxWeight) * 100);
      const barClass = {
        severe:     'bar-risk',
        high:       'bar-high-risk',
        moderate:   'bar-moderate',
        protective: 'bar-protective'
      }[level] || 'bar-moderate';

      const row = document.createElement('div');
      row.className = 'word-bar-row';
      row.innerHTML = `
        <span class="word-bar-label"></span>
        <div class="word-bar-track">
          <div class="word-bar-fill ${barClass}" style="width: 0%" data-pct="${pct}"></div>
        </div>
        <span class="word-bar-score">${weight > 0 ? '+' : ''}${weight.toFixed(1)}</span>
      `;
      const lbl = row.querySelector('.word-bar-label');
      lbl.textContent = word; lbl.title = word;
      container.appendChild(row);
    });

    // Animate bars after paint
    requestAnimationFrame(() => {
      container.querySelectorAll('.word-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.pct + '%';
      });
    });
  }

  /**
   * Update the vocal analysis column in the report view.
   */
  function renderVocalResults(metrics) {
    const container = document.getElementById('vocal-results');
    if (!container) return;

    if (!metrics) {
      container.innerHTML = '<p class="no-voice-msg">No voice recording detected. Use the microphone in Check-in to enable vocal biomarker analysis.</p>';
      return;
    }

    const rows = [
      {
        name: 'Fundamental Frequency (F0)',
        val:  `${metrics.pitchMean} Hz`,
        pct:  Math.min((metrics.pitchMean / 400) * 100, 100),
        insight: metrics.pitchMean < 100
          ? 'Low average pitch. Lower pitch is discussed in depression research, but it varies a lot between people (and by voice type), so it is not diagnostic.'
          : 'Pitch within a typical conversational range.'
      },
      {
        name: 'Voice Energy',
        val:  (metrics.energyMean * 1000).toFixed(1) + ' (norm.)',
        pct:  Math.min(metrics.energyMean * 500, 100),
        insight: metrics.energyMean < 0.02
          ? 'Low vocal energy can reflect reduced affect or fatigue — or simply a quiet room or distant microphone.'
          : 'Energy level suggests engaged speech.'
      },
      {
        name: 'Monotony Score',
        val:  metrics.monoScore !== null
          ? (metrics.monoScore < 0.35 ? 'Varied' : metrics.monoScore < 0.65 ? 'Medium' : 'Flat')
          : '—',
        pct:  metrics.monoScore !== null ? Math.round(metrics.monoScore * 100) : 0,
        insight: metrics.monoScore > 0.6
          ? 'Flat pitch (low F0 variation) is associated with reduced expressiveness in research literature.'
          : 'Good pitch variation detected.'
      },
      {
        name: 'Jitter (period variation)',
        val:  (metrics.jitter * 100).toFixed(1) + '%',
        pct:  Math.min(metrics.jitter * 1000, 100),
        insight: 'Frame-to-frame variation of the pitch period. Defined like clinical jitter but measured on 50 ms frames with a browser microphone, so values are higher than lab (Praat) values.'
      },
      {
        name: 'Shimmer (amplitude variation)',
        val:  (metrics.shimmer * 100).toFixed(1) + '%',
        pct:  Math.min(metrics.shimmer * 300, 100),
        insight: 'Frame-to-frame variation of peak amplitude. Sensitive to microphone and distance.'
      },
      {
        name: 'Speech Rate',
        val:  metrics.speechRate != null ? metrics.speechRate + ' syll./s (est.)' : '—',
        pct:  metrics.speechRate != null ? Math.min(metrics.speechRate / 8 * 100, 100) : 0,
        insight: 'Estimated from peaks in the loudness envelope. Typical conversational speech is roughly 3–6 syllables per second.'
      },
      {
        name: 'Pause Ratio',
        val:  metrics.pauseRatio != null ? Math.round(metrics.pauseRatio * 100) + '%' : '—',
        pct:  metrics.pauseRatio != null ? Math.round(metrics.pauseRatio * 100) : 0,
        insight: 'Share of the recording that was silence. Longer pauses are studied as a marker of psychomotor slowing.'
      }
    ];

    container.innerHTML = '';
    rows.forEach(({ name, val, pct, insight }) => {
      const div = document.createElement('div');
      div.className = 'vocal-meter-row';
      div.innerHTML = `
        <div class="vocal-meter-header">
          <span class="vocal-meter-name">${name}</span>
          <span class="vocal-meter-val">${val}</span>
        </div>
        <div class="vocal-meter-bar">
          <div class="vocal-meter-fill" style="width: 0%" data-pct="${pct}"></div>
        </div>
        <p class="vocal-insight">${insight}</p>
      `;
      container.appendChild(div);
    });

    // Animate fills
    requestAnimationFrame(() => {
      container.querySelectorAll('.vocal-meter-fill').forEach(el => {
        el.style.width = el.dataset.pct + '%';
      });
    });
  }

  return { highlightText, renderWordChart, renderVocalResults };
})();
