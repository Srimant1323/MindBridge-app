/**
 * MindBridge API Client v2
 * Save as: js/api_client.js
 * Add to index.html before </body>:  <script src="js/api_client.js"></script>
 *
 * Then in your existing check-in code replace scoreText(text) with:
 *   const result = await MindBridgeAPI.predict(text);
 */

const MindBridgeAPI = (() => {
  // ── UPDATE THIS after deploying to Render ────────────────────────────────
  const BASE_URL   = "https://mindbridge-app.onrender.com";  // ← your Render URL
  const TIMEOUT_MS = 5000;

  let _status = null;   // null = unchecked, true = up, false = down

  async function ping() {
    if (_status !== null) return _status;
    try {
      const ctrl = new AbortController();
      setTimeout(() => ctrl.abort(), TIMEOUT_MS);
      const res = await fetch(`${BASE_URL}/`, { signal: ctrl.signal });
      const data = await res.json();
      _status = res.ok;
      console.log(`[MindBridge] Backend: ${data.nlp_model} | Lifestyle: ${data.lifestyle_model}`);
    } catch {
      _status = false;
      console.warn("[MindBridge] Backend offline — using browser fallback");
    }
    return _status;
  }

  async function post(endpoint, body) {
    const ctrl = new AbortController();
    const tid  = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      clearTimeout(tid);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      clearTimeout(tid);
      throw err;
    }
  }

  // ── Public methods ────────────────────────────────────────────────────────

  /** NLP text risk prediction (Phase 2 model) */
  async function predict(text) {
    if (await ping()) {
      try {
        const data = await post("/predict", { text });
        return { ...data, source: "backend" };
      } catch (e) {
        console.warn("[MindBridge] /predict failed, falling back:", e.message);
      }
    }
    return { ...browserHeuristic(text), source: "browser" };
  }

  /**
   * Lifestyle risk prediction (Phase 1 model)
   * @param {object} lifestyle - keys must match your Phase 1 CSV columns exactly
   * Example: { Age:22, Gender:"Male", Sleep_Hours:5, Physical_Activity_Hours:1,
   *             Work_Study_Hours:10, Social_Support_Score:3,
   *             Financial_Stress:4, Education_Level:"Undergraduate",
   *             Employment_Status:"Student" }
   */
  async function predictLifestyle(lifestyle) {
    if (await ping()) {
      try {
        return await post("/predict_lifestyle", lifestyle);
      } catch (e) {
        console.warn("[MindBridge] /predict_lifestyle failed:", e.message);
      }
    }
    return null;  // no browser fallback for tabular model
  }

  async function phq9(answers) {
    if (await ping()) {
      try { return await post("/phq9", { answers }); } catch {}
    }
    return browserPHQ9(answers);
  }

  async function gad7(answers) {
    if (await ping()) {
      try { return await post("/gad7", { answers }); } catch {}
    }
    return browserGAD7(answers);
  }

  // ── Browser fallbacks ─────────────────────────────────────────────────────
  function browserHeuristic(text) {
    const HIGH = {
      hopeless:9,worthless:9,suicidal:10,suicide:10,"kill myself":10,
      "end it all":10,"self harm":9,"self-harm":9,cutting:7,depressed:7,
      depression:7,numb:6,empty:6,hollow:6,trapped:8,burden:7,"give up":7,
      pointless:6,meaningless:6,exhausted:5,drained:5,crying:5,lonely:6,
      alone:5,isolated:6,anxious:5,panic:6,overwhelmed:6,failure:6,useless:7,
      "hate myself":8,disappear:7,nirasha:8,bekar:7,akela:6,dard:5,
      takleef:5,dukh:5,rona:4,"thak gaya":5,"thak gayi":5,
      "zindagi se tang":9,"marna chahta":10,"marna chahti":10,
      "khatam karna":8,"haar gaya":6,"haar gayi":6,
      "koi nahi":6,"akela hoon":6,tanha:6,
    };
    const PROT = {
      happy:-4,hopeful:-5,grateful:-5,loved:-5,joy:-4,excited:-4,calm:-4,
      peaceful:-4,supported:-5,better:-3,improving:-4,healing:-4,therapy:-3,
      friends:-3,family:-3,khush:-4,khushi:-4,umeed:-5,shukriya:-4,pyaar:-4,
    };
    const CRISIS = ["suicidal","suicide","kill myself","marna chahta","marna chahti",
                    "wish i was dead","tired of living","zindagi se tang","khatam karna"];
    const t = text.toLowerCase();
    let total = 0;
    const contributions = {};
    for (const [p,w] of Object.entries(HIGH)) if (t.includes(p)) { contributions[p]=w; total+=w; }
    for (const [p,w] of Object.entries(PROT)) if (t.includes(p)) { contributions[p]=w; total+=w; }
    const wc = Math.max(text.split(/\s+/).length, 1);
    let score = Math.min(Math.max(Math.round(total/Math.sqrt(wc)*6),0),100);
    if (CRISIS.some(c=>t.includes(c))) score = Math.max(score,70);
    const rl = score<25?"Low":score<50?"Moderate":score<75?"High":"Severe";
    return { score, risk_level:rl, crisis:score>=75||CRISIS.some(c=>t.includes(c)),
             contributions, model_used:"browser_heuristic" };
  }

  function browserPHQ9(answers) {
    const total = answers.reduce((a,b)=>a+b,0);
    const s = total<=4?"Minimal":total<=9?"Mild":total<=14?"Moderate":total<=19?"Moderately Severe":"Severe";
    return { total, severity:s, crisis:answers[8]>=1, max_possible:27,
             percentage:Math.round(total/27*100*10)/10 };
  }
  function browserGAD7(answers) {
    const total = answers.reduce((a,b)=>a+b,0);
    const s = total<=4?"Minimal":total<=9?"Mild":total<=14?"Moderate":"Severe";
    return { total, severity:s, max_possible:21, percentage:Math.round(total/21*100*10)/10 };
  }

  return { predict, predictLifestyle, phq9, gad7, ping };
})();

window.MindBridgeAPI = MindBridgeAPI;

