# 🧠 MindBridge

> *Built from a conference paper. Deployed as a live app.*

**Live demo:** https://mindbridgehealth-ai.netlify.app/

MindBridge is a browser-based, on-device **research prototype** for mental-wellness screening. It combines text emotion analysis, vocal features, PHQ-9 / GAD-7 questionnaires and word-level explanations in one privacy-first web app. It is **not** a clinical diagnostic tool.

---

## Research background

The app grew out of the paper *"Explainable AI For Early Prediction of Anxiety and Depression in India: A Dual-Modality Framework"* (Bhardwaj & Bharadvaja), accepted and presented at IC3-2026. Proceedings are being published by IEEE.

The paper describes a two-phase framework:

- **Phase 1 – Lifestyle risk:** XGBoost + SMOTE on tabular lifestyle data (~80% accuracy), explained with SHAP.
- **Phase 2 – Text analysis:** TF-IDF + XGBoost on 53,000+ statements (93.73% accuracy, 96.10% precision), explained with LIME.

The trained Phase 1/Phase 2 models (`models/*.pkl`) are used by the optional Flask backend (`app.py`). **The live browser app runs without the backend** (see below).

---

## What the live app does today

| Feature | How it works |
| --- | --- |
| Text emotion | Pretrained **DistilRoBERTa** emotion classifier (`Bachkippe/emotion-english-distilroberta-base_onnx`) run in the browser with Transformers.js. English only. |
| Hindi-English text | Rule-based lexicon with negation handling (simpler than the transformer path) |
| Word-level explanation | Per-word contribution from the lexicon (LIME-style, not LIME run on the model) |
| Vocal features | Web Audio API: pitch (F0), energy, monotony, jitter, shimmer, speech rate, pause ratio. Simple DSP, not Praat-grade. |
| Facial affect | Experimental, in-browser |
| Screening | PHQ-9 and GAD-7 questionnaires (public-domain instruments), with a crisis alert on item 9 |
| Risk score | Hand-weighted fusion: **40% text + 30% voice + 30% face** (weights re-normalised when a modality is missing) |
| Also included | Lifestyle tracker, private journal, CBT exercises, FHIR-style summary export, crisis helplines (including Tele-MANAS 14416) |

All processing happens in the user's browser. Nothing is uploaded and the app has no accounts.

---

## Honest limitations

- This is a **research demonstrator, not a validated diagnostic tool.** It does not diagnose, and it is not a substitute for professional care.
- The fusion weights are hand-chosen and have **not** been validated against clinical ground truth.
- The emotion model is English-only and can be wrong on sarcasm, mixed emotions, and short or ambiguous text.
- The browser app does **not** yet run the paper's trained XGBoost/TF-IDF models. Those are served only by the optional Flask backend.
- Vocal and facial features are exploratory and have not been clinically evaluated.
- The app has not been tested for HIPAA / GDPR compliance.

**Roadmap**
- [ ] Clinical validation of the risk score, especially for Hindi-English text
- [ ] Deploy the Phase 1/2 XGBoost models for the browser app
- [ ] Model-based SHAP / LIME explanations
- [ ] Multilingual emotion model

---

## Repository layout

```
models/                  Web app (static front-end) + trained Phase 1 / Phase 2 pickles
  index.html, manifest.json, sw.js
  css/  js/  icons/  assets/lexicon.js
  *.pkl                  XGBoost / TF-IDF models (scikit-learn 1.6.1)
app.py, requirements.txt Optional Flask backend that serves the paper models
```

## Run locally

Front end (no install needed): serve the `models/` folder, e.g.

```bash
cd models
python -m http.server 8000
```

then open http://localhost:8000 (opening `index.html` directly can block the service worker and the model download). The first load downloads the emotion model, so allow about 30 seconds.

Optional backend:

```bash
pip install -r requirements.txt
python app.py
```

Then set `BACKEND_URL` in `models/js/app.js` to your backend address. It is empty by default, so the app uses the browser models only.

## Deployment

Front-end: Netlify (static). Backend (optional): any Python host (e.g. Render).

---

## Author

**Srimant Bhardwaj**, M.Tech Bioinformatics, Delhi Technological University (DTU)

[LinkedIn](https://www.linkedin.com/in/srimant-bhardwaj-13s23a) · [GitHub](https://github.com/Srimant1323) · [ORCID](https://orcid.org/0009-0007-6395-1216)

*Built because transparent AI has to start with a transparent researcher.*
