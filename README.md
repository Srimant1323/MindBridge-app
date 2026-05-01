# 🧠 MindBridge

> *Built from a conference paper. Deployed as a live app.*

**Explainable AI for Early Prediction of Anxiety and Depression in India** — a dual-modality XAI framework turned into a browser-accessible mental health tool.

---

## The Problem

India has a 13.7% prevalence of mental disorders, with 280 million affected globally. Early detection remains limited due to stigma, underreporting, and AI systems that give predictions without explanations. A black-box model in mental health is not just unhelpful — it can be harmful.

---

## The Research Behind It

This app is built from a conference paper — *"Explainable AI for Early Prediction of Anxiety and Depression in India: A Dual-Modality Framework"* — currently submitted for review.

The paper proposes a two-phase XAI framework:

**Phase 1 — Lifestyle Risk Analysis**
- 1,200 records covering sleep, activity, stress and social support
- XGBoost + SMOTE → ~80% accuracy
- SHAP for global feature importance — the model explains *why* it flagged a risk

**Phase 2 — Behavioral Text Analysis**
- 53,000+ text statements
- TF-IDF + XGBoost → 93.73% accuracy · 96.10% precision
- LIME for word-level reasoning — highlights exactly which words drove the prediction

---

## What the App Does

MindBridge takes the research off the page and into a browser — accessible to anyone with a phone or laptop.

| Feature | Description |
| --- | --- |
| 📊 Lifestyle Tracker | Daily sleep, activity and stress logs mapped to Phase 1 |
| 💬 NLP Text Check-in | Emotional text input with LIME-style word attribution |
| 🎙️ Vocal Biomarkers | Real-time pitch, RMS energy, jitter and shimmer via WebAudio API |
| 🤖 BERT Emotion AI | Emotion classification via Transformers.js in-browser |
| 📋 PHQ-9 & GAD-7 | WHO-validated depression and anxiety screening |
| 🌐 Bilingual Support | Hindi and English language support |
| 📓 Private Journal | In-app journaling for self-reflection |
| 🧘 CBT Exercises | Cognitive behavioural therapy exercises |
| 🆘 Crisis Helplines | Direct links to India mental health crisis helplines |

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Backend** | Python, Flask, Flask-CORS |
| **ML Models** | XGBoost, scikit-learn, joblib |
| **NLP** | TF-IDF Vectorizer, clinical heuristic lexicon |
| **Explainability** | SHAP (Phase 1), LIME-style attribution (Phase 2) |
| **Frontend** | HTML, CSS, JavaScript |
| **Browser AI** | MediaPipe, Transformers.js, WebAudio API |
| **Screening** | PHQ-9, GAD-7 (WHO-validated) |
| **Deployment** | Vercel |

---

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Model status and version check |
| `/predict` | POST | NLP text analysis → risk score + word contributions |
| `/predict_lifestyle` | POST | Lifestyle data → XGBoost Phase 1 risk score |
| `/phq9` | POST | PHQ-9 depression screening (9 answers, 0–3 scale) |
| `/gad7` | POST | GAD-7 anxiety screening (7 answers, 0–3 scale) |

---

## Honest Disclaimer

The browser app currently uses research-inspired heuristics that approximate the paper's concepts — not the exact trained models. It is a **research demonstrator, not a clinical diagnostic tool**.

Work in progress:
- [ ] Integrating the full XGBoost Phase 1 lifestyle model
- [ ] Retraining and deploying the TF-IDF + XGBoost Phase 2 NLP pipeline
- [ ] Moving toward full model-based SHAP/LIME explainability

---

## Run Locally

```bash
git clone https://github.com/Srimant1323/MindBridge-app.git
cd MindBridge-app
pip install -r requirements.txt
python app.py
```

Then open `index.html` in your browser.

---

## Author

**Srimant Bhardwaj**
M.Tech Bioinformatics · Delhi Technological University (DTU)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Srimant_Bhardwaj-blue)](https://www.linkedin.com/in/srimant-bhardwaj-13s23a)
[![GitHub](https://img.shields.io/badge/GitHub-Srimant1323-black)](https://github.com/Srimant1323)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--6395--1216-brightgreen?logo=orcid)](https://orcid.org/0009-0007-6395-1216)

---

*Built because transparent AI has to start with a transparent researcher.*
