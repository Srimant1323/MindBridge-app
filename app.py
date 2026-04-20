"""
MindBridge Backend — Production Flask API v2.1
================================================
Phase 1 (lifestyle): uses real xgboost_phase1.pkl ✅
Phase 2 (NLP text):  uses improved clinical heuristic — xgboost_phase2.pkl
                     needs to be retrained in Colab (see RETRAIN.md)

NOTE: xgboost_phase2.pkl predicts 99% At-Risk for ALL inputs including
"the sky is blue" — the model pkl was saved before training completed
or had a data issue in Colab. Retrain and re-export to enable real model.
"""

import os, re
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️  Not found: {path}")
        return None
    obj = joblib.load(path)
    print(f"✅ Loaded: {filename}")
    return obj

model_phase1       = load("xgboost_phase1.pkl")
_vectorizer_raw    = load("tfidf_vectorizer.pkl")
_model_phase2_raw  = load("xgboost_phase2.pkl")

def _validate_phase2():
    if _vectorizer_raw is None or _model_phase2_raw is None:
        return False
    try:
        X = _vectorizer_raw.transform(["the sky is blue today"])
        p = _model_phase2_raw.predict_proba(X)[0][1]
        if p > 0.95:
            print("⚠️  xgboost_phase2.pkl returns degenerate predictions — needs retraining.")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Phase 2 validation failed: {e}")
        return False

NLP_MODEL_VALID = _validate_phase2()
vectorizer      = _vectorizer_raw   if NLP_MODEL_VALID else None
model_phase2    = _model_phase2_raw if NLP_MODEL_VALID else None
LIFESTYLE_READY = model_phase1 is not None

print(f"\n🧠 NLP model:       {'xgboost_tfidf_phase2 ✅' if NLP_MODEL_VALID else 'clinical_heuristic (phase2 pkl invalid)'}")
print(f"📊 Lifestyle model: {'xgboost_phase1 ✅' if LIFESTYLE_READY else 'unavailable'}\n")

# ── Clinical Heuristic ────────────────────────────────────────────────────────
SEVERE    = {'suicide':3.5,'suicidal':3.5,'kill myself':3.5,'want to die':3.5,
             'end my life':3.5,'hopeless':3.0,'worthless':3.0,'meaningless':3.0,
             'marna chahta':3.5,'marna chahti':3.5,'zindagi se tang':3.0}
HIGH      = {'empty':2.0,'numb':2.0,'trapped':2.0,'burden':2.0,'despair':2.0,
             'broken':2.0,'failure':2.0,'useless':2.0,'abandoned':2.0,
             'dead inside':2.5,'hate myself':2.5,'nirasha':2.0,'tanha':2.0}
MODERATE  = {'tired':1.0,'exhausted':1.0,'sad':1.0,'crying':1.0,'anxious':1.0,
             'anxiety':1.0,'worried':1.0,'depressed':1.0,'depression':1.0,
             'lonely':1.0,'overwhelmed':1.0,'stressed':1.0,'helpless':1.0,
             'drained':1.0,'akela':1.0,'dukh':1.0,'dard':1.0}
PROTECTIVE= {'happy':-1.5,'grateful':-1.5,'hopeful':-1.5,'calm':-1.5,'peaceful':-1.5,
             'content':-1.5,'motivated':-1.5,'better':-1.5,'joy':-1.5,'khush':-1.5,'umeed':-1.5}

CRISIS_TERMS = list(SEVERE.keys())

def heuristic_nlp(text):
    t = text.lower()
    raw, contributions = 0.0, {}
    for lexicon in [SEVERE, HIGH, MODERATE, PROTECTIVE]:
        for phrase, weight in lexicon.items():
            if phrase in t:
                contributions[phrase] = round(weight, 2)
                raw += weight
    wc = max(len(text.split()), 4)
    rn = raw / (wc ** 0.5)
    if rn <= 0:    score = max(0, 12 + rn * 8)
    elif rn < 0.3: score = 25 + rn * 80
    elif rn < 1.0: score = 49 + rn * 26
    else:          score = min(100, 75 + (rn - 1.0) * 12)
    if any(c in t for c in CRISIS_TERMS): score = max(score, 75)
    return int(round(score)), contributions

def risk_band(s):
    return 'Low' if s<25 else 'Moderate' if s<50 else 'High' if s<75 else 'Severe'

PHASE1_FEATURES = [
    'Age','Gender','Education_Level','Employment_Status',
    'Sleep_Hours','Physical_Activity_Hrs','Social_Support_Score',
    'Stress_Level','Family_History_Mental_Illness','Chronic_Illnesses',
    'Therapy','Meditation','Financial_Stress','Work_Stress',
    'Self_Esteem_Score','Life_Satisfaction_Score','Loneliness_Score'
]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':          'ok',
        'nlp_model':       'xgboost_tfidf_phase2' if NLP_MODEL_VALID else 'clinical_heuristic',
        'lifestyle_model': 'xgboost_phase1' if LIFESTYLE_READY else 'unavailable',
        'version':         '2.1.0'
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS': return jsonify({}), 200
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text: return jsonify({'error': 'text field is required'}), 400
    if len(text) > 5000: return jsonify({'error': 'Text too long'}), 400
    try:
        if NLP_MODEL_VALID:
            X     = vectorizer.transform([text])
            prob  = float(model_phase2.predict_proba(X)[0][1])
            score = int(prob * 100)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_vals    = X.toarray()[0]
            top_idx       = tfidf_vals.argsort()[-15:][::-1]
            contributions = {feature_names[i]: round(float(tfidf_vals[i]*prob*10),3)
                             for i in top_idx if tfidf_vals[i] > 0}
            model_used = 'xgboost_tfidf_phase2'
        else:
            score, contributions = heuristic_nlp(text)
            model_used = 'clinical_heuristic'
        crisis = any(c in text.lower() for c in CRISIS_TERMS) or score >= 75
        return jsonify({'score':score,'risk_level':risk_band(score),
                        'crisis':crisis,'contributions':contributions,'model_used':model_used})
    except Exception as e:
        app.logger.error(f'/predict error: {e}')
        return jsonify({'error': 'Prediction failed', 'detail': str(e)}), 500

@app.route('/predict_lifestyle', methods=['POST', 'OPTIONS'])
def predict_lifestyle():
    if request.method == 'OPTIONS': return jsonify({}), 200
    if not LIFESTYLE_READY: return jsonify({'error': 'Lifestyle model not loaded'}), 503
    data = request.get_json(silent=True) or {}
    missing = [f for f in PHASE1_FEATURES if f not in data]
    if missing: return jsonify({'error': f'Missing fields: {missing}'}), 400
    try:
        row  = np.array([[data[f] for f in PHASE1_FEATURES]], dtype=float)
        prob = float(model_phase1.predict_proba(row)[0][1])
        return jsonify({'score':int(prob*100),'risk_level':risk_band(int(prob*100)),'model_used':'xgboost_phase1'})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'detail': str(e)}), 500

@app.route('/phq9', methods=['POST', 'OPTIONS'])
def phq9():
    if request.method == 'OPTIONS': return jsonify({}), 200
    answers = (request.get_json(silent=True) or {}).get('answers', [])
    if len(answers) != 9: return jsonify({'error': 'Provide 9 answers (0-3)'}), 400
    total = sum(answers)
    sev = 'Minimal' if total<=4 else 'Mild' if total<=9 else 'Moderate' if total<=14 else 'Moderately Severe' if total<=19 else 'Severe'
    return jsonify({'total':total,'severity':sev,'crisis':answers[8]>=1,'max_possible':27,'percentage':round(total/27*100,1)})

@app.route('/gad7', methods=['POST', 'OPTIONS'])
def gad7():
    if request.method == 'OPTIONS': return jsonify({}), 200
    answers = (request.get_json(silent=True) or {}).get('answers', [])
    if len(answers) != 7: return jsonify({'error': 'Provide 7 answers (0-3)'}), 400
    total = sum(answers)
    sev = 'Minimal' if total<=4 else 'Mild' if total<=9 else 'Moderate' if total<=14 else 'Severe'
    return jsonify({'total':total,'severity':sev,'max_possible':21,'percentage':round(total/21*100,1)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
