"""
MindBridge — Flask API Backend Stub
====================================
Replace the heuristic scoring below with your actual trained model:
  - TF-IDF vectorizer (from Phase 2 of the research paper)
  - XGBoost classifier trained on PHQ-9 labeled data
  - LIME explainer for word importance

Run: python app.py
API: POST http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re, math, json

app = Flask(__name__)
CORS(app)  # Allow requests from the browser app

# ── Stub lexicon (mirror of the JS lexicon for Python) ───────────────────────
WEIGHTS = {
    # Severe
    'suicide': 3.0, 'suicidal': 3.0, 'hopeless': 3.0, 'worthless': 3.0,
    'meaningless': 3.0, 'end my life': 3.5, 'want to die': 3.5,
    # High
    'empty': 2.0, 'numb': 2.0, 'trapped': 2.0, 'burden': 2.0,
    'despair': 2.0, 'broken': 2.0, 'failure': 2.0, 'useless': 2.0,
    'alone': 2.0, 'abandoned': 2.0, 'devastated': 2.0, 'miserable': 2.0,
    # Moderate
    'tired': 1.0, 'exhausted': 1.0, 'sad': 1.0, 'crying': 1.0,
    'anxious': 1.0, 'anxiety': 1.0, 'worried': 1.0, 'depressed': 1.0,
    'lonely': 1.0, 'overwhelmed': 1.0, 'stressed': 1.0, 'helpless': 1.0,
    'guilty': 1.0, 'ashamed': 1.0,
    # Protective (negative weight)
    'happy': -1.5, 'grateful': -1.5, 'hopeful': -1.5, 'calm': -1.5,
    'peaceful': -1.5, 'content': -1.5, 'motivated': -1.5, 'strong': -1.5,
    'better': -1.5, 'good': -1.5,
}

SEVERE_WORDS = {'suicide', 'suicidal', 'hopeless', 'worthless', 'meaningless',
                'end my life', 'want to die'}


def tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z\u0900-\u097F]+', text.lower())


def score_text(text: str) -> dict:
    """
    STUB: Replace this function body with your real model inference.

    Expected interface:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model      = joblib.load('xgboost_model.pkl')
        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]  # P(depressed)
        score = int(prob * 100)
    """
    tokens = tokenize(text)
    # Build bigrams
    bigrams = [tokens[i] + ' ' + tokens[i+1] for i in range(len(tokens) - 1)]
    all_tokens = tokens + bigrams

    word_hits = {}
    raw = 0.0
    severe_found = False

    for tok in all_tokens:
        if tok in WEIGHTS and tok not in word_hits:
            w = WEIGHTS[tok]
            level = 'protective' if w < 0 else ('severe' if w >= 3.0 else ('high' if w >= 2.0 else 'moderate'))
            word_hits[tok] = {'weight': w, 'level': level}
            raw += w
            if tok in SEVERE_WORDS:
                severe_found = True

    # Normalize
    token_weight = max(len(tokens), 4)
    raw_norm     = raw / token_weight

    if raw_norm <= 0:
        score = max(0, 12 + raw_norm * 8)
    elif raw_norm < 0.5:
        score = 25 + raw_norm * 50
    elif raw_norm < 1.5:
        score = 50 + (raw_norm - 0.5) * 25
    else:
        score = min(100, 75 + (raw_norm - 1.5) * 10)

    if severe_found and score < 75:
        score = max(score, 75)
    score = int(round(score))

    if   score < 25: risk = 'low'
    elif score < 50: risk = 'moderate'
    elif score < 75: risk = 'high'
    else:            risk = 'severe'

    sorted_hits = sorted(word_hits.items(), key=lambda x: abs(x[1]['weight']), reverse=True)
    top_words = [{'word': w, **d} for w, d in sorted_hits[:8]]

    return {
        'score':       score,
        'risk_level':  risk,
        'raw_score':   round(raw, 3),
        'token_count': len(tokens),
        'top_words':   top_words,
        'word_hits':   word_hits,
        'severe':      severe_found,
        'model':       'stub_heuristic'  # Change to 'tfidf_xgboost' when you plug in the real model
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'stub — plug in TF-IDF+XGBoost here'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "text": "...", "lang": "en" }
    Returns: { score, risk_level, top_words, ... }
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'text field is required'}), 400

    result = score_text(text)
    result['lang'] = lang
    return jsonify(result)


@app.route('/explain', methods=['POST'])
def explain():
    """
    POST /explain — LIME explanation endpoint stub.
    Body: { "text": "..." }
    Returns: LIME explanation data (plug in real LIME here).

    Example real implementation:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=['not depressed', 'depressed'])
        exp = explainer.explain_instance(text, model.predict_proba, num_features=10)
        return jsonify(exp.as_list())
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    result = score_text(text)
    return jsonify({'lime_weights': result['top_words']})


if __name__ == '__main__':
    print('\n🧠 MindBridge Backend running on http://localhost:5000')
    print('📌 Plug in your TF-IDF + XGBoost model in the score_text() function.')
    print('📌 See /health to verify status.\n')
    app.run(debug=True, port=5000)
