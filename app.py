import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Paths to models
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.pkl')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'selected_features.json')

# Global variables for models
model = None
app_label_encoder = None
selected_features = []

def load_models():
    global model, app_label_encoder, selected_features
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        app_label_encoder = joblib.load(LE_PATH)
        with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
            selected_features = json.load(f)
        print("Modelos carregados com sucesso!")
    else:
        print("Aviso: Modelos não encontrados na pasta /models/")

# Triggering reload to detect new models
load_models()
print("Servidor SymptoAI pronto e aguardando requisições.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    # Retorna as 40 features esperadas pelo modelo para gerar a UI dinamicamente
    return jsonify({"features": selected_features})

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo não carregado no servidor."}), 500
        
    data = request.json
    if not data or 'symptoms' not in data:
        return jsonify({"error": "Nenhum sintoma fornecido."}), 400
        
    user_symptoms = data.get('symptoms', []) # Ex: ["cough", "fever"]
    
    # Construir o vetor binário X para as 40 features
    X_input = [1 if feature in user_symptoms else 0 for feature in selected_features]
    X_input = np.array(X_input).reshape(1, -1)
    
    # Realizar predição
    pred_idx = model.predict(X_input)[0]
    pred_disease = app_label_encoder.inverse_transform([pred_idx])[0]
    
    # Pegar confiança (probabilidade da classe predita)
    proba = model.predict_proba(X_input)[0]
    confidence = float(proba[pred_idx] * 100)
    
    return jsonify({
        "prediction": pred_disease,
        "confidence": confidence
    })

if __name__ == '__main__':
    # Rodar publicamente em todas as interfaces para fácil acesso
    app.run(host='0.0.0.0', port=5000, debug=True)
