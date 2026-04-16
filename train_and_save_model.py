import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.preprocessing import LabelEncoder

# Configurações Iniciais
DATA_PATH = 'data/dataset.csv'
MODELS_DIR = 'models'
DISEASES = ['pneumonia', 'acute bronchitis', 'cystitis']
K_FEATURES = 40

def main():
    print("Iniciando o treinamento do modelo...")
    
    # 1. Carregamento dos Dados
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Arquivo {DATA_PATH} não encontrado.")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset original: {df.shape}")
    
    # 2. Filtragem e Preparação
    df = df[df['diseases'].isin(DISEASES)].copy()
    
    # Separar Features do Alvo
    feature_cols = [c for c in df.columns if c != 'diseases']
    X = df[feature_cols].values
    y = df['diseases'].values
    print(f"Amostras após filtragem: {X.shape[0]}")
    
    # Codificação do Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Feature Selection
    print("Realizando Seleção de Features...")
    
    # Variância Zero
    var_thresh = VarianceThreshold(threshold=0)
    X_nzv = var_thresh.fit_transform(X)
    feature_names_nzv = np.array(feature_cols)[var_thresh.get_support()]
    
    # Chi-Quadrado
    chi2_selector = SelectKBest(chi2, k=K_FEATURES)
    X_selected = chi2_selector.fit_transform(X_nzv, y_encoded)
    
    # Guardar o nome das features selecionadas
    final_features = list(feature_names_nzv[chi2_selector.get_support()])
    print(f"Features finais (Top {K_FEATURES}): {final_features[:5]}...")

    # 4. Treinamento
    print("Treinando o Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_selected, y_encoded)
    print("Modelo treinado com sucesso!")
    
    # 5. Salvar Modelo e Artefatos
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Salvar o classificador
    model_path = os.path.join(MODELS_DIR, 'rf_model.pkl')
    joblib.dump(rf_model, model_path)
    
    # Salvar o extrator chi2 (para inferência precisa do pipeline, ou podemos fazer inferencia apenas filtrando as colunas no pandas)
    # Como só precisamos das colunas exatas, salvamos o nome das features:
    features_path = os.path.join(MODELS_DIR, 'selected_features.json')
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(final_features, f, ensure_ascii=False, indent=2)
        
    # Salvar o LabelEncoder para decodificar a predição depois
    le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    joblib.dump(le, le_path)
    
    print("\n[OK] Artefatos salvos em /models/")
    print(f" - {model_path}")
    print(f" - {le_path}")
    print(f" - {features_path}")

if __name__ == '__main__':
    main()
