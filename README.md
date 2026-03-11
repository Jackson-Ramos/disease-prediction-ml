#  Previsão de Doenças com Aprendizagem de Máquina

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-green?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen?style=for-the-badge)

**Projeto Final — Aprendizagem de Máquina**  
Sistemas de Informação | UNIFACISA  
Prof. Bruno Rafael Araújo Vasconcelos

</div>

---

##  Sumário

- [Introdução](#-introdução)
- [Objetivo](#-objetivo)
- [Doenças Classificadas](#-doenças-classificadas)
- [Dataset](#-dataset)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Resultados Esperados](#-resultados-esperados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Como Instalar](#-como-instalar)
- [Como Rodar](#-como-rodar)
- [Como Testar](#-como-testar)
- [Métricas Avaliadas](#-métricas-avaliadas)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

---

##  Introdução

O diagnóstico médico correto é um dos maiores desafios da saúde moderna. Um mesmo conjunto de sintomas pode indicar doenças completamente distintas, e um diagnóstico incorreto pode trazer consequências graves ao paciente.

Este projeto aplica técnicas de **Aprendizagem de Máquina** para automatizar a classificação de doenças a partir de sintomas clínicos binários (presente/ausente). Utilizamos o algoritmo **Random Forest**, reconhecido por sua robustez em problemas de classificação com alta dimensionalidade de features, combinado com técnicas de **Seleção de Features** para reduzir a complexidade do modelo e evitar overfitting.

O trabalho segue o fluxo completo de um projeto de ciência de dados: da análise exploratória até a avaliação final com múltiplas métricas e validação cruzada.

---

##  Objetivo

Desenvolver um modelo de aprendizagem de máquina capaz de **prever, a partir de um conjunto de sintomas, qual doença o paciente possui** dentre 3 condições clínicas selecionadas, avaliando o modelo com métricas consistentes e documentando todos os resultados em formato científico.

**Objetivos específicos:**
- Filtrar e preparar os dados do dataset original para as 3 doenças escolhidas
- Aplicar técnicas de Seleção de Features (Feature Selection) para reduzir dimensionalidade
- Treinar um classificador Random Forest com os dados processados
- Avaliar o modelo com Matriz de Confusão, Acurácia, Precisão, Recall, F1-Score e ROC-AUC
- Validar a estabilidade do modelo com Validação Cruzada Estratificada (5-fold)
- Comparar o desempenho com e sem seleção de features

---

##  Doenças Classificadas

| # | Doença | Descrição |
|---|--------|-----------|
| 1 | **Pneumonia** | Infecção pulmonar que inflama os alvéolos, com sintomas como febre, tosse e dificuldade respiratória |
| 2 | **Acute Bronchitis** | Inflamação aguda dos brônquios, geralmente de origem viral, causando tosse persistente e produção de muco |
| 3 | **Cystitis** | Inflamação da bexiga, frequentemente causada por infecção bacteriana, com sintomas urinários característicos |

Essas três doenças foram escolhidas por apresentarem **perfis de sintomas distintos**, o que permite avaliar se o modelo consegue separar condições com sobreposição parcial de sintomas (ex: tosse em pneumonia e bronquite).

---

##  Dataset

- **Fonte:** [Kaggle — Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)
- **Total de colunas:** 378 (1 target + 377 sintomas)
- **Total de doenças no dataset original:** 773
- **Doenças utilizadas neste projeto:** 3
- **Tipo de features:** binárias (0 = sintoma ausente, 1 = sintoma presente)

>  **O arquivo CSV não está incluído neste repositório** por restrições de tamanho. Faça o download diretamente no Kaggle pelo link acima.

---

##  Arquitetura do Modelo

```
Dataset Completo (377 features)
        │
        ▼
┌─────────────────────────┐
│  Remoção Variância Zero │  → Remove sintomas constantes
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Seleção Chi² (K=40)   │  → Mantém as 40 features mais ||  relevantes
└─────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Split 80% / 20%           │  → Treino e Teste estratificados
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Random Forest (200 árvores) │  → Treinamento
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Avaliação + Cross-Val      │  → Métricas e validação 5-fold
└─────────────────────────────┘
```

---

##  Resultados Esperados

Com base nas características do dataset e na escolha das 3 doenças, esperamos os seguintes resultados ao executar o projeto:

| Métrica | Resultado Esperado |
|---|---|
| **Acurácia** | ≥ 90% |
| **Precisão (macro)** | ≥ 88% |
| **Recall (macro)** | ≥ 88% |
| **F1-Score (macro)** | ≥ 88% |
| **ROC-AUC (OvR)** | ≥ 95% |
| **CV 5-fold (média)** | ≥ 89% ± 3% |

**Gráficos gerados automaticamente:**

| Arquivo | Descrição |
|---|---|
| `distribuicao_classes.png` | Quantidade de amostras por doença |
| `top_sintomas.png` | Top 10 sintomas mais frequentes por doença |
| `feature_selection_chi2.png` | Score Chi² das features selecionadas |
| `metricas_modelo.png` | Gráfico das métricas do modelo final |
| `matriz_confusao.png` | Matriz de confusão com TP/FP/FN |
| `feature_importance_rf.png` | Importância das features pelo Random Forest |
| `cross_validation.png` | Acurácia por fold na validação cruzada |
| `comparacao_modelos.png` | Comparação com/sem Feature Selection |

---

##  Estrutura do Projeto

```
disease-prediction-ml/
│
├── 📓 disease_prediction.ipynb   # Notebook principal com todo o código
├── 📄 README.md                  # Este arquivo
├── 📄 requirements.txt           # Dependências do projeto
│
├── 📂 data/
│   └── dataset.csv               # Baixar do Kaggle (não incluso)
│
└── 📂 outputs/                   # Gerado ao executar o notebook
    ├── distribuicao_classes.png
    ├── top_sintomas.png
    ├── feature_selection_chi2.png
    ├── metricas_modelo.png
    ├── matriz_confusao.png
    ├── feature_importance_rf.png
    ├── cross_validation.png
    └── comparacao_modelos.png
```

---

##  Pré-requisitos

- **Python** 3.8 ou superior
- **Jupyter Notebook** ou **JupyterLab**
- **pip** atualizado

---

##  Como Instalar

### 1. Clone o repositório

```bash
git clone https://github.com/Jackson-Ramos/disease-prediction-ml.git
cd disease-prediction-ml
```

### 2. (Recomendado) Crie um ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixe o dataset

Acesse o link abaixo e faça o download do arquivo CSV:  
🔗 [https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)

Salve o arquivo como `dataset.csv` na pasta `data/` do projeto:

```
disease-prediction-ml/
└── data/
    └── dataset.csv   ← aqui
```

---

##  Como Rodar

### Opção 1 — Interface Jupyter (recomendado)

```bash
jupyter notebook
```

No navegador, abra o arquivo `disease_prediction.ipynb` e execute com:  
**Cell → Run All** ou `Shift + Enter` célula por célula.

### Opção 2 — JupyterLab

```bash
jupyter lab
```

### Opção 3 — Linha de comando (sem interface)

```bash
jupyter nbconvert --to notebook --execute disease_prediction.ipynb --output disease_prediction_executado.ipynb
```

> **Atenção:** na célula 2 do notebook, verifique se o `CSV_PATH` aponta para o local correto do seu arquivo:
> ```python
> CSV_PATH = 'data/dataset.csv'  # ajuste se necessário
> ```

---

##  Como Testar

### Verificar se a instalação está correta

```bash
python -c "import pandas; import sklearn; import matplotlib; import seaborn; print('Tudo instalado corretamente!')"
```

### Testar o carregamento do dataset

Execute apenas as células 1 e 2 do notebook. A saída esperada é:

```
Bibliotecas importadas com sucesso!
Dataset completo: XXXXX linhas, 378 colunas
Total de doenças únicas: 773
Dataset filtrado: XXXX linhas, 378 colunas
```

### Testar o modelo treinado

Após executar o notebook completo, a saída esperada na seção de métricas é:

```
=============================================
         MÉTRICAS DO MODELO (TESTE)
=============================================
  Acurácia   : 0.XXXX (XX.XX%)
  Precisão   : 0.XXXX
  Recall     : 0.XXXX
  F1-Score   : 0.XXXX
  ROC-AUC    : 0.XXXX
=============================================
```

### Fazer uma predição manual

Após executar o notebook, você pode testar manualmente com um novo paciente:

```python
import numpy as np

# Crie um vetor de sintomas (use os nomes das features selecionadas)
# 1 = sintoma presente, 0 = sintoma ausente
novo_paciente = np.zeros((1, len(selected_features)))

# Exemplo: paciente com tosse, febre e dificuldade para respirar
novo_paciente[0, list(selected_features).index('cough')]              = 1
novo_paciente[0, list(selected_features).index('fever')]              = 1
novo_paciente[0, list(selected_features).index('difficulty breathing')] = 1

predicao = rf_model.predict(novo_paciente)
probabilidade = rf_model.predict_proba(novo_paciente)

print(f"Diagnóstico previsto : {le.inverse_transform(predicao)[0]}")
print(f"Probabilidades       : {dict(zip(le.classes_, probabilidade[0].round(3)))}")
```

---

##  Métricas Avaliadas

| Métrica | Descrição |
|---|---|
| **Acurácia** | Proporção de classificações corretas sobre o total |
| **Precisão** | Dos casos classificados como doença X, quantos realmente tinham X |
| **Recall** | Dos casos que realmente têm doença X, quantos foram identificados |
| **F1-Score** | Média harmônica entre Precisão e Recall |
| **ROC-AUC** | Área sob a curva ROC, avalia a separabilidade das classes |
| **Matriz de Confusão** | Visualização completa dos erros e acertos por classe |
| **Validação Cruzada** | Avaliação da estabilidade do modelo em 5 divisões diferentes dos dados |

---

##  Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| `pandas` | ≥ 1.3 | Manipulação e filtragem do dataset |
| `numpy` | ≥ 1.21 | Operações numéricas |
| `scikit-learn` | ≥ 1.0 | Modelo, métricas e seleção de features |
| `matplotlib` | ≥ 3.4 | Geração de gráficos |
| `seaborn` | ≥ 0.11 | Visualizações estatísticas |
| `jupyter` | ≥ 1.0 | Ambiente de desenvolvimento |

---

##  Licença

Este projeto foi desenvolvido para fins acadêmicos — **UNIFACISA, 2026**.

---

<div align="center">
Desenvolvido para a disciplina de Aprendizagem de Máquina — Sistemas de Informação | UNIFACISA
</div>