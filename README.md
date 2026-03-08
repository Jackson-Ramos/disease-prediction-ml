# Previsão de Doenças a partir de Sintomas

Este projeto prático de **Machine Learning** tem como objetivo desenvolver um modelo de classificação capaz de prever qual doença um paciente possui, com base em um conjunto de sintomas relatados ou observados.

O escopo do projeto foca em três doenças específicas:
1. **Pneumonia**
2. **Bronquite Aguda** (*Acute Bronchitis*)
3. **Cistite** (*Cystitis*)

## 🎯 Objetivo
Avaliar como algoritmos de aprendizado de máquina podem atuar como sistemas especialistas de triagem, distinguindo entre condições clínicas. Optamos por duas doenças do trato respiratório/pneumológico (que tendem a compartilhar sintomas, tornando a classificação difícil) e uma infecção sistêmica/urinária (que serve de contraste).

## 📊 Dataset
A base de dados utilizada provém do Kaggle:
[Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)
* A base é composta por 378 colunas (1 com nomes de doenças variadas e as restantes mapeando a presença ou ausência de cada sintoma no formato 0/1).

## 🛠️ Tecnologias e Ferramentas
- **Linguagem**: Python
- **Ambiente**: Jupyter Notebook (`.ipynb`)
- **Bibliotecas**:
  - `pandas` (Manipulação e estruturação de dados)
  - `scikit-learn` (Criação do modelo, métricas e split)
  - `matplotlib` & `seaborn` (Plotagem gráfica e visualizações)

## 🧠 Modelagem e Divisão dos Dados
O arquivo principal deste projeto encontra-se em `analise_sintomas_doencas.ipynb`.
1. **Split dos Dados**: Foi adotada uma distribuição de dados segura de:
   - **70%** para Treinamento
   - **15%** para Validação
   - **15%** para Teste Cego
   
2. **Modelo Escolhido**: O escolhido foi o classificador `Random Forest` (Floresta Aleatória), pela sua exímia robustez e facilidade em escalar pesos de variáveis categóricas binárias altamente esparsas contendo muito ruído.

## 📈 Resultados e Métricas Obtidas
* Para avaliar a performance e testar efetivamente o modelo perante o escopo da vida-real, efetuamos a inferência final utilizando o conjunto estrito de **Teste Cego** (15% inéditos pro modelo).
* As métricas de *Classification Report* demonstraram índices de Accuracy (Acurácia Global), Precision (Precisão) e Recall (Revocação) superando consistentemente a casa dos ~95%.
* A **Matriz de Confusão** gráfica nos provou que, mesmo com semelhança sintomática severa (Pneumonia e Bronquite), o modelo foi capaz de isolar precisamente o caso (falsa rejeição mínima/nula comparada entre as opções respiratórias ou a infecção de Cistite).
* Um gráfico de *Feature Importances* foi plotado extraindo organicamente os pesos do robô para descobrir quais variáveis (sintomas) foram decisivos matematicamente na separação. (Tosse, febre e dores locais vs Dificuldade aguda para respirar).

## 🚀 Como Executar
1. Certifique-se de que tenha a pasta `data/` com o arquivo `.csv` na raiz junto do Notebook.
2. Ative seu ambiente virtual (ex: `.venv\Scripts\Activate.ps1`).
3. Instale os requisitos necessários: `pip install pandas scikit-learn matplotlib seaborn`.
4. Abra o `analise_sintomas_doencas.ipynb` e rode todas as células.
