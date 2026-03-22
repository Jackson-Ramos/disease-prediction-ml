"""
rebuild_notebook.py
-------------------
Transforma o notebook disease_prediction.ipynb em um artigo bem-escrito,
substituindo as células markdown (cabeçalhos curtos) por texto explicativo
de artigo acadêmico. Nenhum código é alterado.
"""

import json
import copy
import sys

# ─── Conteúdo de artigo para cada célula markdown ────────────────────────────

MARKDOWN_CELLS = {

# ── Cell 0: Título e Introdução ──────────────────────────────────────────────
0: [
    "# Previsão de Doenças com Aprendizagem de Máquina\n",
    "## Um Estudo Sobre Classificação de Doenças e Seleção de Variáveis\n",
    "\n",
    "**Projeto Final — Sistemas de Informação | UNIFACISA**  \n",
    "**Prof. Bruno Rafael Araújo Vasconcelos**\n",
    "\n",
    "---\n",
    "\n",
    "### Introdução\n",
    "\n",
    "A aplicação de técnicas de **Aprendizagem de Máquina (Machine Learning)** na área da saúde tem ganhado "
    "destaque nos últimos anos, possibilitando diagnósticos mais rápidos e auxiliando profissionais de saúde "
    "na tomada de decisão clínica. Neste contexto, a **previsão de doenças a partir de sintomas** é um problema "
    "de classificação multiclasse — onde o modelo recebe um conjunto de sintomas relatados pelo paciente e "
    "deve prever qual doença está mais fortemente associada a esse conjunto.\n",
    "\n",
    "O presente artigo implementa e avalia um modelo de **Random Forest (Floresta Aleatória)** para classificar "
    "três condições clínicas distintas a partir de sintomas binários (presente/ausente):\n",
    "\n",
    "| Doença | Descrição |\n",
    "|--------|-----------|\n",
    "| **Pneumonia** | Infecção que inflama os alvéolos pulmonares, podendo causar tosse, febre e dificuldade respiratória. |\n",
    "| **Bronquite Aguda** (*acute bronchitis*) | Inflamação dos brônquios, geralmente causada por vírus, com sintomas como tosse e produção de muco. |\n",
    "| **Cistite** (*cystitis*) | Infecção do trato urinário inferior, com sintomas como dor ao urinar e frequência urinária aumentada. |\n",
    "\n",
    "### Objetivos\n",
    "\n",
    "Os principais objetivos deste estudo são:\n",
    "\n",
    "1. **Desenvolver um modelo de classificação** capaz de distinguir entre as três doenças com alta acurácia.\n",
    "2. **Investigar técnicas de Seleção de Features (Feature Selection)** para reduzir o número de variáveis "
    "de entrada, avaliando o impacto dessa redução no desempenho do modelo.\n",
    "3. **Demonstrar a importância da redução de dimensionalidade** na mitigação de *overfitting*, na diminuição "
    "do tempo de treinamento e na melhoria da interpretabilidade do modelo.\n",
    "4. **Avaliar o modelo de forma rigorosa**, utilizando múltiplas métricas (acurácia, precisão, recall, "
    "F1-Score, ROC-AUC) e validação cruzada estratificada.\n",
    "\n",
    "### Metodologia Resumida\n",
    "\n",
    "O fluxo de trabalho segue as etapas clássicas de um pipeline de Machine Learning:\n",
    "\n",
    "```\n",
    "Dados Brutos → Filtragem → EDA → Preparação → Seleção de Features → Treinamento → Avaliação\n",
    "```\n",
    "\n",
    "A escolha do algoritmo **Random Forest** foi motivada por sua robustez, capacidade de lidar com dados "
    "de alta dimensionalidade, resistência a *overfitting* (quando bem parametrizado) e pela facilidade de "
    "interpretação através dos *feature importances* que o algoritmo naturalmente fornece.\n",
],

# ── Cell 1: Importação das Bibliotecas ───────────────────────────────────────
1: [
    "## 1. Importação das Bibliotecas\n",
    "\n",
    "O primeiro passo é importar todas as bibliotecas necessárias para o projeto. Cada biblioteca desempenha "
    "um papel específico no pipeline:\n",
    "\n",
    "- **pandas** e **numpy**: Manipulação e transformação de dados tabulares.\n",
    "- **matplotlib** e **seaborn**: Visualização de dados — essenciais para a análise exploratória e para "
    "a apresentação de resultados.\n",
    "- **scikit-learn**: Framework principal de Machine Learning, fornecendo:\n",
    "  - `RandomForestClassifier` — o algoritmo de classificação;\n",
    "  - `train_test_split` — divisão dos dados em treino e teste;\n",
    "  - `SelectKBest` e `chi2` — técnica de seleção de features baseada no teste Chi-Quadrado;\n",
    "  - `VarianceThreshold` — remoção de features com variância zero;\n",
    "  - `StratifiedKFold` e `cross_val_score` — validação cruzada estratificada;\n",
    "  - Métricas de avaliação: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, "
    "`roc_auc_score`, `confusion_matrix` e `classification_report`.\n",
    "- **LabelEncoder**: Codificação da variável alvo (doença) em valores numéricos.\n",
],

# ── Cell 3: Carregamento e Filtragem ────────────────────────────────────────
3: [
    "## 2. Carregamento e Filtragem dos Dados\n",
    "\n",
    "O dataset utilizado neste estudo contém registros de pacientes com diversas doenças, representados por "
    "colunas binárias indicando a presença (1) ou ausência (0) de cada sintoma. A coluna alvo (`prognosis`) "
    "indica o diagnóstico associado.\n",
    "\n",
    "Como o foco deste artigo é a classificação de **três doenças específicas** — pneumonia, bronquite aguda "
    "e cistite — o primeiro passo é **filtrar o dataset** para manter apenas os registros relevantes. Essa "
    "filtragem é importante porque:\n",
    "\n",
    "- **Reduz a complexidade** do problema, tornando-o mais tratável e analisável;\n",
    "- Permite uma **análise mais profunda** das relações entre sintomas e cada doença;\n",
    "- Facilita a **interpretação dos resultados** do modelo.\n",
    "\n",
    "O dataset original é carregado a partir de um arquivo CSV e, em seguida, filtrado para conter apenas "
    "as observações das três doenças de interesse.\n",
],

# ── Cell 6: EDA ──────────────────────────────────────────────────────────────
6: [
    "## 3. Análise Exploratória dos Dados (EDA)\n",
    "\n",
    "A **Análise Exploratória dos Dados (EDA)** é uma etapa fundamental em qualquer projeto de ciência de dados. "
    "Antes de treinar um modelo, é necessário compreender a estrutura, a distribuição e as características "
    "dos dados disponíveis.\n",
    "\n",
    "Nesta seção, serão analisados três aspectos principais:\n",
    "\n",
    "1. **Distribuição das classes (doenças)**: Verificar se o dataset é balanceado, ou seja, se há uma "
    "quantidade similar de amostras para cada doença. Datasets desbalanceados podem enviesar o modelo, "
    "fazendo-o favorecer a classe majoritária.\n",
    "\n",
    "2. **Valores ausentes**: Dados faltantes podem comprometer a qualidade do modelo. É fundamental "
    "verificar a integridade dos dados antes de prosseguir.\n",
    "\n",
    "3. **Sintomas mais frequentes por doença**: Identificar quais sintomas são mais prevalentes para cada "
    "doença ajuda a entender os *padrões discriminativos* que o modelo deverá aprender.\n",
    "\n",
    "### 3.1 Distribuição das Classes\n",
    "\n",
    "O gráfico a seguir mostra a contagem de amostras para cada uma das três doenças. Um dataset razoavelmente "
    "balanceado é desejável, pois evita que o modelo fique enviesado em favor de classes com mais exemplos. "
    "Caso haja desbalanceamento, estratégias como *class_weight='balanced'* no Random Forest podem ser "
    "utilizadas para compensar.\n",
],

# ── Cell 10: Preparação dos Dados ────────────────────────────────────────────
10: [
    "## 4. Preparação dos Dados\n",
    "\n",
    "Com a análise exploratória concluída, o próximo passo é **preparar os dados** para o treinamento do modelo. "
    "Esta etapa envolve:\n",
    "\n",
    "### 4.1 Separação de Features e Variável Alvo\n",
    "\n",
    "As **features (variáveis independentes, X)** são todas as colunas de sintomas — variáveis binárias que "
    "indicam a presença ou ausência de cada sintoma. A **variável alvo (y)** é a coluna `prognosis`, que "
    "contém o nome da doença.\n",
    "\n",
    "### 4.2 Codificação da Variável Alvo (Label Encoding)\n",
    "\n",
    "Como a maioria dos algoritmos de Machine Learning trabalha com valores numéricos, a variável alvo "
    "precisa ser convertida de texto (*string*) para números inteiros. O `LabelEncoder` realiza essa "
    "transformação de forma automática, atribuindo um índice numérico a cada classe.\n",
    "\n",
    "### 4.3 Divisão Treino/Teste\n",
    "\n",
    "Os dados são divididos em dois conjuntos:\n",
    "\n",
    "- **Treino (80%)**: Usado para treinar o modelo.\n",
    "- **Teste (20%)**: Usado para avaliar o desempenho em dados *nunca vistos* durante o treinamento.\n",
    "\n",
    "A divisão é feita com **estratificação** (`stratify=y`), o que garante que a proporção de cada doença "
    "seja mantida tanto no conjunto de treino quanto no de teste. Essa técnica é especialmente importante "
    "quando há classes desbalanceadas, pois evita que uma classe fique sub-representada em um dos conjuntos.\n",
],

# ── Cell 13: Seleção de Features ─────────────────────────────────────────────
13: [
    "## 5. Seleção de Features (Feature Selection)\n",
    "\n",
    "A **Seleção de Features** é uma das etapas mais importantes em projetos de Machine Learning, especialmente "
    "quando o número de variáveis de entrada é elevado. Neste dataset, temos **centenas de colunas de sintomas**, "
    "porém nem todas são igualmente relevantes para distinguir entre as três doenças.\n",
    "\n",
    "### Por que reduzir o número de variáveis?\n",
    "\n",
    "Reduzir a quantidade de features oferece benefícios significativos:\n",
    "\n",
    "| Benefício | Descrição |\n",
    "|-----------|----------|\n",
    "| **Redução de Overfitting** | Modelos com muitas variáveis tendem a memorizar o conjunto de treino em vez de "
    "generalizar. Ao remover features irrelevantes, o modelo aprende padrões mais robustos. |\n",
    "| **Menor Tempo de Treinamento** | Menos variáveis significam menos cálculos durante o treinamento, "
    "acelerando significativamente o processo. |\n",
    "| **Melhor Interpretabilidade** | Um modelo com poucas features é mais fácil de entender e explicar, "
    "o que é crucial em aplicações clínicas onde médicos precisam confiar nas decisões do modelo. |\n",
    "| **Menor Risco de Ruído** | Features irrelevantes introduzem ruído nos dados, confundindo o modelo "
    "e degradando o desempenho. |\n",
    "\n",
    "### Técnicas Utilizadas\n",
    "\n",
    "Neste estudo, aplicamos duas técnicas de seleção de features em sequência:\n",
    "\n",
    "#### Etapa 1 — Remoção de Features com Variância Zero\n",
    "\n",
    "Features com **variância zero** são colunas onde todos os valores são iguais (ex: um sintoma que nunca "
    "aparece ou sempre aparece em todos os pacientes). Essas features não carregam informação discriminativa "
    "e devem ser removidas. Utilizamos o `VarianceThreshold` do scikit-learn para essa filtragem automática.\n",
    "\n",
    "#### Etapa 2 — Teste Chi-Quadrado (χ²)\n",
    "\n",
    "O **teste Chi-Quadrado** é um método estatístico que avalia a independência entre cada feature e a "
    "variável alvo. Features com **score χ² mais alto** possuem maior associação estatística com a doença — "
    "ou seja, são mais relevantes para a classificação.\n",
    "\n",
    "Utilizamos o `SelectKBest` com `chi2` para selecionar as **K melhores features** conforme seu score χ². "
    "Esse valor de K foi definido como **40**, reduzindo drasticamente o espaço de features de centenas para "
    "apenas 40 variáveis, sem perda significativa de desempenho.\n",
    "\n",
    "O gráfico a seguir mostra os scores χ² das features selecionadas, permitindo visualizar quais sintomas "
    "possuem a relação mais forte com o diagnóstico.\n",
],

# ── Cell 17: Treinamento ─────────────────────────────────────────────────────
17: [
    "## 6. Treinamento do Modelo — Random Forest\n",
    "\n",
    "Com as features selecionadas, o próximo passo é treinar o modelo de classificação. O algoritmo escolhido "
    "foi o **Random Forest (Floresta Aleatória)**, um dos métodos mais populares e eficazes para problemas "
    "de classificação.\n",
    "\n",
    "### Como funciona o Random Forest?\n",
    "\n",
    "O Random Forest é um método *ensemble* que combina múltiplas **árvores de decisão** (estimadores), cada "
    "uma treinada em uma amostra aleatória dos dados (*bootstrap*). A previsão final é determinada por "
    "**votação majoritária** — a classe mais votada entre todas as árvores é escolhida como a previsão.\n",
    "\n",
    "```\n",
    "            ┌─── Árvore 1 ──→ classe A ───┐\n",
    "Dados ──→  ├─── Árvore 2 ──→ classe B ───├──→ Votação ──→ classe A (maioria)\n",
    "            ├─── Árvore 3 ──→ classe A ───┤\n",
    "            └─── ...       ──→ ...    ───┘\n",
    "```\n",
    "\n",
    "### Hiperparâmetros Utilizados\n",
    "\n",
    "| Parâmetro | Valor | Justificativa |\n",
    "|-----------|-------|----- |\n",
    "| `n_estimators` | 200 | Número de árvores na floresta. Mais árvores geralmente melhoram a estabilidade. |\n",
    "| `class_weight` | `'balanced'` | Ajusta automaticamente os pesos das classes inversamente proporcionais à sua frequência, mitigando possíveis desbalanceamentos. |\n",
    "| `random_state` | 42 | Garante reprodutibilidade dos resultados. |\n",
    "| `n_jobs` | -1 | Utiliza todos os núcleos do processador para acelerar o treinamento. |\n",
],

# ── Cell 19: Avaliação do Modelo ─────────────────────────────────────────────
19: [
    "## 7. Avaliação do Modelo\n",
    "\n",
    "A avaliação é uma etapa crítica para determinar a qualidade do modelo treinado. Neste estudo, utilizamos "
    "**múltiplas métricas** de avaliação, pois cada uma captura um aspecto diferente do desempenho:\n",
    "\n",
    "| Métrica | O que mede | Quando é importante |\n",
    "|---------|-----------|--------------------|\n",
    "| **Acurácia** | Proporção total de previsões corretas | Útil quando as classes são balanceadas |\n",
    "| **Precisão** (*Precision*) | Dos diagnosticados com a doença, quantos realmente a têm | Importante para evitar *falsos positivos* |\n",
    "| **Recall** (*Sensibilidade*) | Dos que realmente têm a doença, quantos foram detectados | Crucial em diagnósticos médicos para não perder casos |\n",
    "| **F1-Score** | Média harmônica entre precisão e recall | Métrica balanceada que penaliza disparidades entre precisão e recall |\n",
    "| **ROC-AUC** | Capacidade de distinguir entre classes | Quanto mais próximo de 1.0, melhor a separação entre doenças |\n",
    "\n",
    "As métricas são calculadas com `average='macro'`, o que significa que o desempenho de cada classe recebe "
    "**peso igual** no cálculo final, independentemente do número de amostras.\n",
    "\n",
    "### Predições e Métricas Globais\n",
    "\n",
    "A seguir, o modelo é aplicado ao conjunto de teste (dados não vistos durante o treinamento) e as métricas "
    "são calculadas. Um bom modelo deve apresentar valores altos e balanceados em todas as métricas.\n",
],

# ── Cell 23: Matriz de Confusão ──────────────────────────────────────────────
23: [
    "## 8. Matriz de Confusão\n",
    "\n",
    "A **Matriz de Confusão** é uma ferramenta visual fundamental para avaliar modelos de classificação. Ela "
    "apresenta, para cada classe real, quantas previsões foram feitas corretamente e quantas foram confundidas "
    "com outras classes.\n",
    "\n",
    "Na diagonal principal da matriz estão as **previsões corretas** (verdadeiros positivos para cada classe). "
    "Valores fora da diagonal indicam **erros de classificação** — por exemplo, pacientes com pneumonia que "
    "foram erroneamente classificados como bronquite.\n",
    "\n",
    "A análise da matriz de confusão é especialmente útil em aplicações médicas porque:\n",
    "\n",
    "- Permite identificar **quais doenças o modelo confunde** entre si;\n",
    "- Ajuda a entender se os erros são clinicamente relevantes (ex: confundir pneumonia com bronquite é mais "
    "preocupante do que outros tipos de confusão);\n",
    "- Serve de base para ajustar o modelo ou coletar mais dados de classes problemáticas.\n",
],

# ── Cell 25: Importância das Features ────────────────────────────────────────
25: [
    "## 9. Importância das Features (Random Forest)\n",
    "\n",
    "Uma das grandes vantagens do Random Forest é sua capacidade nativa de calcular a **importância de cada "
    "feature** (*feature importance*). Esse cálculo é baseado na **redução média de impureza (Gini)** — "
    "features que mais contribuem para dividir corretamente os dados nas árvores de decisão recebem scores "
    "de importância mais altos.\n",
    "\n",
    "### Por que isso é relevante?\n",
    "\n",
    "A análise de importância das features complementa a etapa de **Seleção de Features** (Seção 5) e oferece "
    "uma perspectiva adicional:\n",
    "\n",
    "- **Feature Selection (Chi²)** avalia a relação estatística entre cada feature e a variável alvo *antes* "
    "do treinamento;\n",
    "- **Feature Importance (Random Forest)** mede a contribuição de cada feature *dentro* do modelo treinado.\n",
    "\n",
    "A concordância entre ambos os métodos — ou seja, features com alto score χ² que também apresentam alta "
    "importância no Random Forest — é um forte indicador de que essas variáveis são genuinamente relevantes "
    "para a classificação.\n",
    "\n",
    "O gráfico a seguir apresenta as features mais importantes conforme calculado pelo Random Forest, "
    "permitindo identificar os **sintomas mais determinantes** para cada diagnóstico.\n",
],

# ── Cell 27: Validação Cruzada ───────────────────────────────────────────────
27: [
    "## 10. Validação Cruzada (Cross-Validation)\n",
    "\n",
    "A avaliação feita na Seção 7 utiliza apenas **uma única divisão** treino/teste, o que pode não refletir "
    "o desempenho real do modelo — afinal, o resultado depende de quais dados caíram em cada conjunto.\n",
    "\n",
    "A **Validação Cruzada Estratificada (Stratified K-Fold Cross-Validation)** resolve esse problema ao:\n",
    "\n",
    "1. Dividir os dados de treino em **K subconjuntos (folds)** de tamanho similar;\n",
    "2. Treinar o modelo K vezes, usando K−1 folds para treino e 1 fold para validação;\n",
    "3. Calcular a métrica de desempenho em cada fold;\n",
    "4. Reportar a **média** e o **desvio padrão** das métricas.\n",
    "\n",
    "```\n",
    "Fold 1: [VALID] [Treino] [Treino] [Treino] [Treino]  →  Acurácia fold 1\n",
    "Fold 2: [Treino] [VALID] [Treino] [Treino] [Treino]  →  Acurácia fold 2\n",
    "Fold 3: [Treino] [Treino] [VALID] [Treino] [Treino]  →  Acurácia fold 3\n",
    "Fold 4: [Treino] [Treino] [Treino] [VALID] [Treino]  →  Acurácia fold 4\n",
    "Fold 5: [Treino] [Treino] [Treino] [Treino] [VALID]  →  Acurácia fold 5\n",
    "                                                         ────────────────\n",
    "                                                         Média ± Desvio\n",
    "```\n",
    "\n",
    "A estratificação garante que **cada fold mantenha a mesma proporção de classes** dos dados originais, "
    "evitando que algum fold fique sem exemplos de uma doença.\n",
    "\n",
    "### Interpretação dos Resultados\n",
    "\n",
    "- Uma **média alta** indica que o modelo generaliza bem;\n",
    "- Um **desvio padrão baixo** indica que o desempenho é **estável** independentemente da divisão dos dados;\n",
    "- Se houver grande variabilidade entre folds, isso pode indicar *overfitting* ou dados insuficientes.\n",
],

# ── Cell 30: Comparação de Modelos ───────────────────────────────────────────
30: [
    "## 11. Comparação: Modelo Completo vs. Modelo com Feature Selection\n",
    "\n",
    "Para demonstrar concretamente o **impacto da Seleção de Features**, esta seção compara dois modelos:\n",
    "\n",
    "| Modelo | Nº de Features | Descrição |\n",
    "|--------|---------------|----------|\n",
    "| **Modelo Completo** | Todas (377) | Treinado com *todas* as features originais, sem nenhuma seleção |\n",
    "| **Modelo com Feature Selection** | 40 | Treinado apenas com as 40 features selecionadas pelo teste Chi² |\n",
    "\n",
    "### O que esperamos observar?\n",
    "\n",
    "Se a seleção de features foi eficaz, o modelo com 40 features deve apresentar desempenho **comparável** "
    "ao modelo completo com 377 features. Isso demonstraria que:\n",
    "\n",
    "- A grande maioria das 377 features originais era **redundante ou irrelevante**;\n",
    "- É possível reduzir o espaço de variáveis em **mais de 89%** (de 377 para 40) sem perda significativa;\n",
    "- O modelo reduzido é **mais eficiente** (menor tempo de treinamento e inferência) e **mais interpretável**.\n",
    "\n",
    "Uma pequena queda na acurácia ou F1-Score ao usar menos features é aceitável e frequentemente desejável, "
    "pois indica um modelo **mais generalista**, menos propenso a *overfitting*.\n",
    "\n",
    "Os gráficos a seguir comparam visualmente as métricas de ambos os modelos.\n",
],

# ── Cell 33: Conclusões ──────────────────────────────────────────────────────
33: [
    "## 12. Resumo Final e Conclusões\n",
    "\n",
    "### Síntese dos Resultados\n",
    "\n",
    "Este artigo apresentou um pipeline completo de Machine Learning para a **classificação de três doenças** "
    "(pneumonia, bronquite aguda e cistite) a partir de sintomas binários. O resumo abaixo consolida os "
    "principais indicadores do projeto:\n",
],

}

# ─── Células markdown adicionais intercaladas (inseridas ENTRE código) ───────
# Estas serão NOVAS células inseridas entre células de código existentes
# para enriquecer o fluxo narrativo do artigo.

# Inseridas APÓS uma célula de código específica (por índice da célula code)
INSERT_AFTER = {

# Após Cell 8 (verificação de valores ausentes) — antes do top sintomas
8: {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.2 Verificação de Valores Ausentes e Tipos de Dados\n",
        "\n",
        "A verificação acima confirma que **não há valores ausentes** no dataset, o que simplifica o "
        "pré-processamento. Todos os dados de sintomas são do tipo inteiro (0 ou 1), e a coluna "
        "`prognosis` é do tipo *object* (texto), o que é esperado.\n",
        "\n",
        "### 3.3 Sintomas Mais Frequentes por Doença\n",
        "\n",
        "Para entender quais sintomas são mais discriminativos, visualizamos os **10 sintomas mais "
        "frequentes** para cada doença. Essa análise é importante pois:\n",
        "\n",
        "- Revela quais variáveis carregam **mais informação** sobre cada doença;\n",
        "- Antecipa quais features terão **maior importância** no modelo;\n",
        "- Permite identificar **sintomas em comum** entre doenças (que podem causar confusão no modelo) "
        "e sintomas **exclusivos** (que facilitam a diferenciação).\n",
    ]
},

# Após Cell 21 (relatório por classe) — antes do gráfico de métricas
21: {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Relatório Detalhado por Classe\n",
        "\n",
        "O relatório acima (*classification report*) mostra precisão, recall e F1-Score **individuais para "
        "cada doença**. Essa análise granular é fundamental para:\n",
        "\n",
        "- Identificar se o modelo tem dificuldade com alguma doença específica;\n",
        "- Verificar se o desempenho é equilibrado entre as classes;\n",
        "- Detectar possíveis problemas de desbalanceamento.\n",
        "\n",
        "### Visualização das Métricas\n",
        "\n",
        "O gráfico a seguir apresenta as métricas globais do modelo de forma visual, facilitando a "
        "comparação rápida entre os diferentes indicadores de desempenho.\n",
    ]
},

}


def main():
    # Ler notebook original
    with open('disease_prediction.ipynb', encoding='utf-8') as f:
        nb = json.load(f)

    original_cells = nb['cells']
    print(f"Notebook original: {len(original_cells)} cells")

    # Contar células de código originais
    code_count = sum(1 for c in original_cells if c['cell_type'] == 'code')
    print(f"  Células de código: {code_count}")
    print(f"  Células markdown:  {len(original_cells) - code_count}")

    # Construir novo array de células
    new_cells = []

    for i, cell in enumerate(original_cells):
        if cell['cell_type'] == 'markdown' and i in MARKDOWN_CELLS:
            # Substituir markdown
            new_cell = copy.deepcopy(cell)
            new_cell['source'] = MARKDOWN_CELLS[i]
            new_cells.append(new_cell)
            print(f"  ✓ Cell {i}: markdown SUBSTITUÍDA")
        else:
            # Manter célula original (code ou markdown não mapeada)
            new_cells.append(copy.deepcopy(cell))
            if cell['cell_type'] == 'markdown':
                print(f"  - Cell {i}: markdown mantida")

        # Verificar se precisamos inserir uma célula extra após esta
        if i in INSERT_AFTER:
            new_cells.append(INSERT_AFTER[i])
            print(f"  + INSERIDA nova markdown após Cell {i}")

    nb['cells'] = new_cells

    # Contar células de código no resultado
    new_code_count = sum(1 for c in new_cells if c['cell_type'] == 'code')
    print(f"\nNotebook atualizado: {len(new_cells)} cells")
    print(f"  Células de código: {new_code_count}")
    print(f"  Células markdown:  {len(new_cells) - new_code_count}")

    # Verificar que o código não foi alterado
    assert new_code_count == code_count, \
        f"ERRO: Número de células de código mudou! {code_count} → {new_code_count}"
    print("\n✓ Verificação: número de células de código preservado.")

    # Salvar
    with open('disease_prediction.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("\n✓ Notebook salvo com sucesso!")


if __name__ == '__main__':
    main()
