document.addEventListener('DOMContentLoaded', () => {
    
    const symptomsGrid = document.getElementById('symptomsGrid');
    const predictionForm = document.getElementById('predictionForm');
    const loadingFeatures = document.getElementById('loadingFeatures');
    const submitBtn = document.getElementById('submitBtn');
    const submitSpinner = document.getElementById('submitSpinner');
    
    // Result card elements
    const resultCard = document.getElementById('resultCard');
    const diseaseNameEl = document.getElementById('diseaseName');
    const confidenceValueEl = document.getElementById('confidenceValue');
    const confidenceFillEl = document.getElementById('confidenceFill');
    const resetBtn = document.getElementById('resetBtn');

    // Dicionários de Tradução para Português (pt-BR)
    const symptomTranslations = {
        "shortness of breath": "Falta de ar",
        "sharp chest pain": "Dor aguda no peito",
        "sore throat": "Dor de garganta",
        "cough": "Tosse",
        "nasal congestion": "Congestão nasal",
        "retention of urine": "Retenção urinária",
        "suprapubic pain": "Dor suprapúbica",
        "sharp abdominal pain": "Dor aguda no abdômen",
        "vomiting": "Vômito",
        "headache": "Dor de cabeça",
        "painful urination": "Dor ao urinar",
        "involuntary urination": "Micção involuntária",
        "frequent urination": "Vontade frequente de urinar",
        "lower abdominal pain": "Dor no baixo abdômen",
        "blood in urine": "Sangue na urina",
        "back pain": "Dor nas costas",
        "pelvic pain": "Dor pélvica",
        "wheezing": "Chiado no peito",
        "weakness": "Fraqueza",
        "side pain": "Dor lateral",
        "fever": "Febre",
        "difficulty breathing": "Dificuldade para respirar",
        "chills": "Calafrios",
        "coughing up sputum": "Tosse com catarro",
        "coryza": "Coriza (nariz escorrendo)",
        "congestion in chest": "Congestão no peito",
        "symptoms of bladder": "Sintomas na bexiga"
    };

    const diseaseTranslations = {
        "pneumonia": "Pneumonia",
        "acute bronchitis": "Bronquite Aguda",
        "cystitis": "Cistite"
    };

    // 1. Fetch Features from Server
    fetch('/api/features')
        .then(response => response.json())
        .then(data => {
            const features = data.features;
            
            if(!features || features.length === 0){
                loadingFeatures.innerText = "Erro: Servidor não encontrou os modelos.";
                return;
            }

            loadingFeatures.classList.add('hidden');
            predictionForm.classList.remove('hidden');

            // Renderizar checkboxes (toggles) para cada sintoma
            features.forEach(feature => {
                // Traduzir texto se disponível, senão formatar texto padrão
                const labelText = symptomTranslations[feature] || feature.split('_').join(' ').replace(/\b\w/g, c => c.toUpperCase());
                
                const labelEl = document.createElement('label');
                labelEl.className = 'symptom-label';
                
                // HTML interno do Toggle
                labelEl.innerHTML = `
                    <span class="symptom-name">${labelText}</span>
                    <input type="checkbox" name="symptoms" value="${feature}">
                    <div class="toggle-switch"></div>
                `;

                // Efeito visual ao clicar
                const checkbox = labelEl.querySelector('input');
                checkbox.addEventListener('change', (e) => {
                    if(e.target.checked) labelEl.classList.add('active');
                    else labelEl.classList.remove('active');
                });

                symptomsGrid.appendChild(labelEl);
            });
        })
        .catch(err => {
            console.error('Error fetching features:', err);
            loadingFeatures.innerText = "Erro de conexão com o servidor.";
        });


    // 2. Submit Form for Prediction
    predictionForm.addEventListener('submit', (e) => {
        e.preventDefault();

        // Pegar todos os checkboxes marcados
        const checkedBoxes = document.querySelectorAll('input[name="symptoms"]:checked');
        const userSymptoms = Array.from(checkedBoxes).map(cb => cb.value);

        // UI State: Loading
        submitBtn.disabled = true;
        submitSpinner.classList.remove('hidden');
        submitBtn.querySelector('span').style.opacity = '0.5';

        // Fechar card antigo se existir
        resultCard.classList.add('hidden');
        confidenceFillEl.style.width = '0%';

        // Request para predição
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptoms: userSymptoms })
        })
        .then(response => response.json())
        .then(data => {
            // Restore UI State
            submitBtn.disabled = false;
            submitSpinner.classList.add('hidden');
            submitBtn.querySelector('span').style.opacity = '1';

            if(data.error) {
                alert("Erro: " + data.error);
                return;
            }

            // Update Result Card
            diseaseNameEl.innerText = diseaseTranslations[data.prediction] || data.prediction;
            confidenceValueEl.innerText = data.confidence.toFixed(2) + "%";
            
            // Exibir card e animar barra
            resultCard.classList.remove('hidden');
            
            // Pequeno delay para a transição CSS da barra funcionar
            setTimeout(() => {
                confidenceFillEl.style.width = data.confidence + "%";
            }, 100);

            // Rolar para cima mobile
            if(window.innerWidth <= 900) {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        })
        .catch(err => {
            console.error('Prediction error:', err);
            alert("Falha na comunicação com o servidor de predição.");
            submitBtn.disabled = false;
            submitSpinner.classList.add('hidden');
            submitBtn.querySelector('span').style.opacity = '1';
        });
    });

    // 3. Reset Button
    resetBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        confidenceFillEl.style.width = '0%';
        
        // Desmarcar todos
        const checkedBoxes = document.querySelectorAll('input[name="symptoms"]:checked');
        checkedBoxes.forEach(cb => {
            cb.checked = false;
            cb.closest('.symptom-label').classList.remove('active');
        });
    });
});
