# Plan: Multi-Agent System for Wind Pressure Cp Time Series Modeling

**TL;DR**: Construir un sistema de dos agentes autónomos orquestados con **LangGraph** (de LangChain) y **Claude API** como LLM backbone. El **Agente 1 (Researcher)** busca exhaustivamente en Semantic Scholar, arXiv y Google Scholar papers sobre ML/DL para predicción de coeficientes de presión (Cp) en time series, extrae metodologías con Claude, y genera un ranking de modelos candidatos. El **Agente 2 (Modeler)** toma esos candidatos, genera código PyTorch, entrena autónomamente en tu GPU local, evalúa con cross-validation temporal, y produce un reporte comparativo final. Todo se trackea con **MLflow** y se integra con tu pipeline existente en `src/DataPreproccesor.py` y `src/Read.py`.

**Costo estimado**: ~$80-120/mes (Claude API ~$40-60, SerpAPI ~$50/mo para Google Scholar, Semantic Scholar y arXiv son gratis).

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (LangGraph)              │
│                                                          │
│  ┌──────────────┐    papers.json    ┌──────────────────┐ │
│  │  AGENT 1:    │ ───────────────> │  AGENT 2:         │ │
│  │  Researcher  │                  │  Modeler          │ │
│  │              │ <─────────────── │                   │ │
│  │  - Semantic  │   feedback loop  │  - Code Generator │ │
│  │    Scholar   │                  │  - Trainer        │ │
│  │  - arXiv     │                  │  - Evaluator      │ │
│  │  - Google    │                  │  - Reporter       │ │
│  │    Scholar   │                  │                   │ │
│  └──────────────┘                  └──────────────────┘ │
│         │                                   │            │
│    Claude API                          PyTorch +         │
│    (extraction)                        MLflow            │
│                                        (local GPU)       │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware disponible

| GPU | VRAM | Ubicación |
|-----|------|-----------|
| NVIDIA RTX A4000 | 16 GB | Alemania |
| NVIDIA RTX 4500 | 24 GB | Alemania |
| NVIDIA RTX 3070 | 8 GB | Colombia |

---

## Step 0 — Infrastructure base

1. Crear `requirements.txt` con todas las dependencias: `langchain`, `langgraph`, `anthropic`, `langchain-anthropic`, `arxiv`, `semanticscholar`, `google-search-results` (SerpAPI), `mlflow`, `optuna`, `torch`, `transformers`, `einops`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `pyyaml`, `python-dotenv`, `pymupdf` (para parsear PDFs)
2. Crear `config/settings.yaml` — configuración centralizada: rutas de datos, hiperparámetros base, lista de ángulos de viento, parámetros de búsqueda
3. Crear `.env` — API keys: `ANTHROPIC_API_KEY`, `SERPAPI_KEY`, `MLFLOW_TRACKING_URI`
4. Crear estructura de directorios:
   ```
   agents/
     __init__.py
     orchestrator.py          # LangGraph workflow principal
     researcher/
       __init__.py
       agent.py               # Agente de búsqueda de literatura
       tools/
         semantic_scholar.py   # Tool: búsqueda en Semantic Scholar API
         arxiv_search.py       # Tool: búsqueda en arXiv API
         google_scholar.py     # Tool: búsqueda via SerpAPI
         pdf_parser.py         # Tool: extracción de texto de PDFs
         paper_analyzer.py     # Tool: análisis con Claude de metodología
       prompts/
         search_queries.yaml   # Queries predefinidas por categoría
         analysis_prompt.txt   # System prompt para análisis de papers
     modeler/
       __init__.py
       agent.py               # Agente de implementación de modelos
       tools/
         code_generator.py     # Tool: genera código PyTorch desde especificación
         trainer.py            # Tool: entrena modelo con early stopping
         evaluator.py          # Tool: evalúa con métricas + cross-validation
         hyperparameter_tuner.py # Tool: Optuna HPO
       templates/
         model_template.py     # Template base para modelos PyTorch
         training_template.py  # Template de training loop
       prompts/
         code_gen_prompt.txt   # System prompt para generación de código
   ```

---

## Step 1 — Agente Researcher: Tools de búsqueda

1. Implementar `semantic_scholar.py`: Usar la API gratuita de Semantic Scholar (sin key necesaria, rate limit 100 req/5min). Búsquedas por keywords combinados: `"wind pressure coefficient" AND ("deep learning" OR "machine learning" OR "neural network")`, `"Cp prediction" AND "building"`, `"time series forecasting" AND "pressure coefficient"`, etc. Extraer: título, abstract, año, citation count, DOI, PDF URL
2. Implementar `arxiv_search.py`: Usar librería `arxiv` de Python. Queries similares, enfocadas en cs.LG, physics.flu-dyn, cs.AI
3. Implementar `google_scholar.py`: Usar SerpAPI (requiere key, ~$50/mo, 5000 búsquedas/mes). Esto es clave porque Semantic Scholar y arXiv no cubren journals de wind engineering (como los PDFs adjuntos que son de *Energy and Buildings* y *Wind and Structures*)
4. Implementar `pdf_parser.py`: Usar `pymupdf` (fitz) para extraer texto de PDFs descargados. Incluir extracción de tablas y figuras descriptivas
5. Implementar `paper_analyzer.py`: Envía abstract + secciones de metodología a Claude API con un prompt estructurado que extrae: (a) tipo de modelo, (b) arquitectura exacta, (c) inputs/features usados, (d) métricas reportadas, (e) dataset usado, (f) si es aplicable a Cp time series, (g) ventajas/limitaciones

---

## Step 2 — Agente Researcher: Lógica del agente

1. Implementar `researcher/agent.py` como un **LangGraph StateGraph** con estos nodos:
   - `generate_queries`: Claude genera queries de búsqueda diversas basadas en el objetivo (reproducir Cp time series sin túnel de viento)
   - `search_papers`: Ejecuta búsquedas en paralelo en las 3 fuentes
   - `deduplicate`: Elimina papers duplicados por DOI/título
   - `rank_relevance`: Claude analiza los abstracts y rankea por relevancia (0-10)
   - `deep_analyze`: Para los top-30 papers, descarga PDFs cuando disponibles y extrae metodología detallada
   - `synthesize`: Genera un JSON estructurado con todos los modelos candidatos, clasificados por tipo: (1) LSTM/GRU variants, (2) Transformer-based (TFT, Informer, Autoformer), (3) CNN-based (TCN, WaveNet), (4) Hybrid (CNN-LSTM, ConvTransformer), (5) Probabilistic (DeepAR, N-BEATS), (6) Classical ML con features avanzados (XGBoost, LightGBM con lag features), (7) Physics-informed neural networks (PINNs)
   - `save_results`: Guarda en `literature_results/papers.json` y `literature_results/model_candidates.json`
2. Edge conditions: si < 15 papers relevantes encontrados → genera queries adicionales y re-busca; si un modelo candidato aparece en >3 papers → marcarlo como "high priority"

---

## Step 3 — Agente Modeler: Tools de implementación

1. Implementar `code_generator.py`: Recibe especificación de modelo (del JSON de Researcher) y genera código PyTorch. Usa Claude API con el template en `templates/model_template.py` como referencia. El código generado hereda de una base class `BaseWindModel` que estandariza: `forward()`, `predict()`, `get_config()`. Se valida que el código compile (exec + type check) antes de guardarlo
2. Implementar `trainer.py`: Training loop genérico que acepta cualquier `BaseWindModel`. Incluye: early stopping, learning rate scheduling (ReduceLROnPlateau), gradient clipping, mixed precision (para aprovechar tus GPUs), checkpoint saving. Se integra con la data pipeline existente de `src/DataPreproccesor.py`
3. Implementar `evaluator.py`: Evaluación completa con: (a) interpolation metrics (one-step-ahead), (b) **true forecasting** multi-step (horizons: 10, 50, 100, 500 steps), (c) `TimeSeriesSplit` cross-validation 5-fold, (d) métricas: RMSE, MAE, R², MAPE, directional accuracy. Compara contra los baselines ya existentes en `model_results/model_comparison_results.csv`
4. Implementar `hyperparameter_tuner.py`: Optuna con pruning (MedianPruner), búsqueda de: learning_rate, hidden_size, num_layers, dropout, sequence_length, batch_size. N_trials configurable (default 50)

---

## Step 4 — Agente Modeler: Lógica del agente

1. Implementar `modeler/agent.py` como LangGraph StateGraph:
   - `load_candidates`: Lee `model_candidates.json` del Researcher
   - `prioritize`: Ordena modelos por: (a) frecuencia en literatura, (b) reported performance en papers similares, (c) compatibilidad con tu dataset (univariate vs multivariate, seq length, etc.)
   - `generate_code`: Para cada modelo candidato (empezando por el de mayor prioridad), genera el código PyTorch
   - `validate_code`: Ejecuta el código con datos dummy para verificar que no crashea
   - `train_model`: Entrena con los datos reales. Primero un "quick run" (5 epochs, subset) para verificar que converge, luego entrenamiento completo
   - `evaluate_model`: Evaluación completa con cross-validation
   - `analyze_results`: Claude analiza los resultados, identifica patrones (¿qué tipo de modelo funciona mejor para true forecasting?)
   - `iterate`: Si ningún modelo supera el baseline en true forecasting, Claude sugiere modificaciones (ensemble, diferentes features, data augmentation) y genera nuevas variantes
   - `generate_report`: Produce un reporte markdown + datos para LaTeX con tablas comparativas, gráficos de predicción vs real, análisis de errores
2. MLflow tracking en cada paso: log de hiperparámetros, métricas, artefactos (modelos guardados, plots)

---

## Step 5 — Orquestador principal

1. Implementar `orchestrator.py`: LangGraph workflow que conecta Researcher → Modeler con un feedback loop. Si el Modeler determina que necesita modelos de un tipo específico no cubierto (ej: "necesitamos papers sobre physics-informed neural networks para Cp"), envía feedback al Researcher para buscar más papers de ese tipo
2. CLI en `agents/run.py`: Interfaz simple para ejecutar: `python -m agents.run --mode full` (todo el pipeline), `--mode research` (solo búsqueda), `--mode model` (solo modelado con candidatos existentes)
3. Logging en español para la interfaz, en inglés para el código

---

## Step 6 — Integración con proyecto existente

1. Adaptar `src/DataPreproccesor.py` para exponer una API limpia que los agentes puedan usar: `get_training_data(angle, face, seq_length) → (X_train, y_train, X_val, y_val, X_test, y_test)`
2. Los modelos generados se guardan en `Deeplearn/saved_models/` con nomenclatura estandarizada: `{model_type}_{timestamp}_{best_metric}.pth`
3. Los resultados se exportan a `model_results/` en formato compatible con los CSVs existentes
4. Actualizar el reporte LaTeX `Wind_pressure_Machine_learning_Report.tex` con una sección nueva de resultados del agente

---

## Step 7 — Modelos prioritarios a implementar

Basado en la literatura actual y los PDFs adjuntos, estos son los modelos que el agente debería priorizar (algunos se implementarán directamente como "seed models" antes de que el Researcher busque más):

1. **Temporal Fusion Transformer (TFT)** — estado del arte para time series forecasting con atención multi-horizon
2. **Informer** — eficiente para secuencias largas (tu dataset tiene 32K timesteps)
3. **N-BEATS** — interpretable, excelente para forecasting puro sin features externas
4. **DeepAR** — probabilístico, genera intervalos de confianza
5. **Temporal Convolutional Network (TCN)** — alternativa a LSTM, paralelizable
6. **XGBoost/LightGBM con lag features** — baseline fuerte, rápido de entrenar
7. **WaveNet** — diseñado para secuencias temporales, convoluciones causales dilatadas
8. **Ensemble** — stacking de los mejores modelos individuales

---

## Verification

1. **Researcher Agent**: Verificar que encuentra al menos los 3 papers adjuntos (o citados en ellos) como sanity check. Debe generar `literature_results/papers.json` con >50 papers relevantes y `model_candidates.json` con >10 modelos únicos
2. **Modeler Agent**: Verificar que cada modelo generado: (a) compila sin errores, (b) acepta input shape correcto (batch, seq_length, features), (c) produce output shape correcto, (d) converge en entrenamiento (loss decrece). Criterio de éxito: al menos 1 modelo debe superar R² > 0 en **true forecasting** (actualmente todos los modelos fallan en esto)
3. **End-to-end**: Ejecutar `python -m agents.run --mode full` y verificar que produce: MLflow dashboard con todos los experimentos, reporte comparativo en `model_results/agent_comparison_report.md`, modelos guardados en `Deeplearn/saved_models/`
4. **Costos**: Monitorear uso de Claude API tokens (~$0.015/1K input, $0.075/1K output para Sonnet) — un run completo no debería superar ~$20-30 en tokens

---

## Decisions

- **LangGraph sobre CrewAI/AutoGen**: LangGraph ofrece control explícito del flujo de estados, mejor debugging, integración nativa con Claude via `langchain-anthropic`, y es más maduro para workflows complejos con feedback loops. CrewAI es más simple pero menos flexible; AutoGen está más orientado a chat multi-agente que a pipelines autónomos
- **Claude API (Sonnet 4) como LLM**: Mejor razonamiento para análisis de papers y generación de código comparado con GPT-4o en benchmarks de coding. Costo competitivo (~$3/MTok input). Tu presupuesto permite ~2M tokens/mes de input + 500K output
- **SerpAPI para Google Scholar**: Es la única forma legal y estable de acceder a Google Scholar programáticamente (no tiene API oficial). Los journals de wind engineering (Wind & Structures, J. Wind Eng., Energy and Buildings) no están bien indexados en arXiv, así que Google Scholar es esencial
- **Optuna sobre Ray Tune**: Más ligero, no requiere setup distribuido, suficiente para una sola GPU. Ray Tune sería overkill
- **MLflow sobre W&B**: Gratuito, self-hosted, sin límites de experimentos. W&B tiene free tier pero con restricciones

---

## Current Project Status (Reference)

### Models already implemented

| Model | RMSE (Interpolation) | R² (Interpolation) | RMSE (True Forecast) | R² (Forecast) |
|---|---|---|---|---|
| Ridge Regression | 0.0391 | **0.9893** | 0.3872 | -0.051 |
| Linear Regression | 0.0392 | 0.9892 | 0.3872 | -0.051 |
| Random Forest | 0.0411 | 0.9882 | 0.6330 | -1.810 |
| Naive Persistence | 0.0473 | 0.9843 | 0.4551 | -0.453 |
| LSTM (4 variants) | TBD | TBD | TBD | TBD |

### Core problem to solve

All current models **fail at true multi-step forecasting** (R² < 0). The agent system must find architectures that can genuinely forecast Cp time series ahead without ground truth history — this is the primary success metric.

### Data characteristics

- Source: TPU Aerodynamic Database wind tunnel measurements
- 11 wind angles (0°–50°, 5° step), 300 pressure taps, ~32,768 timesteps at 1000 Hz
- Target: Cp difference between windward/leeward face sensor pairs
- Format: `.npy` arrays in `Data/Processed/`
