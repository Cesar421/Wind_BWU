# Resultados — plan GPU 2026-10 (ventana de entrada + restricción estadística)

Plan: [`Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md`](../../../Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md).
Código: `patchtst/train_patchtst_v2.py`, `stat_loss.py`, `data_loader.load_all_buildings_ctx`,
`stationarity_blocks.py`, `run_gpu_plan.ps1`, `analyze_gpu_plan.py`.

**Estado (2026-10-22):** el **protocolo B** terminó: 13 corridas en 3.2 h, sin fallos. El
**protocolo A** (primer lote, 40 min) queda archivado en `protocolo_A/`: tenía dos problemas,
descritos abajo, pero sus resultados documentan el control de alineación y el artefacto de plantilla.

## Resultados del protocolo B

Las RMSE están en Cp y son de test. El RMSE medio es sobre h=1..500. Skill = correlación de la
varianza añadida con el error del modelo MSE (0 = no informa).

| Corrida | h=1 | h=10 | h=500 | RMSE medio | RMSE máx. (h) | potencia | skill |
|---|---|---|---|---|---|---|---|
| naive R2 | 0.031 | 0.064 | 0.236 | 0.212 | 0.236 | 0 | — |
| publicado R2 (seq 100) | 0.044 | 0.074 | 0.172 | — | — | — | — |
| 1a seq 100 | 0.037 | **0.060** | 0.172 | 0.164 | 0.175 | 0.01 | — |
| 1b seq 1000 | 0.048 | 0.071 | 0.169 | 0.161 | 0.171 | 0.02 | — |
| **1c seq 2000 (`SEQ*`)** | 0.046 | 0.067 | **0.167** | **0.159** | 0.168 | 0.02 | — |
| 1a-orig (alineación original) | 0.042 | 0.073 | 0.174 | 0.165 | 0.175 | 0.01 | — |
| 2d λ_psd 0 | 0.183 | 0.138 | 0.176 | 0.169 | 0.183 (1) | 0.14 | 0.04 |
| 2a λ_psd 0.0003 | 0.147 | 0.128 | 0.175 | 0.169 | 0.205 (245) | 0.20 | 0.04 |
| 2b λ_psd 0.003 | 0.107 | 0.110 | 0.171 | 0.170 | 0.236 (241) | 0.35 | 0.04 |
| 2c λ_psd 0.03 | 0.079 | 0.088 | 0.170 | 0.174 | 0.246 (242) | 0.63 | 0.02 |
| 2f λ_psd 0.1 | 0.080 | 0.094 | 0.169 | 0.177 | 0.287 (250) | 0.80 | 0.03 |
| 2e λ_psd 0.003, bins lineales | 0.142 | 0.133 | 0.175 | 0.170 | 0.198 (244) | 0.20 | 0.03 |
| 2c-legacy λ_psd 0.03, StatLoss original | 0.087 | 0.097 | 0.169 | 0.173 | 0.297 (244) | 0.61 | 0.03 |
| naive R3 | 0.032 | 0.066 | 0.231 | 0.209 | 0.231 | 0 | — |
| publicado R3 (seq 100) | 0.043 | 0.073 | 0.175 | — | — | — | — |
| **4a R3, seq 2000** | 0.043 | 0.064 | **0.167** | **0.158** | 0.167 | 0.01 | — |

Fase 5 (leave-buildings-out), sobre las **mismas** 64 series y 28 288 ventanas de las
geometrías `1_1_4` y `3_1_3`:

| Modelo | h=1 | h=10 | h=500 | RMSE medio |
|---|---|---|---|---|
| 4a (entrenó con esas geometrías) | 0.038 | 0.057 | 0.159 | 0.151 |
| **5a (nunca las vio)** | 0.036 | 0.056 | 0.160 | 0.151 |
| naive | 0.030 | 0.060 | 0.224 | 0.200 |

Conclusiones:

1. **Calendario corregido:** 1a reproduce el publicado (val MSE 0.1416 frente a 0.1430; RMSE@500
   0.1722 frente a 0.1724). El stride 30 no cambia nada y el infraentrenamiento del protocolo A
   queda resuelto.
2. **Ventana larga (Fase 1):** con seq 2000 el error baja un 3% respecto a seq 100, tanto a
   h=500 como en el RMSE medio. A h ≤ 10 empeora, porque los patches de 128 ms difuminan las
   últimas muestras. En R3 el RMSE@500 es 0.1665, un 5% mejor que el publicado (0.1752).
3. **Alineación de patches:** −11% de error a h=1 y −18% a h=10; a h=500 la mejora es menor del 1%.
4. **Generalización (Fase 5):** sobre geometrías no vistas, el modelo rinde igual que el que sí
   las vio (+0.3% a h=500). La pregunta que R2/R3 decían responder queda respondida en positivo.
5. **Restricción estadística (Fase 2): resultado negativo.** El arreglo elimina la plantilla fija
   (fracción 0.157 → 0.000), pero el modelo la sustituye por una **familia** de oscilaciones en
   h≈240–250 con signo variable (PC1 0.49–0.71). La potencia sube de 0.02 a 0.80, pero la
   varianza añadida no informa (skill ≤ 0.04) y el RMSE medio empeora entre un 6% y un 11%. El
   RMSE@500 baja con λ porque no mira el resto del horizonte; la regla de λ del plan elegiría 2f,
   que es el peor en error medio. Con validación sobre todo el horizonte, todo λ > 0 empeora:
   **λ = 0**. Un modelo determinista no puede reproducir la variabilidad impredecible sin
   inventarla; para eso haría falta un modelo probabilístico que genere muestras.
6. **Bandas de octava frente a lineales:** 2e (lineal, λ 0.003) cae en el mismo punto de la curva
   que 2a (octava, λ 0.0003). Las octavas solo reescalan λ; no cambian la frontera.

## Estructura

| Ruta | Contenido |
|---|---|
| `metrics.csv` | Tabla maestra del protocolo B: una fila por corrida × split (val/test) × h ∈ {1, 10, 50, 100, 500} |
| `runs/<corrida>/` | Una carpeta por corrida: `config.json`, `summary.json`, `train.log`, `history.csv`, `training_curves.png`, `rmse_curve.npy` (test, h=1..500), `power_ratio.csv`, `psd.npz`, `series_meta.csv`, `forecast_examples.png` |
| `runs/<corrida>/big/` | `preds.npy`, `trues.npy`, `series_id.npy`, `target_start.npy` — **no va a git** |
| `runs/naive_*` | Naive (último valor observado) sobre exactamente las mismas ventanas de val y test (r2, r3, r3_lbo) |
| `analisis/` | Salida de `analyze_gpu_plan.py`: tabla, curvas de Pareto, Fase 1, PSD y pronósticos comparados |
| `run_gpu_plan.log` | Log del lanzador: comandos, tiempos y decisión de `SEQ*` |
| `fase0_estacionariedad/` | Fase 0: estadísticos por bloques de 10 s, `series_stats_train.csv` (Fase 3) y figura para la tesis |
| `protocolo_A/` | Primer lote, archivado (misma estructura) |
| `_smoke/` | Pruebas de humo — **no va a git** |

Checkpoints: `patchtst/checkpoints/gpu_plan_2026-10/[protocolo_A/]<corrida>.pt` (gitignored).
Nombre de corrida = `<tag>_s<seq>_p<patch_len>-<patch_stride>_<pérdida>_<scope>`, p. ej.
`2b_s2000_p128-64_lpsd0.003_lstd1_r2`.

## Experimento 6: quitar a cada serie su media de train (2026-10-22)

Idea: a cada serie se le resta **la media de su parte de train** (primeros 23 s, por cara), en
las tres partes: train, val y test. Después se le vuelve a sumar antes de medir el error, así que
la comparación con 4a es sobre exactamente las mismas ventanas y en Cp. Es causal, porque solo usa
el pasado de la serie, y se hace en memoria: `--demean` en `train_patchtst_v2.py`. El dataset no
cambia: los 3169 archivos de `Data/` tienen la misma huella SHA-256 antes y después, y git no ve
cambios. Lanzador: `run_gpu_demean.ps1` (6 corridas en 3.8 h). Motivación medida: a 500 ms, el
modelo 4a era peor (0.1665) que predecir simplemente la media de train de la serie (0.1630),
porque estimaba el nivel solo con 2 s de historia. Tras quitar la media de cada serie, σ_w pasa de
0.452 a 0.163: el 87% de la varianza eran diferencias de nivel entre series.

Test R3, media ± desviación de 3 semillas (42, 43, 44):

| Modelo | h=1 | h=10 | h=500 | RMSE medio h=1..500 |
|---|---|---|---|---|
| naive | 0.0322 | 0.0656 | 0.2309 | 0.2092 |
| 4a (seq 2000) | 0.0425 ± 0.0011 | 0.0626 ± 0.0009 | 0.1665 ± 0.0001 | 0.1576 ± 0.0002 |
| **6a (4a + media por serie)** | **0.0330 ± 0.0014** | **0.0574 ± 0.0011** | **0.1632 ± 0.0000** | **0.1546 ± 0.0001** |

Geometrías no vistas (`1_1_4` + `3_1_3`, las mismas 64 series y ventanas):

| Modelo | h=1 | h=10 | h=500 | RMSE medio |
|---|---|---|---|---|
| 4a (las vio) | 0.0382 | 0.0567 | 0.1592 | 0.1508 |
| 5a (no las vio) | 0.0361 | 0.0556 | 0.1597 | 0.1512 |
| 6a media por serie (las vio) | 0.0295 | 0.0522 | 0.1560 | 0.1484 |
| **6b media por serie (no las vio)** | 0.0322 | 0.0507 | **0.1558** | **0.1482** |
| naive | 0.0295 | 0.0596 | 0.2236 | 0.2000 |

Conclusiones:

1. **Mejora real en todos los horizontes:** −2.0% a h=500, −1.9% en el RMSE medio, −8% a h=10 y
   −22% a h=1, donde iguala prácticamente al naive. La variación entre semillas (±0.0001 a h=500)
   es unas 30 veces menor que la mejora.
2. **La generalización se mantiene:** 6b, que no vio esas geometrías, rinde igual que 6a, que sí las
   vio, y mejora a 5a un 2.4%.
3. **La variabilidad entre semillas es muy pequeña** (±0.0001–0.0002 a h=500 y en el RMSE medio),
   lo que respalda también las diferencias de la Fase 1.
4. **Límite que hay que declarar:** el modelo necesita conocer la media de la serie (o de esa
   configuración) a partir de una medición previa de unos 23 s. Vale para el escenario de
   pronóstico online o de diseño con historia; para un edificio nuevo sin ninguna medición, el
   modelo es 4a/5a.

## Protocolo B: qué cambió y qué se corrió

Cambios respecto al protocolo A:

1. **Calendario equivalente en pasos de optimización** al de los runs publicados (stride 10):
   máximo de épocas, paciencia de early stopping y paciencia del LR = 200 / 15 / 5 × step/10, es
   decir, **600 / 45 / 15 con stride 30**. El protocolo A mantenía 300 / 15 / 5 épocas, que
   equivalen a un tercio de los pasos entre reducciones del LR: el modelo quedaba infraentrenado.
2. **StatLoss sin plantilla:** la potencia y la std se calculan sobre las desviaciones de cada
   ventana **menos la forma media del batch**. Una forma común a todas las ventanas ya no aporta
   nada a la restricción. `--legacy-statloss` reproduce la versión original.
3. **Métricas nuevas** por corrida y split: RMSE medio y máximo sobre h=1..500, fracción de
   plantilla (energía de la forma media / varianza dentro de ventana; ≈0 en la señal real) y
   fracción del primer componente principal. `analyze_gpu_plan.py` añade el **skill** de la
   varianza añadida, `corr(pred − pred_SEQ*, true − pred_SEQ*)`: 0 = no informa.
4. Los batches de val usados para el early stopping mezclan series, en un orden fijo.

| Tag | Fase | Config | Pregunta |
|---|---|---|---|
| 1a / 1b / 1c | 1 | seq 100 / 1000 / 2000, MSE, R2 | ¿Más contexto baja el error? → `SEQ*` = menor val MSE |
| 2d / 2a / 2b / 2c / 2f | 2 | `SEQ*`, StatLoss sin plantilla, λ_std 1, λ_psd 0 / 0.0003 / 0.003 / 0.03 / 0.1 | Curva de Pareto error–potencia |
| 4a | 4 | `SEQ*`, MSE, **R3** | Confirmación en el dataset completo (no depende de λ) |
| 5a | 5 | `SEQ*`, MSE, R3 sin `1_1_4` ni `3_1_3`; test en esas 64 series | ¿Generaliza a geometrías no vistas? |
| 2e | control | `SEQ*`, λ_psd 0.003 con bins lineales | Justifica las bandas de octava |
| 2c-legacy | control | `SEQ*`, λ_psd 0.03, StatLoss original | Efecto del arreglo de plantilla |
| 1a-orig | control | seq 100, alineación de patches original | Efecto de la alineación |

**Decisiones:** `SEQ*` = 2000 (menor val MSE en la Fase 1: 0.1341 frente a 0.1372 y 0.1416) ·
λ = 0 (recomendado: todo λ > 0 empeora el val RMSE medio, de 0.161 a 0.170–0.178).

## Protocolo A (archivado en `protocolo_A/`)

Test R2. Las RMSE están en Cp; el RMSE medio es sobre h=1..500.

| Corrida | h=1 | h=10 | h=500 | RMSE medio | potencia | plantilla | skill |
|---|---|---|---|---|---|---|---|
| naive | 0.031 | 0.064 | 0.236 | 0.212 | 0 | — | — |
| publicado (seq 100, stride 10) | 0.044 | 0.074 | 0.172 | — | — | — | — |
| 1a seq 100 | 0.032 | 0.060 | 0.180 | 0.169 | 0.01 | — | — |
| 1b seq 1000 | 0.044 | 0.067 | 0.169 | 0.161 | 0.02 | — | — |
| 1c seq 2000 (`SEQ*`) | 0.052 | 0.073 | 0.167 | 0.160 | 0.02 | 0.001 | — |
| 2d λ_psd 0 | 0.230 | 0.143 | 0.174 | 0.169 | 0.12 | 0.022 | 0.06 |
| 2a λ_psd 0.0003 | 0.135 | 0.132 | 0.172 | 0.169 | 0.21 | 0.042 | 0.06 |
| 2b λ_psd 0.003 | 0.116 | 0.120 | 0.170 | 0.170 | 0.33 | 0.080 | 0.05 |
| 2c λ_psd 0.03 | 0.091 | 0.100 | 0.169 | 0.173 | 0.62 | 0.172 | 0.04 |
| 1a-orig (alineación original) | 0.041 | 0.074 | 0.179 | 0.170 | 0.01 | — | — |

- **Ventana larga:** con seq 2000, el RMSE@500 baja un 7% respecto a seq 100 y mejora el publicado
  (0.172). A h ≤ 10 empeora, porque los patches de 128 ms difuminan las muestras más recientes.
- **Alineación de patches:** −22% de error a h=1 y −19% a h=10; a h=500 queda igual (0.1795 frente a 0.1793).
- **Problema 1, infraentrenamiento:** la train loss final de 1a es 0.148, frente a 0.141 del run
  publicado. Son ~49 k pasos frente a ~168 k, y el LR cayó a 4e-6 en la época 99.
- **Problema 2, plantilla:** la StatLoss original se cumple insertando **la misma bajada y salto en
  h≈236 en todas las ventanas**, que es el 17% de la varianza con λ_psd = 0.03. Ahí el RMSE es 0.279,
  peor que el naive en cualquier horizonte. El RMSE@500 no lo ve, y la regla de λ del plan elegía
  justamente esa corrida. La varianza añadida tiene un skill de 0.04–0.06: no se relaciona con lo
  que el modelo MSE falla.

## Fase 0 — estacionariedad (hecha, 340 series)

Desviación de cada bloque de 10 s respecto a la media de su serie (`stationarity_summary.csv`):

| Estadístico | mediana \|desv.\| | p95 | series monótonas (1/3 = sin tendencia) | Wilcoxon bloque 1 vs 3 |
|---|---|---|---|---|
| media | 0.0095 Cp | 0.035 Cp | 34 % | p = 0.12 |
| std | 3.6 % | 10.3 % | 33 % | p = 0.44 |
| asimetría | 0.10 | 0.29 | 39 % | p = 0.11 |
| curtosis | 0.20 | 0.69 | 32 % | p = 0.94 |
| potencia AC | 10.0 % | 27.0 % | 31 % | p = 0.55 |
| fracción 2–10 Hz | 3.6 pp | 11.1 pp | 34 % | p = 0.10 |

Sin tendencia en ningún estadístico: se confirma usar estadísticos por serie, no por bloque.
Matiz: el **pico espectral** solo cae en el mismo bin de Welch en los 3 bloques en el 2 % de
las series. No indica no-estacionariedad, porque el espectro es ancho y dominado por bajas
frecuencias y el argmax de 3 segmentos de Welch es muy ruidoso, pero el "mismo pico" del plan
no se sostiene como estadístico robusto.

## Notas de implementación (desviaciones del plan, con su porqué)

1. **Alineación de patches.** El PatchTST original empieza los patches en el índice 0 y deja
   sin cubrir las `(seq − patch_len) % stride` muestras **más recientes**: 4 ms con seq 100,
   8 ms con 1000 y 16 ms con 2000. `train_patchtst_v2.py` las descarta por el extremo antiguo
   (`--align end`, por defecto); el modelo original sigue disponible con `--align start`.
2. **Ventanas en GPU, sin materializar.** `load_all_buildings_ctx` devuelve las series
   normalizadas (45 MB en R2, 178 MB en R3) más un índice (fila, inicio) por ventana, y
   `WindowLoader` corta cada batch en la GPU. Son las mismas ventanas que con `build_sequences`
   (verificado). La Fase 4 ya no necesita `--step 50`.
3. **μ/σ en float64.** `load_all_buildings` reduce en float32 y da σ_w = 0.4502 en R3. El valor
   correcto es 0.4521, que es el sanity check del plan. No afecta a resultados anteriores.
4. **Calendario y StatLoss:** ver protocolo B.

## Tiempos medidos (RTX A4000)

| Config | s/época | pasos/época | GPU pico | RAM pico |
|---|---|---|---|---|
| R2 seq 100–2000, MSE | 2.9–3.3 | 495–453 | 114–199 MB | 1.3 GB |
| R2 seq 2000 + StatLoss | 3.5–4.3 | 453 | 199 MB | 1.4 GB |
| R3 seq 2000, MSE | 12.4 | 1812 | 337 MB | 2.3 GB |
| R3 leave-buildings-out | 10.0 | 1471 | 332 MB | 1.4 GB |
