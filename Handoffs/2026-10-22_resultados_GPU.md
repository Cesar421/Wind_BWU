# Resultados del plan GPU — ventana de entrada + restricción estadística (PatchTST)

> Resultado del plan [`2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md`](2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md).
> Ejecutado el 2026-10-21/22 en la máquina GPU (RTX A4000). Todo está medido; las tablas
> completas, figuras y logs están en [`Agent_Test/results/gpu_plan_2026-10/`](../Agent_Test/results/gpu_plan_2026-10/README.md).

## Resumen en 30 segundos

- **Ventana larga: sí ayuda.** Con 2 s de historia (seq 2000, patches 128/64), el RMSE@500 en
  R3 es **0.1665**, un 5% mejor que el publicado (0.1752). En R2 la mejora frente a seq 100 es
  de un 3%.
- **Generaliza a geometrías no vistas (Fase 5).** Entrenado sin `1_1_4` ni `3_1_3`, sobre esas
  64 series da 0.1597 a h=500, frente a 0.1592 del modelo que sí las vio.
- **Restricción estadística: resultado negativo → λ = 0.** Devuelve potencia espectral (hasta
  0.80), pero con variabilidad que no informa (skill ≤ 0.04) y concentrada en h≈240–250. El
  error medio sobre el horizonte empeora entre un 6% y un 11%. La regla de λ del plan habría
  elegido el peor modelo.
- **Dos correcciones del pipeline** que cambian números publicados a horizonte corto: la
  alineación de patches (−18% de error a h=10) y la σ de normalización (float64).
- **No falló ninguna corrida.** El lote final tuvo 13 corridas en 3.2 h.

## 1. Qué se hizo (y en qué se desvió del plan)

| Paso | Estado | Nota |
|---|---|---|
| Código de la sección 4 | Hecho | `load_all_buildings_ctx`, `WindowLoader`, `train_model(loss_fn, history)`, `stat_loss.py`, `train_patchtst_v2.py`, `run_gpu_plan.ps1/.sh`, `stationarity_blocks.py`, `analyze_gpu_plan.py`, `plot_forecast.py` |
| Fase 0 | Hecha | Estacionariedad confirmada; el "mismo pico" no se sostiene (ver §4) |
| Prueba de humo | Hecha | 3–4 s/época en R2, 12 s/época en R3; < 340 MB de GPU, < 2.3 GB de RAM |
| Fases 1 y 2 | Hechas, dos veces | Protocolo A (archivado) → protocolo B (válido) |
| Fase 3 (token de estadísticos) | **No hecha** | Opcional; `series_stats_train.csv` está listo |
| Fase 4 | Hecha (4a) | Con el modelo MSE: λ = 0 tras la Fase 2 |
| Fase 5 | Hecha (5a) | Leave-buildings-out |

Desviaciones, con su porqué:

1. **Alineación de patches.** El PatchTST original deja sin ver las `(seq − patch_len) % stride`
   muestras **más recientes**: 4 ms con seq 100, 16 ms con seq 2000. Prueba: su RMSE@1 (0.044)
   era el del naive a h=5. Ahora se alinea al final por defecto (`--align end`); la corrida de
   control `1a-orig` mide el efecto. Los checkpoints antiguos no cambian (`align_end=False` por
   defecto en el modelo).
2. **Ventanas cortadas en GPU, sin materializar.** Son las mismas ventanas que con
   `build_sequences` (verificado), pero ocupan 45 MB en R2 y 178 MB en R3 en lugar de varios GB.
   La Fase 4 no necesitó `--step 50`.
3. **μ/σ en float64.** `load_all_buildings` (float32) da σ_w = 0.4502 en R3; el valor correcto es
   0.4521, que es el sanity check del plan. No afecta a resultados anteriores.
4. **Calendario de entrenamiento (protocolo B).** El plan escalaba `MAX_EPOCHS` por el stride 30
   pero no las paciencias, así que con 5/15 épocas el modelo recibía un tercio de los pasos entre
   reducciones del LR y quedaba infraentrenado (protocolo A). Ahora es 600 / 45 / 15 =
   200 / 15 / 5 × step/10. Con ello 1a reproduce el run publicado: val MSE 0.1416 frente a
   0.1430, y RMSE@500 0.1722 frente a 0.1724.
5. **StatLoss sin plantilla.** La versión original se cumplía insertando la misma bajada y salto
   en h≈236 en todas las ventanas (17% de la varianza). Ahora los estadísticos se calculan tras
   restar la forma media del batch (`--legacy-statloss` reproduce la original).
6. **Criterio de selección.** El plan elegía λ por el RMSE@500. Se añadió el RMSE medio sobre
   h=1..500, la fracción de plantilla, la fracción del primer componente principal y el skill de
   la varianza añadida.
7. **Corridas extra:** λ_psd = 0.1 (2f), 2e (bins lineales), 2c-legacy y 1a-orig como controles.

## 2. Resultados (protocolo B, test)

| Corrida | h=1 | h=10 | h=500 | RMSE medio h=1..500 | potencia | skill |
|---|---|---|---|---|---|---|
| naive R2 | 0.031 | 0.064 | 0.236 | 0.212 | 0 | — |
| publicado R2 (seq 100) | 0.044 | 0.074 | 0.172 | — | — | — |
| 1a seq 100 | 0.037 | 0.060 | 0.172 | 0.164 | 0.01 | — |
| 1b seq 1000 | 0.048 | 0.071 | 0.169 | 0.161 | 0.02 | — |
| **1c seq 2000 = SEQ\*** | 0.046 | 0.067 | **0.167** | **0.159** | 0.02 | — |
| 1a-orig (alineación original) | 0.042 | 0.073 | 0.174 | 0.165 | 0.01 | — |
| 2d λ_psd 0, λ_std 1 | 0.183 | 0.138 | 0.176 | 0.169 | 0.14 | 0.04 |
| 2a λ_psd 0.0003 | 0.147 | 0.128 | 0.175 | 0.169 | 0.20 | 0.04 |
| 2b λ_psd 0.003 | 0.107 | 0.110 | 0.171 | 0.170 | 0.35 | 0.04 |
| 2c λ_psd 0.03 | 0.079 | 0.088 | 0.170 | 0.174 | 0.63 | 0.02 |
| 2f λ_psd 0.1 | 0.080 | 0.094 | 0.169 | 0.177 | 0.80 | 0.03 |
| naive R3 | 0.032 | 0.066 | 0.231 | 0.209 | 0 | — |
| publicado R3 (seq 100) | 0.043 | 0.073 | 0.175 | — | — | — |
| **4a R3, seq 2000** | 0.043 | 0.064 | **0.167** | **0.158** | 0.01 | — |

Fase 5, sobre las mismas 64 series y 28 288 ventanas de `1_1_4` + `3_1_3`:

| Modelo | h=1 | h=10 | h=500 | RMSE medio |
|---|---|---|---|---|
| 4a (vio esas geometrías) | 0.038 | 0.057 | 0.159 | 0.151 |
| **5a (no las vio)** | 0.036 | 0.056 | 0.160 | 0.151 |
| naive | 0.030 | 0.060 | 0.224 | 0.200 |

Decisiones: **`SEQ*` = 2000** (menor val MSE: 0.1341, frente a 0.1372 y 0.1416). **λ = 0**: todo
λ > 0 empeora el val RMSE medio, de 0.161 a 0.170–0.178.

Cómo leer la Fase 2: la potencia de la predicción sube con λ, pero el skill se queda en ≈0. La
varianza añadida no se relaciona con lo que falla el modelo MSE: aparece como una familia de
oscilaciones de signo variable en h≈240–250 (fracción PC1 0.49–0.71; la señal real tiene 0.28),
y ahí el RMSE llega a 0.25–0.29, peor que el naive. Un pronóstico determinista no puede
reproducir la parte impredecible sin inventarla. Para pronósticos con la variabilidad correcta
haría falta un modelo probabilístico que genere muestras. Las bandas de octava solo reescalan λ:
2e (lineal, 0.003) cae en el mismo punto de la curva que 2a (octava, 0.0003).

Figuras clave (en `Agent_Test/results/gpu_plan_2026-10/`):
- `analisis/pareto_rmse_vs_potencia.png`, `analisis/pareto_rmse500.png`: la curva de Pareto con
  los dos criterios.
- `analisis/fase1_ventana.png`, `analisis/psd_predicciones.png`,
  `analisis/forecast_comparison.png`.
- `runs/4a_s2000_p128-64_mse_r3/forecast_continuous.png` y `forecast_detail.png`: la serie real
  frente a las predicciones del modelo final. Se generan con `python plot_forecast.py`.
- `fase0_estacionariedad/stationarity_blocks.png`: figura de la Fase 0 para la tesis.

## 3. Tiempos reales (RTX A4000)

Protocolo B: Fase 1 en 31 min (19.1 + 4.4 + 8.0); Fase 2 en 48 min (8–11.5 min por corrida);
4a en 41.6 min; 5a en 40.4 min; controles en 32 min. **Total: 3.2 h.** El protocolo A tardó
40 min (8 corridas, infraentrenadas). No hubo ningún TDR ni ningún NaN.

## 4. Anomalías y matices

- **Artefacto al final de las series.** En 158 de 340 series, las últimas ~5 muestras saltan más
  que el percentil 99.9 de los saltos normales. Por ejemplo, Alpha1_4/2_1_3/0 pasa de 0.2 a 1.25
  en t = 32.76 s. Afecta a menos del 0.1% de los targets de test, así que no cambia resultados,
  pero conviene recortar las últimas ~10 muestras en futuras corridas.
- **El naive sigue ganando a h=1** (0.031 frente a 0.037–0.046), y con patches grandes (seq 2000)
  el modelo empeora a h ≤ 10 frente a seq 100. Un patching multiescala podría recuperar el corto
  plazo.
- **Fase 0:** sin tendencia entre bloques (Wilcoxon p ≥ 0.10 en todos los estadísticos; std ±3.6%,
  potencia ±10%). Pero el pico espectral solo coincide entre bloques en el 2% de las series: es un
  estadístico ruidoso con estos espectros anchos, y el "mismo pico" del plan no se sostiene.
- **Regla de λ del plan:** con RMSE@500 frente al naive, todas las λ pasan y la regla elige la
  más alta. Ese criterio no detecta artefactos en otros horizontes.

## 5. Qué falta / siguientes pasos

1. **Tesis (máquina sin GPU):** añadir el resultado de la Fase 1/4, la corrección de alineación,
   la Fase 5 y el resultado negativo de la Fase 2 con su explicación. Siguen pendientes las
   tareas de la sección 9 del plan: correcciones de texto y la evaluación de los modelos directos
   sobre las 680 trayectorias.
2. **Opcional:** Fase 3 (token de estadísticos por serie), recorte del final de las series, y
   patching multiescala para horizontes cortos.
3. **Si se quiere variabilidad realista:** un modelo probabilístico (entrada de ruido +
   entrenamiento con energy score o CRPS) en lugar de la StatLoss.

## 6. Reproducir

```powershell
cd Agent_Test
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python stationarity_blocks.py                      # Fase 0 (CPU, minutos)
.\run_gpu_plan.ps1 -Python <python del entorno con CUDA>   # protocolo B completo, ~3.2 h
python analyze_gpu_plan.py                         # tabla + figuras en results/gpu_plan_2026-10/analisis/
python plot_forecast.py                            # serie real frente a pronóstico (run 4a)
```

En la máquina GPU, el entorno es `ML_Cesar` (`C:\Users\verwalter\anaconda3\envs\ML_Cesar`). Los
checkpoints (`*.pt`) y las predicciones (`runs/*/big/`) **no están en git**: se regeneran con el
lote de arriba, o se suben a Drive si hacen falta en la otra máquina.
