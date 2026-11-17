# Bias gaussiano AR(1) sobre 4a

> Propuesta del asesor (noviembre 2026): usar la PSD de las series como restricción y modelar el
> error del mejor modelo con una distribución normal y un término exponencial con una constante
> entre −1 y 1. Script: [`bias_gaussian_ar1.py`](../../../bias_gaussian_ar1.py). Todo corre en
> CPU y sin reentrenar.

## Resumen en 30 segundos

- **Modelo.** Al pronóstico de 4a (PatchTST, 2 s de historia, MSE, R3) se le añade una capa de
  error, sin cambiar el pronóstico medio:
  ```
  Cp(t+h) = 4a(t+h) + μ(h) + σ(h)·z_h
  z_h = ρ·z_(h−1) + √(1−ρ²)·ε_h        ε_h ~ N(0, 1)
  ```
  μ(h) y σ(h) salen de los errores de validación. ρ = 0.992 se elige en validación para que la
  PSD de las trayectorias se ajuste a la real en bandas de octava (las mismas de la StatLoss):
  esa es la restricción PSD.
- **Frente a 2f, en las mismas 37 570 ventanas de test:** misma potencia espectral (0.81 frente
  a 0.80), RMSE medio un 10% menor (0.1585 frente a 0.1765) y CRPS un 36% menor (0.0884 frente a
  0.1385). Además da bandas de incertidumbre calibradas: el intervalo del 90% contiene el 90.4%
  de los valores reales.
- **Limitaciones.** Una sola exponencial no reproduce la forma fina del espectro por encima de
  32 Hz (errores de hasta ±4–6 dB por banda). Los errores tienen colas más pesadas que la normal
  (curtosis 4.2–7.0). Y desde h = 116 ms el CRPS es algo peor que el de la climatología (~4%),
  porque 4a ya no aporta información sobre la media de la serie.

## Resultados en test

R3, 150 280 ventanas:

| Pronóstico | RMSE medio h=1..500 | RMSE h=500 | CRPS medio | Cobertura 50/80/90/95 % | Potencia | dB <10 Hz | dB 10–50 Hz | dB >50 Hz |
|---|---|---|---|---|---|---|---|---|
| **4a + bias AR(1)** | 0.1576 | 0.1664 | **0.0874** | 53.4 / 82.1 / 90.7 / 94.9 | 0.82 | −0.9 | −1.0 | +0.3 |
| 4a (pronóstico puntual) | 0.1576 | 0.1664 | 0.1219 (MAE) | — | 0.01 | −17.8 | −31.0 | −28.2 |
| Climatología N(μ_s, σ_s²) | 0.1631 | 0.1630 | 0.0892 | — | 0 | — | — | — |

R2, las 37 570 ventanas de la corrida 2f:

| Pronóstico | RMSE medio h=1..500 | RMSE h=500 | CRPS medio | Potencia | dB <10 Hz | dB 10–50 Hz | dB >50 Hz | LSD <50 Hz |
|---|---|---|---|---|---|---|---|---|
| **4a + bias AR(1)** | **0.1585** | **0.1670** | **0.0884** | 0.81 | −1.0 | −0.9 | +0.6 | 2.6 dB |
| 2f (PSD en la pérdida) | 0.1765 | 0.1692 | 0.1385 (MAE) | 0.80 | −1.2 | 0.0 | −0.1 | 1.3 dB |
| 4a (pronóstico puntual) | 0.1585 | 0.1670 | 0.1238 (MAE) | 0.01 | −17.7 | −30.8 | −27.9 | 29.5 dB |
| Climatología N(μ_s, σ_s²) | 0.1642 | 0.1633 | 0.0906 | 0 | — | — | — | — |

Cómo leerlas:
- El CRPS de un pronóstico puntual es su MAE, así que la columna compara las dos familias.
- El RMSE del pronóstico medio no cambia porque el bias medido es despreciable (|μ(h)| ≤ 0.003).
- Una sola trayectoria muestreada tiene RMSE 0.2232, √2 veces el de la media (0.1576 × 1.414 =
  0.2229). Es lo esperado y por eso las trayectorias se evalúan con CRPS y PSD, no con RMSE.
- La climatología es la media y la desviación típica de la parte de train de cada serie.

## La constante ρ (el término exponencial)

| Estimación | ρ | τ = −1/ln ρ |
|---|---|---|
| Restricción PSD (validación, bandas de octava) | 0.992 | 125 ms |
| Series de train: exp(−1/τ), con τ la escala integral (mediana) | 0.986 | 73 ms |
| Correlación lag-1 de los errores en h = 500 (validación) | 0.981 | 53 ms |

Las tres caen entre 0.98 y 0.99, pero la autocorrelación real de los errores (figura b de
`diagnostics.png`) no es una sola exponencial: cae rápido al principio y tiene una cola lenta.
Con la mejor ρ, el error por banda de octava es −0.4, −1.5, −2.0, −0.6, +3.6, +4.0, +1.2 y
−5.6 dB (de 2–4 Hz a 256–500 Hz). Por debajo de 32 Hz, donde está casi toda la energía, el ajuste
queda dentro de ±2 dB. La búsqueda completa está en `rho_search.csv`.

## Normalidad

Errores estandarizados de validación: asimetría entre +0.42 y +0.56, y curtosis 7.0 (h = 10 ms),
4.7 (h = 100 ms) y 4.2 (h = 500 ms), frente a 3 de la normal. Las series en sí son casi normales
(curtosis mediana 3.09), pero los errores tienen colas más pesadas, sobre todo a corto plazo,
donde el modelo no anticipa los saltos bruscos. Aun así, la normal da coberturas correctas en los
intervalos centrales.

## Siguientes pasos posibles

1. Un segundo término (otra exponencial o un ruido blanco) para el espectro por encima de 32 Hz.
2. Una distribución con colas más pesadas: t de Student, o la exponencial-potencia de Box–Tiao
   con β > 0.
3. Desde ~110 ms, mezclar con la climatología (regresión tipo EMOS con μ_s como predictor).

## Archivos

| Archivo | Contenido |
|---|---|
| `psd_4a_bias_vs_2f.png` | Figura principal: espectro de 4a + bias, 2f y 4a frente a la verdad (R2) |
| `diagnostics.png` | QQ de los errores, autocorrelación frente a ρ^k, cobertura y CRPS por horizonte (R3) |
| `forecast_examples.png` | Dos ventanas de test con bandas, trayectorias, 2f y la media de 4a |
| `comparison_test_r3.csv`, `comparison_test_r2_vs_2f.csv` | Las tablas de arriba |
| `metrics_by_h_test_r3.csv` | μ(h), σ(h), correlación lag-1, RMSE, CRPS y cobertura por horizonte |
| `rho_search.csv`, `summary.json` | Búsqueda de ρ y números clave |
| `big/` (915 MB, fuera de git) | Predicciones de 4a (validación y test) y de 2f (test) |

## Reproducir

```powershell
cd Agent_Test
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python bias_gaussian_ar1.py      # ~9 min de predicciones en CPU + ~1.5 min de análisis
```

Comprobaciones que hace el script: las predicciones reproducen las curvas de RMSE de test
guardadas de 4a y 2f (diferencia máxima 1.4e-8 y 4.0e-8), y los espectros de 2f coinciden con
su `psd.npz` (3e-6).
