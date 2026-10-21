# Plan GPU — ventana de entrada + restricción estadística (PatchTST)

> **Si eres la persona en la máquina GPU:** empieza por la sección **«Arranque»**, justo
> debajo del resumen.
> **Si eres una sesión de Claude Code:** lee el archivo entero; tu checklist es la sección 5.
>
> Este archivo es autocontenido. Se escribió el 2026-10-21 en la máquina sin GPU, después
> de una revisión completa del proyecto. Todos los números están medidos; donde algo es
> una hipótesis, se dice.
>
> El `HANDOFF.md` de la raíz (junio 2026) está **reemplazado**. No ejecutes lo que pide.

## Resumen en 30 segundos

Tres fases de GPU, todas con **PatchTST directo** (el mejor modelo actual):

1. **Ventana de entrada larga.** Hoy el modelo ve 100 ms y la dinámica dominante tarda
   400–4000 ms. Probar 1 s y 2 s.
2. **Restricción estadística.** Una pérdida que obliga a la predicción a tener la
   potencia espectral y la desviación estándar de la señal real, porque hoy colapsa a la media.
3. **Confirmar en R3** la configuración ganadora.

Más dos fases opcionales: condicionar en los estadísticos de la serie y un test
leave-buildings-out. **Unas 3–4.5 h de GPU en total.**

---

## Arranque — para ti, paso a paso

**1. Traer los cambios.** Si estás leyendo esto en la máquina GPU, ya está hecho.
```powershell
cd <ruta del repo>\Wind_BWU
git status      # comprueba que no haya cambios locales sin guardar
git pull
```
Si `git pull` se queja de cambios locales, haz antes `git stash`.

**2. Comprobar la GPU y el entorno**
```powershell
conda env list
conda activate <el entorno que tenga torch con CUDA>
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Tiene que imprimir `True NVIDIA RTX A4000`. Si sale `False`, ese entorno tiene torch sin
CUDA y hay que usar otro. (El entorno `ML_Cesar` del README no existe.)

**3. Que el PC no se suspenda.** Son 3–4 h de entrenamiento, y si Windows entra en
suspensión, el entrenamiento muere:
```powershell
powercfg /change standby-timeout-ac 0     # para restaurarlo después: ... 30
```

**4. Abrir Claude Code en la carpeta del repo** (VS Code con la carpeta abierta, o
`claude` en la terminal). El `CLAUDE.md` se carga solo y apunta a este archivo.

**5. Pegarle este mensaje:**

> Lee entero `Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md` y sigue
> su checklist (sección 5). Implementa el código de la sección 4, haz la Fase 0 y la
> prueba de humo en GPU (1 época por cada configuración de la Fase 1), y **para ahí**:
> enséñame la memoria usada, el tiempo por época y la primera fila del CSV antes de
> lanzar las Fases 1 y 2.

Ese "para ahí" es importante: así confirmas que todo funciona antes de gastar casi 2 h de GPU.

**6. Lanzar las Fases 1 y 2** (~1h40m, sin supervisión). Cuando la prueba de humo salga
bien, lánzalo en **una ventana de PowerShell aparte**, no dentro de Claude. Si se cierra
VS Code o la sesión, un proceso arrancado desde ahí puede morir.
```powershell
cd <ruta del repo>\Wind_BWU\Agent_Test
conda activate <entorno>
$env:KMP_DUPLICATE_LIB_OK="TRUE"
.\run_gpu_plan.ps1
```
El script `run_gpu_plan.ps1` lo crea Claude en el paso 5. Si sale un error de política
de ejecución: `powershell -ExecutionPolicy Bypass -File .\run_gpu_plan.ps1`.

**7. Cuando termine,** vuelve a Claude:

> Las Fases 1 y 2 terminaron. Analiza los resultados en
> `Agent_Test/results/gpu_plan_2026-10/` y hazme la curva de Pareto.

Con eso decides el λ y lanzas la Fase 4 (~1 h), y las fases 3 y 5 si hay tiempo.

**8. Devolver los resultados.** Claude escribe `Handoffs/AAAA-MM-DD_resultados_GPU.md` y
hace commit + push (sección 8). En la máquina sin GPU basta con un `git pull`.

**Si algo falla:**
- Si la pantalla parpadea o sale "el controlador de pantalla dejó de responder", es un
  TDR, el mismo que mató a LSTM-direct. PatchTST con batch 128 no lo tuvo, pero si pasa,
  dile a Claude que baje el batch.
- Si en la prueba de humo la memoria no alcanza, la Fase 4 (sección 6) trae la solución:
  `--step 50` y evaluación serie a serie.

---

## 1. Estado de partida (verificado el 2026-10-21)

| | |
|---|---|
| Mejor modelo | PatchTST directo, h=500: RMSE **0.1724** (R2) / **0.1752** (R3), R² 0.85 |
| Tiempos de referencia (A4000) | R2 **18.2 min**, R3 **71.2 min** (parada temprana en la época 111) |
| Naive a h=500 | 0.231 (mismas ventanas que los modelos directos) |
| Random Forest multi-trayectoria | Hecho — `Agent_Test/results/long_horizon_multitraj_round3.csv` |
| Checkpoints antiguos | No hacen falta: el baseline se reentrena con el mismo script que las variantes |

PatchTST está saturado en R2 (R2 y R3 difieren ≤0.003 de RMSE en todos los horizontes),
así que **todo se experimenta en R2** y solo el ganador va a R3.

---

## 2. Por qué (diagnóstico medido)

### 2.1 La ventana de entrada es demasiado corta

Reparto de la energía del Cp windward (Welch, nperseg=4096, df=0.24 Hz):

| serie | <2 Hz | 2–10 Hz | 10–50 Hz | >50 Hz | periodo del pico |
|---|---|---|---|---|---|
| Alpha1_4/2_1_3/a0 | 49.2% | 38.6% | 7.5% | 1.3% | 1365 ms |
| Alpha1_4/2_1_3/a45 | 41.0% | 44.0% | 7.2% | 1.7% | 410 ms |
| Alpha1_6/1_1_3/a0 | 50.4% | 35.7% | 8.8% | 1.2% | 4096 ms |
| Alpha1_6/3_1_3/a60 | 37.2% | 47.5% | 9.3% | 2.3% | 4096 ms |

Alrededor del 90% de la energía está por debajo de 10 Hz. Con `seq_length=100` (100 ms),
el modelo ve unos **0.13 ciclos** de la dinámica dominante.
**Hipótesis (no certeza):** el "suelo" de RMSE≈0.175 a h=500 es falta de contexto, no
ruido irreducible.

### 2.2 La predicción no diverge: colapsa a la media

Medido sobre las predicciones de PatchTST R3 (12 800 ventanas, espacio normalizado):
la std dentro de cada ventana de 500 pasos es **0.059**, frente a **0.291** en la señal
real, y la potencia espectral es ~63× menor. La restricción estadística sirve para
**devolver varianza**, no para frenar.

### 2.3 La señal es estacionaria → estadísticos por serie, no por bloque

Los bloques de 10 s de una misma serie dan los mismos estadísticos: std ±3%, potencia
±12% sin tendencia, mismo pico. Consecuencias:

- Los estadísticos se calculan **por serie**, no por bloque temporal.
- **No hay que entrenar "con los últimos 10 s".** Cada serie dura 32.77 s y el split
  actual es train 0–22.94 s, val 22.94–27.85 s, test 27.85–32.77 s. Los últimos 10 s
  (22.77–32.77 s) contienen el **100% del val y el 100% del test**: sería leakage
  total. Además, en una señal estacionaria lo reciente no aporta nada.
- Se entrena con **toda** la porción de train de cada serie.

---

## 3. Reglas comunes a TODAS las corridas

1. **Split** 70/15/15 cronológico por serie, sin cambios. z-score ajustado solo en train.
   Sanity check: en R3 deben salir μ_w = 0.2386 y σ_w = 0.4521.
2. **Mismas ventanas objetivo para todas las configuraciones.** Val y test anteponen el
   contexto del segmento anterior, para que los targets sean idénticos sea cual sea
   `seq_length`:
   ```python
   train_seg = ts_n[:t1]
   val_seg   = ts_n[t1 - seq_length : t2]   # contexto tomado del final del train
   test_seg  = ts_n[t2 - seq_length :]      # contexto tomado del final del val
   ```
   Es legítimo: el contexto es pasado y ningún target sale de su split. Sin esto,
   comparar seq100 con seq1000 sería comparar conjuntos de test distintos.
3. **Stride:** train 30 y val 30; **test 10, fijo para todas las configuraciones**.
4. **Elegir con validación, reportar en test.** Nunca elijas una configuración mirando el test.
5. **Naive** (último valor observado, repetido) evaluado sobre las mismas ventanas de val y de test.
6. **Nombres únicos** de checkpoints y resultados, p. ej.
   `patchtst_s1000_p64-32_lpsd0.003_lstd1_r2.pt`. Ya pasó una vez que R3 sobrescribió
   los forecasts de R1/R2.
7. Semilla 42 · batch 128 (con 128 PatchTST no tuvo TDR) · Adam lr 1e-3, wd 1e-4 ·
   `MAX_EPOCHS=300` (con stride 30 hay 3× menos pasos por época) · `PATIENCE=15` ·
   `clip_grad_norm_(1.0)`, como ya hace `train_model`.
8. `$env:KMP_DUPLICATE_LIB_OK="TRUE"` siempre.

### Métricas por corrida (en test, Cp desnormalizado)

- RMSE@h y R²@h para h ∈ {1, 10, 50, 100, 500}, más la curva RMSE completa (h=1..500, `.npy`).
- **Ratio de potencia P_pred/P_true**, total AC y por banda (2–10, 10–50, 50–500 Hz):
  periodograma con ventana hann sobre los 500 pasos (fs=1000, sin media, df=2 Hz),
  promediado sobre ventanas. Ojo: `spectral_analysis.py` usa nperseg=256 (df=3.9 Hz);
  aquí se usa la ventana completa.
- std media dentro de ventana, pred frente a true.
- Tiempo, épocas, best val loss y val MSE puro.
- El `series_id` de cada ventana de test, guardado junto a preds/trues (lo necesita la Fase 5).

Todo va en `Agent_Test/results/gpu_plan_2026-10/`: un `metrics.csv` (una fila por
corrida×h, en modo append), más curvas, plots y logs.

---

## 4. Código a implementar

No hay que romper los scripts existentes: todo lo nuevo lleva valores por defecto que
mantienen el comportamiento anterior.

| Archivo | Cambio |
|---|---|
| `Agent_Test/data_loader.py` | Nueva `load_all_buildings_ctx(seq_length, step, horizon, test_step=10, alphas=None, ratios=None, exclude_ratios=None, only_ratios=None, mu_sigma=None)`. Es igual que `load_all_buildings`, pero aplica la regla 2 y devuelve `test_meta` (alpha, ratio y ángulo de cada ventana). Si se pasa `mu_sigma`, reutiliza esa normalización en lugar de ajustarla (Fase 5) |
| `Agent_Test/train_utils.py` | `train_model(..., loss_fn=None)`; por defecto `nn.MSELoss()`, y se usa en train y en val. El early stopping y el checkpoint usan la **misma** pérdida compuesta. Registrar también el val MSE puro |
| `Agent_Test/stat_loss.py` | Nuevo — `StatLoss` (sección 4.1) |
| `Agent_Test/patchtst/train_patchtst_v2.py` | Nuevo, basado en `train_patchtst.py` (mismos límites de recursos). Flags: `--scope {round2,all}`, `--seq`, `--patch-len`, `--patch-stride`, `--step 30`, `--test-step 10`, `--lam-psd`, `--lam-std`, `--max-epochs 300`, `--batch 128`, `--exclude-ratios`, `--tag` |
| `Agent_Test/run_gpu_plan.ps1` (+ `.sh`) | Nuevo. Lanza la Fase 1, elige `SEQ*` automáticamente (menor val MSE, e imprime la decisión) y sigue con la Fase 2. Un log por corrida; se detiene si una falla |
| `Agent_Test/stationarity_blocks.py` | Nuevo — Fase 0 |

### 4.1 `StatLoss` — verificada en CPU antes de escribir este plan

```python
import numpy as np
import torch
import torch.nn.functional as F

OCTAVE_EDGES = [2, 4, 8, 16, 32, 64, 128, 256, 501]   # Hz


class StatLoss(torch.nn.Module):
    """MSE + lam_psd * log-PSD en bandas de octava (a nivel de batch)
             + lam_std * (std_pred - std_true)^2 por ventana.
    Opera en espacio normalizado sobre la salida directa (B, H); fs = 1000 Hz."""

    def __init__(self, H=500, lam_psd=0.0, lam_std=0.0, eps=1e-8, fs=1000.0):
        super().__init__()
        self.lam_psd, self.lam_std, self.eps = lam_psd, lam_std, eps
        self.register_buffer("win", torch.hann_window(H))
        f = np.fft.rfftfreq(H, 1 / fs)[1:]                      # sin DC
        band = torch.as_tensor(np.digitize(f, OCTAVE_EDGES) - 1)
        self.register_buffer("M", F.one_hot(band).float())     # (bins, octavas)

    def _psd(self, x):
        x = x - x.mean(-1, keepdim=True)
        P = torch.fft.rfft(self.win * x).abs().pow(2)[:, 1:]
        return P @ self.M                                       # (B, octavas)

    def forward(self, pred, true):
        loss = F.mse_loss(pred, true)
        if self.lam_psd > 0:
            Pp, Pt = self._psd(pred).mean(0), self._psd(true).mean(0)
            loss = loss + self.lam_psd * (torch.log(Pp + self.eps)
                                          - torch.log(Pt + self.eps)).pow(2).mean()
        if self.lam_std > 0:
            loss = loss + self.lam_std * (pred.std(-1) - true.std(-1)).pow(2).mean()
        return loss
```

Hay que crearla con `StatLoss(...).to(DEVICE)`: los buffers tienen que estar en la GPU.

**Decisiones de diseño, con su evidencia:**

- **PSD a nivel de batch, no por ventana.** El periodograma de una sola ventana de 500
  muestras es muy ruidoso. Promediar sobre el batch compara *distribuciones* espectrales,
  que es lo que significa la restricción. El término de std por ventana impide concentrar
  toda la energía en unas pocas ventanas.
- **Bandas de octava, no bins lineales.** Con bins lineales, el 90% de los bins cae por
  encima de 50 Hz, donde está ~1.5% de la energía, así que la pérdida dedica casi todo su
  peso a esa banda. En el smoke test, con el mismo λ y el mismo coste de MSE (0.148 frente
  a 0.147), la versión lineal gastaba la potencia en >50 Hz (0.89 frente a 0.36), mientras
  que la de octavas sube la banda 2–10 Hz, la que tiene la energía, de 0.16 a **0.26** y la
  potencia total de 0.23 a **0.30**.
- **λ calibrado, no adivinado.** Sobre las predicciones reales de PatchTST R3 (12 800
  ventanas, normalizado), en el estado colapsado: MSE = 0.1368 (cuadra con su best_val de
  0.1406), L_logPSD (octavas) = 44.46 y L_std = 0.0647. Por tanto, **λ_psd\* = 0.0031**
  hace que el término espectral valga lo mismo que el MSE, y λ_std\* ≈ 2.1.

**Smoke test** (CPU, 1 edificio = 21 series, seq=1000 patch 64/32, 3 épocas). Es evidencia
direccional, **no** un resultado:

| pérdida | val MSE | potencia total | 2–10 Hz | 10–50 Hz | >50 Hz | std pred/true |
|---|---|---|---|---|---|---|
| solo MSE | 0.131 | 0.09 | 0.08 | 0.08 | 0.24 | 0.29 |
| lineal, λ_psd=0.003, λ_std=1 | 0.147 | 0.23 | 0.16 | 0.49 | 0.89 | 0.55 |
| **octava, λ_psd=0.003, λ_std=1** | 0.148 | 0.30 | 0.26 | 0.52 | 0.36 | 0.54 |
| octava, λ_psd=0.03, λ_std=1 | 0.158 | 0.59 | 0.53 | 0.86 | 1.25 | 0.61 |

No hubo ningún NaN. Si en GPU aparecen NaN o picos de pérdida: sube `eps` a 1e-6 (y
recalibra λ\*) o baja lr a 5e-4.

---

## 5. Checklist de ejecución (para Claude)

1. `git pull` en la máquina GPU.
2. Activa un entorno con torch + CUDA, sklearn, scipy, pandas y matplotlib, y comprueba
   `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
   (El entorno `ML_Cesar` del README no existe.)
3. Implementa la sección 4.
4. **Prueba de humo en GPU:** 1 época de cada configuración de la Fase 1
   (`--scope round2 --max-epochs 1`), para confirmar memoria, tiempos por época y que
   el CSV se escribe bien.
5. Fase 0 (CPU).
6. `./run_gpu_plan.ps1` → Fases 1 y 2 (~1h40m, una sola orden).
7. Revisa la curva de Pareto, decide el λ y lanza la Fase 4 a mano (y la 3 y la 5 si hay tiempo).
8. Devuelve los resultados (sección 8).

---

## 6. Fases

### Fase 0 — Estacionariedad + estadísticos por serie · CPU, minutos, cualquier máquina

`stationarity_blocks.py`: para las 340 series (windward), calcula media, std, skew,
curtosis y potencia por banda en bloques de 10 s (0–10, 10–20 y 20–30 s).

- **Salida 1:** `stationarity_blocks.csv` y una figura de la variación entre bloques.
  Es una **figura para la tesis** que justifica el diseño (estadísticos por serie, no por bloque).
- **Salida 2:** `series_stats_train.csv`, con los estadísticos de cada serie calculados
  **solo con la porción de train** (los usa la Fase 3).

### Fase 1 — Ventana de entrada · 3 corridas, ~40 min · solo MSE · R2

| corrida | seq | patch_len / stride | tokens | parámetros |
|---|---|---|---|---|
| 1a | 100 | 16 / 8 | 11 | 1.11M |
| 1b | 1000 | 64 / 32 | 30 | 2.35M |
| 1c | 2000 | 128 / 64 | 30 | 2.39M |

- **1a es el baseline reentrenado con el mismo pipeline** (stride 30 y evaluación
  nueva). Si sale ≈0.172, como el publicado con stride 10, el cambio de stride no
  importa y la comparación queda limpia.
- Para seq=2000 **no uses patch 64/32**: da los mismos beneficios con el doble de coste
  (medido: 6.9× frente a 3.3× el baseline por ventana).
- **Decisión:** la seq con menor val MSE → `SEQ*`.
- **Cómo leerlo:** si RMSE@500 baja claramente de 0.172, el suelo era la ventana. Si no
  baja, el suelo es real, y eso también es un resultado que hay que reportar.

### Fase 2 — Restricción estadística · 4 corridas, ~60 min · R2 · con `SEQ*`

| corrida | λ_psd (octava) | λ_std | qué responde |
|---|---|---|---|
| 2a | 0.0003 | 1.0 | restricción débil |
| 2b | 0.003 | 1.0 | equilibrada (= MSE en el estado colapsado) |
| 2c | 0.03 | 1.0 | dominante (en el smoke test ya pasaba de la potencia real en >50 Hz) |
| 2d | 0 | 1.0 | ¿basta con igualar la varianza, sin PSD? |
| *2e (opcional)* | 0.003, bins **lineales** | 1.0 | justifica en la tesis la elección de octavas |

- **El resultado es la curva de Pareto:** RMSE@500 en el eje x y ratio de potencia total
  en el eje y, con el ganador de la Fase 1 y las corridas 2a–2d, más el naive.
- **Expectativa honesta: el RMSE va a subir.** Igualar toda la potencia llevaría el
  RMSE@500 a ~√2 × la parte impredecible: del orden de 0.25 con la ventana de 100 ms, y
  menos si la Fase 1 funciona. El naive está en 0.231 con un ratio de potencia de
  **0**, porque es una constante dentro de la ventana.
- **Regla para elegir el λ que pasa a la Fase 4:** el λ_psd más alto cuyo RMSE@500 en
  **val** no supere al del naive en val. Si prefieres otro criterio (p. ej. ratio de
  potencia ≥ 0.8), cámbialo aquí antes de lanzar.

### Fase 3 (opcional) — Bias: condicionar en los estadísticos de la serie · 1 corrida, ~30 min

Añadir un token extra a PatchTST: `Linear(n_stats → d_model)` con los estadísticos de
`series_stats_train.csv` (media, std, skew, curtosis y potencia por banda, normalizados),
antepuesto a los tokens de patch. La cabeza pasa a ser `(n_patches + 1) · d_model`. El
bucle de entrenamiento tiene que admitir batches `(x, stats, y)`. Se corre con `SEQ*` y
el λ elegido.
**Límite que hay que declarar:** vale en el escenario de diseño (ya conoces los
estadísticos de esa configuración por el túnel), pero no para un edificio nuevo.

### Fase 4 — Confirmación en R3 · 1 corrida, ~60–80 min

La configuración ganadora con `--scope all`.

- **Memoria:** `X_train` en R3 ocupa 3.9 GB con seq1000 y 7.4 GB con seq2000. Con 16 GB
  de RAM y seq2000, usa `--step 50`.
- Evalúa el test **serie a serie** (en streaming, como `naive_dense_baseline.py`), sin
  materializar las ~150 000 ventanas: con seq2000 serían 4.8 GB.

### Fase 5 (opcional) — Leave-buildings-out · 1 corrida, ~50–65 min

La misma configuración que la Fase 4, pero **excluyendo del train las geometrías `1_1_4`
y `3_1_3` en los dos alphas**: 64 de 340 series (~19%). μ/σ se ajustan sin ellas, y su
test se construye después con `only_ratios` + `mu_sigma`. Se evalúa sobre el test de esas
64 series y se compara con la Fase 4 **sobre las mismas 64 series** (para eso se guarda
el `series_id`).
Responde a "¿generaliza a una geometría que no ha visto?", que es la pregunta que R2/R3
dicen responder y que el split actual no prueba.

---

## 7. Tiempos (A4000)

Base: PatchTST seq100 en R2 = 18.2 min medidos. Coste por ventana medido en CPU:
seq1000 con patch 64/32 = 3.1×, seq2000 con patch 128/64 = 3.3×; la pérdida estadística
añade un 2–9%. Con stride 30 hay 3× menos ventanas, así que una corrida cuesta más o
menos lo mismo que el baseline. Además, en GPU el baseline está limitado por overhead
(~6.5 ms/batch), así que puede ir incluso más rápido. **Rango por corrida en R2: 10–30 min.**

| Fase | Corridas | GPU |
|---|---|---|
| 0 | — | minutos (CPU) |
| 1 | 3 | ~40 min |
| 2 | 4 (+1 opcional) | ~60 min |
| 3 (opcional) | 1 | ~30 min |
| 4 | 1 | ~60–80 min |
| 5 (opcional) | 1 | ~50–65 min |
| **Total** | **8–11** | **~3–4.5 h** |

---

## 8. Al terminar — qué devolver

1. **Commit + push** del código nuevo y de `Agent_Test/results/gpu_plan_2026-10/`
   (CSV, curvas, plots y logs).
2. **No subas** checkpoints ni `*_preds.npy`: están en el gitignore y pesan mucho. Si se
   necesitan en la otra máquina, súbelos a Drive, como el bundle de junio.
3. Escribe `Handoffs/AAAA-MM-DD_resultados_GPU.md` con: la tabla de resultados (Fases 1–2
   + naive), la curva de Pareto, `SEQ*` y el λ elegidos con su porqué, los tiempos reales
   y cualquier anomalía. Añádelo al índice de `Handoffs/README.md` y marca este plan como
   completado.

## 9. Fuera de alcance en la máquina GPU

Las correcciones de texto de la tesis (diagnóstico de R1, "u, v, w", MAPE, la fila de RF,
matizar el take-away #5) y la evaluación de los modelos directos sobre las 680
trayectorias se hacen en la máquina sin GPU. No las toques aquí.
