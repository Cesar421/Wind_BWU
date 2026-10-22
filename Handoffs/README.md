# Handoffs

Instrucciones para pasar trabajo de una máquina a otra. La conversación con Claude no
se sincroniza entre máquinas; git sí. Cada handoff es un archivo autocontenido: el
contexto, qué hacer, cómo lanzarlo y qué devolver.

**Convención:** `AAAA-MM-DD_<tema>.md`. El más reciente que esté pendiente es el vigente.
Cuando uno se completa, se marca aquí y su resultado va en un handoff nuevo
(`AAAA-MM-DD_resultados_<tema>.md`).

**Para empezar en otra máquina:** `git pull`, abre el handoff marcado como **Pendiente**
y sigue su sección «Arranque».

| Fecha | Archivo | Estado |
|---|---|---|
| 2026-10-22 | [Resultados del plan GPU (ventana + restricción estadística)](2026-10-22_resultados_GPU.md) | **Vigente** — leer este |
| 2026-10-21 | [Plan GPU: ventana de entrada + restricción estadística (PatchTST)](2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md) | Completado (resultados en el de 2026-10-22) |
| 2026-06-25 | [`../HANDOFF.md`](../HANDOFF.md) — reentreno de modelos directos + RF | Reemplazado (RF hecho; el reentreno ya no hace falta) |
