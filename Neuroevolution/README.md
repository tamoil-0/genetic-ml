# Bloque 3 — Neuroevolución con Algoritmo Genético (Keras/TensorFlow) — Colab Ready

Este notebook ejecuta una **búsqueda evolutiva** de arquitecturas MLP simples usando un **Algoritmo Genético (GA)** sobre un dataset sintético (clasificación). 
La clase `GeneticNeuroevolution` entrena modelos Keras para evaluar **fitness** y aplica selección, cruzamiento, mutación y **elitismo**.

## Representación (cromosoma)
`[n_layers, units_layer1, units_layer2, dropout_rate, learning_rate_log10]`
- `n_layers` ∈ {1, 2}
- `units` por capa ∈ {32, 64, 128}
- `dropout_rate` ∈ [0.0, 0.5]
- `learning_rate` = 10^`learning_rate_log10` con rango [1e-4, 1e-2]

## Fitness
- **val_accuracy** máxima durante 5 épocas de validación (`validation_split=0.2`)
- Penalización por complejidad: `0.001 * (suma de unidades) / 100`

## GA
- Población: 8 (demo rápida en Colab) — puedes subirla a 50
- Generaciones: 4 (demo) — puedes subir a 30
- Selección: ruleta (probabilidades ∝ fitness, con clipping a ≥0)
- Cruce: 1 punto + armonización de longitud según `n_layers`
- Mutación: cambios en nº de capas, unidades, dropout y learning rate
- Elitismo: 20%

## Salidas
- **Console log** por generación (mejor y promedio)
- **Gráfico** `evolution_progress.png` con Mejor vs Promedio
- **Arquitectura ganadora** (capas, unidades, dropout, lr)

## Cómo usar en Colab
1. Abre `03_neuroevolución_GA_colab.ipynb` en Google Colab.
2. Ejecuta la celda de **instalación** (opcional si Colab ya trae TF).
3. Ejecuta la celda de **código** para definir la clase.
4. Ejecuta la celda **RUN** (ajusta `n_population`/`n_generations` si quieres).

## Requerimientos mínimos
Se adjunta `requirements.txt` para ejecución local. En Colab ya están preinstalados la mayoría.
