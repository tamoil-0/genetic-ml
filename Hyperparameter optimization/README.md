# Bloque 2 — Hyperparameter Optimization con Algoritmo Genético (RandomForest)

Este notebook optimiza los **hiperparámetros** de `RandomForestClassifier` usando un **Algoritmo Genético (GA)** con validación cruzada.

## Dataset de ejemplo
Incluye el CSV: `Accidentes de tránsito en carreteras-2020-2021-Sutran.csv` (formato original). Puedes sustituir por tu propio dataset (clasificación).

## Espacio de búsqueda (genes)
- `n_estimators` ∈ {50..300 paso 10}
- `max_depth` ∈ {None, 3..15}
- `min_samples_split` ∈ {2..10}
- `min_samples_leaf` ∈ {1..5}
- `max_features` ∈ {'sqrt','log2', 0.3..1.0 (paso 0.1)}

> El cromosoma codifica **índices** a listas de valores válidos. El notebook documenta el mapeo índice ↔ valor.

## Flujo del GA
1. **Inicialización:** población aleatoria (semilla fija 42).
2. **Evaluación (fitness):** `accuracy` con `cross_val_score` (cv=5).
3. **Selección:** torneo k=3.
4. **Cruce:** uniforme (prob. 0.5 por gen).
5. **Mutación:** cambiar 1–2 genes por valores válidos.
6. **Elitismo:** conservar los 2 mejores por generación.
7. **Terminación:** 35 generaciones con población de 50.

## Salidas esperadas
- **Mejor conjunto de hiperparámetros** (decodificados).
- **Accuracy CV** del mejor individuo.
- **Accuracy en TEST** (una vez reentrenado con train).
- **Curva** del mejor fitness por generación.

## Requisitos
Ver `requirements_hpo.txt` en la raíz.
