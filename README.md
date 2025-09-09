# genetic-ml — Proyecto completo (FS, HPO, Neuroevolución)

**Objetivo general:** Comprender y demostrar la aplicación de **Algoritmos Genéticos (AG)** en tres tareas de *Machine Learning*:
1) **Feature Selection (FS)** — selección de subconjunto de características.
2) **Hyperparameter Optimization (HPO)** — búsqueda de hiperparámetros para RandomForest.
3) **Neuroevolución (NE)** — búsqueda de arquitectura de una MLP simple (Keras/TensorFlow).

Estructura actual del repositorio (según tus carpetas):
```
genetic-ml/
├─ feature selection/
│  ├─ 01_feature_selection_GA.ipynb
│  ├─ feature_selection.py
│  ├─ utils/ (data.py, ga_tools.py, ...)
│  └─ results/ (artefactos del GA: curva, máscara, joblib, etc.)
├─ Hyperparameter optimization/
│  ├─ Hyperparameter_Optimization.ipynb
│  └─ (datasets de prueba, p.ej. Accidentes Sutran *.csv)
└─ Neuroevolution/
   ├─ neuroevolution.py
   ├─ evolution_progress.png
   └─ 03_neuroevolution_GA_colab.ipynb (opcional, formato Colab)
```

---

## Requerimientos
Instalación local (Python 3.10+ recomendado):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```
**`requirements.txt`** incluido en este bundle contiene:
- `numpy, pandas, scikit-learn, matplotlib, joblib, jupyter`
- `deap` (para FS basada en DEAP)
- `tensorflow>=2.12` (para neuroevolución con Keras)

> Si usarás solo Colab: puedes omitir instalar localmente. Colab ya incluye la mayoría; si falta algo, ejecuta la celda de instalación del notebook.

---

## 1) Feature Selection (FS)
**Ubicación:** `feature selection/01_feature_selection_GA.ipynb` o `feature selection/feature_selection.py`

- **Representación:** cromosoma binario (longitud = nº de *features*).
- **Fitness:** `accuracy_cv(LogReg) − α · k`, con `k` = nº de *features* usadas.
- **Operadores GA:** torneo, cruce 1 punto, mutación flip bit, **elitismo**.
- **Salidas:** 
  - `results/fitness_curve.png`
  - `results/selected_features.csv` (nombres de *features*)
  - `results/bloque1_selection.joblib` (máscara booleana + curva + métricas)

**Ejecución rápida (script):**
```bash
python feature selection/feature_selection.py --csv data/tu_dataset.csv --target <columna_objetivo> --alpha 0.005 --pop-size 50 --n-gen 50
```
> En el notebook, usa **Restart & Run All** y ajusta hiperparámetros en celdas.

---

## 2) Hyperparameter Optimization (HPO)
**Ubicación:** `Hyperparameter optimization/Hyperparameter_Optimization.ipynb`

- **Modelo:** `RandomForestClassifier`
- **Genes/dominios:**  
  `n_estimators` ∈ {50..300 paso 10}, `max_depth` ∈ {None, 3..15},  
  `min_samples_split` ∈ {2..10}, `min_samples_leaf` ∈ {1..5},  
  `max_features` ∈ {'sqrt','log2', 0.3..1.0}  
  (*el cromosoma codifica índices → valores*).
- **Fitness:** `accuracy` con `cross_val_score(cv=5)` (semilla 42).  
- **GA:** torneo k=3, cruce uniforme, mutación cambiando 1–2 genes, **elitismo** (2).
- **Salidas:** mejores hiperparámetros (decodificados), *accuracy* CV y *accuracy* en **test**, curva del mejor fitness.

> Si usas las *features* de FS, carga `results/bloque1_selection.joblib` y filtra `X_train/X_test` con la máscara booleana.

---

## 3) Neuroevolución (NE)
**Ubicación:** `Neuroevolution/neuroevolution.py` y/o `Neuroevolution/03_neuroevolution_GA_colab.ipynb`

- **Cromosoma:** `[n_layers, units_layer1, units_layer2, dropout_rate, learning_rate_log10]`  
  `n_layers` ∈ {1,2}, `units` ∈ {32, 64, 128}, `dropout` ∈ [0, 0.5], `lr` = 10^log10 ∈ [1e-4, 1e-2].
- **Fitness:** mejor `val_accuracy` en entrenamiento corto (p.ej., 5 épocas) − pequeña penalización por complejidad (suma de unidades).
- **GA:** selección por ruleta (con clipping), cruce 1 punto (ajustando tamaño según `n_layers`), mutación de 1–2 genes, **elitismo** (~20%).
- **Salidas:** arquitectura ganadora y `evolution_progress.png` con mejor vs promedio por generación.

**Colab ready:** abre `03_neuroevolution_GA_colab.ipynb`, ejecuta instalación → clase → RUN. Ajusta `n_population` y `n_generations` según el tiempo disponible.

---

## Cómo reproducir (orden sugerido)
1. Corre **FS** para obtener `results/bloque1_selection.joblib` (máscara de *features*).
2. Corre **HPO** y, si deseas, aplica el subset de FS antes de entrenar el RF.
3. Corre **NE** para la búsqueda de arquitectura MLP.
4. Registra las métricas finales y exporta imágenes de curvas.

---

## Tabla de resultados (rellenar al final)
| Bloque | Dataset | Mejor individuo | Métrica CV | Accuracy Test | Artefactos |
|---|---|---|---|---|---|
| FS | _(nombre)_ | _(nº features / nombres)_ | _(acc CV)_ | _(acc test vs baseline)_ | `selected_features.csv`, `fitness_curve.png` |
| HPO | _(nombre)_ | _(hiperparámetros RF)_ | _(acc CV)_ | _(acc test)_ | `best_curve.png` |
| NE | _(nombre)_ | _(capas/unidades/dropout/lr)_ | _(acc CV/val)_ | _(acc test)_ | `evolution_progress.png` |

---

## Tips de evaluación (alineado a la rúbrica)
- Cada ejemplo debe **mostrar claramente** el ciclo AG: representación, inicialización, fitness, selección, cruce, mutación y terminación.
- Explicar **por qué** se eligió cada codificación y operador (p. ej., binario para FS; mixto/índices para HPO; entero/real para NE).
- Mostrar **curvas** por generación y el **mejor individuo**.
- Verificar que todo **corre de principio a fin** (Restart & Run All).
- Reproducibilidad: fija semilla (42), deja `requirements.txt` y guarda artefactos.

---

## Troubleshooting rápido
- Si `tensorflow` falla localmente, usa el notebook Colab para NE.
- Si el HPO es lento, baja población/generaciones o usa menos CV.
- Si FS selecciona 0 *features*, fuerza al menos 1 (ya está manejado en el código).

---

## Licencia y créditos
Código educativo para demostración de AG en ML. Créditos del equipo en el README del repo principal.

