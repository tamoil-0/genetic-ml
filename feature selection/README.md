# genetic-ml

Bloque 1 — **Feature Selection con Algoritmos Genéticos (GA)** + **Setup del repo**.

## Estructura
```
genetic-ml/
├─ 01_feature_selection_GA.ipynb
├─ 02_hyperparam_optimization_GA.ipynb
├─ 03_neuroevolution_GA.ipynb
├─ utils/
│  ├─ data.py
│  └─ ga_tools.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Entorno rápido
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook  # o jupyter lab
```

## Notebook principal (Bloque 1)
Abrir `01_feature_selection_GA.ipynb`, ejecutar todas las celdas.
- Fitness = `accuracy_CV(LogReg) - alpha * k`
- Población=50, Generaciones=50, Torneo, Cruce 1 punto, Mutación flip (p=0.05), Elitismo=1
- Salidas: subset de features, accuracy baseline vs seleccionado, curva mejor fitness/generación
- Guardado opcional en `results/bloque1_selection.joblib`
