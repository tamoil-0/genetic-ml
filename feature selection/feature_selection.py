# feature_selection.py
# USO (DESDE LA RAÍZ DEL PROYECTO):
#   py feature_selection.py --csv data\productos_puno.csv --target producto
#   py feature_selection.py --csv data\productos_puno.csv --target indice --alpha 0.003
# DEPENDENCIAS: numpy, pandas, scikit-learn, matplotlib, deap, joblib

import os, argparse, random, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge

from deap import base, creator, tools

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# UTILIDADES (EN ESPAÑOL)
# ==============================

def fijar_semilla(semilla: int = 42):
    """
    FIJO LA SEMILLA PARA QUE LOS RESULTADOS SEAN REPRODUCIBLES.
    """
    random.seed(semilla)
    np.random.seed(semilla)

def penalizacion_k(k: int, alpha: float = 0.005) -> float:
    """
    AQUI APLICO LA PENALIZACIÓN POR COMPLEJIDAD: ALPHA * NUMERO_DE_CARACTERISTICAS.
    LO RESTO AL SCORE PARA PREMIAR SUBSETS MÁS PEQUEÑOS.
    """
    return alpha * float(k)

def cargar_csv_personalizado(ruta_csv: str, objetivo: str, test_size: float = 0.2, random_state: int = 42):
    """
    CARGO MI CSV Y LO PREPARO:
    - INTENTO LEER EN UTF-8 Y SI VEO CARACTERES 'Ã' REINTENTO EN LATIN-1.
    - NORMALIZO NOMBRES DE COLUMNAS (SOPORTA VARIANTES CON ACENTOS RAROS).
    - HAGO ONE-HOT A LAS CATEGÓRICAS.
    - SI ES CLASIFICACIÓN, LABEL-ENCODER AL TARGET.
    - DEVUELVO: X_ENTRENO, X_TEST, y_ENTRENO, y_TEST, NOMBRES_DE_CARACTERISTICAS, TIPO_DE_PROBLEMA.
    """
    # Intento leer con tolerancia a encoding
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8")
        if any("Ã" in c for c in df.columns.astype(str)):
            df = pd.read_csv(ruta_csv, encoding="latin1")
    except Exception:
        df = pd.read_csv(ruta_csv, encoding="latin1")

    # Normalizo nombres de columnas
    mapa_cols = {
        "Ciudad": "ciudad",
        "Provincia": "provincia",
        "Producto_Regional": "producto_regional",
        "Producción_Toneladas": "produccion_toneladas",
        "ProducciÃ³n_Toneladas": "produccion_toneladas",
        "Índice_Industrialización": "indice_industrializacion",
        "Ã\x8dndice_IndustrializaciÃ³n": "indice_industrializacion",
        "Ãndice_IndustrializaciÃ³n": "indice_industrializacion",
    }
    df = df.rename(columns={c: mapa_cols.get(c, c) for c in df.columns})

    esperadas = {"ciudad", "provincia", "producto_regional", "produccion_toneladas", "indice_industrializacion"}
    faltan = [c for c in esperadas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}\nDisponibles: {list(df.columns)}")

    # Limpieza y tipos
    df = df.dropna(subset=list(esperadas))
    df["produccion_toneladas"] = pd.to_numeric(df["produccion_toneladas"], errors="coerce")
    df["indice_industrializacion"] = pd.to_numeric(df["indice_industrializacion"], errors="coerce")
    df = df.dropna(subset=["produccion_toneladas", "indice_industrializacion"])

    # Defino el problema
    if objetivo == "producto":
        tipo_problema = "classification"
        y = df["producto_regional"].astype(str)
        X = df[["ciudad", "provincia", "produccion_toneladas", "indice_industrializacion"]].copy()
        cols_cat = ["ciudad", "provincia"]
    elif objetivo == "indice":
        tipo_problema = "regression"
        y = df["indice_industrializacion"].astype(float)
        X = df[["ciudad", "provincia", "producto_regional", "produccion_toneladas"]].copy()
        cols_cat = ["ciudad", "provincia", "producto_regional"]
    else:
        raise ValueError("objetivo debe ser 'producto' (clasificación) o 'indice' (regresión)")

    # One-hot a categóricas
    X = pd.get_dummies(X, columns=cols_cat, drop_first=False, dtype=float)

    # LabelEncoder si es clasificación
    if tipo_problema == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split (OJO: estratifico si es clasificación)
    X_entreno, X_test, y_entreno, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state,
        stratify=y if tipo_problema == "classification" else None
    )
    nombres_caracteristicas = np.array(X.columns, dtype=str)
    return X_entreno, X_test, y_entreno, y_test, nombres_caracteristicas, tipo_problema

def construir_estimador(tipo_problema: str):
    """
    ARMO EL PIPELINE PARA EVALUAR EN CV:
    - METO EL STANDARD SCALER DENTRO PARA EVITAR FUGA DE DATOS.
    - CLASIFICACIÓN: LOGISTIC REGRESSION.
    - REGRESIÓN: RIDGE.
    """
    if tipo_problema == "classification":
        return Pipeline([
            ("escalador", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
        ])
    else:
        return Pipeline([
            ("escalador", StandardScaler()),
            ("reg", Ridge(alpha=1.0))
        ])

# ============================================
# GA CON DEAP: SELECCIÓN DE CARACTERÍSTICAS
# ============================================

def ejecutar_ga_seleccion_caracteristicas(
    X_entreno, y_entreno, nombres_caracteristicas, tipo_problema,
    tam_poblacion=50, n_generaciones=50, prob_cruce=0.8, prob_mutacion=0.2,
    indpb=None, tam_torneo=3, alpha=0.005, elitismo=1, semilla=42
):
    """
    AQUI PROGRAME MI ALGORITMO GENÉTICO PARA ELEGIR LAS MEJORES COLUMNAS.
    - CROMOSOMA: VECTOR BINARIO DEL TAMAÑO DE LAS FEATURES.
    - FITNESS: SCORE_CV DEL ESTIMADOR (ACCURACY O -RMSE) - ALPHA * K.
    - OPERADORES: TORNEO, CRUCE 1 PUNTO, MUTACIÓN FLIP, ELITISMO.
    - DEVUELVO LA MÁSCARA, K, MEJOR FITNESS, LOG DE EVOLUCIÓN Y EL INDPB USADO.
    """
    fijar_semilla(semilla)
    n_caract = X_entreno.shape[1]
    if indpb is None:
        indpb = max(1 / n_caract, 0.01)  # ~1 bit mutado por individuo

    # Definiciones DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    caja = base.Toolbox()
    caja.register("gen_binario", random.randint, 0, 1)
    caja.register("individuo", tools.initRepeat, creator.Individual, caja.gen_binario, n=n_caract)
    caja.register("poblacion", tools.initRepeat, list, caja.individuo)

    def asegurar_minimo_1(ind):
        """ME ASEGURO DE QUE NINGÚN INDIVIDUO QUEDE TODO EN 0."""
        if sum(ind) == 0:
            ind[random.randrange(len(ind))] = 1

    def evaluar_individuo(ind):
        """CALCULO EL FITNESS DEL INDIVIDUO (LO EXPLICO COMO ESTUDIANTE)."""
        asegurar_minimo_1(ind)
        mascara = np.array(ind, dtype=bool)
        k = int(mascara.sum())
        X_sel = X_entreno[:, mascara]
        estimador = construir_estimador(tipo_problema)

        if tipo_problema == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
            score = cross_val_score(estimador, X_sel, y_entreno, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=semilla)
            score = cross_val_score(estimador, X_sel, y_entreno, cv=cv,
                                    scoring="neg_root_mean_squared_error", n_jobs=-1).mean()

        return (float(score - penalizacion_k(k, alpha)),)

    # Registro de operadores
    caja.register("evaluate", evaluar_individuo)
    caja.register("mate", tools.cxOnePoint)                       # cruce 1 punto
    caja.register("mutate", tools.mutFlipBit, indpb=indpb)        # mutación flip
    caja.register("select", tools.selTournament, tournsize=tam_torneo)

    # Población inicial
    poblacion = caja.poblacion(n=tam_poblacion)
    for ind in poblacion:
        ind.fitness.values = caja.evaluate(ind)

    # LOG GEN 0 (ME GUSTA VER QUE NO ESTÁ COLGADO)
    mejor_ahora = tools.selBest(poblacion, 1)[0]
    print(f"[GA] Gen 0/{n_generaciones} | best={mejor_ahora.fitness.values[0]:.6f} | k={int(np.sum(mejor_ahora))}", flush=True)

    # Evolución
    registro = []
    for gen in range(n_generaciones):
        elites = tools.selBest(poblacion, elitismo) if elitismo > 0 else []
        hijos = list(map(caja.clone, caja.select(poblacion, len(poblacion) - elitismo)))

        # Cruce
        for i in range(0, len(hijos), 2):
            if i + 1 < len(hijos) and random.random() < prob_cruce:
                caja.mate(hijos[i], hijos[i + 1])
                if hasattr(hijos[i].fitness, "values"): del hijos[i].fitness.values
                if hasattr(hijos[i + 1].fitness, "values"): del hijos[i + 1].fitness.values

        # Mutación
        for i in range(len(hijos)):
            if random.random() < prob_mutacion:
                caja.mutate(hijos[i])
                if hasattr(hijos[i].fitness, "values"): del hijos[i].fitness.values

        # Re-evaluo hijos inválidos
        invalidos = [ind for ind in hijos if not ind.fitness.valid]
        for ind in invalidos:
            ind.fitness.values = caja.evaluate(ind)

        # Nueva población
        poblacion = elites + hijos

        # Estadísticas
        fits = np.array([ind.fitness.values[0] for ind in poblacion], dtype=float)
        registro.append({"gen": gen + 1, "avg": float(fits.mean()), "min": float(fits.min()), "max": float(fits.max())})

        mejor_ahora = tools.selBest(poblacion, 1)[0]
        print(f"[GA] Gen {gen+1}/{n_generaciones} | best={mejor_ahora.fitness.values[0]:.6f} | k={int(np.sum(mejor_ahora))}", flush=True)

    # Mejor individuo final
    mejor = tools.selBest(poblacion, 1)[0]
    mascara_mejor = np.array(mejor, dtype=bool)
    k_mejor = int(mascara_mejor.sum())
    fitness_mejor = float(mejor.fitness.values[0])

    return mascara_mejor, k_mejor, fitness_mejor, registro, indpb

# ==============================
# PROGRAMA PRINCIPAL (CLI)
# ==============================

def main():
    """
    AQUI PARSEO LOS ARGUMENTOS, CARGO DATOS, CORRO EL GA, COMPARO VS BASELINE,
    GRAFICO LA CURVA Y GUARDO LOS ARTEFACTOS PARA EL SIGUIENTE BLOQUE.
    """
    parser = argparse.ArgumentParser(description="GA Feature Selection (DEAP) con CSV propio (variables y comentarios en español)")
    parser.add_argument("--csv", required=True, help="Ruta al CSV (ej: data/productos_puno.csv)")
    parser.add_argument("--target", choices=["producto", "indice"], default="producto",
                        help="producto=clasificación (Producto_Regional) | indice=regresión (Índice_Industrialización)")
    parser.add_argument("--alpha", type=float, default=0.005, help="Penalización por nº de features (α)")
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--n-gen", type=int, default=50)
    parser.add_argument("--cx-pb", type=float, default=0.8)
    parser.add_argument("--mut-pb", type=float, default=0.2)
    parser.add_argument("--indpb", type=float, default=None, help="Probabilidad de flip por gen (default ~1/n_features)")
    parser.add_argument("--tour-size", type=int, default=3)
    parser.add_argument("--elitism", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fijar_semilla(args.seed)

    # CARGO DATOS
    X_entreno, X_test, y_entreno, y_test, nombres_caracteristicas, tipo_problema = cargar_csv_personalizado(
        args.csv, objetivo=args.target, test_size=0.2, random_state=args.seed
    )
    print(f"[DATA] {args.csv} | problema={tipo_problema} | X_entreno={X_entreno.shape} | y={len(y_entreno)}", flush=True)

    # CORRO EL GA
    mascara_mejor, k_mejor, fitness_mejor, registro, indpb = ejecutar_ga_seleccion_caracteristicas(
        X_entreno, y_entreno, nombres_caracteristicas, tipo_problema,
        tam_poblacion=args.pop_size, n_generaciones=args.n_gen, prob_cruce=args.cx_pb, prob_mutacion=args.mut_pb,
        indpb=args.indpb, tam_torneo=args.tour_size, alpha=args.alpha, elitismo=args.elitism, semilla=args.seed
    )

    # BASELINE VS SELECCIONADAS (CV, PARA MOSTRAR EVIDENCIA)
    est_todas = construir_estimador(tipo_problema)
    if tipo_problema == "classification":
        cv_all = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        baseline_cv = cross_val_score(est_todas, X_entreno, y_entreno, cv=cv_all, scoring="accuracy", n_jobs=-1).mean()
        est_sel = construir_estimador(tipo_problema)
        selected_cv = cross_val_score(est_sel, X_entreno[:, mascara_mejor], y_entreno, cv=cv_all, scoring="accuracy", n_jobs=-1).mean()
    else:
        cv_all = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        baseline_cv = cross_val_score(est_todas, X_entreno, y_entreno, cv=cv_all,
                                      scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
        est_sel = construir_estimador(tipo_problema)
        selected_cv = cross_val_score(est_sel, X_entreno[:, mascara_mejor], y_entreno, cv=cv_all,
                                      scoring="neg_root_mean_squared_error", n_jobs=-1).mean()

    caracteristicas_seleccionadas = list(nombres_caracteristicas[mascara_mejor])

    # --------- SALIDAS EN CONSOLA (EN ESPAÑOL) ---------
    print("\n=== SELECCIÓN DE CARACTERÍSTICAS CON AG (DEAP) ===")
    print(f"CSV: {args.csv}")
    print(f"Problema: {tipo_problema}")
    print(f"Población={args.pop_size}  Generaciones={args.n_gen}  CX_PB={args.cx_pb}  MUT_PB={args.mut_pb}  INDPB={indpb:.4f}")
    print(f"Torneo={args.tour_size}  Elitismo={args.elitism}  Alpha={args.alpha}")
    print(f"\nMejor k (seleccionadas): {k_mejor}")
    print(f"Mejor fitness: {fitness_mejor:.6f}")

    if tipo_problema == "classification":
        print(f"\nAccuracy CV (todas):        {baseline_cv:.6f}")
        print(f"Accuracy CV (seleccionadas): {selected_cv:.6f}")
    else:
        print("\n- RMSE (negativo, más cerca de 0 es mejor)")
        print(f"-RMSE CV (todas):           {baseline_cv:.6f}")
        print(f"-RMSE CV (seleccionadas):   {selected_cv:.6f}")

    print("\nPrimeras 20 características elegidas:")
    for f in caracteristicas_seleccionadas[:20]:
        print(" -", f)

    # --------- ASERCIONES (PARA DEMOSTRAR QUE “FUNCIONA”) ---------
    assert k_mejor >= 1, "La máscara no puede estar vacía."
    # Permitimos hasta -0.01 por ruido de CV
    assert selected_cv >= baseline_cv - 0.01, "El subset no debería degradar mucho la métrica."
    print("\n✅ Pruebas básicas OK")

    # --------- GUARDADOS ---------
    os.makedirs("results", exist_ok=True)

    # Curva del mejor fitness
    generaciones = [r["gen"] for r in registro]
    curva_mejor = [r["max"] for r in registro]
    plt.figure()
    plt.plot(generaciones, curva_mejor, marker="o")
    plt.xlabel("Generación"); plt.ylabel("Mejor fitness")
    plt.title("Evolución del mejor fitness"); plt.grid(True)
    ruta_curva = "results/fitness_curve.png"
    plt.savefig(ruta_curva, dpi=150, bbox_inches="tight")

    # Artefacto .joblib
    salida = {
        "csv": args.csv,
        "tipo_problema": tipo_problema,
        "mascara_mejor": mascara_mejor,
        "caracteristicas_seleccionadas": caracteristicas_seleccionadas,
        "fitness_mejor": float(fitness_mejor),
        "baseline_cv": float(baseline_cv),
        "selected_cv": float(selected_cv),
        "log": registro,
        "ga_params": {
            "tam_poblacion": args.pop_size, "n_generaciones": args.n_gen, "prob_cruce": args.cx_pb,
            "prob_mutacion": args.mut_pb, "indpb": indpb, "tam_torneo": args.tour_size,
            "alpha": args.alpha, "elitismo": args.elitism, "semilla": args.seed
        }
    }
    joblib.dump(salida, "results/bloque1_selection.joblib")

    # CSV con lista de features seleccionadas
    pd.Series(caracteristicas_seleccionadas, name="caracteristica").to_csv(
        "results/selected_features.csv", index=False, encoding="utf-8"
    )

    print(f"\n💾 Guardado: results/bloque1_selection.joblib")
    print(f"💾 Guardado: {ruta_curva}")
    print(f"💾 Guardado: results/selected_features.csv")

if __name__ == "__main__":
    main()
