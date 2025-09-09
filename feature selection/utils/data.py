import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_custom_csv(
    csv_path: str,
    target: str = "producto",            # "producto" (clasificación) o "indice" (regresión)
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True
):
    """
    Carga tu CSV y lo deja listo:
    - Arregla posibles issues de encoding (Ã, etc.)
    - One-hot a columnas categóricas
    - Escalado opcional
    Retorna: X_train, X_test, y_train, y_test, feature_names (np.ndarray[str]), problem_type ("classification"/"regression")
    """
    # 1) Intentar leer con utf-8 y, si hay mojibake (Ã), reintentar con latin-1
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        if df.columns.astype(str).str.contains("Ã").any():
            df = pd.read_csv(csv_path, encoding="latin1")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin1")

    # 2) Normalizar nombre de columnas (aceptamos originales y con mojibake)
    colmap = {
        "Ciudad": "ciudad",
        "Provincia": "provincia",
        "Producto_Regional": "producto_regional",
        "Producción_Toneladas": "produccion_toneladas",
        "ProducciÃ³n_Toneladas": "produccion_toneladas",
        "Índice_Industrialización": "indice_industrializacion",
        "Ã\x8dndice_IndustrializaciÃ³n": "indice_industrializacion",
        "Ãndice_IndustrializaciÃ³n": "indice_industrializacion",
    }
    df = df.rename(columns={c: colmap.get(c, c) for c in df.columns})

    # 3) Chequear que están las columnas clave
    esperadas = {"ciudad","provincia","producto_regional","produccion_toneladas","indice_industrializacion"}
    faltan = [c for c in esperadas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}. Columnas disponibles: {list(df.columns)}")

    # 4) Limpieza mínima
    df = df.dropna(subset=["ciudad","provincia","producto_regional","produccion_toneladas","indice_industrializacion"])
    # Asegurar tipos
    df["produccion_toneladas"] = pd.to_numeric(df["produccion_toneladas"], errors="coerce")
    df["indice_industrializacion"] = pd.to_numeric(df["indice_industrializacion"], errors="coerce")
    df = df.dropna(subset=["produccion_toneladas","indice_industrializacion"])

    # 5) Definir target y features según el problema
    problem_type = "classification" if target == "producto" else "regression"

    if target == "producto":
        # Clasificación: predecir Producto_Regional
        y = df["producto_regional"].astype(str)
        # Features: ciudad, provincia, produccion_toneladas, indice_industrializacion
        X = df[["ciudad","provincia","produccion_toneladas","indice_industrializacion"]].copy()
        cat_cols = ["ciudad","provincia"]
    else:
        # Regresión: predecir Índice_Industrialización
        y = df["indice_industrializacion"].astype(float)
        # Features: ciudad, provincia, producto_regional, produccion_toneladas
        X = df[["ciudad","provincia","producto_regional","produccion_toneladas"]].copy()
        cat_cols = ["ciudad","provincia","producto_regional"]

    # 6) One-hot a las categóricas
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)

    # 7) LabelEncode si es clasificación
    if problem_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 8) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y if problem_type=="classification" else None
    )
    feature_names = np.array(X.columns, dtype=str)

    # 9) Escalado (solo si quieres; funciona bien con LR y regresión)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, problem_type
