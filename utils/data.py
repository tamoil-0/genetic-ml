from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(test_size: float = 0.2, random_state: int = 42, scale: bool = True):
    """
    Carga breast_cancer de sklearn, hace split y (opcional) escalado estándar.
    Retorna: X_train, X_test, y_train, y_test, feature_names (np.ndarray[str])
    """
    ds = load_breast_cancer()
    X = ds.data.astype(float)
    y = ds.target.astype(int)
    feature_names = np.array(ds.feature_names, dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names
