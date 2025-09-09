import random
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def cv_score(estimator, X, y, cv: int = 5, scoring: str = "accuracy", random_state: int = 42) -> float:
    """
    Retorna el promedio de CV estratificado con shuffle.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring=scoring, n_jobs=None)
    return float(np.mean(scores))

def penalty_k(k: int, alpha: float = 0.005) -> float:
    return alpha * float(k)
