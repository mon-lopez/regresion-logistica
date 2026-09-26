import numpy as np
from .modelo import sigmoid

def generar_datos(n, seed=2026):
    """
    Genera datos según el modelo P(Y=1|X=x) = sigma(-1 + 2x).
    X ~ U[-2, 2]
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    p = sigmoid(-1 + 2 * X)
    Y = rng.binomial(1, p)
    return X, Y