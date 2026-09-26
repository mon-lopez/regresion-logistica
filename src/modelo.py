import numpy as np

def sigmoid(t):
    """Función sigmoide numéricamente estable."""
    return 1.0 / (1.0 + np.exp(-np.clip(t, -500, 500)))

def proba(x, theta):
    """Calcula p_theta(x) = sigma(theta_0 + theta_1 * x)."""
    return sigmoid(theta[0] + theta[1] * x)

def riesgo_logistico(theta, x, y):
    """Calcula el riesgo empírico logístico R_S(theta)."""
    p = proba(x, theta)
    # Clip para evitar log(0)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))