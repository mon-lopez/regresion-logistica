from scipy.optimize import minimize
import numpy as np

def ajustar_modelo(x, y, theta0=(0.0, 0.0), maxiter=None):
    """
    Ajusta el modelo logístico minimizando el riesgo empírico.
    Retorna: theta_hat, riesgo_minimo, objeto_resultado
    """
    options = {'maxiter': maxiter} if maxiter is not None else {}
    res = minimize(
        riesgo_logistico, 
        theta0, 
        args=(x, y), 
        method='BFGS', 
        options=options
    )
    return res.x, res.fun, res