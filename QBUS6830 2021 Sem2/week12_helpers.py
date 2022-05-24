import warnings

import numpy as np
import pandas as pd


def caviar_loss(b, r, p, q0):
    ''' CaViaR loss function - More efficient than qregCaViaR
    
        Parameters
        ----------
        b: vector of model parameters.
        r: vector of returns.
        p: scalar, indicating probability level.
        q0: scalar, initial quantile.
        
        Returns
        -------
        returns: scalar, loss function value.
    '''
    
    n = len(r)
    q = np.empty(n)
    q[0] = q0
    for t in range(1, n):
        q[t] = b[0] + b[1] * q[t - 1] + b[2] * np.abs(r[t - 1])
    
    x1 = r[r > q]
    x2 = r[r < q]
    f1 = q[r > q]
    f2 = q[r < q]
    return p * np.sum(x1 - f1) + (1 - p) * np.sum(f2 - x2)


def caviar_est(b, r, p, q0):
    n = len(r)
    q = np.empty(n)
    q[0] = q0
    for t in range(1, n):
        q[t] = b[0] + b[1] * q[t - 1] + b[2] * np.abs(r[t - 1])
        
    return q    


def caviar_update(b, r, p, q0):
    ''' CaViaR update function
    
        Parameters
        ----------
        b: vector of model parameters.
        r: vector of returns.
        p: scalar, indicating probability level.
        q0: scalar, initial quantile.
        
        Returns
        -------
        returns: vector of one-step-ahead forecasts, aligned with r.
    '''
    q = pd.Series(index=r.index, dtype='float64')
    q[0] = q0
    for t in range(1, len(r)):
        q[t] = b[0] + b[1] * q[t - 1] + b[2] * np.abs(r[t - 1])
    return q


def igarch_update(r, v0):
    ''' IGARCH (RiskMetrics) update function
    
        Parameters
        ----------
        r: vector of returns.
        v0: scalar, initial variance.
        
        Returns
        -------
        returns: vector of one-step-ahead forecasts, aligned with r.
    '''
    warnings.warn("THIS SHOULD ONLY BE USED IN WEEK 12 TUTORIAL - THE COEFFICEINTS ARE HARD CODED AND CANNOT CHANGE.")
    
    v = pd.Series(index=r.index, dtype='float64')
    v[0] = v0
    for t in range(1, len(r)):
        v[t] = 0.94 * v[t - 1] + 0.06 * r[t - 1] ** 2
    return v