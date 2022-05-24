import numpy as np
from scipy import stats
import pandas as pd


def igarch_update(r, v0):
    '''
    IGARCH (RiskMetrics) update function
    r: vector of returns.
    v0: scalar, initial variance.
    returns: vector of one-step-ahead forecasts, aligned with r.
    '''
    
    v = pd.Series(index=r.index, dtype='float64')
    v[0] = v0
    for t in range(1, len(r)):
        v[t] = 0.94 * v[t - 1] + 0.06 * r[t - 1] ** 2
    return v

# For VaR and ES calculations under Gaussian and Standardised-t distributions
def qn(p):
    return stats.norm.ppf(p)

def qt(p, df):
    return stats.t.ppf(p, df) * np.sqrt((df - 2) / df)

def esn(p):
    ninv = stats.norm.ppf(p)
    return -stats.norm.pdf(ninv) / p

def est(p, df):
    tinv = stats.t.ppf(p, df)
    f = lambda x: stats.t.pdf(x, df)
    return -f(tinv) / p * (df + tinv ** 2) / (df - 1) * np.sqrt((df - 2) / df)

def mcn(p, n):
    z = np.random.normal(size=n)
    var = np.quantile(z, p)
    es = np.mean(z[z < var])
    return (var, es)

def mct(p, df, n):
    t = np.random.standard_t(df, size=n)
    z = t * np.sqrt((df - 2) / df)
    var = np.quantile(z, p)
    es = np.mean(z[z < var])
    return (var, es)

def es(r, p):
    var = np.quantile(r, p)
    return np.mean(r[r < var])

def es_to_var_n(p):
    ninv = stats.norm.ppf(p)
    return stats.norm.cdf((-1 / p) * stats.norm.pdf(ninv))

def es_to_var_t(p, df):
    tinv = stats.t.ppf(p, df)
    tpdf = stats.t.pdf(tinv, df)
    temp = (-1 / p) * tpdf * (df + tinv ** 2) / (df - 1)
    return stats.t.cdf(temp, df)

# ES residuals
def es_resid(es, var, s, r):
    xi = r[r < var] - es[r < var]
    return (xi.dropna(), (xi / s).dropna())

def ttest(x, mu):
    n = len(x)
    xbar = np.mean(x)
    s = np.std(x, ddof=1)
    t = (xbar - mu) / (s / np.sqrt(n))
    pval = 2 * stats.t.sf(np.abs(t), df=(n - 1))
    return pval, t

# Unconditional coverage test
def uctest(hit, a):
    n = len(hit)
    p = np.sum(hit) / n
    z = (p - a) / np.sqrt(a * (1 - a) / n)
    pval = 2 * stats.norm.sf(np.abs(z))
    return pval, p

# Independence test
def indtest(hits):
    n = len(hits)

    r5 = hits.values[1:]
    r51 = hits.values[:-1]
    i11 = r5*r51
    i01 = r5*(1-r51)
    i10 = (1-r5)*r51
    i00 = (1-r5)*(1-r51)

    t00 = np.sum(i00)
    t01 = np.sum(i01)
    t10 = np.sum(i10)
    t11 = np.sum(i11)
    p01 = t01/(t00+t01)
    p11 = t11/(t10+t11)
    p1 = (t01+t11)/n

    ll1 = t00 * np.log(1-p01) + (p01>0) * t01 * np.log(p01) + t10 * np.log(1-p11)
    if p11>0:
        ll1=ll1+t11*np.log(p11)
  
    ll0=(t10+t00)*np.log(1-p1)+(t01+t11)*np.log(p1)

    lrind=2*(ll1-ll0)
    pcc=1-stats.chi2.cdf(lrind,1)
    return pcc, lrind

# Dynamic quantile test
def dqtest(y,f,a,lag):
    n = len(y)
    hits = ((y<f)*1)*(1-a)
    hits = (hits)*1+(y>f)*(-a)
    q=2+lag
    
    if np.sum((y<f)*1) > 0:
        ns = n - lag
        xmat = np.column_stack([np.ones((ns,1)), f[lag:n+1]])
        for k in range(1,lag+1):
            lk = lag-k
            xmat = np.column_stack([xmat, hits[lk:n-k]])
    
        hx = np.dot((hits[lag:n+1]), xmat)
        xtx = np.linalg.lstsq(np.matmul(xmat.T, xmat), np.eye(q), rcond = None)[0]
        dq = np.dot(np.dot(hx, xtx), hx.T)
        dq = dq/(a*(1-a))
        pdq = 1 - stats.chi2.cdf(dq,q)
    else:
        pdq = np.nan
        dq = np.nan
    return pdq, dq

# Quantile loss function
def qloss(q,r,p):
    q = np.array(q)
    x1 = r[r > q]
    x2 = r[r < q]
    f1 = q[r > q]
    f2 = q[r < q]
    l = p * np.sum(x1-f1) + (1-p) * np.sum(f2-x2)
    return l

# Joint loss function
def jointloss(es,q,r,p):
    m = len(r)
    q = np.array(q)
    es = np.array(es)
    i1 = (r < q).astype(int)
    aes = es ** (-1) * (p-1)
    ees = (r-q) * (p - i1)
    l =  np.sum(-np.log(aes)) - np.sum(ees / es) / p
    l = l / m
    return l

# Accuracy checks for VaR
def check_var_fc(var_fc, r, p):
    hit = r < var_fc
    n_hit = np.sum(hit)
    pval_uc, p_hat = uctest(hit, p)
    pval_ind = indtest(hit)[0]
    pval_dq = dqtest(r, var_fc, p, 4)[0]
    qtl_loss = qloss(var_fc, r, p)
    return [n_hit, p_hat, p_hat / p, pval_uc, pval_ind, pval_dq, qtl_loss]

# Accuracy checks for ES
def check_es_fc(es, var, s, r):
    hit = r < var
    n_hit = np.sum(hit)
    xi, xis = es_resid(es, var, s, r)
    t_xi = ttest(xi, 0)[1]
    t_xis = ttest(xis, 0)[1]
    p_xis = ttest(xis, 0)[0]
    return [n_hit, np.mean(xi), t_xi, np.mean(xis), t_xis, p_xis]

# More accuracy checks for ES
def check_es_fc_ex(es, var, s, r, p):
    xi = es_resid(es, var, s, r)[0]
    rmse = np.sqrt(np.mean(xi ** 2))
    mad = np.mean(np.abs(xi))
    jloss = jointloss(es, var, r, p)
    lst = check_var_fc(es, r, p)
    lst.append(jloss)
    lst.append(rmse)
    lst.append(mad)
    return lst