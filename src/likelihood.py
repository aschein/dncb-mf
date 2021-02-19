import numpy as np
import numpy.random as rn
import scipy.special as sp
import scipy.stats as st

def _dncb_pdf(x, a1, a2, mu1, mu2):
    """Alt implementation that calls hchg. Numerically unstable."""
    out = st.beta.pdf(x, a1, a2, loc=0)
    out *= np.exp(-mu1-mu2)
    out *= hchg(x, a1, a2, mu1, mu2)
    return out

def hchg(x, a1, a2, mu1, mu2):
    """Series-based implementation of Humbert's Confluent Hypergeometric Function.
    
    See equation 22 on page 13 of Orsi (2017):
    https://arxiv.org/pdf/1706.08557.pdf

    mu1 is $lambda_1 / 2$ and mu2 is $lambda_1 / 2$ in the paper. 

    This method computes the following special case of Humbert's function:

        H(a1+a2, a1, a2, mu1 * x, mu2 * (1-x))

    This implementation may be unstable; use log_hchg.
    """
    a = a1 + a2
    j = np.arange(250)
    if np.isscalar(x):
        x = np.array([x])
    x = x[:, np.newaxis]
    
    out = (mu1 * x) ** j / sp.factorial(j)
    out *= sp.poch(a1+a2, j) / sp.poch(a1, j)
    out *= sp.hyp1f1(a1+a2+j, a2, mu2*(1-x))
    out = out.sum(axis=1)
    return out if out.size > 1 else float(out)

def dncb_pdf(x, a1, a2, mu1, mu2):
    return np.exp(dncb_logpdf(x, a1, a2, mu1, mu2))

def dncb_logpdf(x, a1, a2, mu1, mu2):
    if not np.alltrue(mu1 <= 200) or not np.alltrue(mu2 <= 200):
        print('WARNING: Unstable and probably inaccurate for mu1>100 or mu2>100.')
        print('\t Use Monte Carlo approximation for this.')
    out = st.beta.logpdf(x, a1, a2, loc=0)
    out -= (mu1 + mu2)
    out += log_hchg(x, a1, a2, mu1, mu2)
    return out

def log_hchg(x, a1, a2, mu1, mu2):
    """Series-based implementation of Humbert's Confluent Hypergeometric Function.
    
    See equation 22 on page 13 of Orsi (2017):
    https://arxiv.org/pdf/1706.08557.pdf

    mu1 is $lambda_1 / 2$ and mu2 is $lambda_1 / 2$ in the paper. 

    This method computes the following special case of Humbert's function:

        H(a1+a2, a1, a2, mu1 * x, mu2 * (1-x))

    This method returns the log of the above function.
    """
    assert np.alltrue(mu1 > 0) and np.alltrue(mu2 > 0)
    assert np.alltrue(a1 > 0) and np.alltrue(a2 > 0)
    
    out_shp = np.broadcast(x, a1, a2, mu1, mu2).shape
    if out_shp == ():
        out_shp = (1,)
    
    x = np.broadcast_to(x, out_shp).ravel()[:, np.newaxis]
    a1 = np.broadcast_to(a1, out_shp).ravel()[:, np.newaxis]
    a2 = np.broadcast_to(a2, out_shp).ravel()[:, np.newaxis]
    mu1 = np.broadcast_to(mu1, out_shp).ravel()[:, np.newaxis]
    mu2 = np.broadcast_to(mu2, out_shp).ravel()[:, np.newaxis]
    
    j = np.arange(250)
    
    out = j * np.log(mu1 * x) - sp.gammaln(j+1)
    out += log_poch(a1+a2, j) - log_poch(a1, j)
    out += np.log(sp.hyp1f1(a1+a2+j, a2, mu2*(1-x)))
    out = sp.logsumexp(out, axis=1)
    return out.reshape(out_shp) if out.size > 1 else float(out)

def log_poch(a,b):
    return sp.gammaln(a+b) - sp.gammaln(a)

def dncb_pdf_mc(x, a1, a2, mu1, mu2, n_samples=1000):
    y1 = rn.poisson(mu1, size=(n_samples,) + x.shape)
    y2 = rn.poisson(mu2, size=(n_samples,) + x.shape)
    return st.beta.pdf(x, a1+y1, a2+y2, loc=0).mean(axis=0)