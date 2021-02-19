"""
Beta-Gamma Non-Negative Matrix Factorization (BG-NMF)

From the paper:
"Variational Bayesian Matrix Factorization for Bounded Support Data"
Ma, Teschendorff, Leijon, Qiao, Zhang (2014)

Model:

X_ij ~ Beta(\sum_k A_ik H_kj, \sum_k B_ik H_kj)
A_ik ~ Gamma(m0, a0)
B_ik ~ Gamma(n0, b0)
H_kj ~ GAmma(r0, z0)

"""
import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import scipy.stats as st
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import scipy.stats as st

from argparse import ArgumentParser
from scipy.special import psi


class BGNMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=14, max_iter=200, eps=1e-10,
                 tol=1e-5, smoothness=100, verbose=True,
                 m0=0.1, a0=0.1, n0=0.1, b0=0.1, r0=0.1, z0=0.1):

        self.n_components = n_components
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose

        self.m0 = m0    # prior shape for A (\mu_0)
        self.a0 = a0    # prior rate for A (\alpha_0)
        self.n0 = n0    # prior shape for B (\nu_0)
        self.b0 = b0    # prior rate for B (\beta_0)
        self.r0 = r0    # prior shape for H (\rho_0)
        self.z0 = z0    # prior rate for H (\zeta_0)

        self.A_IK = np.empty((1, self.n_components))
        self.B_IK = np.empty((1, self.n_components))
        self.H_KJ = np.empty((self.n_components, 1))

    def fit(self, data, fix_state={}):
        if isinstance(data, np.ndarray):
            data = np.ma.core.MaskedArray(data, mask=None)
        assert isinstance(data, np.ma.core.MaskedArray)

        mask = None if not data.mask.any() else data.mask.copy()
        data = data.filled(fill_value=0.0)  # missing entries will be imputed

        assert data.ndim == 2
        self._init(data)
        self.set_state(fix_state)
        schedule = defaultdict(int, [(k, np.inf) for k in fix_state.keys()])
        self._update(data, mask, schedule)
        return self

    def set_state(self, fix_state={}):
        if 'A_IK' in fix_state.keys():
            self.A_IK[:] = fix_state['A_IK']
        elif 'B_IK' in fix_state.keys():
            self.B_IK[:] = fix_state['B_IK']
        elif 'H_KJ' in fix_state.keys():
            self.H_KJ[:] = fix_state['H_KJ']

    def _init(self, data):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert (0 <= data).all() and (data <= 1).all()

        I, J = data.shape
        K = self.n_components
        s = self.smoothness

        self.shp_A_IK = s * rn.gamma(s, 1. / s, size=(I, K))
        self.rte_A_IK = s * rn.gamma(s, 1. / s, size=(I, K))
        self.A_IK = self.shp_A_IK / self.rte_A_IK

        self.shp_B_IK = s * rn.gamma(s, 1. / s, size=(I, K))
        self.rte_B_IK = s * rn.gamma(s, 1. / s, size=(I, K))
        self.B_IK = self.shp_B_IK / self.rte_B_IK

        self.shp_H_KJ = s * rn.gamma(s, 1. / s, size=(K, J))
        self.rte_H_KJ = s * rn.gamma(s, 1. / s, size=(K, J))
        self.H_KJ = self.shp_H_KJ / self.rte_H_KJ

    def _update_A(self, data, mask=None):
        lnX = np.log(data)
        # if mask is not None:
            # lnX[mask] = 0

        self.rte_A_IK[:] = self.a0 - np.dot(lnX, self.H_KJ.T)
        assert np.isfinite(self.rte_A_IK).all()

        foo_IJ = psi(np.dot((self.A_IK + self.B_IK), self.H_KJ))
        bar_IJ = psi(np.dot(self.A_IK, self.H_KJ))
        baz_IK = np.dot((foo_IJ - bar_IJ), self.H_KJ.T)
        self.shp_A_IK[:] = self.m0 + baz_IK * self.A_IK
        assert np.isfinite(self.shp_A_IK).all()

        self.A_IK[:] = self.shp_A_IK / self.rte_A_IK
        assert np.isfinite(self.A_IK).all()
        # assert (self.A_IK > 0).all()

    def _update_B(self, data, mask=None):
        ln1mX = np.log1p(-data)
        # if mask is not None:
            # ln1mX[mask] = 0

        self.rte_B_IK[:] = self.b0 - np.dot(ln1mX, self.H_KJ.T)
        assert np.isfinite(self.rte_B_IK).all()

        foo_IJ = psi(np.dot((self.A_IK + self.B_IK), self.H_KJ))
        bar_IJ = psi(np.dot(self.B_IK, self.H_KJ))
        baz_IK = np.dot((foo_IJ - bar_IJ), self.H_KJ.T)
        self.shp_B_IK[:] = self.n0 + baz_IK * self.B_IK
        assert np.isfinite(self.shp_B_IK).all()

        self.B_IK[:] = self.shp_B_IK / self.rte_B_IK
        assert np.isfinite(self.B_IK).all()
        # assert (self.B_IK > 0).all()

    def _update_H(self, data, mask=None):
        lnX = np.log(data)
        ln1mX = np.log1p(-data)
        # if mask is not None:
            # lnX[mask] = 0
            # ln1mX[mask] = 0

        foo_KJ = np.dot(self.A_IK.T, lnX)
        bar_KJ = np.dot(self.B_IK.T, ln1mX)
        self.rte_H_KJ[:] = self.z0 - foo_KJ - bar_KJ
        assert np.isfinite(self.rte_H_KJ).all()

        ApB_IK = self.A_IK + self.B_IK
        foo_KJ = np.dot(ApB_IK.T, psi(ApB_IK.dot(self.H_KJ)))
        foo_KJ -= np.dot(self.A_IK.T, psi(self.A_IK.dot(self.H_KJ)))
        foo_KJ -= np.dot(self.B_IK.T, psi(self.B_IK.dot(self.H_KJ)))

        self.shp_H_KJ[:] = self.r0 + foo_KJ * self.H_KJ
        assert np.isfinite(self.shp_H_KJ).all()

        self.H_KJ[:] = self.shp_H_KJ / self.rte_H_KJ
        assert np.isfinite(self.H_KJ).all()
        # assert (self.H_KJ > 0).all()

    def reconstruct(self):
        alpha1_IJ = np.dot(self.A_IK, self.H_KJ)
        alpha2_IJ = np.dot(self.B_IK, self.H_KJ)
        return alpha1_IJ / (alpha1_IJ + alpha2_IJ)

    def _impute(self, data, mask=None):
        """
        mask---bool array. True=missing, False=observed.
        """
        if mask is not None:
            data[mask] = self.reconstruct()[mask]

    def _update(self, data, mask=None, schedule=defaultdict(int)):
        if mask is not None:
            data[mask] = rn.beta(1, 1, size=mask.sum())   
        data = np.clip(data, a_min=self.eps, a_max=1-self.eps)

        curr_elbo = -np.inf
        deltas = []
        for itn in range(self.max_iter):
            s = time.time()
            if schedule['A_IK'] <= itn:
                self._update_A(data, mask)
            if schedule['B_IK'] <= itn:
                self._update_B(data, mask)
            if schedule['H_KJ'] <= itn:
                self._update_H(data, mask)
            self._impute(data, mask)

            bound = self._elbo(data, mask)
            delta = (bound - curr_elbo) / abs(curr_elbo) if itn > 0 else np.nan
            deltas.append(delta) 
            e = time.time() - s
            if self.verbose:
                print ('ITERATION %d:\t\
                                       Time: %f\t\
                                       Objective: %.5f\t\
                                       Change: %.5f\t'\
                                       % (itn, e, bound, np.mean(deltas[-5:])))
            # assert ((delta >= 0.0) or (itn == 0))  # not true for monte carlo
            curr_elbo = bound
            
            if np.mean(deltas[-5:]) < self.tol:
                break

    def _elbo(self, data, mask=None, S=10):
        """Monte carlo approximation of ELBO (as implemented by Ma et al. (2015).
        
                B = E_{Q(Z)} [ log P(X,Z) / Q(Z) ]
        """   
        I, J = data.shape
        K = self.n_components
        A_SIK = rn.gamma(self.shp_A_IK, 1. / self.rte_A_IK, size=(S, I, K))
        B_SIK = rn.gamma(self.shp_B_IK, 1. / self.rte_B_IK, size=(S, I, K))
        H_SKJ = rn.gamma(self.shp_H_KJ, 1. / self.rte_H_KJ, size=(S, K, J))

        lnqz = st.gamma.logpdf(A_SIK,
                               self.shp_A_IK,
                               scale=1. / self.rte_A_IK,
                               loc=0).sum(axis=(1, 2)).mean()

        lnqz += st.gamma.logpdf(B_SIK,
                                self.shp_B_IK,
                                scale=1. / self.rte_B_IK,
                                loc=0).sum(axis=(1, 2)).mean()

        lnqz += st.gamma.logpdf(H_SKJ,
                                self.shp_H_KJ,
                                scale=1. / self.rte_H_KJ,
                                loc=0).sum(axis=(1, 2)).mean()

        lnpxz = st.gamma.logpdf(A_SIK,
                                self.m0,
                                scale=1. / self.a0,
                                loc=0).sum(axis=(1, 2)).mean()

        lnpxz += st.gamma.logpdf(B_SIK,
                                 self.n0,
                                 scale=1. / self.b0,
                                 loc=0).sum(axis=(1, 2)).mean()

        lnpxz += st.gamma.logpdf(H_SKJ,
                                 self.r0,
                                 scale=1. / self.z0,
                                 loc=0).sum(axis=(1, 2)).mean()

        alpha1_SIJ = np.einsum('sik,skj->sij', A_SIK, H_SKJ, optimize=True)
        alpha2_SIJ = np.einsum('sik,skj->sij', B_SIK, H_SKJ, optimize=True)

        if mask is not None:
            lnpxz += st.beta.logpdf(data[~mask],
                                    alpha1_SIJ[:, ~mask],
                                    alpha2_SIJ[:, ~mask],
                                    loc=0).sum(axis=1).mean()
        else:
            lnpxz += st.beta.logpdf(data,
                                    alpha1_SIJ,
                                    alpha2_SIJ,
                                    loc=0).sum(axis=(1, 2)).mean()

        return lnpxz - lnqz

if __name__ == '__main__':
    data = np.load('../dat/methylation/breast_ovary_colon_lung/data.npz')
    Beta_IJ = data['Beta_IJ']
    train_Beta_IJ = Beta_IJ[:300]
    test_Beta_IJ = Beta_IJ[-100:, :4500]
    heldout_Beta_IJ = Beta_IJ[-100:, 4500:]

    rn.seed(444)
    train_model = BGNMF(tol=0, max_iter=10)
    train_model.fit(train_Beta_IJ)

    H_KJ = train_model.H_KJ.copy()

    test_model = BGNMF(tol=0, max_iter=10)
    test_model.fit(test_Beta_IJ, fix_state={'H_KJ': H_KJ[:, :4500]})

    A_IK = test_model.A_IK
    B_IK = test_model.B_IK
    assert np.allclose(H_KJ[:, :4500], test_model.H_KJ)

    alpha1_IJ = A_IK.dot(H_KJ[:, 4500:])
    alpha2_IJ = B_IK.dot(H_KJ[:, 4500:])

    recon_IJ = alpha1_IJ / (alpha1_IJ + alpha2_IJ)
    err_IJ = np.abs(heldout_Beta_IJ - recon_IJ)
    ll_IJ = st.beta.logpdf(heldout_Beta_IJ, alpha1_IJ, alpha2_IJ)
    print('Test error: %f' % err_IJ.mean())
    print('Average test likelihood: %f' % np.exp(ll_IJ.mean()))
