# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
import scipy.stats as st
import scipy.special as sp
cimport numpy as np
from libc.math cimport sqrt, exp, log, log1p
from collections import defaultdict

from cython.parallel import parallel, prange
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads

from mcmc_model_parallel cimport MCMCModel
from sample cimport _sample_gamma, _sample_beta, _sample_dirichlet
from bessel cimport _sample as _sample_bessel
from bessel cimport _mode as _bessel_mode

from bgnmf import BGNMF


cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass


cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    double gsl_ran_beta_pdf(double x, double a, double b)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef extern from "gsl/gsl_sf_hyperg.h" nogil:
    double gsl_sf_hyperg_1F1(double a, double b, double x)

cdef extern from "gsl/gsl_sf_gamma.h" nogil:
    double gsl_sf_gamma(double x)
    double gsl_sf_lngamma(double x)


cdef class DNCBMF(MCMCModel):

    cdef:
        int I, J, K, debug, any_missing
        double bm, bu, shp_a, rte_a, shp_h, rte_h
        double[:,::1] A_IK, B_IK, H_KJ, Beta_IJ, P_TK, Lambda_IJ, zeta_IK, zeta_KJ
        double[:,:,::1] Mu_2IJ
        long[:,::1] Y_KJ
        long[:,:,::1] Y_2IJ, Y_TKJ, Y_2IK
        unsigned int[:,::1] N_TK, mask_IJ

    def __init__(self, int I, int J, int K, double bm=1., double bu=1., 
                 double shp_a=0.1, double rte_a=0.1, double shp_h=0.1, double rte_h=0.1,
                 int debug=0, object seed=None, object n_threads=None):

        if n_threads is None:
            n_threads = omp_get_max_threads()
            print('Max threads: %d' % n_threads)
            
        super(DNCBMF, self).__init__(seed=seed, n_threads=n_threads)

        # Params
        self.I = self.param_list['I'] = I
        self.J = self.param_list['J'] = J
        self.K = self.param_list['K'] = K
        self.bm = self.param_list['bm'] = bm
        self.bu = self.param_list['bu'] = bu
        self.shp_a = self.param_list['shp_a'] = shp_a
        self.shp_h = self.param_list['shp_h'] = shp_h
        self.rte_a = self.param_list['rte_a'] = rte_a
        self.rte_h = self.param_list['rte_h'] = rte_h
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.A_IK = np.zeros((I, K))
        self.B_IK = np.zeros((I, K))
        self.H_KJ = np.zeros((K, J))

        self.Y_KJ = np.zeros((K, J), dtype=int)
        self.Y_2IK = np.zeros((2, I, K), dtype=int)
        self.Y_2IJ = np.zeros((2, I, J), dtype=int)
        self.Lambda_IJ = np.zeros((I, J))

        # Cache 
        self.Mu_2IJ = np.zeros((2, I, J))
        self.zeta_IK = np.zeros((I, K))
        self.zeta_KJ = np.zeros((K, J))

        # Auxiliary data structures
        self.P_TK = np.zeros((n_threads, K))
        self.N_TK = np.zeros((n_threads, K), dtype=np.uint32)
        self.Y_TKJ = np.zeros((n_threads, K, J), dtype=int)

        # Copy of the data
        self.Beta_IJ = np.zeros((I, J))
        
        # Masks (1 means observed, 0 means unobserved)
        self.mask_IJ = np.ones((I, J), np.uint32)  # default is all beta_ij are observed
        self.any_missing = 0

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('A_IK', self.A_IK, self._update_A_IK_and_B_IK),
                     ('B_IK', self.B_IK, self._dummy_update),
                     ('H_KJ', self.H_KJ, self._update_H_KJ),              
                     ('Y_2IJ', self.Y_2IJ, self._update_Y_2IJ),
                     ('Y_2IK', self.Y_2IK, self._update_Y_2IJK),
                     ('Y_KJ', self.Y_KJ, self._dummy_update),
                     ('Lambda_IJ', self.Lambda_IJ, self._update_Lambda_IJ)]
        return variables

    cdef void _dummy_update(self, int update_mode) nogil:
        pass

    def reset_total_itns(self):
        self._total_itns = 0

    def fit(self, Beta_IJ, n_itns=1000, verbose=1, initialize=True, 
            schedule={}, fix_state={}, init_state={}):

        assert Beta_IJ.shape == (self.I, self.J)
        assert (0 <= Beta_IJ).all() and (Beta_IJ <= 1).all()
        if isinstance(Beta_IJ, np.ndarray):
            Beta_IJ = np.ma.core.MaskedArray(Beta_IJ, mask=None)
        assert isinstance(Beta_IJ, np.ma.core.MaskedArray)
        self.Beta_IJ = Beta_IJ.filled(fill_value=0.0)  # missing entries are marginalized out
        self.set_mask(~Beta_IJ.mask)

        if initialize:
            if verbose:
                print('\nINITIALIZING...\n')
            self._init_state()
        
        self.set_state(init_state)
        self.set_state(fix_state)

        if 'Lambda_IJ' not in fix_state.keys():
            self._update_Lambda_IJ(update_mode=self._INFER_MODE)
        
        for k in fix_state.keys():
            schedule[k] = lambda x: False

        if verbose:
            print('\nSTARTING INFERENCE...\n')

        self._update(n_itns=n_itns, verbose=int(verbose), schedule=schedule)

    def set_mask(self, mask_IJ=None):
        self.any_missing = 0
        if mask_IJ is not None:
            assert mask_IJ.shape == (self.I, self.J)
            if mask_IJ.sum() / float(mask_IJ.size) < 0.3:
                print('WARNING: Less than 30 percent observed entries.')
                print('REMEMBER: 1 means observed, 0 means unobserved.')
            self.mask_IJ = mask_IJ.astype(np.uint32)
            self.any_missing = int((mask_IJ == 0).any())

    def set_state(self, state):
        for k, v, _ in self._get_variables():
            if k in state.keys():
                state_v = state[k]
                assert v.shape == state_v.shape
                for idx in np.ndindex(v.shape):
                    v[idx] = state_v[idx]
        self.update_cache()

    def update_cache(self):
        self._compose()

    cdef void _print_state(self):
        cdef:
            int num_tokens
            double sparse, fano, theta, phi, pi

        sparse = 100 * (1 - np.count_nonzero(self.Y_2IJ) / float(self.Y_2IJ.size))
        num_tokens = np.sum(self.Y_2IJ)
        mu = np.mean(np.sum(self.Mu_2IJ, axis=0))

        print('ITERATION %d: percent of zeros: %f, num_tokens: %d, mu: %f\n' % \
                      (self.total_itns, sparse, num_tokens, mu))

    cdef void _init_state(self):
        """
        Initialize internal state.
        """
        self._generate_global_state()

    def compose(self):
        self._compose()
        return np.array(self.Mu_2IJ)

    cdef void _compose(self):
        self.Mu_2IJ = np.stack([np.dot(self.A_IK, self.H_KJ), 
                                np.dot(self.B_IK, self.H_KJ)])

    def reconstruct(self, subs=()):
        Mu_2IJ = self.compose()
        return (Mu_2IJ[0] / Mu_2IJ.sum(axis=0))[subs]

    def generate_state(self):
        self._generate_state()

    cdef void _generate_state(self):
        """
        Generate internal state (all model parameters and latent variables).
        """
        self._generate_global_state()
        self._generate_local_state()

    cdef void _generate_global_state(self):
        """
        Generate the global (shared) model parameters.
        """
        for key, _, update_func in self._get_variables():
            if key not in ['Y_2IJ', 'Y_2IK', 'Lambda_IJ']:
                update_func(self, update_mode=self._GENERATE_MODE)
        self._compose()

    cdef void _generate_local_state(self):
        """
        Generate the local latent variables.
        """
        self._update_Y_2IJ(update_mode=self._GENERATE_MODE)
        self._update_Y_2IJK(update_mode=self._GENERATE_MODE)

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """

        cdef:
            int j, i, tid
            double bm, bu, shp_mij, shp_uij
        
        bm, bu = self.bm, self.bu

        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                shp_mij = bm + self.Y_2IJ[0, i, j]
                shp_uij = bu + self.Y_2IJ[1, i, j]
                self.Beta_IJ[i, j] = _sample_beta(self.rngs[tid], shp_mij, shp_uij)

        self._update_Lambda_IJ(update_mode=self._GENERATE_MODE)

    def likelihood(self, subs=(), missing_vals=None):
        """Calculates the Beta likelihood at given points."""
        Y_2IJ = np.array(self.Y_2IJ)

        vals = missing_vals if missing_vals is not None else np.array(self.Beta_IJ)[subs]
        return st.beta.pdf(vals,
                           self.bm + Y_2IJ[0][subs], 
                           self.bu + Y_2IJ[1][subs])

    def marginal_likelihood(self, subs=(), missing_vals=None, n_mc_samples=1000):
        """Calculates the doubly non-central Beta likelihood at given points.
        
        Marginalizes over Poisson latent variables using Monte Carlo approximation.
        """
        Mu_2IJ = self.compose()
        Mu_m_ = Mu_2IJ[0][subs]
        Mu_u_ = Mu_2IJ[1][subs]

        Y_m_M_ = rn.poisson(Mu_m_, size=(n_mc_samples,) + Mu_m_.shape)
        Y_u_M_ = rn.poisson(Mu_u_, size=(n_mc_samples,) + Mu_u_.shape)

        vals = missing_vals if missing_vals is not None else np.array(self.Beta_IJ)[subs]
        return st.beta.pdf(vals, 
                           self.bm + Y_m_M_, 
                           self.bu + Y_u_M_).mean(axis=0)

    cdef void _update_Lambda_IJ(self, int update_mode):
        cdef:
            int i, j, tid
            double b, shp_ij

        b = self.bm + self.bu
        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                shp_ij = b + self.Y_2IJ[0, i, j] + self.Y_2IJ[1, i, j]
                self.Lambda_IJ[i, j] = _sample_gamma(self.rngs[tid], shp_ij, 1.)

    cdef void _update_Y_2IJ(self, int update_mode):
        cdef:
            int j, tid, i, x, y_xij
            double bm, bu, lam_ij, beta_ij, mu_xij, beta_xij, shp_x, sca_xij

        self._compose()

        if update_mode != self._INFER_MODE:
            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()
                for j in range(self.J):
                    for x in range(2):
                        mu_xij = self.Mu_2IJ[x, i, j]
                        self.Y_2IJ[x, i, j] = gsl_ran_poisson(self.rngs[tid], mu_xij)

        elif update_mode == self._INFER_MODE:
            bm, bu = self.bm, self.bu

            self.Y_2IJ[:] = 0  # don't delete! loops assume default value is zero
            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()
                for j in range(self.J):
                    if self.mask_IJ[i, j]:  # sample y_xij from Bessel if beta_ij is observed
                        lam_ij = self.Lambda_IJ[i, j]
                        beta_ij = self.Beta_IJ[i, j]
                        for x in range(2):
                            mu_xij = self.Mu_2IJ[x, i, j]
                            beta_xij = beta_ij if x == 0 else 1 - beta_ij
                            if beta_xij > 0:
                                shp_x = bm - 1 if x == 0 else bu - 1
                                sca_xij = 2 * sqrt(mu_xij * lam_ij * beta_xij)
                                if sca_xij > 0:
                                    y_xij = _sample_bessel(self.rngs[tid], shp_x, sca_xij)
                                    if self.debug:
                                        with gil:
                                            # assert y_xij >= 0  # comment in for debugging (uses GIL)
                                            if not (y_xij >= 0):
                                                print(shp_x, sca_xij)
                                    self.Y_2IJ[x, i, j] = y_xij

                    else:  # if beta_ij is unobserved, impute y_xij from prior
                        for x in range(2):
                            mu_xij = self.Mu_2IJ[x, i, j]
                            self.Y_2IJ[x, i, j] = gsl_ran_poisson(self.rngs[tid], mu_xij)

    cdef void _update_Y_2IJK(self, int update_mode):
        cdef:
            int K, i, j, x, k, y_xij, tid
            double[:,::1] X_IK

        K = self.K
        
        self.Y_2IK[:] = 0
        self.Y_TKJ[:] = 0

        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                for x in range(2):
                    y_xij = self.Y_2IJ[x, i, j]
                    if y_xij > 0:
                        if x == 0:
                            for k in range(K):
                                self.P_TK[tid, k] = self.A_IK[i, k] * self.H_KJ[k, j]
                        else:
                            for k in range(K):
                                self.P_TK[tid, k] = self.B_IK[i, k] * self.H_KJ[k, j]
                        
                        gsl_ran_multinomial(self.rngs[tid], K, y_xij, &self.P_TK[tid, 0], &self.N_TK[tid, 0])

                        for k in range(K):
                            self.Y_2IK[x, i, k] += self.N_TK[tid, k]
                            self.Y_TKJ[tid, k, j] += self.N_TK[tid, k]

        self.Y_KJ = np.sum(self.Y_TKJ, axis=0)

    cdef void _update_H_KJ(self, int update_mode):
        cdef:
            int k, j, i, tid
            double shp_kj, rte_kj
            double [::1] zeta_K

        if update_mode == self._INFER_MODE:
            zeta_K = np.sum(self.A_IK, axis=0) + np.sum(self.B_IK, axis=0)

        for k in prange(self.K, schedule='static', nogil=True):
            tid = self._get_thread()

            for j in range(self.J):
                shp_kj = self.shp_h
                rte_kj = self.rte_h

                if update_mode == self._INFER_MODE:
                    shp_kj = shp_kj + self.Y_KJ[k, j]
                    rte_kj = rte_kj + zeta_K[k]

                self.H_KJ[k, j] = _sample_gamma(self.rngs[tid], shp_kj, 1 / rte_kj)

    cdef void _update_A_IK_and_B_IK(self, int update_mode):
        cdef:
            int x, k, i, j, tid
            double shp_ik, rte_ik
            double [::1] zeta_K

        if update_mode == self._INFER_MODE:
            zeta_K = np.sum(self.H_KJ, axis=1)

        for k in prange(self.K, schedule='static', nogil=True):
            tid = self._get_thread()

            for x in range(2):
                for i in range(self.I):
                    shp_ik = self.shp_a
                    rte_ik = self.rte_a

                    if update_mode == self._INFER_MODE:
                        shp_ik = shp_ik + self.Y_2IK[x, i, k]
                        rte_ik = rte_ik + zeta_K[k]

                    if x == 0:
                        self.A_IK[i, k]= _sample_gamma(self.rngs[tid], shp_ik, 1 / rte_ik)
                    else:
                        self.B_IK[i, k]= _sample_gamma(self.rngs[tid], shp_ik, 1 / rte_ik)


def initialize_DNCBMF_with_BGNMF(model, data_IJ, verbose=0, n_itns=50):
    params = model.get_params()
    K = params['K']

    print('Initializing with BGNMF...')

    bg_model = BGNMF(n_components=K, 
                     tol=1e-2, 
                     max_iter=500, 
                     verbose=bool(verbose))

    bg_model.fit(data_IJ)

    A_IK = bg_model.A_IK
    B_IK = bg_model.B_IK
    H_KJ = bg_model.H_KJ

    fix_state = {}
    fix_state['A_IK'] = A_IK
    fix_state['B_IK'] = B_IK
    fix_state['H_KJ'] = H_KJ

    model.fit(Beta_IJ=data_IJ,
              n_itns=n_itns,
              initialize=True,
              fix_state=fix_state,
              verbose=verbose)

    print('\n------------------\nInitialized.\n')
    model.reset_total_itns()
