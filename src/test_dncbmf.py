import sys
import numpy as np
import numpy.random as rn
import scipy.stats as st

from dncbmf import DNCBMF
from IPython import embed

if __name__ == '__main__':
    I = 7
    J = 8
    K = 3
    bm = 1
    bu = 0.1
    shp_a = 0.1
    rte_a = 0.1
    shp_h = 0.01
    rte_h = 0.01
    debug = 0
    n_threads = 3

    # seed = rn.randint(10000)
    seed = 760
    print(seed)

    rn.seed(seed)
    mask_IJ = (rn.random(size=(I, J)) < 0.75).astype(int)

    model = DNCBMF(I=I,
                   J=J,
                   K=K,
                   bm=bm,
                   bu=bu,
                   shp_a=shp_a,
                   rte_a=rte_a,
                   shp_h=shp_h,
                   rte_h=rte_h,
                   debug=debug,
                   seed=seed,
                   n_threads=n_threads)

    model.set_mask(mask_IJ)

    def get_schedule_func(burnin=0, update_every=1):
        return lambda x: x >= burnin and x % update_every == 0

    schedule = {'Lambda_IJ': get_schedule_func(0, 1),
                'Y_2IJ': get_schedule_func(0, 1),
                'Y_2IK': get_schedule_func(0, 1),
                'Y_KJ': get_schedule_func(0, 1),
                'A_IK': get_schedule_func(0, 1),
                'B_IK': get_schedule_func(0, 1),
                'H_KJ': get_schedule_func(0, 1)}

    entropy_funcs = {'Entropy min': lambda x: np.min(st.entropy(x)),
                     'Entropy max': lambda x: np.max(st.entropy(x)),
                     'Entropy mean': lambda x: np.mean(st.entropy(x)),
                     'Entropy var': lambda x: np.var(st.entropy(x))}
    var_funcs = {}

    model.alt_geweke(1000, var_funcs=var_funcs, schedule=schedule)
