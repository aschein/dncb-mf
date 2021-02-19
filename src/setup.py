import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np
from path import Path


# if sys.platform == 'darwin':
#     os.environ['CC'] = '/anaconda3/bin/gcc'
#     os.environ['CXX'] = '/anaconda3/bin/g++'
 

include_gsl_dir = '/usr/local/include/'
lib_gsl_dir = '/usr/local/lib/'


EXT_MODULES = [Extension(str(x.namebase),
					     [str(x.name)],
					     library_dirs=[lib_gsl_dir],
                     	 libraries=['gsl', 'gslcblas'],
					     extra_compile_args=['-fopenmp'],
					     extra_link_args=['-fopenmp'],
					     include_dirs=[include_gsl_dir, np.get_include()])
			   for x in Path('.').files('*.pyx')]

setup(name='dncb-mf',
  	  cmdclass={"build_ext": build_ext},
      ext_modules=EXT_MODULES)
