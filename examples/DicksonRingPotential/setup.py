from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import os, glob
import numpy

Cython.Compiler.Options.annotate = True
 
numpy_include = numpy.get_include()

_include = ['.', numpy_include]
_extra = ['-O3',]
 
 
extensions = [
        Extension('modelsim', ['modelsim.pyx', 'randomkit.c'], 
            depends=['randomkit.h',],
            include_dirs=_include,
            extra_compile_args=_extra),
        ]

 
cy_extensions = cythonize(extensions)
 
setup(
    name = "",
    ext_modules = cy_extensions
)
