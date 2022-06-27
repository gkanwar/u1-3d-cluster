from distutils.core import setup, Extension
import numpy

cluster = Extension(
    'cluster', sources=['cluster.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3']
    # undef_macros=['NDEBUG']
)
setup(ext_modules=[cluster])
