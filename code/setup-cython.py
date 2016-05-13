from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("base", ["base.pyx"])]

setup(
  name = 'convDBN base ',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
