from distutils.core import setup#, Extension
from distutils.extension import Extension

# define the extension module
em_env_module = Extension('em_env_module', sources=['C:/Users/Jeffrey Ede/source/repos/machine_learning/machine_learning/em_env.cpp'])

# run the setup
setup(ext_modules=[em_env_module])
