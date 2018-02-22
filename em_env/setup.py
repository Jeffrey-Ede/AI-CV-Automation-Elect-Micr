from distutils.core import setup, Extension

# define the extension module
em_env_module = Extension('em_env_module', sources=['em_env.cpp'])

# run the setup
setup(ext_modules=[em_env_module])
