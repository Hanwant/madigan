# from sys import platform
from setuptools import setup, find_packages
# from setuptools import Extension
# from Cython.Build import cythonize
# from Cython.Compiler import Options
# import numpy

# python setup.py develop build_ext --inplace for sym-linking local directory to site-packages
# python setup.py install build_ext --inplace for proper installation

# Options.annotate = True

with open("README.org", 'r') as f:
    long_description = f.read()


packages = find_packages()
print('Packages: ', packages)
setup(
    name = 'madigan',
    version = '0.0.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    packages=packages,
    include_package_data=True,
    # ext_modules = cythonize(EXT_MODULES, compiler_directives={'language_level' : "3"}, emit_linenums=True),
    # include_dirs=[numpy.get_include()]
)

# EXT_MODULES=None
# if platform == "win32":
#     EXT_MODULES = [Extension("preprocessing.sampling_cy", ["preprocessing/sampling_cy.pyx"],# language='c++',
#                              extra_compile_args=['/openmp', '-O3', '-Zi' ],
#                              extra_link_args=[ '/openmp', '-debug:full']),
#                    Extension("preprocessing.rollers", ["preprocessing/rollers.pyx"],  # language='c++',
#                              extra_compile_args=['/openmp', '-O3', '-Zi'],
#                              extra_link_args=['/openmp', '-debug:full']
#                    )
#     ]
# elif platform in ["linux", "linux2"]:
#     EXT_MODULES = [Extension("preprocessing.sampling_cy", ["preprocessing/sampling_cy.pyx"],# language='c++',
#                              extra_compile_args=['-fopenmp', '-O3'],
#                              extra_link_args=[ '-fopenmp', '-debug:full']),
#                    Extension("preprocessing.rollers", ["preprocessing/rollers.pyx"],  # language='c++',
#                              extra_compile_args=['-fopenmp', '-O3'],
#                              extra_link_args=['-fopenmp', '-debug:full']
#                    )
#     ]
# else:
#     raise EnvironmentError("Ext_modules definitions are only available for windows or linux, specify compiler directives/args for your installation")
