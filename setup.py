"""
Installs the madigan library.

Can be installed via either:
    python setup.py install build_ext
OR
    pip install .


If a development version is desired, the following allows
changes to take effect without reinstalling:
    python setup.py develop build_ext --inplace
OR
    pip install -e .

"""
import os
import sys
import platform
import subprocess
import re
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from Cython.Build import cythonize
from Cython.Compiler import Options
# import numpy
Options.annotate = True

with open("README.md", 'r') as f:
    long_description = f.read()

###### Cython Modules #####
EXT_MODULES = None
if platform.system() == "Windows":
    EXT_MODULES = [
        Extension("madigan.environments.reward_shaping",
                  ["madigan/environments/reward_shaping.pyx"],
                  language='c++',
                  extra_compile_args=['/openmp', '-O3', '-Zi'],
                  extra_link_args=['/openmp'])
        # extra_link_args=['/openmp', '-debug:full']),
    ]
elif platform.system() == "Linux":
    EXT_MODULES = [
        Extension("madigan.environments.reward_shaping",
                  ["madigan/environments/reward_shaping.pyx"],
                  language='c++',
                  extra_compile_args=['-fopenmp', '-O3'],
                  extra_link_args=['-fopenmp'])
        # extra_link_args=['-fopenmp', '-debug:full']),
    ]
else:
    raise EnvironmentError(
        "Ext_modules definitions are only available for windows or linux,"
        " specify compiler directives/args for your installation")


###### Pybind11 Modules (C++) #####
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp,
                              env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


packages = find_packages(exclude=['arena'])
print('Packages: ', packages)

EXT_MODULES = [cythonize(EXT_MODULES,
                         compiler_directives={'language_level': "3"}),
               [CMakeExtension('madigan.environments.cpp')]]

setup(
    name='madigan',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    packages=packages,
    include_package_data=True,
    ext_modules=EXT_MODULES[0],
    # include_dirs=[numpy.get_include()]
)
