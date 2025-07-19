import io
import sys
import os
import numpy as np
import platform
from setuptools import find_packages, setup, Extension

def get_compile_args():
    """Get platform-specific compilation arguments."""
    if platform.system() == 'Windows':
        # MSVC compiler flags
        return ['/O2']
    else:
        # GCC/Clang flags for Unix-like systems
        return ['-O3', '-fPIC', '-std=c99', '-ffast-math']
    
def get_libraries():
    """Get platform-specific libraries to link."""
    if platform.system() == 'Windows':
        return []
    else:
        return ['m']
    
fastdfa_extension = Extension(
    'pyhctsa.Toolboxes.Max_Little.fastdfa',
    sources=['pyhctsa/Toolboxes/Max_Little/ML_fastdfa_core.c'],
    include_dirs=["pyhctsa/Toolboxes/Max_Little", np.get_include()],
    extra_compile_args=get_compile_args(),
    libraries=get_libraries(),
    extra_link_args=[],
    define_macros=[],
)
    
sampen_extension = Extension(
    'pyhctsa.Toolboxes.physionet.sampen',
    sources=['pyhctsa/Toolboxes/physionet/sampen.c'],  
    include_dirs=["pyhctsa/Toolboxes/physionet", np.get_include()], 
    extra_compile_args=get_compile_args(),
    libraries=get_libraries(),
    extra_link_args=[],
    define_macros=[],
)

close_returns_extension = Extension(
    'pyhctsa.Toolboxes.Max_Little.close_returns',
    sources=['pyhctsa/Toolboxes/Max_Little/ML_close_ret.c'],  
    include_dirs=["pyhctsa/Toolboxes/Max_Little", np.get_include()], 
    extra_compile_args=get_compile_args(),
    libraries=get_libraries(),
    extra_link_args=[],
    define_macros=[],
)

periodicity_wang_module = Extension(
    "pyhctsa.Toolboxes.c22.PD_PeriodicityWang",
    sources=[
        "pyhctsa/Toolboxes/c22/PD_PeriodicityWang.c",
        "pyhctsa/Toolboxes/c22/splinefit.c",
        "pyhctsa/Toolboxes/c22/stats.c",
        "pyhctsa/Toolboxes/c22/helper_functions.c"
    ],
    include_dirs=["pyhctsa/Toolboxes/c22", np.get_include()],
    extra_compile_args=get_compile_args(),
    libraries=get_libraries(),
    extra_link_args=[],
    define_macros=[],
)

shannon_entropy_module = Extension(
    "pyhctsa.Toolboxes.Michael_Small.shannon",
    sources=[
        "pyhctsa/Toolboxes/Michael_Small/MS_shannon.c"
    ],
    include_dirs=["pyhctsa/Toolboxes/Michael_Small", np.get_include()],
    extra_compile_args=get_compile_args(),
    libraries=get_libraries(),
    extra_link_args=[],
    define_macros=[]
)

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name="pyhctsa",
    version="0.0.1",
    description="project_description",
    long_description=read("README.md"),
    author="Joshua B. Moore",
    packages=find_packages(exclude=["tests", ".github"]),
    ext_modules=[periodicity_wang_module, close_returns_extension, 
                 sampen_extension, fastdfa_extension, shannon_entropy_module],
    install_requires=read_requirements("requirements.txt"),
    zip_safe=False,
)
