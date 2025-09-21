import platform
import numpy as np
from setuptools import Extension

def _compile_args():
    if platform.system() == "Windows":
        # MSVC: no -std=c99; use defaults (C11-ish) and enable O2
        return ["/O2"]
    return ["-O3", "-fPIC", "-std=c99", "-ffast-math"]

def _libraries():
    return [] if platform.system() == "Windows" else ["m"]

def build_extensions():
    np_inc = np.get_include()

    fastdfa = Extension(
        "pyhctsa.Toolboxes.Max_Little.fastdfa",
        sources=["pyhctsa/Toolboxes/Max_Little/ML_fastdfa_core.c"],
        include_dirs=["pyhctsa/Toolboxes/Max_Little", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    sampen = Extension(
        "pyhctsa.Toolboxes.physionet.sampen",
        sources=["pyhctsa/Toolboxes/physionet/sampen.c"],
        include_dirs=["pyhctsa/Toolboxes/physionet", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    close_returns = Extension(
        "pyhctsa.Toolboxes.Max_Little.close_returns",
        sources=["pyhctsa/Toolboxes/Max_Little/ML_close_ret.c"],
        include_dirs=["pyhctsa/Toolboxes/Max_Little", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    periodicity_wang = Extension(
        "pyhctsa.Toolboxes.c22.PD_PeriodicityWang",
        sources=[
            "pyhctsa/Toolboxes/c22/PD_PeriodicityWang.c",
            "pyhctsa/Toolboxes/c22/splinefit.c",
            "pyhctsa/Toolboxes/c22/stats.c",
            "pyhctsa/Toolboxes/c22/helper_functions.c",
        ],
        include_dirs=["pyhctsa/Toolboxes/c22", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    shannon = Extension(
        "pyhctsa.Toolboxes.Michael_Small.shannon",
        sources=["pyhctsa/Toolboxes/Michael_Small/MS_shannon.c"],
        include_dirs=["pyhctsa/Toolboxes/Michael_Small", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    return [periodicity_wang, close_returns, sampen, fastdfa, shannon]
