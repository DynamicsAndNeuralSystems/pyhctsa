import platform
import numpy as np
from setuptools import Extension

def _compile_args():
    if platform.system() == "Windows":
        # MSVC: no -std=c99; use defaults (C11-ish) and enable O2
        return ["/O2"]
    return ["-O3", "-fPIC", "-std=c99", "-ffast-math"]

def _strict_compile_args():
    # The TISEAN kernels turn floating-point values into discrete decisions:
    # d2 truncates log()-derived length-scale indices to int, poincare tests
    # samples against a crossing level, and lyap_r both truncates coordinates
    # into boxes and compares squared distances against a search radius, so a
    # reassociated or approximated expression moves a pair to the neighbouring
    # bin, drops a cut entirely, or picks a different nearest neighbour.
    # Keep IEEE semantics here (no -ffast-math).
    if platform.system() == "Windows":
        return ["/O2", "/fp:precise"]
    return ["-O2", "-fPIC", "-std=c99"]

def _libraries():
    return [] if platform.system() == "Windows" else ["m"]

def build_extensions():
    np_inc = np.get_include()

    fastdfa = Extension(
        "pyhctsa.toolboxes.Max_Little.fastdfa",
        sources=["pyhctsa/toolboxes/Max_Little/ML_fastdfa_core.c"],
        include_dirs=["pyhctsa/toolboxes/Max_Little", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    sampen = Extension(
        "pyhctsa.toolboxes.physionet.sampen",
        sources=["pyhctsa/toolboxes/physionet/sampen.c"],
        include_dirs=["pyhctsa/toolboxes/physionet", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    close_returns = Extension(
        "pyhctsa.toolboxes.Max_Little.close_returns",
        sources=["pyhctsa/toolboxes/Max_Little/ML_close_ret.c"],
        include_dirs=["pyhctsa/toolboxes/Max_Little", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    periodicity_wang = Extension(
        "pyhctsa.toolboxes.c22.PD_PeriodicityWang",
        sources=[
            "pyhctsa/toolboxes/c22/PD_PeriodicityWang.c",
            "pyhctsa/toolboxes/c22/splinefit.c",
            "pyhctsa/toolboxes/c22/stats.c",
            "pyhctsa/toolboxes/c22/helper_functions.c",
        ],
        include_dirs=["pyhctsa/toolboxes/c22", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    shannon = Extension(
        "pyhctsa.toolboxes.Michael_Small.shannon",
        sources=["pyhctsa/toolboxes/Michael_Small/MS_shannon.c"],
        include_dirs=["pyhctsa/toolboxes/Michael_Small", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    tisean_d2 = Extension(
        "pyhctsa.toolboxes.Tisean_3_0_1.d2",
        sources=["pyhctsa/toolboxes/Tisean_3_0_1/TS_d2.c"],
        include_dirs=["pyhctsa/toolboxes/Tisean_3_0_1", np_inc],
        extra_compile_args=_strict_compile_args(),
        libraries=_libraries(),
    )

    tisean_poincare = Extension(
        "pyhctsa.toolboxes.Tisean_3_0_1.poincare",
        sources=["pyhctsa/toolboxes/Tisean_3_0_1/TS_poincare.c"],
        include_dirs=["pyhctsa/toolboxes/Tisean_3_0_1", np_inc],
        extra_compile_args=_strict_compile_args(),
        libraries=_libraries(),
    )

    return [periodicity_wang, close_returns, sampen, fastdfa, shannon, tisean_d2,
            tisean_poincare]
