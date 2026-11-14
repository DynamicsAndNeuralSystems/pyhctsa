import platform
import numpy as np
from setuptools import Extension

def _compile_args():
    # IEEE semantics on every platform (no -ffast-math): the extensions turn floating-point
    # values into discrete decisions (template matches within a tolerance, peaks and troughs of
    # an autocorrelation function, length-scale indices truncated to int, samples tested against
    # a crossing level, coordinates hashed into grid boxes, distances compared with a search
    # radius), so a reassociated or approximated expression can move a pair to the neighbouring
    # bin or pick a different neighbour. -ffast-math also implies -ffinite-math-only, under
    # which a compiler may drop the isnan() checks the code relies on.
    if platform.system() == "Windows":
        return ["/O2", "/fp:precise"]
    return ["-O2", "-fPIC", "-std=c99"]

def _libraries():
    return [] if platform.system() == "Windows" else ["m"]

def build_extensions():
    np_inc = np.get_include()

    sampen = Extension(
        "pyhctsa.toolboxes.physionet.sampen",
        sources=["pyhctsa/toolboxes/physionet/sampen.c"],
        include_dirs=["pyhctsa/toolboxes/physionet", np_inc],
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

    tisean_d2 = Extension(
        "pyhctsa.toolboxes.Tisean_3_0_1.d2",
        sources=["pyhctsa/toolboxes/Tisean_3_0_1/TS_d2.c"],
        include_dirs=["pyhctsa/toolboxes/Tisean_3_0_1", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    tisean_poincare = Extension(
        "pyhctsa.toolboxes.Tisean_3_0_1.poincare",
        sources=["pyhctsa/toolboxes/Tisean_3_0_1/TS_poincare.c"],
        include_dirs=["pyhctsa/toolboxes/Tisean_3_0_1", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    tisean_false_nearest = Extension(
        "pyhctsa.toolboxes.Tisean_3_0_1.false_nearest",
        sources=["pyhctsa/toolboxes/Tisean_3_0_1/TS_false_nearest.c"],
        include_dirs=["pyhctsa/toolboxes/Tisean_3_0_1", np_inc],
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    )

    return [periodicity_wang, sampen, tisean_d2, tisean_poincare,
            tisean_false_nearest]
