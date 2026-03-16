from __future__ import annotations


"""Small real-kernel QBX dual-tree parity reproducer.

This script is intended for environments like `ipa` where a CPU OpenCL device
is available and the local editable stack has been installed. It exercises a
non-FFT QBX/sumpy path and compares classic and dual-tree QBX outputs.
"""

import importlib.metadata as importlib_metadata
import sys
from functools import partial

import numpy as np
import pyopencl as cl


_orig_version = importlib_metadata.version
importlib_metadata.version = lambda name: (
    "0" if name in {"sumpy", "pytential", "meshmode", "modepy"} else _orig_version(name)
)

sys.path.append("/home/xywei/Work/fmm/pytential/test")

from pytential.array_context import PyOpenCLArrayContext
from pytential import GeometryCollection, sym
from pytential.qbx import get_flat_strengths_from_densities
from pytential.qbx.fmm import (
    QBXSumpyTreeIndependentDataForWrangler,
    drive_dual_tree_fmm,
    drive_fmm,
)
from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
from sumpy.kernel import LaplaceKernel
from test_cost_model import get_lpot_source


def main() -> None:
    plat = next(
        p for p in cl.get_platforms() if p.name == "Portable Computing Language"
    )
    dev = next(d for d in plat.get_devices() if d.type & cl.device_type.CPU)
    queue = cl.CommandQueue(cl.Context([dev]))
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    lpot_source = get_lpot_source(actx, 2)
    places = GeometryCollection(lpot_source)
    density_discr = places.get_discretization(places.auto_source.geometry)
    geo_data = lpot_source.qbx_fmm_geometry_data(
        places,
        places.auto_source.geometry,
        target_discrs_and_qbx_sides=((density_discr, 1),),
    )

    base_kernel = LaplaceKernel(2)
    expansion_factory = lpot_source.expansion_factory
    mpole_expn_class = expansion_factory.get_multipole_expansion_class(base_kernel)
    local_expn_class = expansion_factory.get_local_expansion_class(base_kernel)
    try:
        qbx_local_expn_class = expansion_factory.get_qbx_local_expansion_class(
            base_kernel
        )
    except AttributeError:
        qbx_local_expn_class = local_expn_class

    m2l_factory = NonFFTM2LTranslationClassFactory()
    m2l_class = m2l_factory.get_m2l_translation_class(base_kernel, local_expn_class)

    tree_indep = QBXSumpyTreeIndependentDataForWrangler(
        actx,
        partial(mpole_expn_class, base_kernel),
        partial(local_expn_class, base_kernel, m2l_translation_override=m2l_class()),
        partial(qbx_local_expn_class, base_kernel),
        target_kernels=(base_kernel,),
        source_kernels=(base_kernel,),
    )
    wrangler = tree_indep.wrangler_cls(
        tree_indep,
        geo_data,
        np.float64,
        lpot_source.qbx_order,
        lpot_source.fmm_level_to_order,
        source_extra_kwargs={},
        kernel_extra_kwargs={},
        _use_target_specific_qbx=lpot_source._use_target_specific_qbx,
    )

    density = density_discr.zeros(actx) + 1
    flat = get_flat_strengths_from_densities(
        actx,
        places,
        lambda expr: density,
        [sym.var("sigma")],
        dofdesc=places.auto_source,
    )[0]

    classic_result = drive_fmm(actx, wrangler, (flat,))[0]
    dual_result = drive_dual_tree_fmm(actx, wrangler, (flat,))[0]

    classic_host = actx.to_numpy(classic_result)
    dual_host = actx.to_numpy(dual_result)

    print("classic nan", np.isnan(classic_host).sum())
    print("dual nan", np.isnan(dual_host).sum())
    print("allclose equal_nan", np.allclose(classic_host, dual_host, equal_nan=True))


if __name__ == "__main__":
    main()
