from __future__ import annotations


import importlib.metadata as importlib_metadata
import os

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts


_orig_metadata_version = importlib_metadata.version


def _patched_metadata_version(name: str) -> str:
    if name in {"sumpy", "pytential", "meshmode", "modepy"}:
        return "0"
    return _orig_metadata_version(name)


importlib_metadata.version = _patched_metadata_version

import meshmode.mesh.generation as mgen

from pytential.array_context import PytestPyOpenCLArrayContextFactory
from test_cost_model import ConstantOneQBXExpansionWrangler, get_lpot_source


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
])


def _build_qbx_constant_one_results(actx):
    from boxtree.fmm import TreeIndependentDataForWrangler
    from pytential import GeometryCollection
    from pytential.qbx.fmm import drive_dual_tree_fmm, drive_fmm

    lpot_source = get_lpot_source(actx, 2)
    places = GeometryCollection(lpot_source)
    density_discr = places.get_discretization(places.auto_source.geometry)
    geo_data = lpot_source.qbx_fmm_geometry_data(
        places,
        places.auto_source.geometry,
        target_discrs_and_qbx_sides=((density_discr, 1),),
    )

    wrangler = ConstantOneQBXExpansionWrangler(
        TreeIndependentDataForWrangler(),
        geo_data,
        use_target_specific_qbx=False,
    )

    src_weights = np.ones(geo_data.tree().nsources)
    classic_result = drive_fmm(actx, wrangler, (src_weights,))
    dual_result = drive_dual_tree_fmm(actx, wrangler, (src_weights,))

    return geo_data, classic_result, dual_result


def _build_real_qbx_sumpy_results(actx):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from pytential import GeometryCollection, sym
    from pytential.qbx import QBXLayerPotentialSource, get_flat_strengths_from_densities
    from pytential.qbx.fmm import drive_dual_tree_fmm, drive_fmm
    from sumpy.kernel import LaplaceKernel

    target_order = 4
    qbx_order = 3
    mesh = mgen.make_curve_mesh(
        mgen.starfish, np.linspace(0, 1, 20), order=target_order
    )
    pre_density_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )
    qbx = QBXLayerPotentialSource(
        pre_density_discr,
        4 * target_order,
        qbx_order,
        fmm_order=qbx_order + 2,
        fmm_backend="sumpy",
        _expansions_in_tree_have_extent=True,
        target_association_tolerance=0.05,
    )

    places = GeometryCollection(qbx)
    density_discr = places.get_discretization(places.auto_source.geometry)
    geo_data = qbx.qbx_fmm_geometry_data(
        places,
        places.auto_source.geometry,
        target_discrs_and_qbx_sides=((density_discr, 1),),
    )

    knl = LaplaceKernel(2)
    tree_indep = qbx._tree_indep_data_for_wrangler(
        source_kernels=(knl,), target_kernels=(knl,)
    )
    wrangler = tree_indep.wrangler_cls(
        tree_indep,
        geo_data,
        np.float64,
        qbx.qbx_order,
        qbx.fmm_level_to_order,
        source_extra_kwargs={},
        kernel_extra_kwargs={},
        _use_target_specific_qbx=qbx._use_target_specific_qbx,
    )

    density = density_discr.zeros(actx) + 1
    flat_strength = get_flat_strengths_from_densities(
        actx,
        places,
        lambda expr: density,
        [sym.var("sigma")],
        dofdesc=places.auto_source,
    )[0]

    classic_result = drive_fmm(actx, wrangler, (flat_strength,))
    dual_result = drive_dual_tree_fmm(actx, wrangler, (flat_strength,))
    return classic_result, dual_result


def _build_real_qbx_sumpy_results_nonfft(actx):
    from functools import partial

    from pytential import GeometryCollection, sym
    from pytential.qbx import get_flat_strengths_from_densities
    from pytential.qbx.fmm import (
        QBXSumpyTreeIndependentDataForWrangler,
        drive_dual_tree_fmm,
        drive_fmm,
    )
    from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
    from sumpy.kernel import LaplaceKernel

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
    flat_strength = get_flat_strengths_from_densities(
        actx,
        places,
        lambda expr: density,
        [sym.var("sigma")],
        dofdesc=places.auto_source,
    )[0]

    classic_result = drive_fmm(actx, wrangler, (flat_strength,))
    dual_result = drive_dual_tree_fmm(actx, wrangler, (flat_strength,))
    return classic_result, dual_result


def _build_real_qbx_sumpy_results_fft(actx):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from pytential import GeometryCollection, sym
    from pytential.qbx import QBXLayerPotentialSource, get_flat_strengths_from_densities
    from pytential.qbx.fmm import drive_dual_tree_fmm, drive_fmm
    from sumpy.kernel import LaplaceKernel

    target_order = 4
    qbx_order = 3
    mesh = mgen.make_curve_mesh(
        mgen.starfish, np.linspace(0, 1, 20), order=target_order
    )
    pre_density_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )
    qbx = QBXLayerPotentialSource(
        pre_density_discr,
        4 * target_order,
        qbx_order,
        fmm_order=qbx_order + 2,
        fmm_backend="sumpy",
        _expansions_in_tree_have_extent=True,
        target_association_tolerance=0.05,
    )

    places = GeometryCollection(qbx)
    density_discr = places.get_discretization(places.auto_source.geometry)
    geo_data = qbx.qbx_fmm_geometry_data(
        places,
        places.auto_source.geometry,
        target_discrs_and_qbx_sides=((density_discr, 1),),
    )

    knl = LaplaceKernel(2)
    tree_indep = qbx._tree_indep_data_for_wrangler(
        source_kernels=(knl,), target_kernels=(knl,)
    )
    wrangler = tree_indep.wrangler_cls(
        tree_indep,
        geo_data,
        np.float64,
        qbx.qbx_order,
        qbx.fmm_level_to_order,
        source_extra_kwargs={},
        kernel_extra_kwargs={},
        _use_target_specific_qbx=qbx._use_target_specific_qbx,
    )

    density = density_discr.zeros(actx) + 1
    flat_strength = get_flat_strengths_from_densities(
        actx,
        places,
        lambda expr: density,
        [sym.var("sigma")],
        dofdesc=places.auto_source,
    )[0]

    classic_result = drive_fmm(actx, wrangler, (flat_strength,))
    dual_result = drive_dual_tree_fmm(actx, wrangler, (flat_strength,))
    return classic_result, dual_result


def _build_qbx_constant_one_stage_data(actx):
    from boxtree.dual_tree_fmm import DualTreeCompatibilityWrangler
    from boxtree.dual_tree_traversal import DualTreeTraversalEngine
    from boxtree.fmm import TreeIndependentDataForWrangler
    from pytential import GeometryCollection

    lpot_source = get_lpot_source(actx, 2)
    places = GeometryCollection(lpot_source)
    density_discr = places.get_discretization(places.auto_source.geometry)
    geo_data = lpot_source.qbx_fmm_geometry_data(
        places,
        places.auto_source.geometry,
        target_discrs_and_qbx_sides=((density_discr, 1),),
    )

    wrangler = ConstantOneQBXExpansionWrangler(
        TreeIndependentDataForWrangler(),
        geo_data,
        use_target_specific_qbx=False,
    )

    traversal = wrangler.trav
    tree = traversal.tree
    host_tree = (
        tree if isinstance(tree.box_centers, np.ndarray) else actx.to_numpy(tree)
    )
    src_weight_vecs = [wrangler.reorder_sources(np.ones(geo_data.tree().nsources))]

    classic_mpoles = wrangler.form_multipoles(
        actx,
        traversal.level_start_source_box_nrs,
        traversal.source_boxes,
        src_weight_vecs,
    )
    classic_mpoles = wrangler.coarsen_multipoles(
        actx,
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        classic_mpoles,
    )
    classic_locals = wrangler.multipole_to_local(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_siblings_starts,
        traversal.from_sep_siblings_lists,
        classic_mpoles,
    )
    classic_locals = classic_locals + wrangler.form_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_bigger_starts,
        traversal.from_sep_bigger_lists,
        src_weight_vecs,
    )
    classic_locals = wrangler.refine_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        classic_locals,
    )

    compat = DualTreeCompatibilityWrangler(
        wrangler.tree_indep,
        host_tree,
        DualTreeTraversalEngine(traversal.well_sep_is_n_away),
        wrangler,
    )
    dual_mpoles = compat.form_multipoles(
        actx,
        traversal.level_start_source_box_nrs,
        traversal.source_boxes,
        src_weight_vecs,
    )
    dual_mpoles = compat.coarsen_multipoles(
        actx,
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        dual_mpoles,
    )
    dual_locals = compat.local_expansion_zeros(actx)
    for pair_batch in compat.traversal_engine.walk(host_tree):
        if pair_batch.interaction_kind == "m2l":
            dual_locals = compat.multipole_to_local_batch(
                actx, pair_batch, dual_mpoles, dual_locals
            )
        elif pair_batch.interaction_kind == "p2l":
            dual_locals = compat.form_locals_batch(
                actx, pair_batch, src_weight_vecs, dual_locals
            )
    dual_locals = compat.finalize_multipole_to_local(actx, dual_mpoles, dual_locals)
    dual_locals = compat.refine_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        dual_locals,
    )

    dual_qbx_locals = dual_locals
    if geo_data.ncenters:
        dual_qbx_locals = wrangler.multipole_to_local(
            actx,
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            dual_mpoles,
        )
        dual_qbx_locals = dual_qbx_locals + wrangler.form_locals(
            actx,
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weight_vecs,
        )
        dual_qbx_locals = wrangler.refine_locals(
            actx,
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            dual_qbx_locals,
        )

    return {
        "classic_m2qbxl": wrangler.translate_box_multipoles_to_qbx_local(
            actx, classic_mpoles
        ),
        "dual_m2qbxl": wrangler.translate_box_multipoles_to_qbx_local(
            actx, dual_mpoles
        ),
        "classic_l2qbxl": wrangler.translate_box_local_to_qbx_local(
            actx, classic_locals
        ),
        "dual_l2qbxl": wrangler.translate_box_local_to_qbx_local(actx, dual_qbx_locals),
        "classic_locals": classic_locals,
        "dual_locals": dual_locals,
        "qbx_center_to_target_box": actx.to_numpy(geo_data.qbx_center_to_target_box()),
    }


@pytest.mark.opencl
def test_qbx_constant_one_dual_tree_smoke(actx_factory):
    actx = actx_factory()
    geo_data, classic_result, dual_result = _build_qbx_constant_one_results(actx)

    assert len(classic_result) == len(dual_result)
    for classic_part, dual_part in zip(classic_result, dual_result, strict=True):
        assert classic_part.shape == dual_part.shape
        assert np.all(dual_part >= 0)
        assert np.all(np.isfinite(dual_part))


@pytest.mark.opencl
def test_qbx_constant_one_dual_tree_matches_classic(actx_factory):
    actx = actx_factory()
    _geo_data, classic_result, dual_result = _build_qbx_constant_one_results(actx)

    assert len(classic_result) == len(dual_result)
    for classic_part, dual_part in zip(classic_result, dual_result, strict=True):
        assert np.array_equal(classic_part, dual_part)


@pytest.mark.opencl
def test_qbx_constant_one_dual_tree_matches_classic_m2qbxl(actx_factory):
    actx = actx_factory()
    stage_data = _build_qbx_constant_one_stage_data(actx)
    assert np.array_equal(stage_data["dual_m2qbxl"], stage_data["classic_m2qbxl"])


@pytest.mark.opencl
def test_qbx_constant_one_dual_tree_matches_classic_l2qbxl(actx_factory):
    actx = actx_factory()
    stage_data = _build_qbx_constant_one_stage_data(actx)
    assert np.array_equal(stage_data["dual_l2qbxl"], stage_data["classic_l2qbxl"])


@pytest.mark.opencl
def test_qbx_constant_one_dual_tree_matches_classic_qbx_box_locals(actx_factory):
    actx = actx_factory()
    stage_data = _build_qbx_constant_one_stage_data(actx)
    qbx_boxes = np.unique(stage_data["qbx_center_to_target_box"])

    assert np.array_equal(
        stage_data["dual_locals"][qbx_boxes],
        stage_data["classic_locals"][qbx_boxes],
    )


@pytest.mark.opencl
@pytest.mark.parametrize("use_target_specific_qbx", [False, True])
def test_qbx_constant_one_dual_tree_smoke_target_specific_variants(
    actx_factory, use_target_specific_qbx
):
    actx = actx_factory()

    from boxtree.fmm import TreeIndependentDataForWrangler
    from pytential import GeometryCollection
    from pytential.qbx.fmm import drive_dual_tree_fmm, drive_fmm

    lpot_source = get_lpot_source(actx, 2)
    places = GeometryCollection(lpot_source)
    density_discr = places.get_discretization(places.auto_source.geometry)
    geo_data = lpot_source.qbx_fmm_geometry_data(
        places,
        places.auto_source.geometry,
        target_discrs_and_qbx_sides=((density_discr, 1),),
    )

    wrangler = ConstantOneQBXExpansionWrangler(
        TreeIndependentDataForWrangler(),
        geo_data,
        use_target_specific_qbx=use_target_specific_qbx,
    )

    src_weights = np.ones(geo_data.tree().nsources)
    classic_result = drive_fmm(actx, wrangler, (src_weights,))
    dual_result = drive_dual_tree_fmm(actx, wrangler, (src_weights,))

    assert len(classic_result) == len(dual_result)
    for classic_part, dual_part in zip(classic_result, dual_result, strict=True):
        assert np.array_equal(classic_part, dual_part)


@pytest.mark.opencl
def test_qbx_sumpy_dual_tree_matches_classic_real_kernel_nonfft(actx_factory):
    if os.environ.get("PYTENTIAL_RUN_REAL_QBX_DUAL_TREE") != "1":
        pytest.skip(
            "set PYTENTIAL_RUN_REAL_QBX_DUAL_TREE=1 to run real-kernel QBX parity"
        )

    actx = actx_factory()
    platform_name = actx.queue.device.platform.name
    device_type = actx.queue.device.type
    if platform_name != "Portable Computing Language" or not (device_type & 2):
        pytest.skip("run this regression on the POCL CPU device")

    classic_result, dual_result = _build_real_qbx_sumpy_results_nonfft(actx)

    assert len(classic_result) == len(dual_result)
    for classic_part, dual_part in zip(classic_result, dual_result, strict=True):
        classic_part = actx.to_numpy(classic_part)
        dual_part = actx.to_numpy(dual_part)
        assert np.allclose(
            classic_part, dual_part, rtol=1e-12, atol=1e-12, equal_nan=True
        )


@pytest.mark.opencl
def test_qbx_sumpy_dual_tree_matches_classic_real_kernel_fft(actx_factory):
    if os.environ.get("PYTENTIAL_RUN_REAL_QBX_DUAL_TREE_FFT") != "1":
        pytest.skip(
            "set PYTENTIAL_RUN_REAL_QBX_DUAL_TREE_FFT=1 to run FFT-based real-kernel QBX parity"
        )

    actx = actx_factory()
    classic_result, dual_result = _build_real_qbx_sumpy_results_fft(actx)

    assert len(classic_result) == len(dual_result)
    for classic_part, dual_part in zip(classic_result, dual_result, strict=True):
        classic_part = actx.to_numpy(classic_part)
        dual_part = actx.to_numpy(dual_part)
        assert np.allclose(
            classic_part, dual_part, rtol=1e-12, atol=1e-12, equal_nan=True
        )


@pytest.mark.opencl
def test_qbx_operator_level_dual_tree_matches_classic_constant_one(
    actx_factory, monkeypatch
):
    actx = actx_factory()

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )
    from pytential import GeometryCollection, bind, sym
    from pytential.qbx import QBXLayerPotentialSource
    from pytential.qbx import fmm as qbx_fmm_mod
    from sumpy.kernel import LaplaceKernel

    target_order = 3
    qbx_order = 2
    mesh = mgen.make_curve_mesh(
        mgen.starfish, np.linspace(0, 1, 12), order=target_order
    )
    pre_density_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )
    qbx = QBXLayerPotentialSource(
        pre_density_discr,
        4 * target_order,
        qbx_order,
        fmm_order=qbx_order + 2,
        fmm_backend="sumpy",
        _expansions_in_tree_have_extent=True,
        target_association_tolerance=0.05,
    )
    places = GeometryCollection(qbx)
    density_discr = places.get_discretization(places.auto_source.geometry)
    sigma = density_discr.zeros(actx) + 1
    op = sym.S(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=+1)

    classic_result = bind(places, op)(actx, sigma=sigma)

    original_drive_fmm = qbx_fmm_mod.drive_fmm
    monkeypatch.setattr(qbx_fmm_mod, "drive_fmm", qbx_fmm_mod.drive_dual_tree_fmm)
    try:
        dual_result = bind(places, op)(actx, sigma=sigma)
    finally:
        monkeypatch.setattr(qbx_fmm_mod, "drive_fmm", original_drive_fmm)

    assert np.allclose(
        actx.to_numpy(classic_result),
        actx.to_numpy(dual_result),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
