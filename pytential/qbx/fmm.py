from __future__ import annotations


__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from boxtree.dual_tree_fmm import DualTreeCompatibilityWrangler
from boxtree.dual_tree_traversal import DualTreeTraversalEngine
from pytools import ProcessLogger, log_process, memoize_method
from sumpy.fmm import (
    FMMLevelToOrder,
    LocalExpansionFromOrderFactory,
    MultipoleExpansionFromOrderFactory,
    SumpyExpansionWrangler,
    SumpyTreeIndependentDataForWrangler,
)

from pytential.qbx.interactions import L2QBXL, M2QBXL, QBXL2P, P2QBXLFromCSR


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from arraycontext import Array

    from pytential.array_context import PyOpenCLArrayContext
    from pytential.qbx.geometry import QBXFMMGeometryData

logger = logging.getLogger(__name__)


def _polyval_ascending(coeffs: np.ndarray, x: float) -> complex:
    value: complex = 0
    for coeff in coeffs[::-1]:
        value = value * x + coeff
    return value


def _fit_pade_from_samples(
    sample_t: np.ndarray,
    sample_values: np.ndarray,
    numerator_degree: int,
    denominator_degree: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    nunknowns = numerator_degree + 1 + denominator_degree
    if sample_t.size < nunknowns:
        return None

    vand_num = np.vander(sample_t, N=numerator_degree + 1, increasing=True)
    vand_den = np.vander(sample_t, N=denominator_degree + 1, increasing=True)[:, 1:]
    system_matrix = np.hstack([vand_num, -sample_values[:, np.newaxis] * vand_den])

    try:
        coeff_vec, *_ = np.linalg.lstsq(system_matrix, sample_values, rcond=None)
    except np.linalg.LinAlgError:
        return None

    numerator = coeff_vec[: numerator_degree + 1]
    denominator = np.concatenate([
        np.array([1], dtype=coeff_vec.dtype),
        coeff_vec[numerator_degree + 1 :],
    ])

    return numerator, denominator


def _chebyshev_nodes_01(n: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.pi * (np.arange(n) + 0.5) / n))


__doc__ = """
.. autoclass:: QBXSumpyTreeIndependentDataForWrangler

.. autoclass:: QBXExpansionWrangler

.. autofunction:: drive_fmm
"""


# {{{ sumpy expansion wrangler


class QBXSumpyTreeIndependentDataForWrangler(SumpyTreeIndependentDataForWrangler):
    qbx_local_expansion_factory: LocalExpansionFromOrderFactory

    def __init__(
        self,
        actx: PyOpenCLArrayContext,
        multipole_expansion_factory: MultipoleExpansionFromOrderFactory,
        local_expansion_factory: LocalExpansionFromOrderFactory,
        qbx_local_expansion_factory: LocalExpansionFromOrderFactory,
        target_kernels,
        source_kernels,
    ):
        super().__init__(
            actx,
            multipole_expansion_factory,
            local_expansion_factory,
            target_kernels=target_kernels,
            source_kernels=source_kernels,
        )

        self.qbx_local_expansion_factory = qbx_local_expansion_factory

    @memoize_method
    def qbx_local_expansion(self, order: int):
        if self.use_rscale is None:
            return self.qbx_local_expansion_factory(order)
        else:
            return self.qbx_local_expansion_factory(order, use_rscale=self.use_rscale)

    @memoize_method
    def p2qbxl(self, order: int):
        return P2QBXLFromCSR(
            self.qbx_local_expansion(order), kernels=self.source_kernels
        )

    @memoize_method
    def m2qbxl(self, source_order: int, target_order: int):
        return M2QBXL(
            self.multipole_expansion_factory(source_order),
            self.qbx_local_expansion_factory(target_order),
        )

    @memoize_method
    def l2qbxl(self, source_order: int, target_order: int):
        return L2QBXL(
            self.local_expansion_factory(source_order),
            self.qbx_local_expansion_factory(target_order),
        )

    @memoize_method
    def qbxl2p(self, order: int):
        return QBXL2P(self.qbx_local_expansion_factory(order), self.target_kernels)

    @property
    def wrangler_cls(self):
        return QBXExpansionWrangler


class QBXExpansionWrangler(SumpyExpansionWrangler):
    """A specialized implementation of the
    :class:`boxtree.fmm.ExpansionWranglerInterface` for the QBX FMM.
    The conventional ('point') FMM is carried out on a filtered
    set of targets
    (see :meth:`pytential.qbx.geometry.QBXFMMGeometryData.\
non_qbx_box_target_lists`),
    and thus all *non-QBX* potential arrays handled by this wrangler don't
    include all targets in the tree, just the non-QBX ones.

    .. rubric:: QBX-specific methods

    .. automethod:: form_global_qbx_locals

    .. automethod:: translate_box_local_to_qbx_local

    .. automethod:: eval_qbx_expansions
    """

    tree_indep: QBXSumpyTreeIndependentDataForWrangler  # pyright: ignore[reportIncompatibleVariableOverride]
    qbx_order: int
    geo_data: QBXFMMGeometryData
    qbx_out_of_disk_eval_mode: Literal["local", "pade"]
    qbx_out_of_disk_pade_sample_count: int | None
    qbx_out_of_disk_pade_smax: float
    qbx_out_of_disk_pade_denominator_tolerance: float

    def __init__(
        self,
        tree_indep: QBXSumpyTreeIndependentDataForWrangler,
        geo_data: QBXFMMGeometryData,
        dtype: np.dtype[np.inexact],
        qbx_order: int,
        fmm_level_to_order: FMMLevelToOrder,
        source_extra_kwargs,
        kernel_extra_kwargs,
        qbx_out_of_disk_eval_mode: Literal["local", "pade"] = "local",
        qbx_out_of_disk_pade_sample_count: int | None = None,
        qbx_out_of_disk_pade_smax: float = 0.9,
        qbx_out_of_disk_pade_denominator_tolerance: float = 1.0e-12,
        translation_classes_data=None,
        _use_target_specific_qbx=None,
    ):
        if _use_target_specific_qbx:
            raise ValueError("TSQBX is not implemented in sumpy")

        traversal = geo_data.traversal()
        base_kernel = tree_indep.get_base_kernel()
        if translation_classes_data is None and base_kernel.is_translation_invariant:
            from pytential.qbx.fmm import translation_classes_builder

            actx = geo_data._setup_actx

            translation_classes_data, _ = translation_classes_builder(actx)(
                actx, traversal, traversal.tree, is_translation_per_level=True
            )

        super().__init__(
            tree_indep,
            traversal,
            dtype,
            fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs,
            translation_classes_data=translation_classes_data,
        )

        self.qbx_order = qbx_order
        self.geo_data = geo_data
        self.using_tsqbx = False
        self.qbx_out_of_disk_eval_mode = qbx_out_of_disk_eval_mode
        self.qbx_out_of_disk_pade_sample_count = qbx_out_of_disk_pade_sample_count
        self.qbx_out_of_disk_pade_smax = qbx_out_of_disk_pade_smax
        self.qbx_out_of_disk_pade_denominator_tolerance = (
            qbx_out_of_disk_pade_denominator_tolerance
        )

    # {{{ data vector utilities

    @override
    def output_zeros(self, actx: PyOpenCLArrayContext):
        """This ought to be called ``non_qbx_output_zeros``, but since
        it has to override the superclass's behavior to integrate seamlessly,
        it needs to be called just :meth:`output_zeros`.
        """
        from pytools import obj_array

        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return obj_array.new_1d([
            actx.np.zeros(nqbtl.nfiltered_targets, dtype=self.dtype)
            for k in self.tree_indep.target_kernels
        ])

    def full_output_zeros(self, actx: PyOpenCLArrayContext):
        # The superclass generates a full field of zeros, for all
        # (not just non-QBX) targets.
        return super().output_zeros(actx)

    def qbx_local_expansion_zeros(self, actx: PyOpenCLArrayContext):
        order = self.qbx_order
        qbx_l_expn = self.tree_indep.qbx_local_expansion(order)

        return actx.np.zeros((self.geo_data.ncenters, len(qbx_l_expn)), self.dtype)

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        raise NotImplementedError(
            "reorder_potentials should not be called on a QBXExpansionWrangler"
        )

        # Because this is a multi-stage, more complicated process that combines
        # potentials from non-QBX targets and QBX targets, reordering takes
        # place in multiple stages below.

    # }}}

    # {{{ source/target dispatch

    # box_source_list_kwargs inherited from superclass

    @override
    def box_target_list_kwargs(self):
        # This only covers the non-QBX targets.

        nqbtl = self.geo_data.non_qbx_box_target_lists()
        return {
            "box_target_starts": nqbtl.box_target_starts,
            "box_target_counts_nonchild": (nqbtl.box_target_counts_nonchild),
            "targets": nqbtl.targets,
        }

    # }}}

    # {{{ qbx-related

    @log_process(logger)
    def form_global_qbx_locals(
        self, actx: PyOpenCLArrayContext, src_weight_vecs: Sequence[Array]
    ):
        local_exps = self.qbx_local_expansion_zeros(actx)

        geo_data = self.geo_data
        if len(geo_data.global_qbx_centers()) == 0:
            return local_exps

        traversal = geo_data.traversal()

        starts = traversal.neighbor_source_boxes_starts
        lists = traversal.neighbor_source_boxes_lists

        kwargs = dict(self.extra_kwargs)
        kwargs.update(self.box_source_list_kwargs())

        p2qbxl = self.tree_indep.p2qbxl(self.qbx_order)

        result = p2qbxl(
            actx,
            global_qbx_centers=geo_data.global_qbx_centers(),
            qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
            qbx_centers=geo_data.flat_centers(),
            qbx_expansion_radii=geo_data.flat_expansion_radii(),
            source_box_starts=starts,
            source_box_lists=lists,
            strengths=src_weight_vecs,
            qbx_expansions=local_exps,
            **kwargs,
        )
        assert local_exps is result

        return result

    @log_process(logger)
    def translate_box_multipoles_to_qbx_local(
        self, actx: PyOpenCLArrayContext, multipole_exps
    ):
        qbx_expansions = self.qbx_local_expansion_zeros(actx)

        geo_data = self.geo_data
        if geo_data.ncenters == 0:
            return qbx_expansions

        traversal = geo_data.traversal()

        wait_for = multipole_exps.events

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):
            m2qbxl = self.tree_indep.m2qbxl(
                self.level_orders[isrc_level], self.qbx_order
            )

            source_level_start_ibox, source_mpoles_view = (
                self.multipole_expansions_view(multipole_exps, isrc_level)
            )

            qbx_expansions_res = m2qbxl(
                actx,
                qbx_center_to_target_box_source_level=(
                    geo_data.qbx_center_to_target_box_source_level(isrc_level)
                ),
                centers=self.tree.box_centers,
                qbx_centers=geo_data.flat_centers(),
                qbx_expansion_radii=geo_data.flat_expansion_radii(),
                src_expansions=source_mpoles_view,
                src_base_ibox=source_level_start_ibox,
                qbx_expansions=qbx_expansions,
                src_box_starts=ssn.starts,
                src_box_lists=ssn.lists,
                src_rscale=self.level_to_rscale(isrc_level),
                wait_for=wait_for,
                **self.kernel_extra_kwargs,
            )

            assert qbx_expansions_res is qbx_expansions

        return qbx_expansions

    @log_process(logger)
    def translate_box_local_to_qbx_local(self, actx: PyOpenCLArrayContext, local_exps):
        qbx_expansions = self.qbx_local_expansion_zeros(actx)

        geo_data = self.geo_data

        if geo_data.ncenters == 0:
            return qbx_expansions

        trav = geo_data.traversal()
        wait_for = local_exps.events

        for isrc_level in range(geo_data.tree().nlevels):
            l2qbxl = self.tree_indep.l2qbxl(
                self.level_orders[isrc_level], self.qbx_order
            )

            target_level_start_ibox, target_locals_view = self.local_expansions_view(
                local_exps, isrc_level
            )

            qbx_expansions_res = l2qbxl(
                actx,
                qbx_center_to_target_box=geo_data.qbx_center_to_target_box(),
                target_boxes=trav.target_boxes,
                target_base_ibox=target_level_start_ibox,
                centers=self.tree.box_centers,
                qbx_centers=geo_data.flat_centers(),
                qbx_expansion_radii=geo_data.flat_expansion_radii(),
                expansions=target_locals_view,
                qbx_expansions=qbx_expansions,
                src_rscale=self.level_to_rscale(isrc_level),
                wait_for=wait_for,
                **self.kernel_extra_kwargs,
            )

            assert qbx_expansions_res is qbx_expansions

        return qbx_expansions

    @log_process(logger)
    def eval_qbx_expansions(self, actx: PyOpenCLArrayContext, qbx_expansions):
        pot = self.full_output_zeros(actx)

        geo_data = self.geo_data

        if len(geo_data.global_qbx_centers()) == 0:
            return pot

        ctt = geo_data.center_to_tree_targets()
        qbxl2p = self.tree_indep.qbxl2p(self.qbx_order)

        pot_res = qbxl2p(
            actx,
            qbx_centers=geo_data.flat_centers(),
            qbx_expansion_radii=geo_data.flat_expansion_radii(),
            global_qbx_centers=geo_data.global_qbx_centers(),
            center_to_targets_starts=ctt.starts,
            center_to_targets_lists=ctt.lists,
            targets=self.tree.targets,
            qbx_expansions=qbx_expansions,
            result=pot,
            **self.kernel_extra_kwargs,
        )

        for pot_i, pot_res_i in zip(pot, pot_res, strict=True):
            assert pot_i is pot_res_i

        if self.qbx_out_of_disk_eval_mode == "pade":
            return self._eval_qbx_expansions_pade_out_of_disk(
                actx,
                pot,
                qbx_expansions,
                qbxl2p,
                ctt,
            )

        return pot

    def _eval_qbx_expansions_pade_out_of_disk(
        self,
        actx: PyOpenCLArrayContext,
        pot,
        qbx_expansions,
        qbxl2p,
        ctt,
    ):
        geo_data = self.geo_data

        global_centers = np.asarray(
            actx.to_numpy(actx.thaw(geo_data.global_qbx_centers())), dtype=np.int64
        )
        if global_centers.size == 0:
            return pot

        centers = actx.to_numpy(actx.thaw(geo_data.flat_centers()))
        expansion_radii = actx.to_numpy(actx.thaw(geo_data.flat_expansion_radii()))
        targets = actx.to_numpy(actx.thaw(self.tree.targets))
        center_to_targets_starts = actx.to_numpy(actx.thaw(ctt.starts))
        center_to_targets_lists = actx.to_numpy(actx.thaw(ctt.lists))

        pair_centers: list[int] = []
        pair_targets: list[int] = []
        pair_ratios: list[float] = []

        for icenter in global_centers:
            radius = float(expansion_radii[icenter])
            if radius <= 0:
                continue

            start = int(center_to_targets_starts[icenter])
            end = int(center_to_targets_starts[icenter + 1])
            if start == end:
                continue

            center = centers[:, icenter]
            for ilist in range(start, end):
                itarget = int(center_to_targets_lists[ilist])
                displacement = targets[:, itarget] - center
                dist = float(np.linalg.norm(displacement))
                if dist <= radius:
                    continue

                pair_centers.append(int(icenter))
                pair_targets.append(itarget)
                pair_ratios.append(dist / radius)

        npairs = len(pair_targets)
        if npairs == 0:
            return pot

        nsamples = self.qbx_out_of_disk_pade_sample_count
        if nsamples is None:
            nsamples = max(16, 2 * self.qbx_order + 6)
        nsamples = int(nsamples)

        if nsamples < 4:
            return pot

        pade_degree = max(1, min(self.qbx_order // 2, (nsamples - 2) // 2))
        numerator_degree = pade_degree
        denominator_degree = pade_degree

        if numerator_degree + 1 + denominator_degree > nsamples:
            return pot

        sample_t = self.qbx_out_of_disk_pade_smax * _chebyshev_nodes_01(nsamples)

        center_pair_lists: list[list[int]] = [[] for _ in range(int(geo_data.ncenters))]
        for ipair, icenter in enumerate(pair_centers):
            center_pair_lists[icenter].append(ipair)

        sample_points = np.empty(
            (targets.shape[0], npairs * nsamples), dtype=targets.dtype
        )
        for ipair, (icenter, itarget, ratio) in enumerate(
            zip(pair_centers, pair_targets, pair_ratios, strict=True)
        ):
            center = centers[:, icenter]
            displacement = targets[:, itarget] - center
            scales = sample_t / ratio
            sample_points[:, ipair * nsamples : (ipair + 1) * nsamples] = (
                center[:, np.newaxis]
                + displacement[:, np.newaxis] * scales[np.newaxis, :]
            )

        sample_lists = np.empty(npairs * nsamples, dtype=center_to_targets_lists.dtype)
        sample_starts = np.empty(
            center_to_targets_starts.size, dtype=center_to_targets_starts.dtype
        )

        iwrite = 0
        for icenter, pair_list in enumerate(center_pair_lists):
            sample_starts[icenter] = iwrite
            for ipair in pair_list:
                ibase = ipair * nsamples
                sample_lists[iwrite : iwrite + nsamples] = np.arange(
                    ibase,
                    ibase + nsamples,
                    dtype=sample_lists.dtype,
                )
                iwrite += nsamples
        sample_starts[-1] = iwrite

        if iwrite != npairs * nsamples:
            return pot

        from pytools import obj_array

        sample_pot = obj_array.new_1d([
            actx.np.zeros(npairs * nsamples, dtype=self.dtype)
            for _ in self.tree_indep.target_kernels
        ])

        sample_pot = qbxl2p(
            actx,
            qbx_centers=geo_data.flat_centers(),
            qbx_expansion_radii=geo_data.flat_expansion_radii(),
            global_qbx_centers=geo_data.global_qbx_centers(),
            center_to_targets_starts=actx.from_numpy(sample_starts),
            center_to_targets_lists=actx.from_numpy(sample_lists),
            targets=actx.from_numpy(sample_points),
            qbx_expansions=qbx_expansions,
            result=sample_pot,
            **self.kernel_extra_kwargs,
        )

        sample_pot_host = [
            actx.to_numpy(sample_pot_i).reshape(npairs, nsamples)
            for sample_pot_i in sample_pot
        ]
        pot_host = [actx.to_numpy(pot_i) for pot_i in pot]

        for ipair, (itarget, ratio) in enumerate(
            zip(pair_targets, pair_ratios, strict=True)
        ):
            for iknl, sample_vals_by_pair in enumerate(sample_pot_host):
                fit = _fit_pade_from_samples(
                    sample_t,
                    sample_vals_by_pair[ipair],
                    numerator_degree,
                    denominator_degree,
                )
                if fit is None:
                    continue

                numerator_coeffs, denominator_coeffs = fit
                denominator_val = _polyval_ascending(denominator_coeffs, ratio)

                if (
                    abs(denominator_val)
                    <= self.qbx_out_of_disk_pade_denominator_tolerance
                ):
                    continue

                numerator_val = _polyval_ascending(numerator_coeffs, ratio)
                pade_val = numerator_val / denominator_val

                if not np.isfinite(np.real(pade_val)):
                    continue
                if np.iscomplexobj(pade_val) and not np.isfinite(np.imag(pade_val)):
                    continue

                if np.isrealobj(pot_host[iknl]):
                    pade_val = np.real(pade_val)

                pot_host[iknl][itarget] = pade_val

        new_pot = obj_array.new_1d([actx.from_numpy(pot_i) for pot_i in pot_host])
        for new_pot_i in new_pot:
            new_pot_i.finish()

        return new_pot

    @log_process(logger)
    def eval_target_specific_qbx_locals(
        self, actx: PyOpenCLArrayContext, src_weight_vecs
    ):
        return self.full_output_zeros(actx)

    # }}}


def translation_classes_builder(actx: PyOpenCLArrayContext):
    from pytools import memoize_in

    @memoize_in(actx, (QBXExpansionWrangler, translation_classes_builder))
    def make_container():
        from boxtree.translation_classes import TranslationClassesBuilder

        return TranslationClassesBuilder(actx)

    return make_container()


# }}}


# {{{ FMM top-level


def drive_fmm(actx: PyOpenCLArrayContext, expansion_wrangler, src_weight_vecs):
    """Top-level driver routine for the QBX fast multipole calculation.

    :arg expansion_wrangler: An object exhibiting the
        :class:`boxtree.fmm.ExpansionWranglerInterface`.
    :arg src_weight_vecs: A sequence of source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.

    Returns the potentials computed by *expansion_wrangler*.

    See also :func:`boxtree.fmm.drive_fmm`.
    """
    wrangler = expansion_wrangler
    geo_data = wrangler.geo_data
    traversal = wrangler.traversal
    tree = traversal.tree

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    fmm_proc = ProcessLogger(logger, "qbx fmm")

    src_weight_vecs = [wrangler.reorder_sources(weight) for weight in src_weight_vecs]

    # {{{ construct local multipoles

    mpole_exps = wrangler.form_multipoles(
        actx,
        traversal.level_start_source_box_nrs,
        traversal.source_boxes,
        src_weight_vecs,
    )

    # }}}

    # {{{ propagate multipoles upward

    mpole_exps = wrangler.coarsen_multipoles(
        actx,
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        mpole_exps,
    )

    # }}}

    # {{{ direct evaluation from neighbor source boxes ("list 1")

    non_qbx_potentials = wrangler.eval_direct(
        actx,
        traversal.target_boxes,
        traversal.neighbor_source_boxes_starts,
        traversal.neighbor_source_boxes_lists,
        src_weight_vecs,
    )

    # }}}

    # {{{ translate separated siblings' ("list 2") mpoles to local

    local_exps = wrangler.multipole_to_local(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_siblings_starts,
        traversal.from_sep_siblings_lists,
        mpole_exps,
    )

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result = wrangler.eval_multipoles(
        actx,
        traversal.target_boxes_sep_smaller_by_source_level,
        traversal.from_sep_smaller_by_level,
        mpole_exps,
    )

    non_qbx_potentials = non_qbx_potentials + mpole_result

    # assert that list 3 close has been merged into list 1
    assert traversal.from_sep_close_smaller_starts is None

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    local_result = wrangler.form_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_bigger_starts,
        traversal.from_sep_bigger_lists,
        src_weight_vecs,
    )

    local_exps = local_exps + local_result

    # assert that list 4 close has been merged into list 1
    assert traversal.from_sep_close_bigger_starts is None

    # }}}

    # {{{ propagate local_exps downward

    local_exps = wrangler.refine_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        local_exps,
    )

    # }}}

    # {{{ evaluate locals

    local_result = wrangler.eval_locals(
        actx, traversal.level_start_target_box_nrs, traversal.target_boxes, local_exps
    )

    non_qbx_potentials = non_qbx_potentials + local_result

    # }}}

    # {{{ wrangle qbx expansions

    # form_global_qbx_locals and eval_target_specific_qbx_locals are responsible
    # for the same interactions (directly evaluated portion of the potentials
    # via unified List 1).  Which one is used depends on the wrangler. If one of
    # them is unused the corresponding output entries will be zero.

    qbx_expansions = wrangler.form_global_qbx_locals(actx, src_weight_vecs)

    local_result = wrangler.translate_box_multipoles_to_qbx_local(actx, mpole_exps)

    qbx_expansions = qbx_expansions + local_result

    local_result = wrangler.translate_box_local_to_qbx_local(actx, local_exps)

    qbx_expansions = qbx_expansions + local_result

    qbx_potentials = wrangler.eval_qbx_expansions(actx, qbx_expansions)

    ts_result = wrangler.eval_target_specific_qbx_locals(actx, src_weight_vecs)

    qbx_potentials = qbx_potentials + ts_result

    # }}}

    # {{{ reorder potentials

    nqbtl = geo_data.non_qbx_box_target_lists()

    all_potentials_in_tree_order = wrangler.full_output_zeros(actx)

    for ap_i, nqp_i in zip(
        all_potentials_in_tree_order, non_qbx_potentials, strict=False
    ):
        ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

    all_potentials_in_tree_order += qbx_potentials

    def reorder_and_finalize_potentials(x):
        # "finalize" gives host FMMs (like FMMlib) a chance to turn the
        # potential back into a CL array.
        return wrangler.finalize_potentials(actx, x[tree.sorted_target_ids])

    from pytools import obj_array

    result = obj_array.vectorize(
        reorder_and_finalize_potentials, all_potentials_in_tree_order
    )

    # }}}

    fmm_proc.done()

    return result


def drive_dual_tree_fmm(
    actx: PyOpenCLArrayContext,
    expansion_wrangler,
    src_weight_vecs,
    dual_tree_engine: DualTreeTraversalEngine | None = None,
):
    wrangler = expansion_wrangler
    geo_data = wrangler.geo_data
    traversal = wrangler.traversal
    tree = traversal.tree
    host_tree = (
        tree if isinstance(tree.box_centers, np.ndarray) else actx.to_numpy(tree)
    )

    if dual_tree_engine is None:
        dual_tree_engine = DualTreeTraversalEngine(traversal.well_sep_is_n_away)

    compat = DualTreeCompatibilityWrangler(
        wrangler.tree_indep,
        host_tree,
        dual_tree_engine,
        wrangler,
    )

    fmm_proc = ProcessLogger(logger, "qbx dual-tree fmm")

    src_weight_vecs = [wrangler.reorder_sources(weight) for weight in src_weight_vecs]

    mpole_exps = compat.form_multipoles(
        actx,
        traversal.level_start_source_box_nrs,
        traversal.source_boxes,
        src_weight_vecs,
    )
    mpole_exps = compat.coarsen_multipoles(
        actx,
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        mpole_exps,
    )

    local_exps = compat.local_expansion_zeros(actx)
    non_qbx_potentials = compat.output_zeros(actx)

    for pair_batch in dual_tree_engine.walk(host_tree):
        if pair_batch.interaction_kind == "p2p":
            non_qbx_potentials = compat.eval_direct_batch(
                actx, pair_batch, src_weight_vecs, non_qbx_potentials
            )
        elif pair_batch.interaction_kind == "m2l":
            local_exps = compat.multipole_to_local_batch(
                actx, pair_batch, mpole_exps, local_exps
            )
        elif pair_batch.interaction_kind == "m2p":
            non_qbx_potentials = compat.eval_multipoles_batch(
                actx, pair_batch, mpole_exps, non_qbx_potentials
            )
        elif pair_batch.interaction_kind == "p2l":
            local_exps = compat.form_locals_batch(
                actx, pair_batch, src_weight_vecs, local_exps
            )
        else:
            raise ValueError(f"unknown interaction kind: {pair_batch.interaction_kind}")

    local_exps = compat.finalize_multipole_to_local(actx, mpole_exps, local_exps)

    local_exps = compat.refine_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        local_exps,
    )
    non_qbx_potentials = non_qbx_potentials + compat.eval_locals(
        actx, traversal.level_start_target_box_nrs, traversal.target_boxes, local_exps
    )

    qbx_expansions = wrangler.form_global_qbx_locals(actx, src_weight_vecs)
    qbx_expansions = qbx_expansions + wrangler.translate_box_multipoles_to_qbx_local(
        actx, mpole_exps
    )
    qbx_expansions = qbx_expansions + wrangler.translate_box_local_to_qbx_local(
        actx, local_exps
    )

    qbx_potentials = wrangler.eval_qbx_expansions(actx, qbx_expansions)
    qbx_potentials = qbx_potentials + wrangler.eval_target_specific_qbx_locals(
        actx, src_weight_vecs
    )

    nqbtl = geo_data.non_qbx_box_target_lists()
    all_potentials_in_tree_order = wrangler.full_output_zeros(actx)

    for ap_i, nqp_i in zip(
        all_potentials_in_tree_order, non_qbx_potentials, strict=False
    ):
        ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

    all_potentials_in_tree_order += qbx_potentials

    def reorder_and_finalize_potentials(x):
        return wrangler.finalize_potentials(actx, x[tree.sorted_target_ids])

    from pytools import obj_array

    result = obj_array.vectorize(
        reorder_and_finalize_potentials, all_potentials_in_tree_order
    )

    fmm_proc.done()
    return result


# }}}

# vim: foldmethod=marker
