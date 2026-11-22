"""Substantial synthetic checks for the unvalidated raster geometry estimator."""

import copy
import json
import math
import unittest

from disc_morphology.geometry import classify_planes, measure_plane
from disc_morphology._spatial import pixel_to_lps


def length_mm(source, points):
    # Independent 3D LPS distance used to check exported 2D line endpoints.
    return math.dist(*(pixel_to_lps(source,*point) for point in points))


def frame(**changes):
    result = {"rows": 40, "columns": 40, "spacing": [1.0, 1.0],
              "position": [12.0, -20.0, 4.0],
              "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
    result.update(changes)
    return result


def mask(*rectangles, rows=40, columns=40):
    result = [[0] * columns for _ in range(rows)]
    for left, top, right, bottom in rectangles:
        for y in range(top, bottom):
            for x in range(left, right):
                result[y][x] = 1
    return result


def measure(pixels, **kwargs):
    return measure_plane(kwargs.pop("geometry", frame()), pixels,
                         kwargs.pop("baseline", [[5, 9.5], [35, 9.5]]),
                         kwargs.pop("outside", [20, 30]),
                         plane=kwargs.pop("plane", "axial"), **kwargs)


def protrusion(plane="axial"):
    return measure(mask((10, 10, 30, 11), (16, 11, 24, 20)), plane=plane)


def extrusion(plane="axial"):
    return measure(mask((17, 10, 23, 15), (10, 15, 30, 24)), plane=plane)


class DiscMorphologyGeometryTests(unittest.TestCase):
    def test_wide_base_and_narrow_dome_have_separate_corresponding_widths(self):
        result = protrusion()
        self.assertTrue(result["geometry_measurable"])
        self.assertEqual(result["candidate_base_width_mm"], 20)
        self.assertEqual(result["candidate_displaced_width_mm"], 8)
        self.assertEqual(result["candidate_depth_mm"], 10)
        self.assertEqual(result["candidate_ratio"], .4)
        self.assertEqual(result["geometric_candidate"], "protrusion")
        # A global mask diameter would necessarily include the 20-mm base and
        # make a protrusion hypothesis impossible. D explicitly excludes it.
        self.assertLess(result["candidate_displaced_width_mm"], result["candidate_base_width_mm"])
        self.assertFalse(result["estimator"]["clinically_validated"])

    def test_narrow_neck_and_wide_dome_support_extrusion_geometry(self):
        result = extrusion()
        self.assertEqual(result["candidate_base_width_mm"], 6)
        self.assertEqual(result["candidate_displaced_width_mm"], 20)
        self.assertEqual(result["candidate_depth_mm"], 14)
        self.assertEqual(result["geometric_candidate"], "extrusion")
        self.assertAlmostEqual(result["candidate_ratio"], 20 / 6)
        for name, metric in (("base", "candidate_base_width_mm"),
                             ("displaced", "candidate_displaced_width_mm"),
                             ("depth", "candidate_depth_mm")):
            self.assertAlmostEqual(length_mm(frame(), result["segments_xy"][name]), result[metric])

    def test_raster_resolution_and_user_uncertainty_prevent_borderline_classification(self):
        pixels = mask((10, 10, 30, 11), (11, 11, 29, 22))
        result = measure(pixels, uncertainty_mm=0)
        self.assertAlmostEqual(result["uncertainty_mm"], math.sqrt(2))
        self.assertEqual(result["geometric_candidate"], "indeterminate")
        self.assertEqual(measure(mask((10, 10, 30, 11), (16, 11, 24, 20)),
                                 uncertainty_mm=8)["geometric_candidate"], "indeterminate")

    def test_scanlines_preserve_gaps_instead_of_using_global_outer_edges(self):
        # Connected at the base, then separated arms: D is one arm's width,
        # not the 20-mm span between their far outer edges.
        result = measure(mask((10, 10, 30, 11), (10, 11, 14, 20), (26, 11, 30, 20)))
        self.assertEqual(result["connected_components"], 1)
        self.assertEqual(result["candidate_displaced_width_mm"], 4)
        self.assertEqual(result["geometric_candidate"], "protrusion")

    def test_oblique_dicom_orientation_and_translation_preserve_mm(self):
        diagonal = 1 / math.sqrt(2)
        orientation = [diagonal, diagonal, 0,
                       -math.cos(.4) * diagonal, math.cos(.4) * diagonal, math.sin(.4)]
        pixels = mask((10, 10, 30, 11), (16, 11, 24, 20))
        original = measure(pixels, geometry=frame(spacing=[1.5, .7]))
        rotated = measure(pixels, geometry=frame(spacing=[1.5, .7], orientation=orientation,
                                                position=[1e20, -1e20, 1e20]))
        for key in ("candidate_base_width_mm", "candidate_displaced_width_mm",
                    "candidate_depth_mm", "candidate_ratio", "uncertainty_mm"):
            self.assertAlmostEqual(original[key], rotated[key])
        self.assertAlmostEqual(original["candidate_base_width_mm"], 14)
        self.assertAlmostEqual(original["candidate_displaced_width_mm"], 5.6)
        self.assertAlmostEqual(original["candidate_depth_mm"], 15)

    def test_rotated_baseline_with_unequal_spacing_is_invariant_under_transposition(self):
        pixels = mask((10, 10, 21, 25), rows=34, columns=32)
        geometry = frame(rows=34, columns=32, spacing=[2, .5])
        first = measure(pixels, geometry=geometry, baseline=[[8, 8], [20, 16]], outside=[10, 25])
        transposed = [list(row) for row in zip(*pixels)]
        second = measure(transposed, geometry=frame(rows=32, columns=34, spacing=[.5, 2]),
                         baseline=[[8, 8], [16, 20]], outside=[25, 10])
        for key in ("candidate_base_width_mm", "candidate_displaced_width_mm",
                    "candidate_depth_mm", "candidate_ratio", "resolution_band_mm"):
            self.assertIsNotNone(first[key])
            self.assertAlmostEqual(first[key], second[key], places=6)
        for name, key in (("base", "candidate_base_width_mm"),
                          ("displaced", "candidate_displaced_width_mm"),
                          ("depth", "candidate_depth_mm")):
            self.assertAlmostEqual(length_mm(geometry, first["segments_xy"][name]), first[key])

    def test_empty_disconnected_clipped_or_tiny_mask_never_establishes_absence_or_type(self):
        for pixels, flag in (
            (mask(), "empty_mask"),
            (mask((17, 10, 23, 15), (10, 17, 30, 24)), "disconnected_mask"),
            (mask((0, 10, 23, 20)), "touches_frame_edge"),
            (mask((20, 10, 21, 12)), "too_few_mask_pixels"),
        ):
            with self.subTest(flag=flag):
                result = measure(pixels)
                self.assertFalse(result["geometry_measurable"])
                self.assertEqual(result["geometric_candidate"], "indeterminate")
                self.assertIn(flag, result["quality_flags"])
                self.assertFalse(result["estimator"]["absence_assessed"])
                self.assertFalse(result["estimator"]["sequestration_assessed"])
                json.dumps(result, allow_nan=False)
        empty = measure(mask())
        self.assertIsNone(empty["candidate_base_width_mm"])
        self.assertIsNone(empty["candidate_displaced_width_mm"])
        self.assertIsNone(empty["candidate_depth_mm"])

    def test_missing_base_and_first_band_approximation_are_explicitly_unresolved(self):
        missing = measure(mask((10, 15, 30, 24)))
        self.assertIn("missing_base", missing["quality_flags"])
        self.assertIsNone(missing["candidate_base_width_mm"])
        near = measure(mask((10, 11, 30, 24)), baseline=[[5, 10], [35, 10]])
        self.assertEqual(near["base_method"], "first_resolution_band_approximation")
        self.assertIn("base_not_intersecting_reference", near["quality_flags"])
        self.assertIsNotNone(near["candidate_base_width_mm"])
        self.assertEqual(near["geometric_candidate"], "indeterminate")

    def test_a_single_unresolved_pixel_band_does_not_supply_displaced_width(self):
        result = measure(mask((10, 10, 30, 11)))
        self.assertIn("no_resolved_displaced_tissue", result["quality_flags"])
        self.assertIsNone(result["candidate_displaced_width_mm"])
        self.assertEqual(result["geometric_candidate"], "indeterminate")

    def test_inputs_are_not_mutated_and_baseline_endpoint_order_does_not_change_result(self):
        pixels = mask((10, 10, 30, 11), (16, 11, 24, 20))
        geometry, baseline, outside = frame(), [[5, 9.5], [35, 9.5]], [20, 30]
        snapshot = copy.deepcopy((pixels, geometry, baseline, outside))
        original = measure_plane(geometry, pixels, baseline, outside, plane="axial")
        reverse = measure_plane(geometry, pixels, baseline[::-1], outside, plane="axial")
        self.assertEqual(snapshot, (pixels, geometry, baseline, outside))
        for key in ("candidate_base_width_mm", "candidate_displaced_width_mm", "candidate_depth_mm"):
            self.assertAlmostEqual(original[key], reverse[key])

    def test_invalid_masks_geometry_reference_and_uncertainty_raise(self):
        invalid = [dict(mask_rows=[]), dict(mask_rows=[[0]]),
                   dict(mask_rows=[[.0] * 40 for _ in range(40)]),
                   dict(frame=frame(spacing=[0, 1])), dict(frame=frame(spacing=[1e308, 1e308])),
                   dict(frame=frame(position=[0, 0, math.nan])),
                   dict(baseline_xy=[[5, 10], [5, 10]]),
                   dict(baseline_xy=[[5, 10], [math.inf, 10]]),
                   dict(baseline_xy=[[5, 10], [40, 10]]), dict(outside_xy=[10, 9.5]),
                   dict(plane="coronal"), dict(uncertainty_mm=-1),
                   dict(uncertainty_mm=math.nan), dict(uncertainty_mm=True)]
        for changes in invalid:
            args = dict(frame=frame(), mask_rows=mask(), baseline_xy=[[5, 9.5], [35, 9.5]],
                        outside_xy=[20, 30], plane="axial")
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                measure_plane(**args)


class DiscMorphologyCombinationTests(unittest.TestCase):
    def test_unreviewed_geometry_retains_candidate_but_blocks_clinical_classification(self):
        result = classify_planes([protrusion(), protrusion("sagittal")])
        self.assertEqual(result["geometric_candidate"], "protrusion")
        self.assertEqual(result["clinical_classification"], "indeterminate")
        self.assertTrue(result["both_planes_measurable"])
        self.assertIn("anatomy_unreviewed", result["quality_flags"])

    def test_extrusion_in_either_plane_wins_candidate_but_missing_plane_stays_unreviewed(self):
        for records in ([extrusion()], [extrusion("sagittal")],
                        [protrusion(), extrusion("sagittal")]):
            with self.subTest(planes=[r["plane"] for r in records]):
                result = classify_planes(records)
                self.assertEqual(result["geometric_candidate"], "extrusion")
                self.assertEqual(result["clinical_classification"], "indeterminate")
        self.assertEqual(classify_planes([protrusion()])["geometric_candidate"], "indeterminate")

    def test_each_review_gate_and_synthetic_source_independently_blocks_release(self):
        records = [protrusion(), extrusion("sagittal")]
        gates = dict(numbering_reviewed=True, registration_reviewed=True, anatomy_reviewed=True)
        reviewed = classify_planes(records, **gates)
        self.assertEqual(reviewed["clinical_classification"], "extrusion")
        self.assertFalse(reviewed["clinically_validated"])
        for name in gates:
            changed = dict(gates, **{name: False})
            self.assertEqual(classify_planes(records, **changed)["clinical_classification"], "indeterminate")
        self.assertEqual(classify_planes(records, source_is_synthetic=True,
                                         **gates)["clinical_classification"], "indeterminate")
        self.assertEqual(classify_planes([extrusion()], **gates)["clinical_classification"], "indeterminate")

    def test_unresolved_plane_prevents_protrusion_and_duplicate_planes_are_rejected_as_evidence(self):
        unresolved = measure(mask(), plane="sagittal")
        for records in ([protrusion(), unresolved], [protrusion(), protrusion(), protrusion("sagittal")]):
            self.assertEqual(classify_planes(records)["geometric_candidate"], "indeterminate")
        self.assertFalse(classify_planes([protrusion(), unresolved])["absence_assessed"])

    def test_combiner_recomputes_geometry_instead_of_trusting_forged_label(self):
        forged = protrusion()
        forged["geometric_candidate"] = "extrusion"
        self.assertEqual(classify_planes([forged, protrusion("sagittal")])["geometric_candidate"], "protrusion")
        forged["candidate_displaced_width_mm"] = math.nan
        result = classify_planes([forged, protrusion("sagittal")])
        self.assertEqual(result["geometric_candidate"], "indeterminate")
        json.dumps(result, allow_nan=False)

    def test_combiner_requires_explicit_boolean_gates_and_known_plane_records(self):
        for records, changes in ((None, {}), ([{}], {}), ([{"plane": "coronal"}], {}),
                                 ([], {"numbering_reviewed": 1}),
                                 ([], {"source_is_synthetic": "false"})):
            with self.subTest(records=records, changes=changes), self.assertRaises(ValueError):
                classify_planes(records, **changes)


if __name__ == "__main__":
    unittest.main()
