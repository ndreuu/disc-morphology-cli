"""Research-only geometry of a supplied lesion mask on a native DICOM frame.

Small public API: ``measure_plane`` measures one supplied mask/reference line;
``classify_planes`` combines its axial and sagittal records conservatively.
Neither function segments tissue, identifies the anatomical disc margin, verifies
registration/numbering, or establishes a diagnosis. A reference line estimated by
another module remains an anatomical hypothesis until reviewed.

Coordinates are zero-based native pixel centers, as in ``measurements.py``.
The mask is the union of occupied pixel *cells*. B is its connected width at the
infinite reference line; D is its greatest connected, parallel cross-section
strictly beyond the first projected pixel-resolution band. D deliberately does
not include B. This raster width estimator is an operational approximation,
not a validated implementation of clinical disc nomenclature or 3-D morphology.
"""

import math

from ._measurements import _frame, _number, _points


__all__ = ["measure_plane", "classify_planes"]

_PLANES = ("axial", "sagittal")
_BLOCKING_FLAGS = {
    "empty_mask", "too_few_mask_pixels", "disconnected_mask",
    "touches_frame_edge", "base_not_intersecting_reference",
    "multiple_base_intervals", "missing_base", "no_resolved_displaced_tissue",
}


def _mask(frame, rows):
    if not isinstance(rows, (list, tuple)) or len(rows) != frame["rows"]:
        raise ValueError("Mask must have the native source rows and columns")
    pixels = set()
    for y, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != frame["columns"]:
            raise ValueError("Mask must have the native source rows and columns")
        for x, value in enumerate(row):
            if type(value) not in (bool, int) or value not in (0, 1):
                raise ValueError("Mask values must be booleans or integer zero/one")
            if value:
                pixels.add((x, y))
    return pixels


def _components(pixels):
    remaining, count = set(pixels), 0
    while remaining:
        count += 1
        todo = [remaining.pop()]
        while todo:
            x, y = todo.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    todo.append(neighbor)
    return count


def _coordinates(frame, baseline_xy, outside_xy):
    """Build a true Euclidean 2-D basis from the DICOM directions and spacing."""
    baseline = _points(frame, baseline_xy, count=2)
    outside = _points(frame, [outside_xy], count=1)[0]
    column = [v * frame["spacing"][1] for v in frame["orientation"][:3]]
    row = [v * frame["spacing"][0] for v in frame["orientation"][3:]]
    column_length = math.hypot(*column)
    row_along = math.fsum(a * (b / column_length) for a, b in zip(row, column))
    row_perpendicular = math.hypot(*(r - row_along * (c / column_length)
                                     for r, c in zip(row, column)))
    if not all(math.isfinite(v) and v > 0
               for v in (column_length, row_perpendicular)) or not math.isfinite(row_along):
        raise ValueError("Mask geometry overflows physical coordinates")

    def physical(point):
        x, y = point
        result = (x * column_length + y * row_along, y * row_perpendicular)
        if not all(math.isfinite(v) for v in result):
            raise ValueError("Mask geometry overflows physical coordinates")
        return result

    origin, end = [physical(point) for point in baseline]
    delta = (end[0] - origin[0], end[1] - origin[1])
    length = math.hypot(*delta)
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Reference line must have positive finite physical length")
    tangent = (delta[0] / length, delta[1] / length)
    normal = (-tangent[1], tangent[0])
    outside_physical = physical(outside)
    side = math.fsum((v - o) * n for v, o, n in zip(outside_physical, origin, normal))
    if not math.isfinite(side) or abs(side) <= 1e-9 * max(1.0, length):
        raise ValueError("Outside point must identify one side of the reference line")
    if side < 0:
        normal = (-normal[0], -normal[1])

    def uv(point):
        p = physical(point)
        try:
            result = tuple(math.fsum((v - o) * axis for v, o, axis in zip(p, origin, unit))
                           for unit in (tangent, normal))
        except OverflowError:
            raise ValueError("Mask geometry overflows reference coordinates") from None
        if not all(math.isfinite(v) for v in result):
            raise ValueError("Mask geometry overflows reference coordinates")
        return result

    def xy(u, v):
        px = origin[0] + u * tangent[0] + v * normal[0]
        py = origin[1] + u * tangent[1] + v * normal[1]
        y = py / row_perpendicular
        x = (px - y * row_along) / column_length
        if not all(math.isfinite(value) for value in (x, y)):
            raise ValueError("Mask overlay overflows native coordinates")
        return [x, y]

    band = abs(normal[0] * column_length) + abs(normal[0] * row_along
                                                        + normal[1] * row_perpendicular)
    resolution = math.hypot(column_length, math.hypot(row_along, row_perpendicular))
    if not all(math.isfinite(v) and v > 0 for v in (band, resolution)):
        raise ValueError("Mask geometry overflows resolution calculations")
    return uv, xy, band, resolution


def _boundary(pixels, uv):
    edges = []
    for x, y in sorted(pixels):
        left, right, top, bottom = x - .5, x + .5, y - .5, y + .5
        for neighbor, a, b in (
            ((x - 1, y), (left, top), (left, bottom)),
            ((x + 1, y), (right, top), (right, bottom)),
            ((x, y - 1), (left, top), (right, top)),
            ((x, y + 1), (left, bottom), (right, bottom)),
        ):
            if neighbor not in pixels:
                edges.append((uv(a), uv(b)))
    return edges


def _sections(edges, height, epsilon):
    """Exact scan-line intervals in the pixel-cell union, preserving gaps."""
    crossings, intervals = [], []
    for a, b in edges:
        if abs(a[1] - b[1]) <= epsilon:
            if abs(height - a[1]) <= epsilon:
                intervals.append(tuple(sorted((a[0], b[0]))))
        elif ((a[1] <= height < b[1]) or (b[1] <= height < a[1])):
            crossings.append(a[0] + (height - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
    crossings.sort()
    # Shared native cell vertices have identical transforms; the half-open rule
    # counts each crossing once, including concave contours and holes.
    for i in range(0, len(crossings) - 1, 2):
        intervals.append((crossings[i], crossings[i + 1]))
    intervals.sort()
    merged = []
    for left, right in intervals:
        if right - left <= epsilon:
            continue
        if merged and left <= merged[-1][1] + epsilon:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def _widest(edges, heights, epsilon):
    heights = sorted(set(heights))
    middle = heights[0] + (heights[-1] - heights[0]) / 2 if heights else 0.0
    best = None
    for height in heights:
        for left, right in _sections(edges, height, epsilon):
            if (best is None or right - left > best[1] - best[0] + epsilon
                    or (abs((right - left) - (best[1] - best[0])) <= epsilon
                        and abs(height - middle) < abs(best[2] - middle))):
                best = (left, right, height)
    return best


def _hypothesis(base, displaced, uncertainty):
    if base is None or displaced is None:
        return "indeterminate"
    if displaced - base > 2 * uncertainty:
        return "extrusion"
    if base - displaced > 2 * uncertainty:
        return "protrusion"
    return "indeterminate"


def measure_plane(frame, mask_rows, baseline_xy, outside_xy, *, plane,
                  uncertainty_mm=None):
    """Return JSON-safe calibrated measurements of a supplied 2-D candidate.

    ``baseline_xy`` contains two distinct native points defining an infinite
    hypothesized disc-space margin. ``outside_xy`` chooses its outward side.
    ``plane`` is ``axial`` or ``sagittal``. Invalid inputs raise ``ValueError``;
    absent/unresolved tissue returns null quantities and explicit quality flags.

    ``uncertainty_mm`` is a nonnegative *per-length* uncertainty floor. Effective
    uncertainty is never below a native pixel diagonal. A geometric hypothesis
    requires nonoverlap of the B±uncertainty and D±uncertainty intervals, and
    satisfactory mask geometry. This bound captures raster resolution only;
    it does not quantify segmentation or anatomical-reference error.
    """
    if plane not in _PLANES:
        raise ValueError("Plane must be axial or sagittal")
    geometry = _frame(frame)
    pixels = _mask(geometry, mask_rows)
    uv, xy, band, resolution = _coordinates(geometry, baseline_xy, outside_xy)
    requested_uncertainty = None if uncertainty_mm is None else _number(uncertainty_mm)
    if requested_uncertainty is not None and requested_uncertainty < 0:
        raise ValueError("Uncertainty must be nonnegative millimeters")
    uncertainty = max(resolution, requested_uncertainty or 0.0)
    epsilon = max(1e-10, resolution * 1e-9)
    count = _components(pixels)
    flags = []
    if not pixels:
        flags.append("empty_mask")
    if len(pixels) < 6:
        flags.append("too_few_mask_pixels")
    if count > 1:
        flags.append("disconnected_mask")
    if any(x in (0, geometry["columns"] - 1) or y in (0, geometry["rows"] - 1)
           for x, y in pixels):
        flags.append("touches_frame_edge")
    edges = _boundary(pixels, uv)
    vertices = [point for edge in edges for point in edge]
    levels = sorted(set(point[1] for point in vertices))
    sample_heights = []
    # Cross-sectional widths are piecewise linear between boundary-vertex
    # heights. Endpoints and one-sided limits cover their maxima without
    # replacing a connected chord by the global mask/convex-hull diameter.
    for height in levels:
        sample_heights.extend((height - 2 * epsilon, height, height + 2 * epsilon))
    sample_heights.extend((a + (b - a) / 2 for a, b in zip(levels, levels[1:])))
    base_intervals = _sections(edges, 0.0, epsilon)
    base_method = "reference_line_intersection" if base_intervals else None
    if len(base_intervals) > 1:
        flags.append("multiple_base_intervals")
    base = max(((a, b, 0.0) for a, b in base_intervals),
               key=lambda section: section[1] - section[0], default=None)
    if base is None:
        # Report a clearly flagged band approximation when the raster nearly
        # reaches the line; it cannot support a morphology hypothesis.
        base = _widest(edges, [h for h in sample_heights if 0 <= h <= band], epsilon)
        if base is not None:
            flags.append("base_not_intersecting_reference")
            base_method = "first_resolution_band_approximation"
        else:
            flags.append("missing_base")
    displaced = _widest(edges, [band + 2 * epsilon]
                        + [h for h in sample_heights if h > band + epsilon], epsilon)
    if displaced is None:
        flags.append("no_resolved_displaced_tissue")
    farthest = max(vertices, key=lambda point: (point[1], -abs(point[0])), default=None)
    depth = farthest[1] if farthest and farthest[1] > epsilon else None
    base_width = base[1] - base[0] if base else None
    displaced_width = displaced[1] - displaced[0] if displaced else None
    ratio = displaced_width / base_width if base_width and displaced_width else None
    if any(value is not None and not math.isfinite(value)
           for value in (base_width, displaced_width, depth, ratio)):
        raise ValueError("Candidate measurements overflow finite physical quantities")
    measurable = not any(flag in _BLOCKING_FLAGS for flag in flags)
    candidate = _hypothesis(base_width, displaced_width, uncertainty) if measurable else "indeterminate"

    def segment(section):
        return [xy(section[0], section[2]), xy(section[1], section[2])] if section else None

    return {
        "schema_version": "1.0", "plane": plane,
        "status": "measured" if measurable else "indeterminate",
        "geometry_measurable": measurable, "geometric_candidate": candidate,
        "candidate_base_width_mm": base_width,
        "candidate_displaced_width_mm": displaced_width,
        "candidate_depth_mm": depth,
        "candidate_ratio": ratio,
        "uncertainty_mm": uncertainty, "requested_uncertainty_mm": requested_uncertainty,
        "resolution_band_mm": band, "mask_pixels": len(pixels),
        "connected_components": count, "base_method": base_method,
        "quality_flags": flags,
        "segments_xy": {
            "base": segment(base), "displaced": segment(displaced),
            "depth": [xy(farthest[0], 0.0), xy(*farthest)] if depth else None,
        },
        "estimator": {
            "name": "native_pixel_cell_parallel_chords_v1",
            "clinically_validated": False,
            "reference_anatomy_reviewed": False,
            "displaced_width_definition": "maximum_connected_parallel_chord_beyond_resolution_band",
            "uncertainty_scope": "raster_resolution_floor_only",
            "absence_assessed": False, "sequestration_assessed": False,
        },
    }


def classify_planes(records, *, numbering_reviewed=False, registration_reviewed=False,
                    anatomy_reviewed=False, source_is_synthetic=False):
    """Combine the geometric candidate separately from reviewed classification.

    An extrusion-shaped geometric candidate in either usable plane wins; a
    protrusion-shaped candidate requires usable, resolved evidence in both.
    ``clinical_classification`` remains indeterminate unless both planes have
    measurable geometry, every supplied review gate is true and the source is
    nonsynthetic. Even when released, this is an unvalidated research estimate.
    Empty masks never establish absence, and disconnected masks do not establish
    sequestration. This function accepts only one record per plane.
    """
    gates = {"numbering_reviewed": numbering_reviewed,
             "registration_reviewed": registration_reviewed,
             "anatomy_reviewed": anatomy_reviewed,
             "source_is_synthetic": source_is_synthetic}
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError("Review and synthetic-source gates must be explicit booleans")
    if not isinstance(records, (list, tuple)):
        raise ValueError("Plane records must be a sequence")
    grouped = {plane: [] for plane in _PLANES}
    for record in records:
        if not isinstance(record, dict) or record.get("plane") not in _PLANES:
            raise ValueError("Every measurement record requires axial or sagittal plane")
        grouped[record["plane"]].append(record)
    flags, usable, hypotheses = [], {}, {}
    for plane, items in grouped.items():
        if len(items) != 1:
            flags.append(("missing_" if not items else "duplicate_") + plane)
            usable[plane], hypotheses[plane] = False, "indeterminate"
            continue
        record = items[0]
        values = [record.get(key) for key in ("candidate_base_width_mm",
                                             "candidate_displaced_width_mm", "uncertainty_mm")]
        def finite_positive(value):
            try:
                return type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                return False

        numbers_ok = all(finite_positive(value) for value in values)
        quality = record.get("quality_flags")
        quality_ok = isinstance(quality, list) and all(isinstance(flag, str) for flag in quality)
        usable[plane] = (record.get("geometry_measurable") is True and numbers_ok
                         and quality_ok and not any(flag in _BLOCKING_FLAGS for flag in quality))
        hypotheses[plane] = _hypothesis(*values) if usable[plane] else "indeterminate"
        if not usable[plane]:
            flags.append("unresolved_" + plane)
    if "extrusion" in hypotheses.values():
        candidate = "extrusion"
    elif all(usable.values()) and all(value == "protrusion" for value in hypotheses.values()):
        candidate = "protrusion"
    else:
        candidate = "indeterminate"
    for gate, value in gates.items():
        if gate == "source_is_synthetic":
            if value:
                flags.append("synthetic_source")
        elif not value:
            flags.append(gate.replace("_reviewed", "_unreviewed"))
    reviewed = (all(usable.values()) and numbering_reviewed and registration_reviewed
                and anatomy_reviewed and not source_is_synthetic)
    return {
        "schema_version": "1.0", "geometric_candidate": candidate,
        "clinical_classification": candidate if reviewed else "indeterminate",
        "classification_scope": "unvalidated_research_geometry",
        "clinically_validated": False, "both_planes_measurable": all(usable.values()),
        "review_gates": gates, "plane_candidates": hypotheses, "quality_flags": flags,
        "absence_assessed": False, "sequestration_assessed": False,
    }
