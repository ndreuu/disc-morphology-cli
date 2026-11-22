"""Calibrated manual measurements on one native DICOM source frame.

Points are zero-based pixel-center coordinates (column x, row y), including
fractional positions. They must lie between 0 and columns/rows minus one.
Polygons close implicitly: do not repeat the first vertex at the end. These
functions measure supplied points, without identifying an anatomical structure,
validating a contour, registering frames or calculating a three-dimensional volume.
"""

import math

from ._spatial import pixel_to_lps


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("Measurement coordinates and geometry require finite numbers")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Measurement coordinates and geometry require finite numbers")
    return float(value)


def _frame(frame):
    if not isinstance(frame, dict):
        raise ValueError("Measurement requires one source-frame geometry object")
    for name in ("rows", "columns"):
        if type(frame.get(name)) is not int or frame[name] < 1:
            raise ValueError("Measurement requires positive integer source dimensions")
    if "frame_count" in frame and (type(frame["frame_count"]) is not int or frame["frame_count"] != 1):
        raise ValueError("Measurement supports native single-frame images only")
    if "image_ref" in frame:
        ref = frame["image_ref"]
        if (not isinstance(ref, dict) or type(ref.get("frame_index")) is not int
                or ref["frame_index"] != 0):
            raise ValueError("Measurement supports native single-frame index zero only")
    for name, count in (("position", 3), ("orientation", 6), ("spacing", 2)):
        values = frame.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError("Measurement requires complete per-frame DICOM geometry")
        for value in values:
            _number(value)
    # Reuse the established DICOM row/column spacing and orthonormality checks.
    pixel_to_lps(frame, 0, 0)
    # Translation does not affect either measurement. Working relative to the
    # source origin avoids cancellation when patient coordinates are large.
    return dict(frame, position=[0.0, 0.0, 0.0])


def _points(frame, value, count=None):
    if not isinstance(value, (list, tuple)) or (count is not None and len(value) != count):
        raise ValueError("Measurement requires exactly two points" if count == 2
                         else "Polygon requires an ordered sequence of points")
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Measurement points must contain column x and row y")
        x, y = (_number(v) for v in point)
        if not (0 <= x <= frame["columns"] - 1 and 0 <= y <= frame["rows"] - 1):
            raise ValueError("Measurement point lies outside source pixel centers")
        points.append((x, y))
    if len(set(points)) != len(points):
        raise ValueError("Measurement points must be distinct; polygons close implicitly")
    return points
