"""Source-frame DICOM LPS geometry; no registration or anatomical confirmation."""

import math


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")


def _vector(value, count):
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Geometry requires finite numeric vectors") from None
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise ValueError("Geometry requires finite numeric vectors")
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _axes(frame):
    position = _vector(frame["position"], 3)
    orientation = _vector(frame["orientation"], 6)
    spacing = _vector(frame["spacing"], 2)
    c, r = orientation[:3], orientation[3:]
    if (min(spacing) <= 0 or abs(dot(c, c) - 1) > 1e-3
            or abs(dot(r, r) - 1) > 1e-3 or abs(dot(c, r)) > 1e-3):
        raise ValueError("DICOM directions must be orthonormal with positive spacing")
    n = [c[1] * r[2] - c[2] * r[1], c[2] * r[0] - c[0] * r[2], c[0] * r[1] - c[1] * r[0]]
    norm = math.sqrt(dot(n, n))
    return position, [v * spacing[1] for v in c], [v * spacing[0] for v in r], [v / norm for v in n]


def pixel_to_lps(frame, x, y):
    x, y = _vector([x, y], 2)
    p, c, r, _ = _axes(frame)
    return [p[i] + c[i] * x + r[i] * y for i in range(3)]


def lps_to_pixel(frame, point):
    point = _vector(point, 3)
    p, c, r, n = _axes(frame)
    delta = [v - o for v, o in zip(point, p)]
    cc, rr, cr = dot(c, c), dot(r, r), dot(c, r)
    denominator = cc * rr - cr * cr
    x = (dot(delta, c) * rr - dot(delta, r) * cr) / denominator
    y = (dot(delta, r) * cc - dot(delta, c) * cr) / denominator
    return x, y, dot(delta, n)
