"""Exact axis/diagonal partition of the conditional ball."""
import numpy as np
import verify_conditional_energy_star as MODEL


def reflect(site, kind):
    x, y, z = site
    if kind == "axis":
        return (-x, y, z)
    return (y, x, z)


def masks(sites):
    result = [0]
    for site in sites:
        bit = 1 << MODEL.BOUNDARY.index(site)
        result += [v | bit for v in result]
    return np.array(result, dtype=np.int32)


def layout(kind):
    side = (lambda v: v[0]) if kind == "axis" else (lambda v: v[0] - v[1])
    left = [v for v in MODEL.BOUNDARY if side(v) < 0]
    right = [reflect(v, kind) for v in left]
    center = [v for v in MODEL.BOUNDARY if side(v) == 0]
    assert sorted(left + right + center) == MODEL.BOUNDARY
    assert (len(left), len(center)) == ((5, 8) if kind == "axis" else (7, 4))
    inner_left = [v for v in MODEL.INTERIOR if side(v) < 0]
    inner_right = [v for v in MODEL.INTERIOR if side(v) > 0]
    assert {reflect(v, kind) for v in inner_left} == set(inner_right)
    for x in MODEL.SITES:
        for delta in MODEL.DIRECTIONS:
            y = MODEL.add(x, delta)
            assert side(x)*side(y) >= 0
    return masks(left), masks(right), masks(center)
