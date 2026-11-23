"""Verify anisotropic Fourier tests for the nearest-neighbor cubic Ising model.

The Fourier integrals and final bounds use integer enclosures. A single finite
linear program proposes multipliers, which are checked on every boundary orbit.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import comb, lcm

import numpy as np

from verify_conditional_energy_star import (
    BallMoments, INDEX, ORIGIN, SCALE, SITES, TOTAL_SCALE, add, mask, walk_resolvent,
)

ONE = {ORIGIN: 1}
X, Y = (1, 0, 0), (0, 1, 0)
KEYS = sorted({tuple(sorted(v)) for v in product(range(5), repeat=3) if sum(v) <= 4})
BASIS = [v for n in range(3) for v in sorted(product(range(n+1), repeat=3)) if sum(v) == n]
GRID = 1 << 48
# All entries are (nonnegative localizing factor, coefficients of the square root).
# The constant and three linear monomials are ordered as 1, z, y, x.
ORDER = [
    (0, [0, -6, 3, 3, -2096, 662, 1000, 604, -1267, 1096]),
    (7, [0, -2121, 2121, 0]),
    (2, [2860, 25, -640, -640]),
    (2, [-2460, -117, 1211, 1211]),
    (0, [1488, 245, 245, 245, -362, -1439, -362, -1439, -1439, -362]),
    (2, [2131, -1981, -517, -517]),
    (4, [2057, 1237, 1237, 1307]),
    (6, [2037, 1272, 1271, 1271]),
]


ENERGY = [
    (2, [0, 0, -2121, 2121]),
    (4, [1893, 1334, 1334, 1363]),
    (0, [1071, 1349, 1349, 1349, 422, 788, 422, 788, 788, 422]),
    (1, [-1801, -1309, -1309, -1526]),
    (0, [0, -1041, 1725, -684, -726, 485, 1203, -1222, 738, -477]),
    (0, [-1824, -610, -610, -610, 478, 1136, 478, 1136, 1136, 478]),
    (2, [-1190, -328, -1933, -1933]),
    (2, [2126, 2110, -119, -119]),
]


MASSIVE = [
    (2, [0, 0, -2121, 2121]),
    (0, [0, -1702, 1040, 661, -1437, -377, 878, -593, 970, 559]),
    (0, [-1975, -1293, -1293, -1293, 77, 149, 77, 149, 149, 77]),
    (6, [1898, 1326, 1349, 1349]),
    (0, [0, 18, -1498, 1480, 13, -1050, -1087, 1063, -13, 1073]),
    (3, [1909, 1355, 1349, 1304]),
    (0, [-1854, -715, -715, -715, 394, 1090, 394, 1090, 1090, 394]),
]


CROSS_PLANE = [
    (2, [0, 0, -2121, 2121]),
    (2, [292, 665, -2058, -2058]),
    (0, [-915, -1289, -1289, -1289, -450, -925, -450, -925, -925, -450]),
    ("extension", [
        256, 306, 306, 306, 306, 196, 43, 43, 59, -99, 59, 43, -99, 43,
        -99, -99, -43, -43, -410, -43, -410, -302, -43, -302, -410, -410, -50, -287,
        -287, -287, -287, -219, -173, 40, 40, 40, 40, 24, 35, 35, 0, -1,
        0, 35, -1, 35, -1, -1, -94, 50, -35, -35, -2, -5, 101, 101,
        10, -12, 4, 15, 24, 15, 24, 20, -94, -35, 50, -2, -35, -5,
        101, 15, 4, 24, 10, 101, -12, 15, 20, 24, -94, -35, -2, 50,
        -35, -5, 15, 101, 4, 24, 10, 15, 20, 101, -12, 24, -94, -2,
        -35, -35, 50, -5, 15, 15, 10, 20, 4, 101, 24, 101, 24, -12,
    ]),
    ("extension", [
        156, 167, 167, 167, 167, -23, -92, -92, -85, -72, -85, -92, -72, -92,
        -72, -72, -271, -271, -107, -271, -107, -134, -271, -134, -107, -107, -52, 246,
        246, 246, 246, 975, -41, -6, -6, -6, -6, -20, -33, -33, 26, 68,
        26, -33, 68, -33, 68, 68, -4, -95, 25, 25, -7, -56, -39, -39,
        2, 129, 31, -19, 37, -19, 37, 38, -4, 25, -95, -7, 25, -56,
        -39, -19, 31, 37, 2, -39, 129, -19, 38, 37, -4, 25, -7, -95,
        25, -56, -19, -39, 31, 37, 2, -19, 38, -39, 129, 37, -4, -7,
        25, 25, -95, -56, -19, -19, 2, 38, 31, -39, 37, -39, 37, 129,
    ]),
    ("extension", [
        -299, -361, -361, -361, -361, -140, 24, 24, 14, 77, 14, 24, 77, 24,
        77, 77, 275, 275, 332, 275, 332, 301, 275, 301, 332, 332, 123, 152,
        152, 152, 152, -346, 180, -42, -42, -42, -42, -19, 0, 0, -14, -35,
        -14, 0, -35, 0, -35, -35, 90, -11, 13, 13, 15, -9, 0, 0,
        -21, -18, -16, -20, -85, -20, -85, -39, 90, 13, -11, 15, 13, -9,
        0, -20, -16, -85, -21, 0, -18, -20, -39, -85, 90, 13, 15, -11,
        13, -9, -20, 0, -16, -85, -21, -20, -39, 0, -18, -85, 90, 15,
        13, 13, -11, -9, -20, -20, -21, -39, -16, 0, -85, 0, -85, -18,
    ]),
    ("extension", [
        0, 363, -363, -363, 363, 0, 0, 0, 361, 138, -361, 0, -138, 0,
        -138, 138, -352, 352, 0, 352, 0, 433, -352, -433, 0, 0, 0, -139,
        139, 139, -139, 0, 0, 38, -38, -38, 38, 0, 0, 0, -29, -80,
        29, 0, 80, 0, 80, -80, -111, 50, 0, 0, -26, 0, 12, 12,
        26, -28, 30, -18, 20, -18, 20, -99, 111, 0, -50, 26, 0, 0,
        -12, 18, -30, -20, -26, -12, 28, 18, 99, -20, 111, 0, 26, -50,
        0, 0, 18, -12, -30, -20, -26, 18, 99, -12, 28, -20, -111, -26,
        0, 0, 50, 0, -18, -18, 26, -99, 30, 12, 20, 12, 20, -28,
    ]),
    ("extension", [
        0, -29, 29, 29, -29, 0, 0, 0, 33, -31, -33, 0, 31, 0,
        31, -31, -39, 39, 0, 39, 0, -84, -39, 84, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 41, -41, -41, 41, 0, 0, 0, 253, -14,
        -253, 0, 14, 0, 14, -14, 54, 22, -85, -85, -149, 16, 182, 182,
        420, 41, -346, -144, 24, -144, 24, -49, -54, 85, -22, 149, 85, -16,
        -182, 144, 346, -24, -420, -182, -41, 144, 49, -24, -54, 85, 149, -22,
        85, -16, 144, -182, 346, -24, -420, 144, 49, -182, -41, -24, 54, -149,
        -85, -85, 22, 16, -144, -144, 420, -49, -346, 182, 24, 182, 24, 41,
    ]),
    ("extension", [
        60, -158, -158, -158, -158, 965, -63, -63, -179, 152, -179, -63, 152, -63,
        152, 152, -128, -128, 207, -128, 207, -381, -128, -381, 207, 207, -76, 13,
        13, 13, 13, 180, -42, -22, -22, -22, -22, 120, 33, 33, 79, -72,
        79, 33, -72, 33, -72, -72, 36, 13, -12, -12, -7, 23, 43, 43,
        17, -141, 42, 12, -47, 12, -47, 3, 36, -12, 13, -7, -12, 23,
        43, 12, 42, -47, 17, 43, -141, 12, 3, -47, 36, -12, -7, 13,
        -12, 23, 12, 43, 42, -47, 17, 12, 3, 43, -141, -47, 36, -7,
        -12, -12, 13, 23, 12, 12, 17, 3, 42, 43, -47, 43, -47, -141,
    ]),
    ("extension", [
        0, 0, 0, 0, 0, 0, 80, -80, 0, 0, 0, -80, 0, 80,
        0, 0, 0, 0, 599, 0, -599, 0, 0, 0, -599, 599, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -227, 227, 0, 0,
        0, 227, 0, -227, 0, 0, 0, 0, 128, -128, 0, 0, 5, -5,
        0, 0, 0, 5, -141, -5, 141, 0, 0, 128, 0, 0, -128, 0,
        5, 5, 0, -141, 0, -5, 0, -5, 0, 141, 0, -128, 0, 0,
        128, 0, -5, -5, 0, 141, 0, 5, 0, 5, 0, -141, 0, 0,
        -128, 128, 0, 0, -5, 5, 0, 0, 0, -5, 141, 5, -141, 0,
    ]),
    ("extension", [
        0, 29, -110, 110, -29, 0, 87, -150, 0, -25, 0, 150, 95, -87,
        -95, 25, -58, 218, 56, -218, -97, 0, 58, 0, 97, -56, 0, 232,
        -872, 872, -232, 0, 0, -19, 72, -72, 19, 0, 17, -30, 0, -15,
        0, 30, 55, -17, -55, 15, 1, 5, -42, 64, 5, -40, 7, -12,
        2, 8, -4, 0, 35, -5, -30, 8, -2, -28, -19, -19, -56, 149,
        5, 7, 14, -19, -6, 11, -29, 9, -29, -1, 2, 56, 19, 19,
        28, -149, -9, -11, -14, 1, 6, -7, 29, -5, 29, 19, -1, -5,
        -64, 42, -5, 40, 5, 0, -2, -8, 4, 12, 30, -7, -35, -8,
    ]),
    (0, [-1653, -613, -613, -613, 609, 1159, 609, 1159, 1159, 609]),
    (2, [2090, 2101, -330, -330]),
    (6, [-1722, -1421, -1417, -1417]),
    ("extension", [
        -205, -129, -129, -129, -129, -688, 32, 32, 69, -104, 69, 32, -104, 32,
        -104, -104, 195, 195, 131, 195, 131, 215, 195, 215, 131, 131, 275, 262,
        262, 262, 262, 534, 168, 8, 8, 8, 8, -189, 17, 17, 107, -103,
        107, 17, -103, 17, -103, -103, 89, -7, -14, -14, -58, -75, 30, 30,
        89, 32, 26, -27, -96, -27, -96, -109, 89, -14, -7, -58, -14, -75,
        30, -27, 26, -96, 89, 30, 32, -27, -109, -96, 89, -14, -58, -7,
        -14, -75, -27, 30, 26, -96, 89, -27, -109, 30, 32, -96, 89, -58,
        -14, -14, -7, -75, -27, -27, 89, -109, 26, 30, -96, 30, -96, 32,
    ]),
    ("extension", [
        0, 47, -47, -47, 47, 0, 0, 0, 78, 23, -78, 0, -23, 0,
        -23, 23, -45, 45, 0, 45, 0, 19, -45, -19, 0, 0, 0, 22,
        -22, -22, 22, 0, 0, 60, -60, -60, 60, 0, 0, 0, 190, 57,
        -190, 0, -57, 0, -57, 57, 51, 18, -51, -51, -167, 17, 215, 215,
        412, 20, -258, -109, -182, -109, -182, 37, -51, 51, -18, 167, 51, -17,
        -215, 109, 258, 182, -412, -215, -20, 109, -37, 182, -51, 51, 167, -18,
        51, -17, 109, -215, 258, 182, -412, 109, -37, -215, -20, 182, 51, -167,
        -51, -51, 18, 17, -109, -109, 412, 37, -258, 215, -182, 215, -182, 20,
    ]),
    ("extension", [
        0, 0, 0, 0, 0, 0, -18, 18, 0, 0, 0, 18, 0, -18,
        0, 0, 0, 0, -123, 0, 123, 0, 0, 0, 123, -123, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, -40, 0, 0,
        0, -40, 0, 40, 0, 0, 0, 0, 55, -55, 0, 0, 98, -98,
        0, 0, 0, 98, -463, -98, 463, 0, 0, 55, 0, 0, -55, 0,
        98, 98, 0, -463, 0, -98, 0, -98, 0, 463, 0, -55, 0, 0,
        55, 0, -98, -98, 0, 463, 0, 98, 0, 98, 0, -463, 0, 0,
        -55, 55, 0, 0, -98, 98, 0, 0, 0, -98, 463, 98, -463, 0,
    ]),
    ("extension", [
        0, -202, 202, 202, -202, 0, 0, 0, -293, -52, 293, 0, 52, 0,
        52, -52, 514, -514, 0, -514, 0, -82, 514, 82, 0, 0, 0, 70,
        -70, -70, 70, 0, 0, -18, 18, 18, -18, 0, 0, 0, 76, 4,
        -76, 0, -4, 0, -4, 4, 61, -35, 11, 11, -7, -55, 19, 19,
        -24, -4, -207, -50, 94, -50, 94, 229, -61, -11, 35, 7, -11, 55,
        -19, 50, 207, -94, 24, -19, 4, 50, -229, -94, -61, -11, 7, 35,
        -11, 55, 50, -19, 207, -94, 24, 50, -229, -19, 4, -94, 61, -7,
        11, 11, -35, -55, -50, -50, -24, 229, -207, 19, 94, 19, 94, -4,
    ]),
    ("extension", [
        31, 67, 67, 67, 67, -375, -36, -36, -81, -47, -81, -36, -47, -36,
        -47, -47, -55, -55, 184, -55, 184, -43, -55, -43, 184, 184, 361, 180,
        180, 180, 180, -825, 75, -79, -79, -79, -79, -130, 18, 18, 36, 60,
        36, 18, 60, 18, 60, 60, 25, 47, 6, 6, -13, -254, -24, -24,
        -126, 10, -23, -11, 68, -11, 68, 243, 25, 6, 47, -13, 6, -254,
        -24, -11, -23, 68, -126, -24, 10, -11, 243, 68, 25, 6, -13, 47,
        6, -254, -11, -24, -23, 68, -126, -11, 243, -24, 10, 68, 25, -13,
        6, 6, 47, -254, -11, -11, -126, 243, -23, -24, 68, -24, 68, 10,
    ]),
    ("extension", [
        0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, -413, 0, 413, 0, 0, 0, 413, -413, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 409, -409, 0, 0,
        0, -409, 0, 409, 0, 0, 0, 0, -250, 250, 0, 0, 66, -66,
        0, 0, 0, 66, 71, -66, -71, 0, 0, -250, 0, 0, 250, 0,
        66, 66, 0, 71, 0, -66, 0, -66, 0, -71, 0, 250, 0, 0,
        -250, 0, -66, -66, 0, -71, 0, 66, 0, 66, 0, 71, 0, 0,
        250, -250, 0, 0, -66, 66, 0, 0, 0, -66, -71, 66, 71, 0,
    ]),
    (0, [0, 1516, -451, -1065, 1353, 878, -402, 372, -1249, -950]),
    (3, [-1772, -1387, -1479, -1322]),
    ("extension", [
        -264, -138, -138, -138, -138, -876, 66, 66, 35, -55, 35, 66, -55, 66,
        -55, -55, 196, 196, 283, 196, 283, 175, 196, 175, 283, 283, 178, 209,
        209, 209, 209, 297, 156, -7, -7, -7, -7, -98, -41, -41, -77, 44,
        -77, -41, 44, -41, 44, 44, 44, -34, 17, 17, 50, -49, -19, -19,
        -82, 90, -63, -22, -12, -22, -12, -41, 44, 17, -34, 50, 17, -49,
        -19, -22, -63, -12, -82, -19, 90, -22, -41, -12, 44, 17, 50, -34,
        17, -49, -22, -19, -63, -12, -82, -22, -41, -19, 90, -12, 44, 50,
        17, 17, -34, -49, -22, -22, -82, -41, -63, -19, -12, -19, -12, 90,
    ]),
    ("extension", [
        0, -100, 100, 100, -100, 0, 0, 0, -241, -63, 241, 0, 63, 0,
        63, -63, 331, -331, 0, -331, 0, -126, 331, 126, 0, 0, 0, -50,
        50, 50, -50, 0, 0, -67, 67, 67, -67, 0, 0, 0, -148, -73,
        148, 0, 73, 0, 73, -73, 0, 23, 47, 47, 173, -81, -154, -154,
        -307, -121, 201, 116, 139, 116, 139, -55, 0, -47, -23, -173, -47, 81,
        154, -116, -201, -139, 307, 154, 121, -116, 55, -139, 0, -47, -173, -23,
        -47, 81, -116, 154, -201, -139, 307, -116, 55, 154, 121, -139, 0, 173,
        47, 47, 23, -81, 116, 116, -307, -55, 201, -154, 139, -154, 139, -121,
    ]),
    ("extension", [
        0, 175, -175, -175, 175, 0, 0, 0, 224, 32, -224, 0, -32, 0,
        -32, 32, -460, 460, 0, 460, 0, 110, -460, -110, 0, 0, 0, -189,
        189, 189, -189, 0, 0, -35, 35, 35, -35, 0, 0, 0, -202, -70,
        202, 0, 70, 0, 70, -70, -99, 55, 32, 32, 120, 1, -116, -116,
        -129, -30, 231, 37, 24, 37, 24, -149, 99, -32, -55, -120, -32, -1,
        116, -37, -231, -24, 129, 116, 30, -37, 149, -24, 99, -32, -120, -55,
        -32, -1, -37, 116, -231, -24, 129, -37, 149, 116, 30, -24, -99, 120,
        32, 32, 55, 1, 37, 37, -129, -149, 231, -116, 24, -116, 24, -30,
    ]),
]


def times(p, q):
    out = {}
    for a, u in p.items():
        for b, v in q.items():
            e = add(a, b)
            out[e] = out.get(e, 0) + u*v
    return {e: v for e, v in out.items() if v}


def polynomial(tag, coeff):
    loc = [ONE]
    for a in (1, -1):
        x = {ORIGIN: 1, X: a}
        loc.append(x)
        for b in (1, -1):
            loc.append(times(x, {ORIGIN: 1, Y: b}))
    loc.append({ORIGIN: 1, (2, 0, 0): -1})
    assert len(coeff) == (10 if tag == 0 else 4)
    root = dict(zip(BASIS, coeff))
    p = times(loc[tag], times(root, root))
    a = [sum(v for e, v in p.items() if tuple(sorted(e)) == k) for k in KEYS]
    norm = sum(map(abs, a))
    assert norm > 0 and sum(a) >= 0
    return [F(v, norm) for v in a]


def angular_rows(ball):
    def pair(z):
        if z == ORIGIN:
            return TOTAL_SCALE
        x = next(x for x in SITES if add(x, z) in INDEX)
        return ball.averaged(mask((x, add(x, z))))

    result = []
    for key in KEYS:
        paths = [[(m-2*j, comb(m, j)) for j in range(m+1)] for m in key]
        den = 2**sum(key)
        acc = np.zeros(len(ball.words), dtype=np.int64)
        for steps in product(*paths):
            (i, x), (j, y), (k, z) = steps
            acc += (x*y*z) * pair((i, j, k))
        result.append((2*acc+den)//(2*den))
    assert np.array_equal(result[KEYS.index((0, 0, 1))], ball.g1)
    return result


def binomial_weights(n, axes):
    """Floor-scaled probabilities for binomial(n, (axes-1)/axes)."""
    den = axes**n
    coefficient = 1
    active = []
    for t in range(n+1):
        scaled = GRID*coefficient//den
        if scaled:
            active.append((t, scaled))
        if t != n:
            coefficient = coefficient*(n-t)*(axes-1)//(t+1)
    return active


@lru_cache(None)
def angular_walks(nmax):
    """Lower integer approximations to E[x^i y^j z^k gamma^n], also for n=0."""
    single = [[(GRID*comb(n+i, (n+i)//2))//(2**(n+i)) if (n+i)%2 == 0 else 0
               for n in range(nmax+1)] for i in range(5)]
    pairs = sorted({key[:2] for key in KEYS})
    pairs_values = {key: [] for key in pairs}
    values = {key: [] for key in KEYS}
    for n in range(nmax+1):
        w2 = binomial_weights(n, 2)
        for i, j in pairs:
            total = sum(w*single[i][t]*single[j][n-t] for t, w in w2)
            pairs_values[i, j].append(total//(GRID*GRID))
        w3 = binomial_weights(n, 3)
        for i, j, k in KEYS:
            total = sum(w*pairs_values[i, j][t]*single[k][n-t] for t, w in w3)
            values[i, j, k].append(total//(GRID*GRID))
    def one_dimensional(j):
        return F(comb(j, j//2), 2**j) if j%2 == 0 else F(0)
    for n in range(min(nmax, 6)+1):
        for i, j, k in KEYS:
            exact = sum(comb(n, a)*comb(n-a, b)*one_dimensional(i+a) *
                        one_dimensional(j+b)*one_dimensional(k+n-a-b)
                        for a in range(n+1) for b in range(n+1-a))/3**n
            assert 0 <= exact*GRID-values[i, j, k][n] <= 2*n+7
    return values


@lru_cache(None)
def sharpen_watson():
    from verify_pair_energy_test import return_terms
    nmax = 10000
    total = 0
    for term, denominator in return_terms(nmax):
        total = 36*total+term
    # Machin's formula pi = 16 atan(1/5) - 4 atan(1/239). Alternating truncations.
    def atan_series(t, length):
        return sum((-1)**j*t**(2*j+1)/(2*j+1) for j in range(length))
    low_pi = 16*atan_series(F(1, 5), 6)-4*atan_series(F(1, 239), 3)
    a, b = F(233291, 10**6), F(133, 1000)
    c = F(299, 900)
    assert (3/low_pi)**3/16 < a*a
    assert 1/(4096*low_pi**3*c**7) < b*b
    q = 1-F(299, 45000)
    assert 2*q**(2*nmax+2)/(1-q*q) < F(1, 10**10)
    upper = F(total, denominator)+2*a/100+2*b/(3*100**3)+F(1, 10**10)
    bound = F(151638634, 10**8)
    assert bound-F(1, 10**8) < upper <= bound
    return bound



class AngularGreen:
    def __init__(self, shift, nmax=2400):
        assert nmax % 2 == 0
        d, self.wlow, self.whigh = walk_resolvent(shift)
        if not shift:
            self.whigh = min(self.whigh, sharpen_watson())
        walks = angular_walks(nmax+2)
        unit = walks[ORIGIN]
        # With exact one-dimensional moments the largest per-moment floor
        # error is (2n+7)/GRID. A difference costs the same one-sided width.
        power = GRID*d.denominator//d.numerator
        center = [0]*len(KEYS)
        for n in range(nmax):
            for j, key in enumerate(KEYS):
                center[j] += (walks[key][n]-unit[n])*power//GRID
            power = power*d.denominator//d.numerator
        radius = F(sum(3*n+9 for n in range(nmax)), GRID)
        p_last = F(unit[nmax]+2*nmax+7, GRID)
        p_lo = F(unit[nmax], GRID)
        gap_hi = p_last-F(unit[nmax+2], GRID)
        self.delta = []
        for j, key in enumerate(KEYS):
            if key == ORIGIN:
                self.delta.append((F(0), F(0)))
            else:
                v = F(center[j], GRID)
                degree = sum(key)
                if shift:
                    power_upper = d*F(power+nmax+1, GRID)
                    self.delta.append((v-radius-degree*p_last*power_upper, v+radius))
                else:
                    av = F(-degree-(degree%2), 2)
                    rem = F(3*degree**2+degree+2, 2)*gap_hi
                    self.delta.append((v-radius+av*p_last-rem, v+radius+av*p_lo+rem))

    def upper(self, a):
        assert sum(a) >= 0
        return sum(a)*self.whigh + sum(
            w*(hi if w >= 0 else lo) for w, (lo, hi) in zip(a, self.delta)
        )


def spectral_cut(a, angular, c0, green):
    j = green.upper(a)/6
    assert j > 0
    weights = [-w for w in a]
    weights[KEYS.index(ORIGIN)] += j*c0
    weights[KEYS.index((0, 0, 1))] -= 2*j
    den = lcm(*(t.denominator for t in weights))
    acc = np.zeros(len(angular[0]), dtype=object)
    for t, row in zip(weights, angular):
        if t:
            acc += int(t*den)*row.astype(object)
    rounded = ((2*acc+den)//(2*den)).astype(np.int64)
    input_error = F(1, 2*SCALE) + F(1, 2*TOTAL_SCALE)
    radius = F(1, 2*TOTAL_SCALE) + sum(abs(w) for k, w in zip(KEYS, weights)
                                        if k != ORIGIN)*input_error
    return rounded, radius, sum(a)


def reflection_features(ball):
    from itertools import combinations
    from verify_conditional_energy_star import orbit
    def theta(p):
        return 1-p[0], p[1], p[2]
    forward = [p for p in SITES if p[0] >= 1 and theta(p) in INDEX]
    reverse = [p for p in SITES if p[0] == 0 and theta(p) in INDEX]
    tests = [(None, a) for j in (1, 3, 5) for a in combinations(forward, j)]
    tests.extend((p, a) for p in reverse for j in (0, 2) for a in combinations(forward, j))
    left = [mask(a) for _, a in tests]
    right = [mask(theta(q) for q in a) for _, a in tests]
    full_l = [a^(mask([p]) if p is not None else 0)
              for (p, _), a in zip(tests, left)]
    full_r = [a^(mask([theta(p)]) if p is not None else 0)
              for (p, _), a in zip(tests, right)]
    ordinary = np.array([[orbit(a^b)[0] for b in full_r] for a in full_l])
    base = np.array([[orbit(left[i]^right[j])[0] if p is not None and p == q else -1
                      for j, (q, _) in enumerate(tests)] for i, (p, _) in enumerate(tests)])
    shared = base >= 0
    base[~shared] = 0
    keys = np.union1d(ordinary, base)
    values = np.array([ball.averaged(int(k)) for k in keys])
    assert len(tests) == 112 and len(keys) == 353
    return np.searchsorted(keys, ordinary), values, np.searchsorted(keys, base), shared


def reflection_cut(z, c0, features):
    ordinary, values, base, shared = features
    q = np.asarray(z, dtype=np.int64)
    prod_q = np.outer(q, q)
    norm = int(q@q)
    a, b = c0.as_integer_ratio()
    weights = np.zeros(len(values), dtype=np.int64)
    np.add.at(weights, ordinary.ravel(), (b*prod_q*(1-2*shared)).ravel())
    np.add.at(weights, base.ravel(), (a*prod_q*shared).ravel())
    den = norm*b
    acc = sum(int(w)*row.astype(object) for w, row in zip(weights, values) if w)
    row = ((2*acc+den)//(2*den)).astype(np.int64)
    radius = F(sum(abs(int(w)) for w in weights), 2*SCALE*den) + F(1, 2*TOTAL_SCALE)
    return row, radius, F(0)


def combined_certificate(tests, ratio=F(19, 12)):
    ball = BallMoments(ratio, 4)
    features = reflection_features(ball)
    c0 = (ratio-1)/(ratio+1)+(ratio+1)/(ratio-1)
    green = AngularGreen(F(0))
    angular = angular_rows(ball)
    reflection = [reflection_cut(parameter, c0, features) for tag, parameter in tests
                  if tag == "extension"]
    spectral = [spectral_cut(polynomial(tag, parameter), angular, c0, green)
                for tag, parameter in tests if tag != "extension"]
    assert len(reflection) == 17 and len(spectral) == 8
    cuts = reflection+spectral
    zero = np.zeros_like(ball.g1)
    result = ball.dual(zero, 1, cuts, tolerance=0.0000005, relax=True)
    bound, coeff = ball.certify(zero, 1, cuts, result)
    print("cross-plane spin tests", len(tests), "upper", float(bound),
          "zero-mode coefficient", float(coeff), flush=True)
    assert bound < 0 and coeff > 0
    lower = -bound/coeff
    print("cross-plane rigorous m^2 >=", float(lower), flush=True)
    return lower



def check_bond_reflection(ratio):
    vertical = [[ratio, F(1)], [F(1), ratio]]
    u = (ratio-1)/(ratio+1)
    c0 = 1/u+u
    for a, b in product(range(2), repeat=2):
        cube = sum(vertical[a][i]*vertical[i][j]*vertical[j][b]
                   for i, j in product(range(2), repeat=2))
        assert cube/((ratio**2-1)*vertical[a][b]) == c0-(-1)**(a+b)


def certificate(ratio, tests, suspect_chi=None, energy=False):
    ball = BallMoments(ratio, 4)
    assert len(ball.words) == 3236 and len(ball.rows) == 102
    u = (ratio-1)/(ratio+1)
    c0 = 1/u+u
    # C/(lambda+C/X) increases with C. Its integral at the rational probe
    # is below 1 at each susceptibility hypothesis used here.
    probe = F(412, 100)
    shift = probe/suspect_chi if suspect_chi else F(0)
    green = AngularGreen(shift)
    if suspect_chi:
        assert probe*green.whigh/6 < 1
    angular = angular_rows(ball)
    cuts = [spectral_cut(polynomial(tag, root), angular, c0, green) for tag, root in tests]
    if energy:
        upper, coefficient = ball.certify(ball.g1, 1, cuts, ball.dual(ball.g1, 1, cuts))
        print("r", ratio, "chi hypothesis", suspect_chi, "energy upper", upper,
              "zero-mode coefficient", coefficient, float(upper), float(coefficient), flush=True)
        return upper, coefficient
    zero = np.zeros_like(ball.g1)
    proposal = ball.dual(zero, 1, cuts, relax=True)
    upper, coefficient = ball.certify(zero, 1, cuts, proposal)
    assert upper < 0 and coefficient > 0
    print("r", ratio, "chi hypothesis", suspect_chi,
          "exact upper", upper, "zero-mode coefficient", coefficient, flush=True)
    lower = -upper/coefficient
    print("required zero-mode mass", float(lower), flush=True)
    return lower


if __name__ == "__main__":
    for ratio in (F(25, 16), F(19, 12), F(832, 525), F(317, 200)):
        check_bond_reflection(ratio)
    assert certificate(F(317, 200), ORDER) > F("0.00073509")
    assert certificate(F(832, 525), ORDER) > F("0.00021100")
    upper, coef = certificate(F(19, 12), ENERGY, energy=True)
    assert upper < F("0.342933")
    assert (F(".3434408514")-upper)/coef > F(".00051388")
    assert combined_certificate(CROSS_PLANE) > F(".00059722")
    assert combined_certificate(CROSS_PLANE, F(18997, 12000)) > F(".00003373")
    r0, r1 = F(19, 12), F(17581, 11100)
    upper, coeff = certificate(r1, ENERGY, energy=True)
    t = F(".34265")
    slope = 1 + 2*F(".1554593759") + 8*F(".2171455219") - 11*t*t
    lower = min(t, F(".3423083784") + slope*(r1-r0)/(2*r1))
    assert (lower-upper)/coeff > F(".00001501")
    print("shared-neighbor lower at", r1, "g >=", lower,
          "m^2 >=", float((lower-upper)/coeff), flush=True)
    upper, _ = certificate(F(25, 16), MASSIVE, 245, energy=True)
    assert upper < F(".317023950") < F(".3170321192")
    print("free-boundary g >= 0.3170321192 proves chi > 245", flush=True)

