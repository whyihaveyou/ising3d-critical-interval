"""Check a fixed certificate for spontaneous magnetization of the cubic Ising model.

Positive fixed multipliers combine the massless infrared and bond-reflection
inequalities. We compute the thermal tables anew at the displayed coupling.
"""
import argparse
from fractions import Fraction as F

import numpy as np

import verify_improved_phase_bound as phase
import verify_local_reflection_energy as local
from verify_conditional_energy_star import BallMoments, SCALE, TOTAL_SCALE


DENOMINATOR = 10**6
TESTS = (
    phase.MASSIVE + phase.ENERGY + phase.ORDER
    + [item for item in phase.CROSS_PLANE if item[0] != "extension"]
    + [item for item in phase.CROSS_PLANE if item[0] == "extension"]
)
TRANSLATIONS = (
    800000, 800000, 800000, -572826, -154252, -800000, -326478, -800000, -800000, -196367, -381686, 800000,
    1242, 800000, 282181, 613456, 800000, -800000, -800000, 800000, 640299, 800000, 95124, -800000,
    -6699, 39095, -29102, -800000, 90057, -416549, -43162, 800000, -800000, 164726, 800000, 535788,
    -299890, -33374, -223905, 800000, -17018, 26448, 195534, -800000, -188761, 67886, -286433, 220769,
    -800000, -224308, 11566, -263367, 24694, 764167, 122746, 508646, 558008, 445073, 486863, 444861,
    800000, -214534, 183029, -800000, 35298, -175391, -35966, -800000, 90870, -14016, -258280, 800000,
    46857, -156099, 183787, 8611, -800000, 57147, -155408, 36873, -154030, 182808, -13547, 179057,
    308046, -800000, -38982, -25710, 800000, 50038, -5281, 190288, 347926, 694, -192200, -1991,
    144615, -338793, 107874, -143196, -155185, -7296,
)
TEST_MULTIPLIERS = (
    (4, 287598), (14, 479595), (23, 4734), (24, 249106), (26, 3647082),
    (27, 195055), (28, 10000000), (29, 713293), (30, 5123303), (31, 3728),
    (32, 5144), (33, 28377), (34, 567), (35, 141), (36, 3406),
    (38, 6284), (39, 3949), (40, 43), (42, 2319), (44, 947),
    (45, 7306), (46, 1511), (47, 2927),
)


def exact_reflection(vector, bare, features):
    """Accumulate dimensionless small integers before using the rational bare coefficient."""
    ordinary, values, base, shared = features
    q = np.asarray(vector, dtype=np.int64)
    # This bounds every entry and every partial sum in the two int64 tables.
    assert sum(abs(int(x)) for x in q)**2 < (1 << 62)
    pair = np.outer(q, q)
    weights_ordinary = np.zeros(len(values), dtype=np.int64)
    weights_shared = np.zeros(len(values), dtype=np.int64)
    np.add.at(weights_ordinary, ordinary.ravel(), (pair*(1-2*shared)).ravel())
    np.add.at(weights_shared, base.ravel(), (pair*shared).ravel())
    weights = [bare.denominator*int(left)+bare.numerator*int(right)
               for left, right in zip(weights_ordinary, weights_shared)]
    divisor = int(q @ q)*bare.denominator
    dot = sum(w*row.astype(object) for w, row in zip(weights, values) if w)
    scaled = (2*dot + divisor)//(2*divisor)
    radius = F(sum(map(abs, weights)), 2*SCALE*divisor) + F(1, 2*TOTAL_SCALE)
    return scaled, radius, F(0)


# Each extended proof lists nonzero (index, multiplier) pairs for degree-eight
# translations and the spectral/reflection family, followed by exterior
# (spin mask, multiplier) and bond-covariance (spin mask, t*1000, multiplier).
EXTENDED_PROOFS = (
    (
        F("877/554"), F("0.0000136"),
        (
            (4, -53484), (6, 53553), (16, 76149), (24, 5964), (26, 7729), (34, -36038),
            (40, -10716), (46, -5813), (56, 3368), (64, -13345), (74, -6585), (80, -3495),
            (84, -17259), (92, -18771), (98, -8245), (104, -6464), (106, 270), (110, -2333),
            (112, -5443), (114, 18814), (120, -6385), (122, -2083), (128, -475), (130, -6804),
            (138, -2762), (140, 558), (146, 1429), (152, 9956), (156, -3525), (158, -64),
            (164, -1450), (170, -379), (172, 27116), (179, -1060), (180, -4546), (184, 5717),
            (185, -3486), (193, 5212), (197, 15996), (201, -6030), (203, -214), (210, 5436),
            (228, 7932), (232, 554), (240, -1046), (241, 3656), (245, 2780), (251, -135),
            (255, -228), (257, -563), (259, -547), (265, -284), (271, 3305), (275, -954),
            (279, -222), (289, 61745), (292, 13028), (296, -466), (299, 5197), (303, 8224),
            (307, 337), (311, -218), (315, 8100), (318, 3574), (322, -173), (325, 337),
            (329, -1253), (333, -727), (337, -457), (342, -25), (345, 4540), (349, -335),
            (353, -1455), (358, -3900), (361, -75), (365, 133), (371, 126), (374, -677),
            (378, 77), (381, -409), (385, -459), (389, -12), (393, -406), (397, 865),
            (401, 136), (405, -272), (408, -18), (409, 33), (412, -44), (415, -258),
            (417, -85),
        ),
        (
            (9, 44670), (14, 615993), (23, 8234), (24, 268479), (26, 3687385),
            (28, 10000000), (29, 973306), (30, 5190849), (31, 3997), (32, 4157),
            (33, 27643), (35, 1211), (36, 4687), (38, 8084), (39, 4072),
            (40, 889), (42, 2430), (45, 7732), (47, 4279),
        ),
        (
            (100608, 65), (1182720, 438), (201216, 305), (51456, 332), (200832, 165),
            (237568, 262), (52224, 333), (1059840, 1153), (104448, 350), (39936, 264),
            (1198080, 214), (6528, 1605), (27136, 1034), (15360, 1130), (98688, 54),
            (221184, 341), (233472, 121), (14464, 111), (205824, 171), (148224, 48),
            (13824, 1304), (152064, 2206), (26752, 732), (153600, 540), (245760, 304),
            (184320, 135), (205312, 971), (6912, 2810),
        ),
        (

        ),
    ),
    (
        F("1735/1096"), F("0.00000317"),
        (
            (4, -53462), (6, 53555), (16, 76131), (24, 5965), (26, 7721), (34, -36046),
            (40, -10720), (46, -5816), (56, 3369), (64, -13352), (74, -6584), (80, -3515),
            (84, -17252), (92, -18767), (98, -8249), (104, -6474), (106, 266), (110, -2342),
            (112, -5429), (114, 18884), (120, -6406), (122, -2085), (128, -456), (130, -6804),
            (138, -2766), (140, 565), (146, 1424), (152, 9971), (156, -3527), (158, -60),
            (164, -1443), (170, -387), (172, 27029), (179, -1054), (180, -4540), (184, 5714),
            (185, -3477), (193, 5210), (197, 15957), (201, -6030), (203, -216), (210, 5436),
            (228, 7880), (232, 506), (240, -1046), (241, 3635), (245, 2756), (251, -136),
            (255, -224), (257, -572), (259, -547), (265, -284), (271, 3284), (275, -617),
            (279, -221), (289, 61748), (292, 12981), (296, -466), (299, 5168), (303, 8179),
            (307, 338), (311, -220), (315, 8078), (318, 3530), (322, -171), (325, 315),
            (329, -574), (333, -725), (337, -457), (342, -25), (345, 4517), (353, -1121),
            (357, 334), (358, -5709), (361, -74), (365, 133), (371, 126), (374, -678),
            (378, 79), (381, -412), (385, -460), (389, -11), (393, -406), (397, 866),
            (401, 137), (405, -275), (408, -17), (409, 33), (412, -45), (415, -257),
            (417, -87),
        ),
        (
            (9, 43334), (14, 617111), (23, 8241), (24, 268834), (26, 3687289),
            (28, 10000000), (29, 973414), (30, 5190282), (31, 4003), (32, 4144),
            (33, 27646), (35, 1212), (36, 4684), (38, 8055), (39, 4070),
            (40, 887), (42, 2436), (45, 7730), (47, 4281),
        ),
        (
            (100608, 66), (1182720, 441), (201216, 304), (51456, 336), (200832, 167),
            (237568, 261), (52224, 333), (1059840, 1156), (104448, 349), (39936, 265),
            (1198080, 215), (6528, 1616), (27136, 1024), (15360, 1134), (98688, 54),
            (221184, 340), (233472, 120), (14464, 106), (205824, 171), (148224, 51),
            (13824, 1308), (152064, 2213), (26752, 732), (153600, 539), (245760, 304),
            (184320, 135), (205312, 967), (6912, 2821),
        ),
        (

        ),
    ),
)


def additional_rows(ball, ratio, shell, covariances):
    if shell:
        shell_measure = local.shell_measure(ratio, ball.words)
        shell_sum = int(sum(shell_measure))
        inverse = local.exterior_weights(ratio, ball.words)
        for mask, v in shell:
            assert mask > 0 and mask.bit_length() <= 25 and mask % 128 == 0
            assert bin(mask).count("1") % 2 == 0
            row = ball.averaged(mask)
            lower = int(sum(row*shell_measure)//shell_sum)
            assert 0 <= lower <= TOTAL_SCALE
            rounded = local.round_row((row.astype(object)-lower)*inverse, TOTAL_SCALE)
            yield v, rounded, 3*local.GRID, F(0)
    if covariances:
        allowed = set(local.shared_bonds())
        for mask, t, v in covariances:
            assert mask in allowed and t >= 0
            raw = (10**6*ball.averaged(mask).astype(object)
                   - 2000*t*ball.g1.astype(object) + t*t*TOTAL_SCALE)
            radius = (1+F(2*t, 1000))*local.ROUND + local.GRID
            yield v, local.round_row(raw, 10**6), radius, F(0)


def check(ratio, green, degree, translations, inequalities, shell=(), covariances=()):
    ball = BallMoments(ratio, degree)
    u = (ratio-1)/(ratio+1)
    bare = u + 1/u
    assert len(ball.words) == 3236 and len(ball.rows) == {4: 102, 8: 418}[degree]
    angular = phase.angular_rows(ball)
    features = phase.reflection_features(ball)
    residual = np.zeros(len(ball.words), dtype=object)
    error = F(sum(abs(v) for _, v in translations), DENOMINATOR*SCALE)
    zero_mode = F(0)
    for index, v in translations:
        assert 0 <= index < len(ball.rows)
        if v:
            residual -= v*ball.rows[index].astype(object)
    rows = list(additional_rows(ball, ratio, shell, covariances))
    for index, v in inequalities:
        assert 0 <= index < len(TESTS) and v > 0
        tag, vector = TESTS[index]
        if tag == "extension":
            row, radius, mass = exact_reflection(vector, bare, features)
        else:
            row, radius, mass = phase.spectral_cut(
                phase.polynomial(tag, vector), angular, bare, green
            )
        rows.append((v, row, radius, mass))
    for v, row, radius, mass in rows:
        assert v > 0 and mass >= 0
        residual += v*row.astype(object)
        error += F(v, DENOMINATOR)*radius
        zero_mode += F(v, DENOMINATOR)*mass
    assert zero_mode > 0
    worst = int(residual.argmax())
    upper = F(int(residual[worst]), DENOMINATOR*TOTAL_SCALE) + error
    lower = -upper/zero_mode
    print(f"r={ratio}  boundary orbits={len(residual)}  worst orbit index={worst}")
    print(f"U={upper}\nQ={zero_mode}\nm_plus^2 >= {lower} = {float(lower):.12g}", flush=True)
    return lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true",
                        help="replay the two degree-eight certificates")
    args = parser.parse_args()
    green = phase.AngularGreen(F(0))
    if args.extended:
        for ratio, floor, translations, inequalities, shell, covariances in EXTENDED_PROOFS:
            assert F(25, 16) < ratio < F(158307, 100000)
            assert check(ratio, green, 8, translations, inequalities, shell, covariances) > floor
    else:
        ratio = F(158307, 100000)
        assert F(25, 16) < ratio < F(18997, 12000)
        assert check(ratio, green, 4, tuple(enumerate(TRANSLATIONS)), TEST_MULTIPLIERS) > F("0.0000178")
