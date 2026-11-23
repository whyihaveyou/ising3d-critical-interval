#!/usr/bin/env python3
"""Exact checks of every scalar rational relation quoted in the manuscript.

Uses only the Python standard library.  Run:  python3 -B check_numbers.py

The large certificates (prism integrals, automaton rows, field traversals)
are checked by the programs listed in Appendix A.  This script re-derives the
scalar relations between their outputs as printed in the paper.  The exact
values of U_ord(r_b) and L_b are copied from the ordered-endpoint record
work/manuscripts/critical_interval/supplement/upper/new_endpoint/
STAGE30_UPPER_REVIEW_3775_2386.json.
"""
from fractions import Fraction as F
from itertools import combinations

ok = []


def check(name, cond):
    assert cond, name
    ok.append(name)


def floor(x):
    return x.numerator // x.denominator


def log_bounds(x, terms=60):
    """Rational enclosure of log(x), x > 1, via log x = 2 artanh((x-1)/(x+1))."""
    z = (x - 1) / (x + 1)
    assert 0 <= z < 1
    s, p = F(0), z
    for j in range(terms):
        s += p / (2 * j + 1)
        p *= z * z
    s *= 2
    tail = 2 * p / ((2 * terms + 1) * (1 - z * z))
    return s, s + tail


def T3(x):
    return 1 + x + x ** 2 / 2 + x ** 3 / 6


# ---------------------------------------------------------------- endpoints
r_lo = F(275442362401406827838944754146320499627, 178669509598593172161055245853679500373)
r_hi = F(18874999999514449, 11930000000000000)
r_b, d1, d0 = F(3775, 2386), F(4075, 10 ** 14), F(407, 10 ** 13)
check("r- decimal", F("1.541630483120638") < r_lo < F("1.541630483120639"))
check("r+ decimal", F("1.582145850755611") < r_hi < F("1.582145850755612"))
check("r- < r+", r_lo < r_hi)
check("r+ = rb - 407e-13", r_b - d0 == r_hi)
check("rb - d1 < r+", r_b - d1 < r_hi)
lo, _ = log_bounds(r_lo)
_, hi = log_bounds(r_hi)
check("K lower", lo / 2 > F("0.2164203058063"))
check("K upper", hi / 2 < F("0.2293910294998"))

# ---------------------------------------------------------------- couplings
u = lambda r: (r - 1) / (r + 1)
u_star, u1, u2 = u(F(30829, 20000)), u(F(19269, 12500)), u(F(1927, 1250))
check("u*", u_star == F(10829, 50829))
check("u1", u1 == F(6769, 31769))
check("u2", u2 == F(677, 3177))
check("u* < u1, u* < u2", u_star < u1 and u_star < u2)

# ---------------------------------------------------------------- allocations
P1 = [0, 4181010088, 2796622230, 1408633261, 0, 4181010088, 2796622230, 1696196670, 1287882389, 549724752]
C1 = [0, 0, 2768775703, 2484813406, 929924211, 0, 2768775703, 2772376814, 2428165575, 1815642656]
P2 = [0, 4191328333, 2699473836, 1309065313, 0, 4191328333, 3406623940, 1425837850, 738905393, 0]
C2 = [0, 0, 2983708843, 2765490332, 1809011447, 0, 2276558740, 2882262869, 2547916991, 2095664091]
DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def vertex_rows(P, C):
    """All instances of condition (A3); returns (count, minimum)."""
    def typ(d, S):
        neg = tuple(-x for x in d)
        return sum(1 for e in S if e not in (d, neg)) + 5 * (neg in S)
    vals = [sum(P[typ(d, S)] if d in pp else C[typ(d, S)] for d in S)
            for k in range(2, 7) for S in combinations(DIRS, k) for pp in combinations(S, 2)]
    return len(vals), min(vals)


gamma1, gamma2 = F(8362020128, 10 ** 12), F(8382656215, 10 ** 12)
theta1, theta2 = F(9, 10000), F(1457355254, 10 ** 12)
n, m = vertex_rows(P1, C1)
check("(A3) table 1: 240 rows, minimum = gamma1", n == 240 and F(m, 10 ** 12) == gamma1)
n, m = vertex_rows(P2, C2)
check("(A3) table 2: 240 rows, minimum = gamma2", n == 240 and F(m, 10 ** 12) == gamma2)

W_TAU = [1127558275, 544046375, 544046375, 94806903, 0]
OMEGA_BAR = [998874, 999457, 999457, 999906, 1000000]
for w, o in zip(W_TAU, OMEGA_BAR):
    x = F(w, 10 ** 12)
    check(f"Omega bar for w = {w}", -floor(-(1 - x + x * x / 2) * 10 ** 6) == o)

# ---------------------------------------------------------------- transfer and Hoelder
Lam1, U2 = F(473271745, 10 ** 8), F(1182583, 250000)
w1 = F(10000, 9991)
check("w1 = 1/(1-theta1)", w1 == 1 / (1 - theta1))
check("row quotient < Lambda1", F(7337944694907297, 1550471750000000) < Lam1)
check("(5/U2)^4 < 5/4", (5 / U2) ** 4 < F(5, 4))
lo125, _ = log_bounds(F(5, 4))
check("theta2 < 1/5 < log(5/4)", theta2 < F(1, 5) < lo125)

# ---------------------------------------------------------------- lower endpoint
g2 = gamma2 - theta2 / 4
check("gamma bar 2", g2 == F(16036634803, 2 * 10 ** 12))
check("endpoint-lower (a) floor", floor(10 ** 12 * (T3(g2) / (U2 * u2) - 1)) == 44289252)
check("endpoint-lower (a) r-", (U2 + T3(g2)) / (U2 - T3(g2)) == r_lo)
check("endpoint-lower (c) floor", floor(10 ** 12 * (T3(gamma1) / (Lam1 * u1) - 1)) == 74673)
check("intermediate points below r-", F(19269, 12500) < r_lo and F(1927, 1250) < r_lo)

# ---------------------------------------------------------------- upper endpoint
Q = F(8943787728743670589, 8943788333028500000)
U_ORD = F("-6270492790229122632162179551923249324561213045958902051380266559007791858425972650003148961406834786201517473043043545770547/"
          "29112770712349929546078302950594870279523440367412164114496068188042923996150279337388595828112031219712000000000000000000000000000")
L_B = F("6270492790229122632162179551923249324561213045958902051380266559007791858425972650003148961406834786201517473043043545770547/"
        "29112768745352832954808301409409608029346777129135814591321381100646492131179912461963312364181901151775490048000000000000000000000")
check("L_b = -U_ord/Q", L_B == -U_ORD / Q)
check("floor 1e15 L_b with positive remainder", floor(10 ** 15 * L_B) == 215386342 and 10 ** 15 * L_B > 215386342)
C_STAR = 8326 + F(468) * F(71, 1000)
check("C* = 2089807/250", C_STAR == F(2089807, 250))
check("interval (e)", L_B - C_STAR * d1 / (Q * (r_b - d1)) > F(84723, 10 ** 15))
check("margin at d0", floor(10 ** 15 * (L_B - C_STAR * d0 / (Q * (r_b - d0)))) == 348897)
dd = F(3, 2 * 10 ** 14)
check("coarse interval with rate 18000000", floor(10 ** 15 * (L_B - 18000000 * dd / (Q * (r_b - dd)))) == 44732026)
check("120 rows", 104 + 4 + 7 + 5 == 120)
check("field counts", 14 * 2 ** 18 == 3670016 and 3670016 - 3661604 == 8412)
check("derivative constants", (2 * 72, 3 * 72 + 12, 112 * (7 * 72 + 12), 32 * (2 * 72 + 300)) == (144, 228, 57792, 14208))
for r in (r_b - d1, r_b):
    check("|d c0/dy| < 12 on the interval", 8 * r * r / (r * r - 1) ** 2 < 12)

print(f"PASS: {len(ok)} exact checks")
print(f"r- = {float(r_lo):.16f}   r+ = {float(r_hi):.16f}   width = {float(r_hi - r_lo):.14f}")
