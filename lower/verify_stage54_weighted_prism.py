"""Audit the priced three-bond inequality using parity subgraphs and actual directions."""

from fractions import Fraction as F
from itertools import product

from STAGE53_PRISM_CHECK import (RING, UNIT, compare_spin_solver, even_statistics,
                                  integral_tables, vertex_gains)
from STAGE54_WEIGHTED_shapes import CHARGES, PRICE


def outward_directions(mask, x):
    axial = [(x, 0, 0)] if mask & 16 else []
    return axial + [(0, y, z) for j, (y, z) in enumerate(RING) if mask & (1 << j)]


def classify(before, after):
    axial_count = (before[0] != 0) + (after[0] != 0)
    if axial_count == 2:
        return 0
    if axial_count == 1:
        return 1
    # The outer endpoints differ by the central bond when the offsets coincide.
    transverse_dot = sum(a*b for a, b in zip(before, after))
    return {1: 2, 0: 3, -1: 4}[transverse_dot]


def check():
    canonical, reps, polynomials = even_statistics()
    u = F(677, 3177)
    lower, upper = integral_tables(polynomials, reps, u, 64)
    compare_spin_solver(u, 64, canonical, lower, upper)
    edge_charge, shape_charge = CHARGES[:20], CHARGES[20:]
    path_slack, chord_slack = [], []
    for (a, b), rep in canonical.items():
        slots = [bin(mask & 15).count("1")+5*(mask >> 4) for mask in (a, b)]
        left, right = outward_directions(a, -1), outward_directions(b, 1)
        for before, after in product(left, right):
            charge = sum(edge_charge[i] for i in slots) + shape_charge[classify(before, after)]
            path_slack.append(lower[rep]-charge)
        if min(len(left), len(right)) >= 2:
            chord_slack.append(lower[rep]+PRICE*UNIT-sum(edge_charge[i+10] for i in slots))
    assert min(path_slack) == 301 and min(chord_slack) == 1301
    stars = vertex_gains(edge_charge)
    h = F(min(row[0] for row in stars), UNIT)
    assert len(stars) == 240 and h == F(8382656215, UNIT)
    ceiling = F(4730332, 10**6)
    assert (5/ceiling)**4 < F(5, 4)
    g = h - PRICE/4
    taylor = 1+g+g**2/2+g**3/6
    endpoint = (ceiling+taylor)/(ceiling-taylor)
    assert endpoint == F(275442362401406827838944754146320499627,
                         178669509598593172161055245853679500373)
    print("actual directions: priced path rows", len(path_slack), "chord rows", len(chord_slack),
          "vertex stars", len(stars), "minimum slacks", min(path_slack), min(chord_slack))
    print("contraction margin floor x1e12", (taylor/(u*ceiling)-1)*UNIT//1)
    print("strict lower endpoint", endpoint, "floor x1e15", endpoint*10**15//1)


if __name__ == "__main__":
    check()
