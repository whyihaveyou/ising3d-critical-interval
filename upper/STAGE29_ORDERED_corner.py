"""Source-flip odds after summing 21 additional ordinary Ising spins."""

from collections import Counter
from fractions import Fraction as F
from itertools import product
from math import floor, prod


CORNERS = tuple(product((-1, 1), repeat=3))
STEPS = tuple(tuple(sign if i == axis else 0 for i in range(3))
              for axis in range(3) for sign in (-1, 1))


def shift(point, step, length=1):
    return tuple(x + length*y for x, y in zip(point, step))


def neighbours(point):
    return {shift(point, step) for step in STEPS}


def external(point):
    return {x for x in neighbours(point) if sum(map(abs, x)) == 4}


REMOVED = {frozenset((c, x)) for c in CORNERS for x in external(c)}


def ordinary_neighbours(point):
    return {x for x in neighbours(point) if frozenset((x, point)) not in REMOVED}


def geometry():
    """Validate the retained edges on the once-augmented periodic graph."""
    for y in CORNERS:
        inward = [tuple(-v if j == i else 0 for j,v in enumerate(y))
                  for i in range(3)]
        outward = [tuple(-v for v in e) for e in inward]
        directions = inward + outward
        first = [shift(y, e) for e in directions]
        second = [shift(y, e, 2) for e in directions]
        core = {y, *first, *second}
        branches = [ordinary_neighbours(q)-{p} for p, q in zip(first, second)]
        assert len(core) == 13 and [len(b) for b in branches] == [2]*3+[5]*3
        leaves = set.union(*branches)
        assert len(leaves) == 21 and leaves.isdisjoint(core)
        keep = leaves | core
        assert {x for x in keep if x in CORNERS} == {y, *second[:3]}
        for c in CORNERS:
            assert keep & external(c) == (set(first[3:]) if c == y else set())
        assert ordinary_neighbours(y) == set(first[:3])
        for i, (p, q, branch) in enumerate(zip(first, second, branches)):
            bonds = ordinary_neighbours(p)
            assert bonds & keep == ({q,y} if i < 3 else {q})
            assert len(bonds-keep) == 4
            assert ordinary_neighbours(q) == {p, *branch}
            for z in branch:
                bonds = ordinary_neighbours(z)
                assert bonds & keep == {q} and len(bonds-keep) == 5


def odds(point):
    s = 1/point
    v = s**3
    def across(x):
        return (s+x)/(1+s*x)
    leaf = across(s**5)
    outer = across(s**4 * across(leaf**5))**3
    mismatch = across(s**4 * across(leaf**2 * s**3))
    match = across(s**4 * across(leaf**2 / s**3))
    assert s < leaf < 1 and v < outer < 1 and 0 < mismatch < match
    result = []
    for n in range(4):
        inner = mismatch**(3-n)*match**n
        result.append((outer-v)*(inner+v)/((1-v*outer)*(1+v*inner)))
    assert 0 < result[0] < result[1] < result[2] < result[3] < 1
    return result


def direct(point, n):
    """Remove only the terminal leaves, then sum all 8192 core spin words."""
    a, b = point.as_integer_ratio()
    def bond(x, y):
        return a if x == y else b
    leaf = [sum(bond(z, w)*bond(w, 1)**5 for w in (0,1)) for z in (0,1)]
    spins = tuple(product((0,1), repeat=6))
    fixed = [0]*n + [1]*(3-n)
    at_ends = {}
    for z in spins:
        at_ends[z] = (prod(leaf[u]**2*bond(u, t)**3 for u,t in zip(z, fixed))
                      * prod(leaf[u]**5 for u in z[3:]))
    out = [0, 0]
    for x in spins:
        j = sum(x[3:])
        p, m = a**(3-j)*b**j, a**j*b**(3-j)
        lam = [a**3*p-b**3*m, a**3*m-b**3*p]
        assert min(lam) >= 0
        first = prod(bond(u, 1)**4 for u in x)
        contraction = sum(at_ends[z]*prod(bond(u,w) for u,w in zip(x,z))
                          for z in spins)
        for t in (0,1):
            middle = sum(bond(y, t)**3*prod(bond(y,u) for u in x[:3]) for y in (0,1))
            out[t] += first*contraction*middle*lam[t]
    return F(out[0], out[1])


def class_graph():
    from itertools import permutations
    owners = []
    maps = [[CORNERS.index(tuple(c[p[i]]*sgn[i] for i in range(3))) for c in CORNERS]
            for p in permutations(range(3)) for sgn in CORNERS]
    for word in range(256):
        images = [sum(((word >> i) & 1) << j for i,j in enumerate(mapping))
                  for mapping in maps]
        owners.append(min(min(images), 255-max(images)))
    sizes = Counter(owners)
    edges = {}
    for new in range(256):
        for j, y in enumerate(CORNERS):
            old = new ^ (1 << j)
            nearby = [k for k,z in enumerate(CORNERS)
                      if sum(abs(v-w) for v,w in zip(y,z)) == 2]
            matches = sum(not ((new >> k) ^ (new >> j)) & 1 for k in nearby)
            e = owners[old], owners[new]
            edges[e] = max(edges.get(e, 0), matches)
    assert len(sizes) == 14 and len(edges) == 40
    assert Counter(edges.values()) == {0:8,1:12,2:12,3:8}
    return sizes, edges


def verify(point):
    geometry()
    rates = odds(point)
    for n, rate in enumerate(rates):
        assert direct(point, n) == rate
    sizes, edges = class_graph()
    # These strict floors can be frozen in a small open temperature interval.
    floors = [F(floor(v*10**10), 10**10) for v in rates]
    assert all(0 < a < b for a,b in zip(floors, rates))
    print(f'r={point} odds_lower_1e10={[floor(v*10**10) for v in rates]}', flush=True)
    print('PASS 8 corner geometries, 21 actual leaves, 4 exact 8192-term sums, '
          '256 words and 40 oriented class inequalities', flush=True)
    return sizes, {e:floors[n] for e,n in edges.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('point', type=F)
    verify(parser.parse_args().point)
