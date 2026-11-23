"""Check the small-torus quantifiers and exact endpoint/sector calculations."""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path


HERE = Path(__file__).resolve().parent
H = F(8362020128, 10**12)
THETA = F(9, 10000)
WALK_RATE = F(20811153908856, 4398046511088)


def upper_from_receipt(name, weight, iteration):
    lines = (HERE / name).read_text().splitlines()
    assert lines[0].startswith(
        "memory 16 states 12994574 labels 53221763 contact "
        f"{weight.numerator}/{weight.denominator} ")
    row = next(line for line in lines if line.startswith(f"iteration {iteration} "))
    fields = row.split()
    fraction = F(fields[5])
    bound = F(int(fields[8]), 10**8)
    assert bound - F(1, 10**8) < fraction <= bound
    return bound


def endpoint_and_sectors():
    upper = upper_from_receipt("STAGE54_CONTACT_ROBUST16.log", F(10000, 9991), 40)
    assert upper == F(473271745, 10**8)
    r = F(19269, 12500)
    u = (r-1) / (r+1)
    old_net = H - THETA/4
    assert F(94639, 20000) * u > 1 / (1-old_net)
    lower_exp = 1 + H + H**2/2 + H**3/6
    margin = (lower_exp/(u*upper) - 1) * 10**12 // 1
    assert margin == 74673
    print("r", r, "all-path contraction margin floor x1e12", margin)
    u = F(109, 509)
    for name, w, d, bound, expect in (
        ("SPARSE", F(4, 5), F(137, 1000), 456604381, 113923862243),
        ("DENSE", F(6, 5), F(253, 1000), 492941352, 132118107306),
    ):
        upper = upper_from_receipt(f"STAGE54_CONTACT_{name}16.log", w, 32)
        assert upper == F(bound, 10**8)
        g = H - d*THETA
        test = (u*upper/(1+g+g**2/2))**d.denominator / w**d.numerator
        margin = (1-test)*10**12 // 1
        assert margin == expect
        print("r 309/200", name.lower(), "density", d,
              "normalized block margin floor x1e12", margin)
    obstruction = u * (1-H) * WALK_RATE
    assert obstruction*10**12//1 == 1004842998330
    print("all-local-successors middle-sector obstruction floor x1e12",
          obstruction*10**12//1)


def subset_comparison():
    rows = 0
    weights = (F(4,5), F(10000,9991), F(6,5))
    for m in range(1,6):
        for entries in product((0,1,9), repeat=m):
            for w in weights:
                exact = max(w**(5-len(A))*sum(entries[i] for i in A)
                            for k in range(1,m+1) for A in combinations(range(m),k))
                total = sum(entries)
                if w <= 1 or min(entries)*w >= total*(w-1):
                    fast = w**(5-m)*total
                else:
                    best = sorted(entries, reverse=True)
                    fast = max(w**(5-k)*sum(best[:k]) for k in range(1,m+1))
                assert fast == exact
                rows += 1
    print("arbitrary subsets versus sorted / one-row shortcut", rows)


def move(point, direction):
    p = list(point)
    p[direction//2] += 1 if direction%2 == 0 else -1
    return tuple(p)


def torus_rows(length, memory, limit):
    start = (0,0,0)
    trail = [start]
    occupied = {start}
    checks = [0,0]

    def visit():
        n = len(trail)-1
        if n == limit:
            return
        tail = trail[-(memory+1):]
        lifts = [move(trail[-1],d) for d in range(6)]
        projected = [tuple(x%length for x in point) for point in lifts]
        allowed = {d for d,p in enumerate(projected) if p not in occupied}
        local = {d for d,p in enumerate(lifts) if p not in tail}
        assert allowed <= local
        if n:
            past = {tuple(x%length for x in p) for p in trail[:-2]}
            contacts = sum(p in past for p in projected)
            assert 5-len(allowed) == contacts
            checks[0] += 1
            checks[1] += int(len(local)>len(allowed))
        for d in allowed:
            occupied.add(projected[d])
            trail.append(lifts[d])
            visit()
            trail.pop()
            occupied.remove(projected[d])

    visit()
    assert checks[1] > 0
    print("torus side", length, "noninitial prefixes", checks[0],
          "prefixes with a hidden contact", checks[1])


if __name__ == "__main__":
    endpoint_and_sectors()
    subset_comparison()
    torus_rows(4, 3, 7)
    torus_rows(8, 3, 7)
