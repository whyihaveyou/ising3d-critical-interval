"""Propagate the exact endpoint caps with the individual physical-row slopes."""

from fractions import Fraction as F
from math import ceil, floor
import json
import sys

from run import HERE, SOURCE, read_inputs
sys.path.insert(0, str(HERE.parent))
import verify_stage33_upper_margin as prior


class Box:
    """Round each exact rational interval outwards to the 2^-120 grid."""
    unit = 1 << 120

    def __init__(self, value):
        self.x = prior.Interval(F(floor(value.lo*self.unit), self.unit),
                                F(ceil(value.hi*self.unit), self.unit))

    @staticmethod
    def exact(other):
        return other.x if isinstance(other, Box) else prior.interval(other)

    def __add__(self, other):
        return Box(self.x + self.exact(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Box(self.x - self.exact(other))

    def __rsub__(self, other):
        return Box(self.exact(other) - self.x)

    def __mul__(self, other):
        return Box(self.x * self.exact(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Box(self.x / self.exact(other))

    def __rtruediv__(self, other):
        return Box(self.exact(other) / self.x)

    def __pow__(self, n):
        return Box(self.x ** n)


def odds_on_interval(point, delta):
    exact = prior.local_factors(1/point)
    safe = [F(floor(x*prior.GRID),prior.GRID) for x in exact]
    assert exact[:4] == prior.odds(point)
    # Partition the interval in s = 1/r, where the local formulas are evaluated.
    start, stop = 1/point, 1/(point-delta)
    partitions = 16
    grid = [start+(stop-start)*i/partitions for i in range(partitions+1)]
    vertices = [prior.local_factors(s) for s in grid]
    excess = []
    for i in range(partitions):
        closed = prior.Interval(grid[i],grid[i+1])
        odds = prior.local_factors(Box(closed))
        for j, (cut, outer) in enumerate(zip(safe,odds)):
            for inner in (vertices[i][j],vertices[i+1][j]):
                assert outer.x.lo <= inner <= outer.x.hi
            assert outer.x.lo > cut > 0
            excess.append(outer.x.lo-cut)
    print("minimum box lower endpoint minus frozen odds >=",
          F(floor(min(excess)*10**14),10**14), flush=True)
    return safe


def run():
    assert __debug__
    _, archive = read_inputs()
    point, radius, U, Q = (F(archive[k]) for k in ("r","radius","U","Q"))
    delta = F(4075,10**14)
    claimed = F(407,10**13)
    assert 0 < claimed < delta and F(3,2) < point-delta < point < F(8,5)
    source = json.loads(SOURCE.read_text())
    rows = source["rows"]
    assert prior.slope_bound(source) == 18000000
    w = rows["ordinary"]
    spectral = set(prior.green.INPUTS)
    reflection = {42,45,76,83}
    assert set(i for i,v in w) == spectral | reflection and spectral.isdisjoint(reflection)
    assert [len(rows[k]) for k in ("translations","ordinary","walsh")] == [104,9,7]
    denominator = source["denominator"]*10**6
    trans = F(sum(abs(v) for i,v in rows["translations"]),denominator)
    spec = F(sum(v for i,v in w if i in spectral),denominator)
    refl = F(sum(v for i,v in w if i in reflection),denominator)
    delete = F(sum(v for i,j,v in rows["walsh"]),denominator)
    rate = 144*trans + 228*spec + 57792*refl + 14208*delete
    assert rate < 8326
    with (HERE/"model.txt").open() as model:
        assert model.readline().strip() == "STAGE34v1"
        n,c,o,h = map(int,model.readline().split())
        assert (n,c,o,h) == (18,8,3236,14)
        scale = int(model.readline())
        for _ in range(n+2):
            next(model)
        residuals = [F(int(next(model).split()[1]),scale) for _ in range(o)]
    center = max(map(abs,residuals)) + radius
    local_max = F(71,1000)
    assert center + 8326*delta/(point-delta) < local_max
    # Cauchy-Schwarz bounds the covariance by sqrt(E[R^2] E[(log W)'^2]).
    assert 468 == 36 + 24*6 + 8*6 + 6*(2*6+16) + 12*6
    cap_rate = 8326 + 468*local_max
    safe = odds_on_interval(point,delta)
    sizes, pairs = prior.class_graph()
    assert U == prior.combine(archive,safe,sizes,pairs)
    assert -U/Q == F(archive["L"]) and Q > 0
    print("individual-row log-r slope bound", rate)
    print("frozen-field log-r slope bound", cap_rate)
    for d in (claimed,delta):
        floor_lower = floor(-(U+cap_rate*d/(point-d))/Q * 10**15)
        assert 0 < F(floor_lower,10**15) < -(U+cap_rate*d/(point-d))/Q
        print("delta", d, "left", point-d, "uniform floor(1e15 lower)",floor_lower)
    print("PASS uniform positive magnetization to",point-delta,
          "and strict critical upper bound",point-claimed,flush=True)


if __name__ == "__main__":
    run()
