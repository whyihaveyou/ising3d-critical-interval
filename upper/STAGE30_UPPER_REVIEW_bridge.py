"""Use the archived shared-bridge flip inequalities to reduce each class cap."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
from math import floor
from pathlib import Path

import numpy as np

import STAGE28_ORDERED_ENDPOINT_precise as precise
import STAGE29_ORDERED_endpoint as endpoint


HERE = Path(__file__).resolve().parent
GRID = 10**9


def geometry(point, ratios):
    s = 1/point
    theta = (s**2+s**4)/(1+s**6)
    boundary = tuple(precise.G.BOUNDARY)
    edges = []
    for i,a in enumerate(boundary):
        if max(map(abs,a)) != 2:
            continue
        for j,b in enumerate(boundary):
            if max(map(abs,b)) != 1:
                continue
            # Share the L1=3 common neighbour of type (2,1,0).
            possible = {precise.G.add(a,d) for d in precise.G.DIRECTIONS}
            shared = [z for z in possible if sorted(map(abs,z)) == [0,1,2] and
                      sum(abs(x-y) for x,y in zip(z,b)) == 1]
            if shared:
                assert len(shared) == 1
                edges.append((i,j))
                assert ratios[j] == theta
    assert len(edges) == 24
    omega = []
    for rho in ratios:
        x = rho/theta
        assert 0 < rho < x <= 1
        geometric = rho*(x-rho)/(1-rho*x)
        # Rounding down makes 1/omega an upper bound on mass transferred.
        safe = F(floor(geometric*10**10),10**10)
        assert 0 < safe <= geometric
        omega.append(safe)
    return edges, omega


def averaged_cap(record, edges, omega):
    upper, low = F(record["upper"]), F(record["low"])
    assert low <= upper
    assignments = record["words"]
    if not assignments:
        assert upper == low
        return upper
    high = {f for f,_,_ in assignments}
    assert len(assignments) == len(high) and all(0 <= f < (1<<18) for f in high)
    bad_mass = defaultdict(F)
    for f,i,j in assignments:
        assert (i,j) in edges and (((f>>i)^(f>>j)) & 1)
        a,b = f ^ (1<<i), f ^ (1<<j)
        assert a not in high and b not in high
        bad_mass[a] += 1/omega[i]
        bad_mass[b] += 1/omega[j]
    congestion = max(bad_mass.values())
    # mass(H) <= congestion * mass(H^c).
    return upper-(upper-low)/(1+congestion)


def verify_caps(model, v, scale, layout, records, raw):
    compact = precise.base.fields.compact
    stage = precise.base.stage
    classes,counts,ratios,trace,_,_ = layout
    bounds = [compact.coefficients(F(int(x),int(trace[0]))) for x in trace]
    lows, highs = [np.asarray(x,dtype=np.int64) for x in zip(*bounds)]
    compact.check_products()
    for n,c in enumerate(classes,1):
        rec = records[c]
        cap = F(rec["upper"])
        low = F(rec["low"])
        high_indices = [f for f,i,j in rec["words"]]
        # First pass verifies all words at 'cap', second verifies the lower
        # cap on the complement of the explicitly checked high set.
        weights = stage.corner_weights(model,c,counts,trace) if raw else None
        for cutoff, exceptions in [(cap,[])] + ([(low,high_indices)] if high_indices else []):
            dif = cutoff.numerator*scale - cutoff.denominator*v
            if raw:
                values = stage.W.fields.transform(weights*dif[model.owners],ratios)
            else:
                residual = [int(w)*int(z) for w,z in zip(model.weights,dif)]
                values = compact.normalize_lower(residual, model.owners)
                for j, tally in enumerate(counts):
                    k = 3-tally if (c>>j)&1 else tally
                    values = compact.multiply_lower(values,lows[k],highs[k])
                values = compact.average_lower(values,ratios)
            if exceptions:
                values[exceptions] = 0
            assert min(values) >= 0, (c, cutoff)
        if n in (1,7,14):
            print("source class",n,"of 14 checked",flush=True)


def run(point, mode):
    if not __debug__ or point != F(5441,3439) or mode not in ("directed", "raw"):
        raise ValueError("use the frozen 5441/3439 certificate with assertions enabled")
    path = HERE / "STAGE30_UPPER_REVIEW_5441_3439.json"
    record = json.loads(path.read_text())
    assert F(record["r"]) == point
    assert hashlib.sha256(precise.SOURCE.read_bytes()).hexdigest() == record["source_sha256"]
    base = precise.base
    base.POINT = point
    source = json.loads(precise.SOURCE.read_text())
    model, pool, layout = base.make_model()
    base.validate_rows(source["rows"],pool)
    v, scale, radius, Q = precise.build_residual(model,source)
    edges, omega = geometry(point, layout[2])
    sizes, rates = endpoint.corner.verify(point)
    records = {int(c): item for c, item in record["class_bounds"].items()}
    dual = record["class_dual"]
    assert sizes.keys() == records.keys()
    caps = {c:averaged_cap(rec,edges,omega) for c,rec in records.items()}
    offset = endpoint.shifts(dual,sizes,rates)
    upper = max(caps[c]+offset[c] for c in caps) + radius
    lower = -upper/Q
    print("r",point,"floor(1e15 m2 lower)",floor(lower*10**15),flush=True)
    assert lower > F(floor(lower*10**15),10**15) > 0
    verify_caps(model,v,scale,layout,records,raw=(mode=="raw"))
    item = dict(r=str(point), source_sha256=hashlib.sha256(precise.SOURCE.read_bytes()).hexdigest(),
                Q=str(Q),radius=str(radius),class_dual=dual, class_denominator=endpoint.GRID,
                U=str(upper),L=str(lower), floor_1e15_L=floor(lower*10**15),
                class_bounds={str(c):rec for c,rec in records.items()})
    assert record == item
    print("PASS",mode,"14 classes, all 3670016 fields with directed shared-bridge constraints",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("r",type=F)
    p.add_argument("mode", choices=("directed","raw"))
    a=p.parse_args()
    run(a.r,a.mode)
