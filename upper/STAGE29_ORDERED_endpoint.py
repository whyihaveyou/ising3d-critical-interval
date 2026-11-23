"""A conditional corner law and its ordered-endpoint integer certificate."""

import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
from math import floor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import STAGE28_ORDERED_ENDPOINT_precise as precise
import STAGE29_ORDERED_corner as corner


HERE = Path(__file__).resolve().parent
GRID = 10**14


def gradient(caps, sizes, rates):
    """Use the finite LP only to propose nonnegative integer multipliers."""
    classes, edges = sorted(sizes), sorted(rates)
    A = np.zeros((len(edges), len(classes)))
    for k, (i, j) in enumerate(edges):
        A[k, classes.index(i)] = float(rates[i,j]/sizes[i])
        A[k, classes.index(j)] = -1.0/sizes[j]
    y = np.array([float(caps[c][0]) for c in classes])
    sol = linprog(-10**8*y, A_ub=A, b_ub=np.zeros(len(A)),
                  A_eq=[np.ones(len(y))], b_eq=[1],
                  bounds=(0, None), method='highs')
    assert sol.success
    weighted = []
    for (i,j), m in zip(edges, sol.ineqlin.marginals):
        weight = max(0, round(-m*GRID/10**8))
        if weight:
            weighted.append([i,j,weight])
    return weighted


def shifts(dual, sizes, rates):
    answer, seen = dict.fromkeys(sizes, F(0)), set()
    for i,j,weight in dual:
        assert type(weight) is int and weight > 0 and (i,j) in rates and (i,j) not in seen
        seen.add((i,j))
        answer[i] -= F(weight, GRID)*rates[i,j]/sizes[i]
        answer[j] += F(weight, GRID)/sizes[j]
    return answer


def run(point, mode):
    assert __debug__ and F(791, 500) < point < F(1666, 1053)
    archive = HERE / f'STAGE29_ORDERED_{point.numerator}_{point.denominator}.json'
    source = json.loads(precise.SOURCE.read_text())
    base = precise.base
    base.POINT = point
    _, check_sizes, check_edges = base.class_geometry()
    sizes, rates = corner.verify(point)
    assert sizes == check_sizes and sorted(rates) == sorted(check_edges)
    model, pool, layout = base.make_model()
    base.validate_rows(source['rows'], pool)
    residual, scale, radius, mass = precise.build_residual(model, source)
    assert radius > 0 and mass == base.stage.MASS > 0
    if mode == 'raw' or (mode == 'directed' and archive.exists()):
        saved = json.loads(archive.read_text())
        caps = {int(k):(F(v), w) for k,(v,w) in saved['caps'].items()}
        dual = saved['class_dual']
    else:
        caps = base.propose_caps(model, residual, scale, layout)
        dual = gradient(caps, sizes, rates)
    assert caps.keys() == sizes.keys()
    shift = shifts(dual, sizes, rates)
    owner = max(sizes, key=lambda k:caps[k][0]+shift[k])
    cap, field = caps[owner]
    weights = base.stage.corner_weights(model, owner, layout[1], layout[3])
    weights = base.stage.direct_field(weights, layout[2], layout[4], layout[5], field)
    witness = F(int(weights @ residual[model.owners]), int(weights.sum())*scale)
    assert 0 <= cap-witness < F(1, 10**7)
    upper = cap + shift[owner] + radius
    lower = -upper/mass
    out = floor(10**15*lower)
    print(f'r={point} active=({owner},{field}) dual_rows={len(dual)} '
          f'floor(1e15 U)={floor(10**15*upper)}, floor(1e15 L)={out}', flush=True)
    if mode == 'probe':
        return
    assert 0 < F(out,10**15) < lower and upper < 0
    base.fields.check_fields(model, residual, scale, layout,
                             {c:v for c,(v,f) in caps.items()}, raw=mode == 'raw')
    certificate = dict(r=str(point), source=precise.SOURCE.name,
                       source_sha256=sha256(precise.SOURCE.read_bytes()).hexdigest(),
                       odds_by_matches=[str(v) for v in sorted(set(rates.values()))],
                       class_dual=dual, class_denominator=GRID,
                       caps={str(c):[str(v),f] for c,(v,f) in caps.items()},
                       local_radius=str(radius), U=str(upper), Q=str(mass), L=str(lower),
                       floor_1e15_L=out, active=[owner,field])
    if mode == 'directed' and not archive.exists():
        archive.write_text(json.dumps(certificate, indent=2)+'\n')
    assert certificate == json.loads(archive.read_text())
    print(f'PASS {mode}, {len(sizes)*len(model.owners)} fields, 1e15 floor {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('point', type=F)
    parser.add_argument('mode', choices=('probe','directed','raw'))
    args = parser.parse_args()
    run(args.point, args.mode)
