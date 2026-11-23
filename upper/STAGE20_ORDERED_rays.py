"""Check the frozen ordered certificate with six extended axial segments."""

import argparse
from fractions import Fraction
from itertools import product
import json
from math import floor
from pathlib import Path

import verify_stage18_ordered_spectral_20261121 as previous
import STAGE20_ORDERED as extended


def axial_factor(ratio, length):
    s = 1 / ratio
    z = s
    for _ in range(length):
        next_z = (s + s**4 * z) / (1 + s**5 * z)
        assert z < next_z < 1
        z = next_z
    # With the end and four side neighbours at each site fixed positive,
    # enumerate the physical chain directly. A spin bit of 1 means minus.
    a, b = ratio.numerator, ratio.denominator
    sums = [0, 0]
    for left in (0, 1):
        for bits in product((0, 1), repeat=length):
            chain = (left, *bits, 0)
            n = sum(p != q for p, q in zip(chain, chain[1:])) + 4*sum(bits)
            sums[left] += b**n * a**(5*length+1-n)
    assert z == Fraction(sums[1], sums[0])
    return z


def validate_geometry(model, length):
    geom = previous.W.geometry
    sphere = set(geom.BOUNDARY)
    shell = {p for v in sphere for p in previous.W.fields.neighbours(v)} - set(geom.INTERIOR)
    assert len(shell) == 38
    extra = {tuple(k*i for i in e) for e in geom.DIRECTIONS for k in range(4, length+3)}
    assert len(extra) == 6*(length-1)
    retained = sphere | shell | extra
    for e in geom.DIRECTIONS:
        line = [tuple(k*i for i in e) for k in range(2, length+3)]
        for k in range(1, len(line)):
            neighbours = previous.W.fields.neighbours(line[k])
            expected = {line[k-1], *line[k+1:k+2]}
            assert neighbours & retained == expected
    return sphere


def extend_rays(model, length):
    assert 2 <= length <= 10
    sphere = validate_geometry(model, length)
    one = axial_factor(model.ratio, 1)
    more = axial_factor(model.ratio, length)
    r = 1 / model.ratio
    D0, D2 = 1+r**6, r**2+r**4
    touched = 0
    for j, point in enumerate(previous.W.geometry.BOUNDARY):
        assert point in sphere
        if max(map(abs, point)) == 2:
            assert model.ratios[j] == (D2/D0)**2 * one
            model.ratios[j] *= more / one
            touched += 1
        else:
            assert model.ratios[j] * model.ratio**2 == D2/D0
    assert touched == 6


def run(args):
    assert 1 < args.ratio < Fraction(1242, 785)
    green = previous.W.fixed.phase.AngularGreen(Fraction(0))
    model = previous.W.Certificate(args.ratio, "order", green, False,
                                   bridges=True, expanded=False, corners=True)
    if args.multipliers == "legacy":
        source = json.loads(previous.TABLE.read_text())["568/359"]
        record = source["multipliers"]
        denominator = source["multiplier_denominator"]
        assert previous.prior.multipliers_digest(record) == source["multiplier_sha256"]
        pool, _ = previous.source()
        previous.verify_domain(model, pool, record, denominator)
    else:
        source_r = Fraction(1579, 998)
        assert args.ratio <= source_r
        source = json.loads(Path(__file__).with_name("STAGE20_ORDERED_1579_998.json").read_text())
        assert Fraction(source["ratio"]) == source_r
        record, denominator = source["rows"], source["denominator"]
        extra = extended.add_new_tests(model, green)
        pool, _ = extended.pool_for(model)
        assert [[i, extra[i]] for i, _ in record["ordinary"] if i in extra] == [[54,7],[69,27],[72,41]]
        for name, rows in record.items():
            allowed = {tuple(entry[:-1]) for entry in pool[name]}
            used = [tuple(entry[:-1]) for entry in rows]
            assert set(used) <= allowed and len(used) == len(set(used))
            assert all(type(v) is int for row in rows for v in row)
            assert all(row[-1] != 0 if name == "translations" else row[-1] > 0 for row in rows)
    assert type(denominator) is int and denominator > 0
    values, radius, q = previous.build_residual(model, record)
    radius /= denominator
    q /= denominator
    assert q == previous.MASS > 0
    if args.length != 1:
        extend_rays(model, args.length)
    scale = previous.W.T * previous.W.DEN * denominator
    layout = previous.corner_geometry(model)
    if args.cap is None:
        cap, (source, field) = previous.prior.bridge.propose(model, values, layout, scale)
    else:
        cap = args.cap
        source, field = args.field
    weights = previous.corner_weights(model, source, layout[1], layout[3])
    weights = previous.direct_field(weights, layout[2], layout[4], layout[5], field)
    assert source in layout[0] and 0 <= field < len(model.owners) and int(weights.sum()) > 0
    one = Fraction(int(weights @ values[model.owners]), int(weights.sum())*scale)
    assert 0 <= cap - one < Fraction(1, 10**7)
    lower = -(cap+radius)/q
    print(f"r={args.ratio} ray length={args.length} rows={args.multipliers} cap={cap} field={source},{field} "
          f"floor(1e15*L)={floor(10**15*lower)} "
          f"floor(1e15*(one+radius))={floor(10**15*(one+radius))}", flush=True)
    if args.mode == "witness":
        assert one + radius > 0
        print("PASS single-field obstruction for this combination", flush=True)
    elif args.mode != "probe":
        assert lower > 0
        checker = (previous.W.base.corner_upper if args.mode == "raw"
                   else previous.CornerCheck(previous.W.base).all_corners)
        cap0 = Fraction(max(map(int, values)), scale)
        assert checker(args.ratio, values, model.weights, model.owners, scale, model.ratios,
                       cap0, cap) == cap
        assert len(layout[0])*len(model.owners) == 3670016
        print(f"PASS {args.mode} all 3670016 fields, Q={q}\nradius={radius}\nU={cap+radius}\nL={lower}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratio", type=Fraction)
    parser.add_argument("mode", choices=("probe", "witness", "directed", "raw"))
    parser.add_argument("--multipliers", choices=("legacy", "stage20"), default="legacy")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--cap", type=Fraction)
    parser.add_argument("--field", type=int, nargs=2)
    options = parser.parse_args()
    assert __debug__ and ((options.cap is None) == (options.field is None))
    run(options)
