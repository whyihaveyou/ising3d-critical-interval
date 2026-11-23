"""Rebuild the Stage 30 finite numerator inputs at 3775/2386."""

from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import STAGE28_ORDERED_ENDPOINT_precise as precise
import STAGE30_UPPER_REVIEW_bridge as bridge

POINT = Fraction(3775, 2386)
ARCHIVE = HERE / "STAGE30_UPPER_REVIEW_3775_2386.json"
SOURCE_SHA = "b8c8cec1644bf8bd29978d514643a7044377eb12fcd0e157d950cd2980bd2cb6"
ARCHIVE_SHA = "711f6a8446ecb17eea03384ab964a9dd60f0586be1401e9edd46a321c873f314"


OUTPUT = HERE


def save(name, data):
    tmp = OUTPUT / (name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(OUTPUT / name)
    return sha256(data).hexdigest()


def run(directed=False):
    assert __debug__
    raw = ARCHIVE.read_bytes()
    source_bytes = precise.SOURCE.read_bytes()
    record = json.loads(raw)
    source = json.loads(source_bytes)
    assert sha256(raw).hexdigest() == ARCHIVE_SHA
    assert sha256(source_bytes).hexdigest() == SOURCE_SHA == record["source_sha256"]
    assert Fraction(record["r"]) == POINT
    precise.base.POINT = POINT
    print("rebuilding physical model", flush=True)
    model, pool, layout = precise.base.make_model()
    precise.base.validate_rows(source["rows"], pool)
    print("rebuilding integer residual", flush=True)
    values, scale, radius, mass = precise.build_residual(model, source)
    assert radius == Fraction(record["radius"]) > 0 and mass == Fraction(record["Q"]) > 0
    assert record["class_denominator"] == bridge.endpoint.GRID
    assert scale > 0 and len(values) == len(model.weights)
    classes, counts, ratios, traces, words, popcount = layout
    assert set(classes) == {int(c) for c in record["class_bounds"]}
    assert len(classes) == 14 and len(words) == 2**18
    assert np.array_equal(words, np.arange(2**18))
    assert len(model.owners) == len(words) and min(model.owners) == 0
    assert max(model.owners) == len(values)-1 < 2**16
    assert min(model.weights) > 0 and min(traces) > 0
    # Compare independent coordinate enumeration with every original corner count.
    boundary = sorted(x for x in product(range(-2, 3), repeat=3) if sum(map(abs, x)) == 2)
    assert list(precise.G.BOUNDARY) == boundary and len(boundary) == len(ratios) == 18
    masks = []
    for corner, old_count in zip(product((-1, 1), repeat=3), counts):
        mask = sum(1 << i for i, x in enumerate(boundary)
                   if sum(abs(a-b) for a,b in zip(x, corner)) == 1)
        assert bin(mask).count("1") == 3 and np.array_equal(popcount[words & mask], old_count)
        masks.append(mask)
    assert len(set(masks)) == len(counts) == 8
    assert all(0 < r <= 1 for r in ratios)
    edges, odds = bridge.geometry(POINT, ratios)
    sizes, rates = bridge.endpoint.corner.verify(POINT)
    shifts = bridge.endpoint.shifts(record["class_dual"], sizes, rates)
    caps = {int(c):bridge.averaged_cap(rec, edges, odds) for c,rec in record["class_bounds"].items()}
    assert max(caps[c]+shifts[c] for c in classes) + radius == Fraction(record["U"]) < 0
    assert -Fraction(record["U"])/mass == Fraction(record["L"]) > 0
    assert int(10**15*Fraction(record["L"])) == record["floor_1e15_L"] == 215386342
    if directed:
        records = {int(c):v for c,v in record["class_bounds"].items()}
        bridge.verify_caps(model,values,scale,layout,records,raw=False)
        print("PASS directed 14 classes, 3670016 full upper and 3661604 complement caps", flush=True)
        return
    data = ["STAGE34v1", f"18 8 {len(values)} {len(classes)}", str(scale)]
    data += [f"{r.numerator} {r.denominator}" for r in ratios]
    data.append(" ".join(str(x) for x in traces))
    data.append(" ".join(str(x) for x in masks))
    data += [f"{w} {v}" for w,v in zip(model.weights, values)]
    for c in classes:
        entry = record["class_bounds"][str(c)]
        a, b = map(Fraction, (entry["upper"], entry["low"]))
        high = [f for f,i,j in entry["words"]]
        assert len(high) == len(set(high)) and all(0 <= f < 2**18 for f in high)
        data.append(f"{c} {a.numerator} {a.denominator} {b.numerator} {b.denominator} {len(high)}")
        data.append(" ".join(map(str, high)))
    file_hashes = {}
    file_hashes["model.txt"] = save("model.txt", ("\n".join(data)+"\n").encode("ascii"))
    file_hashes["owners.u16le"] = save("owners.u16le", np.asarray(model.owners, dtype="<u2").tobytes())
    manifest = {"point":str(POINT), "archive_sha256":ARCHIVE_SHA, "source_sha256":SOURCE_SHA,
                "files":file_hashes, "classes":list(map(int,classes)), "fields_per_class":len(words)}
    save("inputs.json", (json.dumps(manifest, indent=2)+"\n").encode("ascii"))
    print("orbits", len(values), "max weight bits", max(int(w).bit_length() for w in model.weights),
          "ratio denominator bits", sorted({r.denominator.bit_length() for r in ratios}))
    print("PASS verified physical rows, margin, masks, positive traces and both input digests", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE)
    options = parser.parse_args()
    OUTPUT = options.output.resolve()
    run(options.directed)
