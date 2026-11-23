"""Compare every reduced-tensor entry and selected physical fields as exact integers."""

import argparse
from fractions import Fraction as F
from itertools import combinations, product
import json
from math import prod
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from run import HERE, EXPECTED_CLASSES, digest, read_inputs, receipt
sys.path.insert(0, str(HERE.parent))
import verify_threshold_certificate_parameterization_20261121 as original


BINARY = HERE/"blocked_raw"


def values_in(log):
    rows = [line.split() for line in log.splitlines() if line.startswith("value ")]
    result = {(label,int(f)):int(z) for _,label,f,z in rows}
    assert len(result) == len(rows)
    return result


def small_case(directory, n):
    words = list(range(1 << n))
    dimension = (1 << (n-1))+1
    owner = [(5*b + b//7) % dimension for b in words]
    weights = [5 + (i*17) % 23 for i in range(dimension)]
    v = [((-1)**i)*(120 + (i*23) % 73) for i in range(dimension)]
    scale = 47
    ratios = [F(2+(j%3), 79+10*j) for j in range(n)]
    trace = [13, 5, 7, 11]
    masks = [sum(1<<j for j in triple) for triple in list(combinations(range(n),3))[:4]]
    classes = [0,(1<<len(masks))-1]
    cuts = {"upper":F(1,3), "low":F(-1,5)}
    header = ["STAGE34v1",f"{n} {len(masks)} {dimension} 2",str(scale)]
    header += [f"{r.numerator} {r.denominator}" for r in ratios]
    header += [" ".join(map(str, trace)), " ".join(map(str, masks))]
    header += [f"{w} {z}" for w,z in zip(weights,v)]
    header += [f"{c} 1 3 -1 5 2\n0 {words[-1]}" for c in classes]
    table, owners = directory/"toy.txt", directory/"toy.u16le"
    table.write_text("\n".join(header)+"\n")
    owners.write_bytes(np.array(owner, dtype="<u2").tobytes())
    reference = [[prod(r.numerator if (b^f) & (1<<j) else r.denominator
                       for j,r in enumerate(ratios)) for b in words] for f in words]
    comparisons = 0
    for c in classes:
        expected = {}
        for name, cut in cuts.items():
            inputs = []
            for b, orbit in zip(words, owner):
                counts = [bin(mask & b).count("1") for mask in masks]
                t = prod(trace[3-x if c&(1<<j) else x] for j,x in enumerate(counts))
                inputs.append(weights[orbit]*t*(cut.numerator*scale-cut.denominator*v[orbit]))
            direct = [sum(a*k for a,k in zip(inputs,row)) for row in reference]
            assert list(original.walsh.fields.transform(inputs,ratios)) == direct
            assert min(direct) < 0 < max(direct)
            expected.update({(name,f):z for f,z in enumerate(direct)})
        for k in sorted({0,1,min(3,n),min(5,n)}):
            args = [str(BINARY), str(table), str(owners), str(c), str(k)]
            replay = subprocess.run(args+["test-dump"], capture_output=True, text=True, check=True)
            assert values_in(replay.stdout) == expected
            comparisons += len(expected)
        rejected = subprocess.run(args, capture_output=True, text=True)
        assert rejected.returncode != 0 and "negative upper numerator" in rejected.stderr
    print("PASS", n, "bits,", len(classes), "classes,", comparisons,
          "signed native entries against direct sums and the original raw kernel", flush=True)
    return comparisons


def small():
    with tempfile.TemporaryDirectory(prefix="ising-small-3775-") as temp:
        cases = [small_case(Path(temp), n) for n in (3,5,8)]
    print("PASS", sum(cases), "reduced-tensor comparisons and six production-mode rejections")


def read_orbits():
    lines = iter((HERE / "model.txt").read_text().splitlines())
    assert next(lines) == "STAGE34v1"
    nbits, corners, orbits, nclasses = map(int, next(lines).split())
    assert (nbits,corners,nclasses) == (18,8,14)
    scale = int(next(lines))
    num_den = [tuple(map(int, next(lines).split())) for _ in range(nbits)]
    ratios = [F(a,b) for a,b in num_den]
    assert all((r.numerator,r.denominator) == pair for r,pair in zip(ratios,num_den))
    traces = np.array(list(map(int, next(lines).split())), dtype=object)
    masks = list(map(int, next(lines).split()))
    weights, residual = zip(*(tuple(map(int,next(lines).split())) for _ in range(orbits)))
    return (scale, ratios, traces, masks, np.array(weights,dtype=object),
            np.array(residual,dtype=object))


def samples():
    manifest, archive = read_inputs()
    cov = json.loads((HERE / "receipts" / "raw_coverage.json").read_text())
    assert cov["binding"]["inputs"] == manifest
    assert cov["binding"]["kernel_source_sha256"] == digest(HERE/"blocked_raw.cpp")
    assert len(cov["classes"]) == len(EXPECTED_CLASSES)
    scale,ratios,traces,masks,weights,residual = read_orbits()
    owners = np.fromfile(HERE/"owners.u16le", dtype="<u2")
    words = np.arange(2**18, dtype=np.int32)
    assert len(owners) == len(words)
    popcount = np.array([bin(w).count("1") for w in range(len(words))], dtype=np.uint8)
    # Derive the corner factors through the original conditionals; by shell word,
    # not by the histogram or prefix coordinates of the C++ implementation.
    boundary = list(original.walsh.geometry.BOUNDARY)
    corners = list(product((-1,1), repeat=3))
    adj = [[i for i,x in enumerate(boundary) if sum(abs(y-z) for y,z in zip(x,p)) == 1]
           for p in corners]
    assert masks == [sum(1<<i for i in neighbours) for neighbours in adj]
    counts = [popcount[words & mask] for mask in masks]
    ratio_groups = [sum(1<<j for j,r in enumerate(ratios) if r == ratio) for ratio in set(ratios)]
    weight_by_word = weights[owners]
    moment_by_word = (weights*residual)[owners]
    trace_digit = np.array([0,1,9,81],dtype=np.int32)
    total = 0
    for c, report in zip(EXPECTED_CLASSES, cov["classes"]):
        file = HERE / "receipts" / f"class_{c}.log"
        record = archive["class_bounds"][str(c)]
        assert receipt(file, c, record) == report
        stored = values_in(file.read_text())
        selected = sorted({f for label,f in stored})
        assert len(selected) == 2
        # In the explicit Hamming sum, identical corner-trace multiplicities
        # and Hamming counts for each ratio have the same positive factor.
        # Add base weights and moments in each group before multiplying by it.
        trace_key = np.zeros(len(words),dtype=np.int32)
        for j,row in enumerate(counts):
            trace_key += trace_digit[3-row if c&(1<<j) else row]
        for f in selected:
            groups = trace_key
            for mask in ratio_groups:
                groups = groups*(bin(mask).count("1")+1) + popcount[(words^f) & mask]
            order = np.argsort(groups,kind="stable")
            sorted_keys = groups[order]
            starts = np.r_[0,1+np.flatnonzero(sorted_keys[:-1] != sorted_keys[1:])]
            representatives = order[starts]
            unit = SimpleNamespace(weights=np.array([1],dtype=object),
                                   owners=np.zeros(len(starts),dtype=np.int32))
            corner = original.corner_weights(unit,c,[row[representatives] for row in counts],traces)
            field = original.direct_field(corner,ratios,words[representatives],popcount,f)
            normalization = int(field @ np.add.reduceat(weight_by_word[order],starts))
            numerator = int(field @ np.add.reduceat(moment_by_word[order],starts))
            assert normalization > 0
            for label in ("upper","low"):
                cut = F(record[label])
                expected = cut.numerator*scale*normalization-cut.denominator*numerator
                assert expected == stored[label,f], (c,f,label)
                total += 1
        print("PASS original direct physical fields", c, selected, flush=True)
    assert total == 56
    print("PASS 56 unnormalized physical integers (14 classes × 2 fields × 2 caps)")


if __name__ == "__main__":
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("small","samples"))
    parser.add_argument("--kernel", type=Path, default=BINARY)
    args = parser.parse_args()
    BINARY = args.kernel.resolve()
    (small if args.mode == "small" else samples)()
