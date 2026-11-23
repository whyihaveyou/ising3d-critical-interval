"""One-point, frozen 136-row check using 150 deleted-graph bonds and eight corners."""
import argparse
from fractions import Fraction as F
import json
from math import ceil, floor
from pathlib import Path

import numpy as np

import verify_ordered_136_endpoint_extension_20261121 as old
from threshold_corner_compact_20261121 import CornerCheck, average_float
from verify_threshold_certificate_parameterization_20261121 import (
    build_residual, corner_geometry, corner_weights, direct_field,
)

W = old.walsh
HERE = Path(__file__).resolve().parent
TABLE = HERE / "ordered_136_bridge_corners_20261121.json"
REFERENCE = F(12421, 7850)


def propose(model, orbit_values, geometry, scale):
    classes, counts, ratios, trace, _, _ = geometry
    max_weight = max(model.weights)
    initial = np.array([w / max_weight for w in model.weights])[model.owners]
    theta = np.array([v/trace[0] for v in trace], dtype=float)
    signed = np.array([int(v)/scale for v in orbit_values])[model.owners]
    best = -np.inf
    candidate = None
    for source in classes:
        envelope = initial.copy()
        for j, count in enumerate(counts):
            envelope *= theta[3-count if source & (1 << j) else count]
        values = average_float(envelope*signed, ratios) / average_float(envelope, ratios)
        assert np.isfinite(values).all()
        field = int(np.argmax(values))
        if values[field] > best:
            best = values[field]
            candidate = source, field
    return F(ceil(float(best)*10**9)+5, 10**9), candidate


def comparison_inputs(model, record):
    deleted = model.deleted
    original = W.fixed.local.shell_measure(model.ratio, model.ball.words)
    sizes = deleted.sizes
    per_word = np.array([int(w)//int(n) for w, n in zip(original, sizes)], dtype=object)
    assert all(p*int(n) == w for p, n, w in zip(per_word, sizes, original))
    before = W.walsh(per_word[model.owners])
    toggles = sorted({p for _, p, _ in record["walsh"]})
    assert toggles == [3, 6, 71, 1158]
    moments = {}
    for toggle in toggles:
        c78 = F(int(before[toggle]), int(before[0]))
        c150 = F(int(deleted.free[toggle]), deleted.norm)
        assert 0 < c78 < c150 < 1
        moments[str(toggle)] = {
            "sites": [list(p) for i, p in enumerate(W.geometry.BOUNDARY) if toggle >> i & 1],
            "multiplicity": sum(t == toggle for _, t, _ in record["walsh"]),
            "c78": str(c78), "c150": str(c150),
        }
    return moments


def reproduce(ratio, raw_integers, propose_only, stored):
    assert ratio < REFERENCE
    record = old.audited_record()
    green = W.fixed.phase.AngularGreen(F(0))
    model = W.Certificate(ratio, "order", green, discovery=False,
                          bridges=True, expanded=False, corners=True)
    assert len(model.terms) == 47 and len(model.words) == 3236
    values, radius, Q = build_residual(model, record)
    assert Q == old.MASS > 0
    moments = comparison_inputs(model, record)
    geom = corner_geometry(model)
    scale = W.DEN*W.T
    if stored is None or propose_only:
        cap, (source, field) = propose(model, values, geom, scale)
    else:
        cap = F(stored["corner_cap"])
        source, field = stored["witness"]
    U = cap + radius
    L = -U/Q
    cls, counts, ratios, trace, words, popcount = geom
    # An exact residual at the suggested word detects bad proposals cheaply.
    weights = corner_weights(model, source, counts, trace)
    kernel = direct_field(weights, ratios, words, popcount, field)
    total = int(kernel.sum())
    assert total > 0
    one = F(int(kernel @ values[model.owners]), total*scale)
    assert one <= cap and cap-one < F(1, 10**7)
    print(f"proposal r={ratio}, word=({source},{field}), cap={cap}, "
          f"L approximately {float(L):+.14g}", flush=True)
    if propose_only or L <= 0:
        print("proposal only; no certification or archive update", flush=True)
        return None
    upper = F(max(map(int, values)), scale)
    if raw_integers:
        assert W.base.corner_upper(ratio, values, model.weights, model.owners, scale,
                                   model.ratios, upper, cap) == cap
    else:
        checker = CornerCheck(W.base)
        assert checker.all_corners(ratio, values, model.weights, model.owners, scale,
                                   model.ratios, upper, cap) == cap
    assert len(cls)*len(words) == 3670016
    return {
        "ratio": str(ratio),
        "source": old.SOURCE,
        "multiplier_sha256": old.SOURCE_HASH,
        "rows": [91, 14, 136],
        "deleted_vertices": 92,
        "deleted_edges": 150,
        "strictly_improved_toggle_masks": [int(t) for t in moments],
        "toggle_moments": moments,
        "expanded": False,
        "corners": True,
        "corner_cap": str(cap),
        "rounding_radius": str(radius),
        "Q": str(Q),
        "U": str(U),
        "L": str(L),
        "floor_1e15_L": floor(10**15*L),
        "witness": [source, field],
        "exact_witness_residual": str(one),
        "source_classes": len(cls),
        "field_count_per_class": len(words),
        "improvement_relative_to_12421_over_7850": str(REFERENCE-ratio),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=F, default=F(1394, 881))
    parser.add_argument("--raw-integers", action="store_true",
                        help="use uncompressed arbitrary-precision field transfers")
    parser.add_argument("--propose-only", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="save the new point only after every field is certified")
    options = parser.parse_args()
    assert __debug__ and options.ratio > 1
    table = json.loads(TABLE.read_text()) if TABLE.exists() else {}
    stored = table.get(str(options.ratio))
    if not (options.record or options.propose_only):
        assert stored is not None
    result = reproduce(options.ratio, options.raw_integers,
                       options.propose_only, stored)
    if result is None:
        return
    if options.record:
        if stored is not None:
            assert result == stored
        table[str(options.ratio)] = result
        pending = TABLE.with_suffix(".pending")
        pending.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
        pending.replace(TABLE)
    else:
        assert result == stored
    print(f"PASS: U<0, Q>0, floor(10^15 L)={result['floor_1e15_L']}", flush=True)


if __name__ == "__main__":
    main()
