"""Rebuild a coupling-specific order certificate from 136 frozen Walsh multipliers.

The archived multipliers are read without modifying their production archive.
Each target supplies its own thermal rows and finite-graph correlation moments.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

SLAB = Path(__file__).resolve().parent / 'physical'
sys.path.insert(0, str(SLAB))
import verify_deleted_walsh_order as walsh

MASS = F(8943787728743670589, 8943788333028500000)
SOURCE = 'order 894/565'
SOURCE_HASH = '18e487426da992bd96c18c7c4a59203311fea15118acd306fad775c006b9e1b6'
VERIFIED = {
    F(7277, 4599): (35284243, 1950, '40223fa46d03a573ae6bbebddfc1f3c5deaffaf2cb1c12ab72927c17b5f1db57'),
    F(8171, 5164): (123627225, 1950, '5c68532cfe6fdb3e79bb0ef704e7e43285c0da04b3ece5278278b83456c204fd'),
}
VERIFIED_CORNERS = {
    F(1394, 881): (-41228258, None, '6f9a750a7fb79e60be850cdd05809e88e3f68f5affd172b6338b836f12f59f1c'),
    F(12421, 7850): (386771774, None, '7252394f66415400bf39e74936cb97a9a5f0c20ac1a4f596ffb9dac0b78b3f09'),
}


def audited_record():
    records = json.loads(walsh.FILE.read_text())
    archived = records[SOURCE]
    multipliers = {key: archived[key] for key in ('translations', 'ordinary', 'walsh')}
    encoded = json.dumps(multipliers, sort_keys=True, separators=(',', ':')).encode()
    assert sha256(encoded).hexdigest() == SOURCE_HASH
    assert tuple(map(len, multipliers.values())) == (91, 14, 136)
    for key in ('order 125/79', 'order 769/486'):
        assert all(records[key][name] == vector for name, vector in multipliers.items())
    _, _, translations, tests, shell, covariances = walsh.fixed.EXTENDED_PROOFS[-1]
    assert len(translations) == 91 and len(shell) == 28 and not covariances
    physical = {i for i, _ in translations}
    assert all(index in physical for index, _ in multipliers['translations'])
    selected = Counter()
    mass = F(0)
    for index, weight in multipliers['ordinary']:
        assert weight > 0 and len(shell) <= index < len(shell) + len(tests)
        global_index = tests[index-len(shell)][0]
        tag, coeff = walsh.fixed.TESTS[global_index]
        selected[tag == 'extension'] += 1
        if tag == 'extension':
            assert sum(abs(v) for v in coeff)**2 < 2**62
        else:
            polynomial = walsh.fixed.phase.polynomial(tag, coeff)
            assert sum(polynomial) >= 0
            mass += F(weight, walsh.DEN) * sum(polynomial)
    assert selected == {False: 6, True: 8}
    assert mass == F(archived['mass']) == MASS
    toggles = Counter(bin(p).count('1') for _, p, _ in multipliers['walsh'])
    assert toggles == {2: 134, 4: 2}
    assert all(a != 0 and bin(a).count('1') % 2 == 0 and weight > 0
               for a, _, weight in multipliers['walsh'])
    return multipliers


def replay(record, green, r, full=False, corners=False, compact_fields=False):
    assert r > 1
    u = (r-1)/(r+1)
    assert (r*r+3)/(r*r-1) == u+1/u-1
    assert (3*r*r+1)/(r*r-1) == u+1/u+1
    model = walsh.Certificate(r, 'order', green, discovery=False, bridges=False,
                              expanded=False, corners=corners)
    assert len(model.terms) == 47 and len(model.words) == 3236
    assert len(model.owners) == 2**18
    with ExitStack() as changes:
        if compact_fields:
            assert corners
            from threshold_corner_compact_20261121 import CornerCheck
            check = CornerCheck(walsh.base)
            changes.enter_context(patch.object(walsh.fields, 'upper_over_fields', check.coarse_bound))
            changes.enter_context(patch.object(walsh.base, 'corner_upper', check.all_corners))
        value, mass, field = model.exact(record)
    floor = value*10**15//1
    digest = sha256(str(value).encode()).hexdigest()
    assert mass == MASS
    expected = VERIFIED_CORNERS if corners else VERIFIED
    if r in expected:
        assert (floor, field, digest) == expected[r]
        stored = json.loads(Path(__file__).with_name('ordered_136_endpoint_extension_20261121.json').read_text())
        assert stored['multiplier_sha256'] == SOURCE_HASH and F(stored['zero_mode_mass']) == mass
        point, = [p for p in stored['points'] if F(p['ratio']) == r and p['corners'] == corners]
        assert (F(point['value']), point['floor_1e15'], point['extremal_field']) == (value, floor, field)
    print(f'r={r} Q={mass} floor(10^15 L)={floor} field={field} sha256(L)={digest}', flush=True)
    if full:
        print(f'L={value}')
    model.deleted.row.cache_clear()
    model.ball.averaged.cache_clear()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ratios', type=F, nargs='*')
    parser.add_argument('--fraction', action='store_true', help='also print the full rational result')
    parser.add_argument('--corners', action='store_true', help='also condition on the eight corner spins')
    parser.add_argument('--compact-fields', action='store_true', help='check the corner caps by directed dyadic arithmetic')
    args = parser.parse_args()
    assert __debug__
    if args.compact_fields and not args.corners:
        parser.error('--compact-fields requires --corners')
    record = audited_record()
    green = walsh.fixed.phase.AngularGreen(F(0))
    default_points = VERIFIED_CORNERS if args.corners else VERIFIED
    for r in args.ratios or list(default_points):
        replay(record, green, r, args.fraction, args.corners, args.compact_fields)


if __name__ == '__main__':
    main()
