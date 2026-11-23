"""Exact finite-energy constant for the eight auxiliary corner-source signs."""

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from math import floor


CORNERS = tuple(product((-1, 1), repeat=3))


def laws(ratio):
    s = 1 / ratio
    v = s**3
    w = ((s+s**5)/(1+s**6))**3
    k = 2*v/(1+v*v)
    odds = k*(w-v)/(1-v*w)
    assert 0 < odds < 1
    return odds/(1+odds)


def verify_local(ratio):
    s = 1 / ratio
    v = s**3
    alpha = laws(ratio)
    neighbourhoods = []
    for y in CORNERS:
        outside = {tuple((2 if i==j else 1)*t for i,t in enumerate(y)) for j in range(3)}
        assert len(outside) == 3
        assert all(sum(map(abs,z)) == 4 and sum(abs(t-u) for t,u in zip(z,y)) == 1 for z in outside)
        assert all(sum(abs(t-u) for t,u in zip(z,d)) > 1 for z in outside for d in CORNERS if d!=y)
        neighbourhoods.append(outside)
    assert len(set.union(*neighbourhoods)) == 24
    counts = sorted(product(range(6), repeat=3))
    actual = []
    for outside_counts in counts:
        decomposition = [Fraction(0), Fraction(0)]
        for word in product((0,1), repeat=3):
            minus = sum(word)
            weights = [s**minus,s**(3-minus)]
            coeff = [(weights[i]-v*weights[1-i])/(1-v*v) for i in range(2)]
            assert min(coeff) >= 0
            assert weights == [coeff[i]+v*coeff[1-i] for i in range(2)]
            unmatched = sum(5-n if e else n for e,n in zip(word,outside_counts))
            for i in (0,1):
                decomposition[i] += s**unmatched * coeff[i]
        for inner_minus in range(4):
            inside = [s**inner_minus, s**(3-inner_minus)]
            traced = [decomposition[i]*(inside[i]+v*inside[1-i]) for i in (0,1)]
            actual.append(traced[0]/sum(traced))
    assert len(actual)==864 and (min(actual),max(actual))==(alpha,1-alpha)
    return alpha


@lru_cache(None)
def classes():
    owner = [None]*256
    for axes in permutations(range(3)):
        for signs in product((-1,1),repeat=3):
            perm = [CORNERS.index(tuple(a*t[j] for a,j in zip(signs,axes))) for t in CORNERS]
            for word in range(256):
                image = sum(((word>>i)&1)<<p for i,p in enumerate(perm))
                rep = min(word, image, 255-image)
                owner[word] = min(rep,owner[word]) if owner[word] is not None else rep
    assert len(set(owner))==14
    return tuple(owner)


def sweep(class_caps, alpha):
    # For any joint law whose one-site probabilities given the other signs lie
    # in [alpha,1-alpha], each elimination is an upper conditional expectation.
    assert 0 < alpha <= Fraction(1,2)
    assert set(class_caps)==set(classes())
    values = [class_caps[rep] for rep in classes()]
    for i in range(8):
        column = []
        for word, a in enumerate(values):
            b = values[word^(1<<i)]
            column.append((1-alpha)*max(a,b)+alpha*min(a,b))
        values=column
    assert len(set(values))==1
    return values[0]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ratio',type=Fraction)
    opts=parser.parse_args()
    assert __debug__ and opts.ratio>1
    p=verify_local(opts.ratio)
    # A product coin realizes the cap at the all-positive event after 8 steps.
    test={rep:int(rep==0) for rep in classes()}
    assert 0 < sweep(test,p) < 1
    print(f'PASS 864 physical field-count patterns, 24 distinct corner neighbours, 14 source classes; '
          f'r={opts.ratio}, floor(1e10*alpha)={floor(p*10**10)}, alpha={p}')
