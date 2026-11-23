"""Audit the active spectral polynomials and Gram rows against 128-spin sums."""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import json

import numpy as np

import STAGE21_ORDERED as current


def active_tests(rows):
    fixed=current.stage.W.fixed
    _,_,_,tests,shell,_=fixed.EXTENDED_PROOFS[-1]
    known=[fixed.TESTS[i] for i,_ in tests]
    known += [test for i,test in enumerate(fixed.TESTS) if i not in dict(tests)]
    custom=json.loads((current.HERE/'STAGE20_ORDERED_price_791_500.json').read_text())['cuts']
    known += [('extension',v) for v in custom['reflections']]
    # The initial shell rows, if used, are listed separately from these tests.
    return {index:known[index-len(shell)] for index,_ in rows if index>=len(shell)}


def audit_polynomial(tag,root):
    origin=(0,0,0)
    x=(1,0,0)
    y=(0,1,0)
    def multiply(p,q):
        out=defaultdict(int)
        for a,va in p.items():
            for b,vb in q.items():
                out[tuple(i+j for i,j in zip(a,b))]+=va*vb
        return out
    def factor(axis,sign):
        return {origin:1,axis:sign}
    local=[{origin:1},factor(x,1),multiply(factor(x,1),factor(y,1)),
           multiply(factor(x,1),factor(y,-1)),factor(x,-1),
           multiply(factor(x,-1),factor(y,1)), multiply(factor(x,-1),factor(y,-1)),
           {origin:1,(2,0,0):-1}]
    basis=sorted((t for t in product(range(3),repeat=3) if sum(t)<=2),key=lambda t:(sum(t),t))
    poly=dict(zip(basis,root))
    raw=multiply(local[tag],multiply(poly,poly))
    by_orbit=defaultdict(int)
    for k,v in raw.items():
        by_orbit[tuple(sorted(k))]+=v
    norm=sum(map(abs,by_orbit.values()))
    phase=current.stage.W.fixed.phase
    result=[Fraction(by_orbit[k],norm) for k in phase.KEYS]
    assert result==phase.polynomial(tag,root)
    assert tag in range(8) and sum(result)>=0
    return sum(result)


def gram_coefficients(ball, orbit):
    geom=current.stage.W.geometry
    mirror=lambda x:(1-x[0],x[1],x[2])
    positive=[p for p in geom.SITES if p[0]>=1 and mirror(p) in geom.INDEX]
    crossed=[p for p in geom.SITES if p[0]==0 and mirror(p) in geom.INDEX]
    patterns=[(None,A) for size in (1,3,5) for A in combinations(positive,size)]
    patterns += [(c,A) for c in crossed for size in (0,2) for A in combinations(positive,size)]
    assert len(patterns)==112
    def mask(A):
        return sum(1<<geom.SITES.index(p) for p in A)
    masks=np.array([mask(A) for _,A in patterns])
    reflected=np.array([mask(tuple(map(mirror,A))) for _,A in patterns])
    full_masks=np.array([m^(mask((c,)) if c is not None else 0) for m,(c,_) in zip(masks,patterns)])
    full_reflected=np.array([m^(mask((mirror(c),)) if c is not None else 0) for m,(c,_) in zip(reflected,patterns)])
    edges=[(i,j) for i in range(7) for j in range(i+1,25)
           if sum(abs(x-y) for x,y in zip(geom.SITES[i],geom.SITES[j]))==1]
    assert len(edges)==36
    # Parity lookups are on two 13-bit pieces of a 25-bit physical word.
    table=np.array([(-1)**bin(n).count('1') for n in range(8192)],dtype=np.int64)
    word=int(ball.words[orbit])
    images={z>>7 for z in geom.orbit(word<<7)}
    p=current.POINT.numerator
    q=current.POINT.denominator
    bare=(current.POINT-1)/(current.POINT+1)
    bare+=1/bare
    for boundary in images:
        spins=np.arange(128,dtype=np.int64)|(boundary<<7)
        def signs(M):
            bits=spins[:,None]&np.asarray(M)[None,:]
            return table[bits&8191]*table[bits>>13]
        weights=[]
        for w in spins:
            d=sum(((int(w)>>i)^(int(w)>>j))&1 for i,j in edges)
            weights.append(p**(36-d)*q**d)
        groups=[]
        for c in crossed:
            ix=np.array([j for j,(pt,_) in enumerate(patterns) if pt==c])
            overlap=signs([mask((c,mirror(c)))])[:,0]
            groups.append((ix,signs(masks[ix]),signs(reflected[ix]),overlap))
        yield signs(full_masks),signs(full_reflected),groups,weights,bare,len(images)


def gram(fixed,selected):
    ball=fixed.BallMoments(current.POINT,8)
    features=fixed.phase.reflection_features(ball)
    u=(current.POINT-1)/(current.POINT+1)
    rows={i:fixed.exact_reflection(v,u+1/u,features) for i,v in selected.items()}
    for o in (0,29,137):
        values=dict.fromkeys(selected,Fraction(0))
        for left,right,groups,weight,bare,nimages in gram_coefficients(ball,o):
            full=sum(weight)
            for i,root in selected.items():
                v=np.array(root,dtype=np.int64)
                norm=sum(t*t for t in root)
                assert 4*sum(map(abs,root))**2<2**62
                normal=(left@v)*(right@v)
                extra=np.zeros_like(normal)
                for ix,L,R,overlap in groups:
                    Z=(L@v[ix])*(R@v[ix])
                    normal-=2*overlap*Z
                    extra+=Z
                num=sum(w*(bare.denominator*int(a)+bare.numerator*int(b))
                        for w,a,b in zip(weight,normal,extra))
                values[i]+=Fraction(num,nimages*full*bare.denominator*norm)
        for i,(row,eps,mass) in rows.items():
            assert mass==0 and abs(values[i]-Fraction(int(row[o]),current.stage.W.T))<=eps
    return len(rows)


if __name__=='__main__':
    assert __debug__
    rec=json.loads(current.ARCHIVE.read_text())
    tests=active_tests(rec['rows']['ordinary'])
    spectra={i:audit_polynomial(tag,v) for i,(tag,v) in tests.items() if tag!='extension'}
    print('all active normalized spectral masses', {i:str(q) for i,q in spectra.items()},flush=True)
    gram_count=gram(current.stage.W.fixed,{i:v for i,(tag,v) in tests.items() if tag=='extension'})
    print(f'PASS {len(spectra)} symbolic spectral polynomials, {gram_count} Gram rows '
          'each on three raw-spin boundary orbits',flush=True)
