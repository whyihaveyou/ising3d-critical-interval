"""Verify the six two-spin branches and an exact classwise dual at r=1791/1132."""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from math import floor
from pathlib import Path

import STAGE20_ORDERED_corner_law as local


R=Fraction(1791,1132)
LOW=Fraction(459,10000)
DUAL=((0,1,42369334),(1,3,27620269),(1,6,357969965),(1,24,7270756),
      (6,7,39127118),(6,22,127571006),(6,25,25653750),(7,15,4355067),
      (7,23,8918756),(22,105,55655689),(25,27,13556269),(25,30,72428269),
      (25,60,1427067))


def geometry():
    dirs=[d for d in product((-1,0,1),repeat=3) if sum(map(abs,d))==1]
    def add(x,y,n=1):
        return tuple(a+n*b for a,b in zip(x,y))
    def near(x):
        return {add(x,d) for d in dirs}
    for y in local.CORNERS:
        x=[add(y,d) for d in dirs]
        z=[add(y,d,2) for d in dirs]
        assert len(set([y,*x,*z]))==13
        outer={w for w in x if sum(map(abs,w))==4}
        inner=set(x)-outer
        assert len(outer)==len(inner)==3 and set(z)&set(local.CORNERS)=={
                  v for v in z if sum(map(abs,v))==3}
        assert len(set(z)&set(local.CORNERS))==3
        for other in local.CORNERS:
            if other!=y:
                other_sources={p for p in near(other) if sum(map(abs,p))==4}
                assert not (set([y,*x,*z]) & other_sources)
        kept=set([y,*x,*z])
        for p,w in zip(x,z):
            assert near(p)&kept=={y,w}
            assert near(w)&kept=={p}
            if w in local.CORNERS:
                inner_w={v for v in near(w) if sum(map(abs,v))==2}
                assert p in inner_w and len(inner_w)==3


def conditional_floor():
    geometry()
    s=1/R
    v=s**3
    a=(s+s**5)/(1+s**6)
    A=(s+s**4*a)/(1+s**5*a)
    # At the second site there are five fixed ordinary spins, or two ordinary
    # and a fixed auxiliary source of strength 3K. Both have at most 5K.
    assert {m+3*t for m in range(3) for t in (0,1)}==set(range(6))
    ratios=[]
    for first,second in product(range(5),range(6)):
        branch=[0,0]
        for left,x,z in product((0,1),repeat=3):
            defects=(left!=x)+(x!=z)+(4-first if x else first)+(5-second if z else second)
            branch[left]+=s**defects
        ratios.append(branch[0]/branch[1])
    assert (min(ratios),max(ratios))==(A,1/A)
    w=A**3
    odds=(w*w-v*v)/(1-v*v*w*w)
    d=odds/(1+odds)
    assert Fraction(0)<LOW<d<Fraction(1,2)
    return d


def bound(caps):
    # Cubic symmetry and global flip imply uniform mass within each orbit.
    owner=local.classes()
    size=Counter(owner)
    adj={(owner[t],owner[t^(1<<j)]) for t in range(256) for j in range(8)}
    assert set(size)==set(caps) and sum(size.values())==256
    entries={c:t*10**9 for c,t in caps.items()}
    assert all(v.denominator==1 for v in entries.values())
    for i,j,x in DUAL:
        assert (i,j) in adj and x>0
        mult=Fraction(x,1000)
        entries[i]-=mult*(LOW/(1-LOW))/size[i]
        entries[j]+=mult/size[j]
    return max(entries.values())/10**9


def run(mode):
    p=Path(__file__).with_name('STAGE20_ORDERED_mix_1791_1132.json')
    saved=json.loads(p.read_text())
    assert Fraction(saved['ratio'])==R
    caps={int(c):Fraction(b) for c,b in saved['caps'].items()}
    eps=Fraction(saved['radius'])
    q=Fraction(saved['Q'])
    assert eps>0 and q==Fraction(8943787728743670589,8943788333028500000)>0
    d=conditional_floor()
    cap=bound(caps)
    lower=-(cap+eps)/q
    assert lower>0
    print(f'r={R}, d={d}, rational probability floor={LOW}, class cap={cap}, '
          f'floor(1e15 L)={floor(10**15*lower)}',flush=True)
    if mode!='local':
        import STAGE20_ORDERED_mixed_caps as mix
        green=mix.stage.W.fixed.phase.AngularGreen(Fraction(0))
        model=mix.stage.W.Certificate(R,'order',green,False,bridges=True,expanded=False,corners=True)
        v,e,m,scale=mix.make_rows(model,green)
        assert e==eps and m==q
        mix.rays.extend_rays(model,3)
        geom=mix.stage.corner_geometry(model)
        mix.check_fields(model,v,scale,geom,caps,raw=(mode=='raw'))
    print(f'PASS {mode}, L={lower}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('local','raw','directed'))
    assert __debug__
    run(parser.parse_args().mode)
