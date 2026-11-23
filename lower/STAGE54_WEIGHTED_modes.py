"""Three consecutive path bonds and the 28-site physical integral at r=309/200."""
from fractions import Fraction
from itertools import product

from STAGE53_DISORDER_prism import rational_grid

UNIT=10**12
GRID=64
COUPLING=Fraction(309,200)
SPARSE=((0,0,1,1),(0,1,1,0),(1,1,0,0),(1,2,0,0),(1,4,0,0))
NAMES=("straight-straight","straight-turn","folded face","twisted turns","planar staircase")


def allowed_other_steps(mask,axial):
    ring=[j for j in range(4) if mask & (1 << j)]
    return ring+([4] if axial else [])


def motif(first,second):
    if first==second==4:
        return 0
    if first==4 or second==4:
        return 1
    if first==second:
        return 2
    if (first-second)%4==2:
        return 4
    return 3


def round_up(x):
    return (x.numerator+x.denominator-1)//x.denominator


def build():
    u=(COUPLING-1)/(COUPLING+1)
    lower,upper=rational_grid(u,GRID)
    sparse=[Fraction(lower[h],UNIT) for h in SPARSE]
    maxloss=Fraction(0)
    witness=None
    constraints=0
    for h,local in lower.items():
        a,b,ma,mb=h
        left=allowed_other_steps(a,ma)
        right=allowed_other_steps(b,mb)
        additional=len(left)+len(right)-2
        for p,q in product(left,right):
            k=motif(p,q)
            deficit=sparse[k]-Fraction(local,UNIT)
            constraints+=1
            if additional:
                loss=deficit/additional
                if loss>maxloss:
                    maxloss=loss
                    witness=h,p,q,additional
            else:
                assert deficit<=0
    chord=min(lower.values())
    price=4*maxloss-Fraction(chord,UNIT)
    assert 0<price<Fraction(1,5)
    # v^- <= exp(-F_sparse) <= v^+, for the true integrals on the five graphs.
    # v^+ also bounds exp(-f_sparse) with f_sparse the certified lower integral.
    fixed=10**6
    down=[(1-Fraction(upper[h],UNIT))*fixed//1 for h in SPARSE]
    up=[round_up((1-f+f*f/2)*fixed) for f in sparse]
    for name,h,lo,hi in zip(NAMES,SPARSE,down,up):
        print(name,"bounds on log gain *1e12",lower[h],upper[h],
              "weight *1e6",lo,hi)
    print("physical marked three-bond comparisons",constraints)
    print("debit per extra neighbour",maxloss,"witness",witness)
    print("all chord lower *1e12",chord,"total price",price)
    return down,up,price,sparse


if __name__=="__main__":
    down,up,_,_=build()
    from pathlib import Path
    import re
    source=Path(__file__).with_name("STAGE54_WEIGHTED_transfer.cpp").read_text()
    for name,values in (("lower_weights",down),("upper_weights",up)):
        row=re.search(r"\b"+name+r"\{(.*?),\s*denominator\}",source).group(1)
        assert [int(v) for v in row.split(",")]==values
