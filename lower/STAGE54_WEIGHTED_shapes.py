"""Physical half-edge and three-bond charge family for the stage 54 test."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import re

from STAGE53_DISORDER_prism import interior_allocations, node_labels, rational_grid
from STAGE54_WEIGHTED_modes import UNIT, allowed_other_steps, motif

R=Fraction(1927,1250)
PANELS=64


def budgets(lower):
    path=[]
    chord=[]
    for h,value in lower.items():
        a,b,ma,mb=h
        left,right=allowed_other_steps(a,ma),allowed_other_steps(b,mb)
        i,j=node_labels(h)
        for shape in sorted({motif(p,q) for p,q in product(left,right)}):
            path.append((i,j,shape,value))
        if min(len(left),len(right))>=2:
            chord.append((i+10,j+10,value))
    return path,chord


def certify(bonds,all_charges,theta):
    path,chords=bonds
    a=all_charges[:20]
    w=all_charges[20:]
    assert min(all_charges)>=0
    slack=min(v-a[i]-a[j]-w[k] for i,j,k,v in path)
    chord_slack=min(v+int(theta*UNIT)-a[i]-a[j] for i,j,v in chords)
    gain=min(sum(x*y for x,y in zip(a,c)) for *_,c in interior_allocations())
    assert min(slack,chord_slack)>=0
    return slack,chord_slack,gain


CHARGES=(0,4191328333,2699473836,1309065313,0,4191328333,3406623940,
         1425837850,738905393,0,0,0,2983708843,2765490332,1809011447,
         0,2276558740,2882262869,2547916991,2095664091,
         1127558275,544046375,544046375,94806903,0)
PRICE=Fraction(728677627,500000000000)


def signed_kernel_weights(name):
    contents=Path(__file__).with_name("STAGE54_WEIGHTED_transfer.cpp").read_text()
    line=re.search(r"\b"+name+r"\{(.*?),\s*denominator\}",contents).group(1)
    return [int(s) for s in line.split(",")]


def check_upper(maximum):
    u=(R-1)/(R+1)
    low,_=rational_grid(u,PANELS)
    rows=budgets(low)
    result=certify(rows,CHARGES,PRICE)
    h=Fraction(result[-1],UNIT)
    print("r",R,"u",u,"rows",len(rows[0]),len(rows[1]),"slacks/min gain",result)
    w=[Fraction(x,UNIT) for x in CHARGES[20:]]
    rational_weights=[10**6*(1-x+x*x/2) for x in w]
    up=[(x.numerator+x.denominator-1)//x.denominator for x in rational_weights]
    assert up==signed_kernel_weights("balanced")
    print("upper weights *1e6",up,"H",h,"theta",PRICE)
    assert (5/maximum)**4 < Fraction(5,4)
    z=h-PRICE/4
    T=1+z+z*z/2+z**3/6
    margin=T/(u*maximum)-1
    print("weighted growth bound",maximum,"contraction margin below, units 1e-12",margin*UNIT//1)
    extended=T/maximum
    r_ext=(1+extended)/(1-extended)
    print("strict extension",r_ext,"lower 1e-15 integer", r_ext*10**15//1)
    return margin


if __name__=="__main__":
    log=Path(__file__).with_name("STAGE54_WEIGHTED_16.log").read_text()
    match=re.search(r"r=1927/1250 priced shape matrix upper in units 1e-6 (\d+)",log)
    assert match and int(match.group(1))<=4730332
    assert "memory 16 rows 12994574 edges 53221763" in log
    result=check_upper(Fraction(4730332,10**6))
    assert result>0
