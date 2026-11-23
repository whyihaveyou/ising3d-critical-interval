"""Sum the 13-site star explicitly with three neighbouring auxiliary sources fixed."""

from fractions import Fraction
from itertools import product

import STAGE20_ORDERED_corner_star as proof


def exact_probability(ratio, side4, side5):
    a,b=ratio.numerator,ratio.denominator
    def bond(left,right):
        return a if left==right else b
    # The first three arms go to shell two; the last three go to shell four.
    # At each of the first three second sites, side5 = n_two_true + 3*t.
    assert all(n in {m+3*t for m,t in product(range(3),range(2))} for n in side5[:3])
    lam=[]
    for j in range(4):
        lam.append((a**(6-j)*b**j - a**j*b**(6-j),
                    a**(3+j)*b**(3-j) - a**(3-j)*b**(3+j)))
    assert all(min(pair)>=0 for pair in lam)
    weights=[0,0]
    for x in product(range(2),repeat=6):
        mixed=[]
        for t in (0,1):
            mixed.append(lam[sum(x[3:])][t]*sum(bond(y,t)**3
                         *bond(y,x[0])*bond(y,x[1])*bond(y,x[2]) for y in (0,1)))
        for z in product(range(2),repeat=6):
            n=sum(i!=j for i,j in zip(x,z))
            n+=sum(4-j if bit else j for bit,j in zip(x,side4))
            n+=sum(5-j if bit else j for bit,j in zip(z,side5))
            p=a**(60-n)*b**n
            for t in (0,1):
                weights[t]+=p*mixed[t]
    return Fraction(weights[0],sum(weights))


if __name__=='__main__':
    assert __debug__
    d=proof.conditional_floor()
    tests=[([4]*6,[5]*6),([0]*6,[0]*6),([2]*6,[3]*6),([4,0,1,3,2,0],[1,3,2,5,4,0])]
    values=[exact_probability(proof.R,h,n) for h,n in tests]
    assert min(values)==values[0]==d and max(values)==values[1]==1-d
    for p in values: print(f'P(+)={float(p):.14g}')
    print('PASS direct 13-site sums over 8192 spin configurations per environment, both extrema exact')
