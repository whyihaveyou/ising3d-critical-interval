"""Finite prism cavity on the cubic lattice used by the stage 53 audit."""
from fractions import Fraction
from itertools import product
from math import prod

RING = range(4)
SIGNS = tuple(product((-1, 1), repeat=4))


def canonical(a, b):
    options = []
    for first, second in ((a, b), (b, a)):
        for k in RING:
            for direction in (-1, 1):
                ix = [(k + direction * i) % 4 for i in RING]
                options.append(tuple(sum(((mask >> j) & 1) << n for n, j in enumerate(ix)) for mask in (first, second)))
    return min(options)


def strength_mask(mask, tied, strong):
    return [tied if mask >> j & 1 else strong for j in RING]


def perimeter_bits(mask):
    return tuple((mask >> i) & 1 for i in RING)


def scaled_side(mask, axial, tied, strong, scale):
    """Weights with the hub fixed to plus or minus, eliminating five outer spins."""
    bonds = strength_mask(mask,tied,strong)
    outer = tied if axial else strong
    results = []
    for spins in SIGNS:
        z = [k*x for k,x in zip(bonds,spins)]
        pos = prod(scale*scale + outer*z_i for z_i in z)
        neg = prod(scale*scale - outer*z_i for z_i in z)
        plus = prod(scale+z_i for z_i in z)*((scale+outer)*pos+(scale-outer)*neg)
        minus = prod(scale-z_i for z_i in z)*((scale-outer)*pos+(scale+outer)*neg)
        results.append((plus+minus,plus-minus))
    return results


def corner_kernel(left_bits,right_bits,tied,strong,scale):
    k = [tied if bit else strong for bit in left_bits]
    l = [tied if bit else strong for bit in right_bits]
    result = []
    for x,y in product((-1,1),repeat=2):
        row = []
        for p,q in product((-1,1),repeat=2):
            val = ((scale**2+k[0]*k[1]*x*y)*(scale**2+l[0]*l[1]*p*q)
                   + tied*scale*(k[0]*x+k[1]*y)*(l[0]*p+l[1]*q))
            join = tied if left_bits[0] and right_bits[0] else strong
            row.append(val*(scale+join*x*p))
        result.append(row)
    return result


SIDE_PAIRS = tuple(tuple((int(x[i]>0)*2 + int(x[(i+1)%4]>0)) for x in SIGNS) for i in RING)
# Indices giving spin-flip-equivalent terms. The two-hub product is unchanged.
HALF = range(8)


def orbit_list():
    return [(a,b,ma,mb) for a,b in product(range(16),repeat=2)
            if (a,b)==canonical(a,b) for ma,mb in product((0,1),repeat=2)]


def integer_sums(u, steps, node, orbits):
    """Return scaled even and two-source polynomials with the hub bond absent.

    The omitted positive multiplier is independent of the node and pattern.
    A grid of 2*steps intervals uses the same denominator on every bond.
    """
    scale = 2*steps*u.denominator
    strong = 2*steps*u.numerator
    tied = node*u.numerator
    sides = {(a,o):scaled_side(a,o,tied,strong,scale)
             for a in range(16) for o in (0,1)}
    kernels = {(la,lb):corner_kernel(la,lb,tied,strong,scale)
               for la in product((0,1),repeat=2) for lb in product((0,1),repeat=2)}
    out = {}
    for a,b in sorted(set((a,b) for a,b,_,_ in orbits)):
        x, y = perimeter_bits(a), perimeter_bits(b)
        local = [kernels[((x[i],x[(i+1)%4]),(y[i],y[(i+1)%4]))] for i in RING]
        weight = []
        for li in HALF:
            fixed = [local[i][SIDE_PAIRS[i][li]] for i in RING]
            weight.append([fixed[0][SIDE_PAIRS[0][j]]*fixed[1][SIDE_PAIRS[1][j]]
                           *fixed[2][SIDE_PAIRS[2][j]]*fixed[3][SIDE_PAIRS[3][j]]
                           for j in range(16)])
        for right_axial in (0,1):
            z0, z1 = [], []
            rs = sides[b,right_axial]
            for row in weight:
                z0.append(sum(w*rs[j][0] for j,w in enumerate(row)))
                z1.append(sum(w*rs[j][1] for j,w in enumerate(row)))
            for left_axial in (0,1):
                ls = sides[a,left_axial]
                even = sum(ls[j][0]*z0[j] for j in HALF)
                odd  = sum(ls[j][1]*z1[j] for j in HALF)
                assert 0 <= odd < even
                out[a,b,left_axial,right_axial] = (even,odd)
    return out


def ceil_ratio(a,b):
    return (a+b-1)//b


def rational_grid(u,steps,fixed=10**12):
    """Rigorous lower/upper integral via positive-coefficient source polynomials."""
    orbits = orbit_list()
    zero = {h:0 for h in orbits}
    low, high = zero.copy(), zero.copy()
    previous = integer_sums(u,steps,0,orbits)
    scale = 2*steps*u.denominator
    for k in range(steps):
        mids = integer_sums(u,steps,2*k+1,orbits)
        ends = integer_sums(u,steps,2*k+2,orbits)
        for h in orbits:
            e0,s0 = previous[h]
            e1,s1 = ends[h]
            sm = mids[h][1]
            # N has nonnegative coefficients, so midpoint <= integral <= trapezoid.
            # D(x) = even + u*x*source is positive and increasing on [0,1].
            d0 = scale*e0+2*k*u.numerator*s0
            d1 = scale*e1+(2*k+2)*u.numerator*s1
            low[h]  += (fixed*2*u.numerator*sm)//d1
            high[h] += ceil_ratio(fixed*u.numerator*(s0+s1),d0)
        previous = ends
    assert all(0<low[h]<=high[h] for h in orbits)
    return low,high


def node_labels(h):
    a,b,ma,mb = h
    return bin(a).count('1')+5*ma,bin(b).count('1')+5*mb


def interior_allocations():
    for d in range(2,7):
        for pairs in range(max(0,d-3),d//2+1):
            opposite = 2*pairs
            single = d-opposite
            for walk_opposite in range(max(0,2-single),min(2,opposite)+1):
                counts = [0]*20
                for label,used,total in ((d+3,walk_opposite,opposite),(d-1,2-walk_opposite,single)):
                    if used:
                        counts[label] += used
                    if total-used:
                        counts[label+10] += total-used
                yield d,pairs,walk_opposite,counts


def edge_options(h):
    i,j=node_labels(h)
    di=i%5+i//5+1
    dj=j%5+j//5+1
    if min(di,dj)>=2:
        yield i,j
    if min(di,dj)>=3:
        yield i+10,j+10


SCALE = 10**12
WALK_GROWTH = Fraction(94639,20000)
CASES = (
    ("degree",Fraction(15407,10000),36,0,
     (4227684207,3528244503,2779480712,2030711740,1273002953,
      3528244503,3405503407,2661480278,1910625383,1176081499)),
    ("walk",Fraction(1541,1000),192,0,
     (0,3722798857,2789516722,1692640637,0,3722798857,2789516722,1692640637,
      1732536417,996622563,0,0,1866564259,2030158208,32784558,0,1866564259,
      2030158208,1973870143,1363088135)),
    ("entropy",Fraction(30829,20000),64,900000000,
     (0,4181010088,2796622230,1408633261,0,4181010088,2796622230,1696196670,
      1287882389,549724752,0,0,2768775703,2484813406,929924211,0,2768775703,
      2772376814,2428165575,1815642656)),
)


def verify_walk_charges(scores,charges,price=0):
    """All removed graph endpoints are handled by the n-13 proof in the report."""
    assert min(charges)>=0
    edge_slacks=[]
    for h,bound in scores.items():
        for i,j in edge_options(h):
            available = bound + (price if i>=10 else 0)
            edge_slacks.append(available-charges[i]-charges[j])
    assert min(edge_slacks)>=0
    vertex_totals=[]
    for d,pairs,walk_opposite,counts in interior_allocations():
        vertex_totals.append((sum(x*y for x,y in zip(charges,counts)),d,pairs,walk_opposite))
    return min(vertex_totals)[0],min(edge_slacks),len(vertex_totals),len(edge_slacks)


def verify_degree_charges(scores,charges):
    edge_slacks=[bound-charges[i]-charges[j]
                 for h,bound in scores.items() for i,j in [node_labels(h)]]
    assert min(charges)>=0 and min(edge_slacks)>=0
    vertex_totals=[]
    for d in range(2,7):
        for pairs in range(max(0,d-3),d//2+1):
            v=2*pairs*charges[d+3] if pairs else 0
            if d>2*pairs:
                v+=(d-2*pairs)*charges[d-1]
            vertex_totals.append(v)
    return min(vertex_totals),min(edge_slacks),len(vertex_totals),len(edge_slacks)


def canonical_full(h):
    a,b,ma,mb=h
    possibilities=[]
    for x,y,m,n in ((a,b,ma,mb),(b,a,mb,ma)):
        for k in RING:
            for direction in (-1,1):
                ids=[(k+direction*i)%4 for i in RING]
                masks=tuple(sum((v>>old&1)<<i for i,old in enumerate(ids)) for v in (x,y))
                possibilities.append((*masks,m,n))
    return min(possibilities)


def check_symmetry(table):
    all_modes=product(range(16),range(16),range(2),range(2))
    covered={}
    for h,v in table.items():
        k=canonical_full(h)
        if k in covered:
            assert covered[k]==v
        covered[k]=v
    assert all(canonical_full(h) in covered for h in all_modes)
    return len(covered)


def lower_log(x,terms=28):
    z=(x-1)/(x+1)
    assert 0<z<1
    return 2*sum(z**(2*j+1)/(2*j+1) for j in range(terms))


def hamilton_path_obstruction():
    """Bound the maximum uniform path gain at r=1.545 using the K0 pressure certificate."""
    u0,u=Fraction(9,41),Fraction(109,509)
    y=u**3
    detour=(4*y+4*y**3)/(1+6*y*y+y**4)
    drop=3*(u0-u)*detour/(1+u0*detour)
    p_upper=Fraction(780611553920666167,10**18)
    normalized_upper=p_upper-lower_log(Fraction(2))-3*lower_log(Fraction(41,40))-drop
    required=lower_log(WALK_GROWTH*u)
    assert normalized_upper < Fraction(127598,10**7)
    assert required > Fraction(132372,10**7)
    print("r=309/200 Hamilton path obstruction, units 1e-12")
    print("max uniform gain <",ceil_ratio(normalized_upper.numerator*SCALE,normalized_upper.denominator))
    print("required gain >",required*SCALE//1,"gap >", (required-normalized_upper)*SCALE//1)


def run_case(name,r,steps,price,charges):
    u=(r-1)/(r+1)
    low,high=rational_grid(u,steps,SCALE)
    unique=check_symmetry(low)
    if len(charges)==10:
        raw,slack,vertex_rows,edge_rows=verify_degree_charges(low,charges)
    else:
        raw,slack,vertex_rows,edge_rows=verify_walk_charges(low,charges,price)
    assert (5/WALK_GROWTH)**4 < Fraction(5,4)
    assert Fraction(price,SCALE)<Fraction(1,5)
    z=Fraction(4*raw-price,4*SCALE)
    margin=(1+z+z*z/2+z**3/6)/(WALK_GROWTH*u)-1
    assert margin>0
    if name == "entropy":
        extended_u = (1+z+z*z/2+z**3/6)/WALK_GROWTH
        print("strict critical lower endpoint from the existing margin", (1+extended_u)/(1-extended_u))
    print(name, "r",r,"u",u,"panels",steps,"tested",len(low),
          "symmetry orbits",unique,"vertex rows",vertex_rows,"edge rows",edge_rows)
    print("edge slack, raw gain, chord price, net gain, Perron margin; units 1e-12")
    print(slack,raw,price,z*SCALE,margin*SCALE//1)
    for label,h in (("straight",(0,0,1,1)),("two distinct turns",(1,2,0,0)),
                    ("all stencil sites occupied",(15,15,1,1))):
        print(label,low[h],high[h])


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true",help="also replay the two intermediate endpoints")
    args=parser.parse_args()
    assert all((5-a)*Fraction(5,4)**a<=5 for a in range(5))
    for case in CASES if args.all else CASES[-1:]:
        run_case(*case)
    hamilton_path_obstruction()


if __name__ == "__main__":
    main()
