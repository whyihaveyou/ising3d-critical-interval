"""Independent spin-layer verifier of the synchronous 28-site prism certificate."""

from fractions import Fraction as Q
from itertools import combinations, product
from math import prod


FROZEN = Q(15407, 10000)
U = Q(94639, 20000)
FOCUS = 10**12
CHARGES = (4227684207, 3528244503, 2779480712, 2030711740, 1273002953,
           3528244503, 3405503407, 2661480278, 1910625383, 1176081499)
SPLIT = (0,3722798857,2789516722,1692640637,0,3722798857,2789516722,
         1692640637,1732536417,996622563,0,0,1866564259,2030158208,32784558,
         0,1866564259,2030158208,1973870143,1363088135)
CONTACT = (0,4181010088,2796622230,1408633261,0,4181010088,2796622230,
           1696196670,1287882389,549724752,0,0,2768775703,2484813406,929924211,
           0,2768775703,2772376814,2428165575,1815642656)
ARMS = ((1, 0), (0, 1), (-1, 0), (0, -1))
CORNERS = tuple((a[0]+b[0], a[1]+b[1]) for a, b in zip(ARMS, ARMS[1:]+ARMS[:1]))
PLANE = ((0, 0),) + ARMS + CORNERS
INNER = [(i,j) for i,j in combinations(range(9), 2)
         if sum(abs(a-b) for a,b in zip(PLANE[i],PLANE[j])) == 1]
assert len(INNER) == 12
SPIN = [tuple(-1 if mask >> i & 1 else 1 for i in range(8)) for mask in range(256)]


def symmetries(mask):
    return tuple(sum(((mask >> ((start+orientation*i) % 4)) & 1) << i for i in range(4))
                 for start, orientation in product(range(4), (-1, 1)))


def orbit(row):
    left,right,out_l,out_r = row
    choices = []
    for a,b in zip(symmetries(left),symmetries(right)):
        choices.extend(((a,b,out_l,out_r),(b,a,out_r,out_l)))
    return min(choices)


def side_messages(mask, outer, tied, strong, scale):
    occupied = (1,) + tuple((mask >> i) & 1 for i in range(4)) + (1,)*4
    weights = [tied if occupied[i] and occupied[j] else strong for i,j in INNER]
    # Direct sums of each intervening outer spoke. Then sum the shared outer hub.
    axial = tied if outer else strong
    cap = {}
    for hub in (1, -1):
        for arms in product((1, -1), repeat=4):
            total = 0
            for outer_hub in (1,-1):
                factor = scale+axial*hub*outer_hub
                for i, arm in enumerate(arms):
                    cross = tied if occupied[1+i] else strong
                    factor *= sum((scale+axial*outer_hub*leaf)*(scale+cross*arm*leaf)
                                  for leaf in (1,-1))
                total += factor
            cap[hub, arms] = total
    even, difference = [], []
    for perimeter in SPIN:
        totals = []
        for hub in (1, -1):
            s = (hub,)+perimeter
            planar = prod(scale+w*s[i]*s[j] for (i,j),w in zip(INNER,weights))
            totals.append(planar*cap[hub,perimeter[:4]])
        even.append(totals[0]+totals[1])
        difference.append(totals[0]-totals[1])
    return even,difference


def transfer(vec, factors):
    answer = vec.copy()
    for bit, weight_same, weight_opposite in factors:
        step = 1 << bit
        for block in range(0, 256, 2*step):
            for i in range(block, block+step):
                a, b = answer[i],answer[i+step]
                answer[i] = weight_same*a+weight_opposite*b
                answer[i+step] = weight_opposite*a+weight_same*b
    return answer


def partition_node(rows, tied, strong, scale):
    sides = {key:side_messages(*key,tied,strong,scale) for key in
             {(a,c) for a,b,c,d in rows} | {(b,d) for a,b,c,d in rows}}
    mapped = {}
    output = {}
    for a,b,c,d in rows:
        rel = a & b
        key = a,c,rel
        if key not in mapped:
            weights = [tied if (rel >> i) & 1 else strong for i in range(4)] + [tied]*4
            links = [(i, scale+w, scale-w) for i,w in enumerate(weights)]
            mapped[key] = [transfer(vec,links) for vec in sides[a,c]]
        first,second = mapped[key]
        even,odd = sides[b,d]
        e = sum(v*w for v,w in zip(first,even))
        s = sum(v*w for v,w in zip(second,odd))
        assert 0 <= s < e
        output[a,b,c,d] = (e,s)
    return output


def table_and_vertex(r, panels, charges, penalty=Q(0)):
    marked = list(product(range(16),range(16),range(2),range(2)))
    indices = {row:orbit(row) for row in marked}
    rows = sorted(set(indices.values()))
    lower = dict.fromkeys(rows, 0)
    u = (r-1)/(r+1)
    scale = 2*panels*u.denominator
    strong = 2*panels*u.numerator
    for panel in range(panels):
        mid = partition_node(rows,(2*panel+1)*u.numerator,strong,scale)
        tied = (2*panel+2)*u.numerator
        end = partition_node(rows,tied,strong,scale)
        for row in rows:
            e,s = end[row]
            lower[row] += (2*FOCUS*u.numerator*mid[row][1])//(scale*e+tied*s)
    slacks = []
    split = (len(charges) == 20)
    for a,b,c,d in marked:
        ai, bi = bin(a).count("1"),bin(b).count("1")
        i,j = ai+5*c,bi+5*d
        degrees = 1+ai+c,1+bi+d
        for offset in ((0,10) if split else (0,)):
            # For a retained edge in the split proof, both ends are interior.
            # An interior chord leaves two other required path edges at each end.
            if split and min(degrees) < (3 if offset else 2):
                continue
            available = lower[indices[a,b,c,d]] + (FOCUS*penalty if offset else 0)
            slacks.append(available-charges[i+offset]-charges[j+offset])
    assert min(slacks) >= 0
    local = []
    for selection in range(64):
        bits = [(selection >> k) & 1 for k in range(6)]
        degree = sum(bits)
        if degree < 2:
            continue
        active = [j for j,v in enumerate(bits) if v]
        for walking in combinations(active,2) if split else [active]:
            half = [charges[degree-1-bits[j^1]+5*bits[j^1]+(0 if j in walking else 10)]
                    for j in active]
            local.append(sum(half))
    assert min(charges) >= 0
    gain = Q(min(local),FOCUS)
    expected = 8362020128 if penalty else 7445597666 if split else 7056488994
    assert gain == Q(expected,FOCUS)
    net = gain - penalty/4
    exp_lower = 1+net+net**2/2+net**3/6
    max_u = exp_lower/U
    frozen_bound = (1+max_u)/(1-max_u)
    assert frozen_bound > Q(154073,100000)
    print("input coupling",r,"panels",panels,"charge classes",len(charges))
    print("labelled endpoint neighbourhoods",len(marked),"physical symmetry orbits",len(rows))
    print("independent plane transfers",len(rows)*2*panels,"tested edge constraints",len(slacks),
          "minimum charge slack at scale 1e12",min(slacks))
    print("six-direction site configurations",len(local),"gain at scale 1e12",int(gain*FOCUS))
    print("endpoint debit at most",12*max(charges),"at scale 1e12; net gain",net)
    print("frozen endpoint r",frozen_bound,"decimal floor at 1e15",frozen_bound*10**15//1)


def embedding():
    vertices = {(x,v,w) for x in (0,1) for v,w in PLANE}
    vertices |= {(x,v,w) for x in (-1,2) for v,w in PLANE[:5]}
    edges = {(v,w) for v,w in combinations(sorted(vertices),2)
             if sum(abs(i-j) for i,j in zip(v,w)) == 1}
    assert len(vertices) == 28 and len(edges) == 51
    count = 0
    for L in (6,8,10):
        for axis in range(3):
            for shift in product(range(L),repeat=3):
                def image(z):
                    return tuple((shift[i]+z[(i-axis)%3])%L for i in range(3))
                assert len({image(v) for v in vertices})==28
                for v,w in edges:
                    sv,sw=image(v),image(w)
                    assert sum(min((a-b)%L,(b-a)%L) for a,b in zip(sv,sw))==1
                count += 1
    print("distinct translated and oriented torus embeddings",count,"vertices 28 edges 51")


def kinetic_torus(L=4, limit=6):
    from collections import Counter
    dirs = [tuple(v if i==j else 0 for i in range(3)) for j in range(3) for v in (-1,1)]
    neighbours = {x:[tuple((i+j)%L for i,j in zip(x,d)) for d in dirs]
                  for x in product(range(L),repeat=3)}
    start = (0,0,0)
    counts = [Counter() for _ in range(limit+1)]

    def explore(path, contacts, denominator):
        end = path[-1]
        excluding = set(path[:-2])
        effects = sum(y in excluding for y in neighbours[end])
        length = len(path)-1
        contacts += effects
        if length:
            induced = sum(v in neighbours[u] for u,v in combinations(path,2))
            assert induced-length == contacts
            counts[length][contacts,denominator] += 1
        if length == limit:
            return
        options = [y for y in neighbours[end] if y not in path]
        assert len(options)==(5-effects if length else 6)
        for y in options:
            explore(path+[y],contacts,denominator*len(options))

    explore([start],0,1)
    base = Q(5,4)
    for n, tally in enumerate(counts[1:],1):
        mass = sum(Q(v,p) for (c,p),v in tally.items())
        weighted = sum(base**c*v for (c,p),v in tally.items())
        assert mass <= 1 and weighted <= 6*base**5 * 5**(n-1) * mass
        assert all(base**c*p <= 6*base**5 * 5**(n-1) for c,p in tally)
    # Three successive positive x-steps have a terminal contact on T4 but not in Z^3.
    assert (3,0,0) in neighbours[start] and abs(3-0)>1
    assert all((5-j)*base**j<=5 for j in range(5)) and (5/U)**4<base
    print("kinetic torus T4^3 paths through length",limit,
          sum(sum(rows.values()) for rows in counts),"wrapped endpoint contact 1 versus lifted 0")


def dense_obstruction():
    # A Hamiltonian snake in each even torus occupies all vertices. Each
    # three-edge square bypass is a series chain; the four bypasses are parallel.
    def lower_log(value):
        z = (value-1)/(value+1)
        assert 0 < z < 1
        return 2*sum(z**(2*j+1)/(2*j+1) for j in range(32))
    u, top = Q(109,509),Q(9,41)
    by = u**3
    conn = ((1+by)**4-(1-by)**4)/((1+by)**4+(1-by)**4)
    p_bound = Q("0.780611553920666167")
    upper = p_bound - lower_log(Q(2)) - 3*lower_log(Q(41,40)) - 3*(top-u)*conn/(1+top*conn)
    needed = lower_log(U*u)
    assert upper < Q(12759795513,FOCUS) and needed > Q(13237245382,FOCUS)
    print("r=1.545 uniform path obstruction: guaranteed gap at scale 1e12",
          (needed-upper)*FOCUS//1)


if __name__ == "__main__":
    import sys
    embedding()
    kinetic_torus()
    dense_obstruction()
    if "--final-only" not in sys.argv:
        table_and_vertex(FROZEN,36,CHARGES)
        table_and_vertex(Q(1541,1000),192,SPLIT)
    table_and_vertex(Q(30829,20000),64,CONTACT,Q(9,10000))
