"""Independent physical-coordinate, high-temperature audit of the weakened prisms."""

from collections import deque
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
RING = ((1, 0), (0, 1), (-1, 0), (0, -1))
HUBS = ((0,0,0), (1,0,0))
UNIT = 10**12
SAW = Fraction(94639,20000)
CASES = (
    (Fraction(15407,10000), 36, 0,
     (4227684207,3528244503,2779480712,2030711740,1273002953,
      3528244503,3405503407,2661480278,1910625383,1176081499)),
    (Fraction(1541,1000), 192, 0,
     (0,3722798857,2789516722,1692640637,0,
      3722798857,2789516722,1692640637,1732536417,996622563,
      0,0,1866564259,2030158208,32784558,
      0,1866564259,2030158208,1973870143,1363088135)),
    (Fraction(30829,20000), 64, 900000000,
     (0,4181010088,2796622230,1408633261,0,
      4181010088,2796622230,1696196670,1287882389,549724752,
      0,0,2768775703,2484813406,929924211,
      0,2768775703,2772376814,2428165575,1815642656)),
)


def prism():
    pts = sorted((x,y,z) for x,y,z in product(range(-1,3),range(-1,2),range(-1,2))
                 if x in (0,1) or abs(y)+abs(z)<=1)
    edges = [(a,b) for a,b in combinations(pts,2)
             if sum(abs(s-t) for s,t in zip(a,b))==1 and (a,b)!=HUBS]
    assert len(pts)==28 and len(edges)==50
    return pts,edges


def physical_mask(pts, edges, la, rb):
    # A label of None is an unconstrained occupation. Hubs belong to the path set.
    mark = {v:None for v in pts}
    for a in HUBS:
        mark[a]=1
    for x,local in ((0,la),(1,rb)):
        for j,(y,z) in enumerate(RING):
            mark[(x,y,z)] = (local>>j)&1
        mark[(-1 if x==0 else 2,0,0)] = (local>>4)&1
    assert sum(q is None for q in mark.values())==16
    mask=0
    for j,(a,b) in enumerate(edges):
        weak=mark[a]!=0 and mark[b]!=0
        for occ_a in ((0,1) if mark[a] is None else (mark[a],)):
            for occ_b in ((0,1) if mark[b] is None else (mark[b],)):
                assert not (occ_a and occ_b) or weak
        mask |= int(weak)<<j
    return mask


def boundary(mask,edges):
    odd=set()
    for i,(a,b) in enumerate(edges):
        if mask&(1<<i):
            odd.symmetric_difference_update((a,b))
    return odd


def cycle_basis(pts, edges):
    root = HUBS[0]
    adj={v:[] for v in pts}
    for j,(a,b) in enumerate(edges):
        adj[a].append((b,j))
        adj[b].append((a,j))
    queue=deque((root,))
    paths,tree={root:0},set()
    while queue:
        a=queue.popleft()
        for b,j in adj[a]:
            if b not in paths:
                paths[b]=paths[a]^(1<<j)
                tree.add(j)
                queue.append(b)
    chords=[j for j in range(len(edges)) if j not in tree]
    bases=[]
    chord_bits=sum(1<<j for j in chords)
    for j in chords:
        a,b=edges[j]
        c=paths[a]^paths[b]^(1<<j)
        assert c & chord_bits == (1<<j) and not boundary(c,edges)
        bases.append(c)
    source=paths[HUBS[1]]
    assert len(paths)==28 and len(bases)==23 and boundary(source,edges)==set(HUBS)
    return bases,source


def transforms(mark):
    for swap,sy,sz in product((False,True),(-1,1),(-1,1)):
        ans=mark&16
        for j,(y,z) in enumerate(RING):
            if swap: y,z=z,y
            site=RING.index((sy*y,sz*z))
            if mark&(1<<j): ans|=1<<site
        yield ans


def physical_orbits():
    all_sym=[tuple(transforms(m)) for m in range(32)]
    canonical={}
    for la,rb in product(range(32),repeat=2):
        choices=[tuple(sorted(pair)) for pair in zip(all_sym[la],all_sym[rb])]
        canonical[(la,rb)]=min(choices)
    return canonical


def even_statistics():
    pts,edges=prism()
    bases,source=cycle_basis(pts,edges)
    canonical=physical_orbits()
    masks={k:physical_mask(pts,edges,*k) for k in canonical}
    reps=sorted(set(canonical.values()))
    serial=' '.join(map(str,[len(edges),len(bases),source,len(reps),*bases,
                             *(masks[k] for k in reps)]))
    with TemporaryDirectory(prefix='STAGE53_PRISM_CHECK_build_',dir=HERE) as build:
        exe=Path(build)/'parity'
        subprocess.run(['clang++','-std=c++17','-O3',str(HERE/'STAGE53_PRISM_CHECK_cycles.cpp'),
                        '-o',str(exe)],check=True)
        run=subprocess.run([str(exe)],input=serial,text=True,capture_output=True,check=True)
    polys={}
    assert len(run.stdout.splitlines())==2*len(reps)
    for line in run.stdout.splitlines():
        data=list(map(int,line.split()))
        ix,parity=data[:2]
        assert len(data[2:])%3==0
        terms=list(zip(data[2::3],data[3::3],data[4::3]))
        assert sum(c for _,_,c in terms)==2**23
        assert all(d%2==parity and k<=d for d,k,c in terms)
        assert (0,0,1) in terms if parity==0 else min(d for d,k,c in terms)==3
        assert max(k for d,k,c in terms)<=bin(masks[reps[ix]]).count('1')
        polys[(reps[ix],parity)]=terms
    print('coordinates',len(pts),'bonds',len(edges),'fundamental cycles',len(bases),
          'per parity',2**23,'labelled masks',len(canonical),'full-geometry orbits',len(reps))
    return canonical,reps,polys


def evaluations(terms,u,grid,weak_max):
    # Integer proportional to E(x) or N(x), with x = grid node / grid.
    # Both polynomials at every node share the factor u.denominator^50 * grid^weak_max.
    coeffs=[0]*(weak_max+1)
    for total,weak,count in terms:
        coeffs[weak]+=count*u.numerator**total * u.denominator**(50-total)
    reverse=[coeffs[j]*grid**(weak_max-j) for j in reversed(range(weak_max+1))]
    vals=[]
    for node in range(grid+1):
        z=0
        for a in reverse:
            z=z*node+a
        vals.append(z)
    return vals


def integral_tables(polys,reps,u,steps):
    grid=2*steps
    low,high={},{}
    for rep in reps:
        terms=[polys[(rep,b)] for b in (0,1)]
        maxw=max(k for row in terms for d,k,c in row)
        even,odd=(evaluations(row,u,grid,maxw) for row in terms)
        denom=[grid*u.denominator*e+node*u.numerator*n
               for node,(e,n) in enumerate(zip(even,odd))]
        assert 0<denom[0] and all(a<=b for a,b in zip(denom,denom[1:]))
        low[rep]=sum((2*UNIT*u.numerator*odd[j+1])//denom[j+2]
                     for j in range(0,grid,2))
        high[rep]=sum(-((-UNIT*u.numerator*(odd[j]+odd[j+2]))//denom[j])
                      for j in range(0,grid,2))
        assert 0<low[rep]<=high[rep]
    return low,high


def half_edge_slot(row):
    return bin(row&15).count('1')+5*(row>>4)


def vertex_gains(charges):
    per_vertex=[]
    for occupied in range(64):
        present=[j for j in range(6) if occupied&(1<<j)]
        for through in combinations(present,2):
            benefit=0
            for central in present:
                outward=central^1
                others=[j for j in present if j//2!=central//2]
                slot=len(others)+5*int(outward in present)
                if len(charges)==20 and central not in through: slot+=10
                benefit+=charges[slot]
            per_vertex.append((benefit,occupied,through))
    return per_vertex


def validate(r,steps,chord_credit,charges,canonical,reps,polys):
    u=(r-1)/(r+1)
    low,high=integral_tables(polys,reps,u,steps)
    constraints=[]
    for ends in sorted(canonical):
        rep=canonical[ends]
        slots=[half_edge_slot(m) for m in ends]
        degrees=[1+bin(m).count('1') for m in ends]
        if len(charges)==10: offsets=(0,)
        else: offsets=[q for required,q in ((2,0),(3,10)) if min(degrees)>=required]
        for offset in offsets:
            assigned=sum(charges[s+offset] for s in slots)
            available=low[rep]+(chord_credit if offset==10 else 0)
            constraints.append((available-assigned,ends,offset,low[rep],assigned,high[rep]))
    assert min(charges)>=0 and min(constraints)[0]>=0
    gains=vertex_gains(charges)
    gain,occupied,through=min(gains)
    # On every simple 6-regular finite graph, after a nonterminal vertex
    # creates j new chords the step has at most 5-j continuations.
    w=Fraction(5,4)
    assert all((5-j)*w**j<=5 for j in range(6))
    assert (5/SAW)**4<w
    assert Fraction(chord_credit,UNIT)<Fraction(1,5)  # 1/5 <= ln(5/4)
    gain_frac=(gain-Fraction(chord_credit,4))/UNIT
    exp_low=1+gain_frac+gain_frac**2/2+gain_frac**3/6
    ratio=exp_low/(SAW*u)
    assert ratio>1
    print('r',r,'subintervals',steps,'colours',len(charges),'edge constraints',len(constraints),
          'interior oriented patterns',len(gains),'least gain',gain,'over',UNIT)
    print('limiting vertex occupied bits',bin(occupied),'path directions',through)
    print('chord allowance',Fraction(chord_credit,UNIT),'net per-origin gain',gain_frac)
    worst=min(constraints)
    print('tight edge row (slack, masks, offset, lower, assigned, upper)',worst)
    print('floor 1e12 * (cubic Taylor lower/(u*SAW)-1)', (UNIT*(ratio-1))//1)
    if r==Fraction(1541,1000):
        assert gain==7445597666 and (UNIT*(ratio-1))//1==1922310
    if r==Fraction(30829,20000):
        assert gain==8362020128 and (UNIT*(ratio-1))//1==38978200
        # Fixed-v cavity correlations and the integration range increase with u.
        # At the extension u = exp_low/SAW, exp(gain_frac) is still strictly larger.
        endpoint=(SAW+exp_low)/(SAW-exp_low)
        assert endpoint==Fraction(1051047402734946162883735364514439,
                                  681844442968178837116264635485561)
        assert Fraction(154147681861212,10**14)<endpoint<Fraction(154147681861213,10**14)
        print('monotone continuation of the same table: r through',endpoint)
    return low,high


def compare_spin_solver(u,steps,canonical,low,high):
    # Optional comparison after completing the independent physical calculation.
    from STAGE53_DISORDER_prism import rational_grid
    spin_low,spin_high=rational_grid(u,steps,UNIT)
    for (a,b,oa,ob),lower in spin_low.items():
        rep=canonical[(a+16*oa,b+16*ob)]
        assert lower==low[rep] and spin_high[a,b,oa,ob]==high[rep]
    assert len(spin_low)==136
    print('two algorithms, both endpoints of all',len(spin_low),'reference integral rows identical')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cross-check',action='store_true')
    args=parser.parse_args()
    canonical,reps,polys=even_statistics()
    for r,steps,credit,charges in CASES:
        low,high=validate(r,steps,credit,charges,canonical,reps,polys)
        if args.cross_check:
            compare_spin_solver((r-1)/(r+1),steps,canonical,low,high)
    print('PASS independent high-temperature, physical weakening, quadrature, colour and gain checks')
