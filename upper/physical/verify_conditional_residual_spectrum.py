"""Local conditional residuals and certified spectral tests on the cubic Ising model."""
import argparse
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.optimize import linprog

import verify_conditional_energy_star as star

D = 10**7


def field_mean(r, s):
    w = r**s
    return (w-1)/(w+1)


def boundary_classes():
    return [(a,b) for a in (1,3,5) for v in range(1,a+1,2) for b in (v,-v)]


def pair_data(r, a, b):
    g = second = e2 = e4 = F(0)
    strength = [F(0)]*4
    edge = [[F(0) for j in range(4)] for i in range(4)]
    for x,y in ((a,b),(b,a),(-a,-b),(-b,-a)):
        weights = {(s,t):r**((s*x+t*y+s*t-1)//2) for s,t in product((-1,1),repeat=2)}
        den = 4*sum(weights.values())
        for (s,t),w in weights.items():
            prob = w/den
            s0,s1 = x+t,y+s
            i,j = abs(s0)//2,abs(s1)//2
            m,n = field_mean(r,s0),field_mean(r,s1)
            g += prob*s*t
            second += prob*F(s0*s0,36)
            e2 += prob*(t-F(s0,6))*(s0==2)
            e4 += prob*(t-F(s0,6))*(s0==4)
            strength[i] += prob*(1-m*m)
            edge[i][j] += prob*(s-m)*(t-n)
    assert all(edge[i][j]==edge[j][i] for i in range(4) for j in range(4))
    return g,second,e2,e4,strength,edge


def walk_returns(nmax):
    from verify_pair_energy_test import return_terms
    out = [F(0)]*(nmax+1)
    for n,(p,q) in enumerate(return_terms(nmax//2)):
        out[2*n] = F(p,q)
    return out


def zero_matrix(n):
    return np.full((n,n),F(0),dtype=object)


def class_matrices(data, c0, chi, degree, ir, massive, atom):
    g,second,_,_,strength,edge = data
    c = c0-2*g
    zero = zero_matrix(5)
    zero[0,0] = chi if chi else F(0)
    corner = zero_matrix(5)
    corner[0,0] = c/12
    for i in range(4):
        corner[0,i+1] = corner[i+1,0] = strength[i]
        zero[0,i+1] = zero[i+1,0] = strength[i]
        for j in range(4):
            diag = strength[i] if i==j else F(0)
            corner[i+1,j+1] = diag-6*edge[i][j]
            zero[i+1,j+1] = diag+6*edge[i][j]
    matrices = [corner,zero if chi else zero[1:,1:]]
    # Use local spins and the once-averaged spin, together with translated
    # neighbour-conditioned residuals under powers of the walk operator.
    if degree>=0:
        ws = walk_returns(2*degree+4)
        modes = [(i,h) for i in range(4) for h in range(degree+1)]
        gram = zero_matrix(2+len(modes))
        gram[:2,:2] = [[1-atom,g-atom],[g-atom,second-atom]]
        for j,(s,h) in enumerate(modes):
            for i in range(2):
                gram[i,2+j] = gram[2+j,i] = strength[s]*ws[i+h]
            for k,(t,l) in enumerate(modes):
                gram[2+j,2+k] = (strength[s]*ws[h+l] if s==t else 0)+6*edge[s][t]*ws[h+l+1]
        matrices.append(gram)
        for poly, spin in [([1,-1],1-g),([1,1],1+g),([1,0,-1],1-second)]:
            weighted=zero_matrix(1+len(modes))
            weighted[0,0]=spin-sum(poly)*atom
            moment=[sum(x*ws[j+i] for i,x in enumerate(poly))
                    for j in range(2*degree+2)]
            for j,(s,h) in enumerate(modes):
                weighted[0,1+j]=weighted[1+j,0]=strength[s]*moment[h]
                for k,(t,l) in enumerate(modes):
                    weighted[1+j,1+k]=((strength[s]*moment[h+l] if s==t else 0)
                                        +6*edge[s][t]*moment[h+l+1])
            matrices.append(weighted)
    if ir:
        # The resolvent parameters give an upper matrix on the integrals of
        # squares (round W upward in the positive rank-one direction).
        d,low,high = massive
        j0,j1,j2 = high/6,(d*high-1)/6,(d*d*high-d)/6
        matrices.append(np.array([[c*j0-1+atom,c*j1-g+atom],
                                  [c*j1-g+atom,c*j2-second+atom]],dtype=object))
        # 1-gamma^2 has negative value at d>1, so the LOWER Green bound is used.
        loc = ((1-d*d)*low+d)/6
        matrices.append(np.array([[c*loc-1+second]],dtype=object))
    return matrices


def class_bounds(r, chi, degree, steps,mode='exclude'):
    u=(r-1)/(r+1)
    c0=1/u+u
    data=[pair_data(r,a,b) for a,b in boundary_classes()]
    # Under a finite susceptibility, there is no magnetization. The two-site
    # rotational test at the specified ratio supplies the indicated energy cap.
    upper_g = {F(25,16):F('0.349748620'),F(19,12):F('0.342979')}.get(r,F(1))
    mass = (c0-2*upper_g)/chi if chi else F(0)
    resolvent = star.walk_resolvent(mass)
    atoms=np.array([F(0)]*len(data),dtype=object)
    if mode=='magnetization':
        atoms=np.concatenate((atoms,atoms+1))
        data=data+data
    cases=[class_matrices(d,c0,chi,degree,mode!='energy',resolvent,atom)
           for d,atom in zip(data,atoms)]
    equality=np.array([[d[2] for d in data],[d[3] for d in data]],dtype=object)
    blocks=[np.stack(block) for block in zip(*cases)]
    approx=[np.asarray(m,dtype=float) for m in blocks]
    g = np.array([d[0] for d in data],dtype=object)
    rows=[]
    lower_inputs = {F(25,16):('0.3161770520','0.1827100670','0.1318704742'),
                    F(19,12):('0.3423083784','0.2171455219','0.1554593759')}
    if r in lower_inputs:
        lower,diag,axis=map(F,lower_inputs[r])
        rows += [g-lower, np.array([d[1] for d in data])-(1+4*diag+axis)/6]
    if mode=='exclude':
        rows.append(upper_g-g)
    if mode=='magnetization' and r==F(25,16):
        rows.append(F('0.477946739723')-g)
    g_trace = (6*chi-(chi-1)*c0)/(4*chi+2) if chi else (6-c0)/4
    rows.append(g-g_trace)
    target = {'energy':g,'magnetization':atoms,'exclude':np.full(len(g),F(0),dtype=object)}[mode]
    # Under the zero-atom hypothesis all these averaged matrices are PSD.
    # Minimize a common slack; a negative certified residual excludes it.
    names=set()
    for n in range(steps+1):
        ub=np.column_stack((-np.asarray(rows,dtype=float).reshape(-1,len(g)),
                            -np.ones(len(rows)))) if rows else None
        eq=np.column_stack((np.vstack([np.ones(len(g)),np.asarray(equality,dtype=float)]),np.zeros(3)))
        cost=np.r_[-np.asarray(target,dtype=float),1. if mode=='exclude' else 100.]
        result=linprog(cost,A_eq=eq,b_eq=[1.,0.,0.],A_ub=ub,
                       b_ub=np.zeros(len(rows)) if rows else None,bounds=(0,None),method='highs')
        if not result.success:
            raise RuntimeError(result.message)
        if n==steps or (mode=='exclude' and result.fun>0.0000005):
            break
        before=len(rows)
        for number, mats in enumerate(approx):
            matrix=np.einsum('k,kij->ij',result.x[:-1],mats)
            ev,vec=np.linalg.eigh(matrix)
            for j in np.where(ev < -1e-8)[0][:3]:
                q=np.rint(10000*vec[:,j]).astype(int)
                key=(number,tuple(q))
                if key in names:
                    continue
                names.add(key)
                q=np.asarray(list(map(int,q)),dtype=object)
                norm=sum(abs(x) for x in q)**2
                rows.append(np.array([F(sum(q*(m @ q)),norm) for m in blocks[number]],dtype=object))
        if len(rows)==before:
            break
    # For any probabilities satisfying the exact inequalities, the expectation
    # of every nonnegative combination of rows + any translation residual is >=0.
    coeff=[F(max(0,round(-float(x)*D)),D) for x in result.ineqlin.marginals]
    free=[F(round(float(x)*D),D) for x in result.eqlin.marginals[1:]]
    residual=sum((c*row for c,row in zip(coeff,rows)),target)
    for c,row in zip(free,equality):
        residual += c*row
    upper=max(residual)
    print('ratio',r,'chi hypothesis',chi,'tests',len(rows),
          {'energy':'unconditional energy upper','magnetization':'m^2 upper',
           'exclude':'maximum exact residual'}[mode],float(upper),
          'excluded',upper<0,flush=True)
    return upper


def check_chain():
    # Infinite chain with G(n)=u^|n|. The conditional mean is linear.
    for r in [F(5,4),F(25,16),F(2)]:
        u=(r-1)/(r+1)
        a=2*u/(1+u*u)
        b=(1-u*u)/(1+u*u)
        for x in [F(-1),F(-2,5),F(0),F(2,3),F(1)]:
            g=(1-u*u)/(1+u*u-2*u*x)
            h=b*(1-a*x)
            assert g*h==b*b
        for n in range(1,8):
            point_to_set = F(0)
            for left,right in product((-1,1),repeat=2):
                weight = (1+left*right*u**(2*n))/4
                mean = (left+right)*u**n/(1+left*right*u**(2*n))
                point_to_set += weight*mean*mean
            assert point_to_set == 2*u**(2*n)/(1+u**(2*n))


def check_cube(r):
    # For an eight-site OPEN cube the three neighbours are distinct. Directly
    # test the conditional identity and the vanishing for every nonedge.
    xyz=list(product((0,1),repeat=3))
    adj=[[j for j,y in enumerate(xyz) if sum(a!=b for a,b in zip(x,y))==1] for x in xyz]
    diag=F(0)
    cross=[F(0)]*8
    noise=[F(0)]*8
    for configuration in product((-1,1),repeat=8):
        equal=sum(configuration[i]==configuration[j] for i in range(8) for j in adj[i] if i<j)
        weight=r**equal
        fields=[sum(configuration[j] for j in a) for a in adj]
        means=[field_mean(r,h) for h in fields]
        eta=[s-m for s,m in zip(configuration,means)]
        diag+=weight*(1-means[0]**2)
        for k in range(8):
            cross[k]+=weight*eta[0]*configuration[k]
            noise[k]+=weight*eta[0]*eta[k]
    assert cross==[diag]+[F(0)]*7 and noise[0]==diag
    assert all(noise[k]==0 for k in range(1,8) if k not in adj[0])


def check_periodic_cube(r):
    # A 2^3 torus has two physical bonds to each of three distinct neighbours.
    # Its on-site conditional mean therefore sees twice their unsigned sum.
    xyz=list(product((0,1),repeat=3))
    adj=[[i^(1<<j) for j in range(3)] for i in range(8)]
    norm=F(0)
    signed_spin=[F(0)]*8
    signed_residual=[F(0)]*8
    signed_cross=[F(0)]*8
    on=near=F(0)
    for spins in product((-1,1),repeat=8):
        edges=2*sum(spins[i]==spins[j] for i in range(8) for j in adj[i] if i<j)
        weight=r**edges
        fields=[2*sum(spins[j] for j in nbrs) for nbrs in adj]
        means=[field_mean(r,s) for s in fields]
        eta=[v-m for v,m in zip(spins,means)]
        norm+=weight
        on+=weight*(1-means[0]**2)
        near+=weight*eta[0]*eta[adj[0][0]]
        for k,wave in enumerate(xyz):
            signs=[(-1)**sum(j*x for j,x in zip(wave,pos)) for pos in xyz]
            a=sum(v*s for v,s in zip(spins,signs))
            b=sum(v*s for v,s in zip(eta,signs))
            signed_spin[k]+=weight*a*a/8
            signed_residual[k]+=weight*b*b/8
            signed_cross[k]+=weight*a*b/8
    for k,wave in enumerate(xyz):
        assert signed_cross[k]==on
        assert signed_residual[k]==on+near*(3-2*sum(wave))
        assert signed_spin[k]*signed_residual[k]>=on*on
    assert norm>0


def square_weights(r):
    # Four equal-field neighbours per vertex outside the four-cycle. At an
    # extremal field they are all fixed to the sign of the corresponding bit.
    table=[]
    for fields in range(16):
        row=[]
        for word in range(16):
            inside=sum((word>>i & 1)==(word>>((i+1)%4) & 1) for i in range(4))
            outside=4*(4-bin(word ^ fields).count('1'))
            row.append(r**(inside+outside))
        table.append(row)
    return table


def least_agreement_event(low,high,pattern):
    norms=[sum(row) for row in high]
    result=F(1)
    for i,j in product(range(16),repeat=2):
        numerator=sum(x*(low[j][s ^ pattern]+low[j][s ^ pattern ^ 15])
                      for s,x in enumerate(low[i]))
        result=min(result,numerator/(norms[i]*norms[j]))
    return result


def free_square_check(r):
    spins=list(product((-1,1),repeat=4))
    weights=[r**sum(s[i]==s[(i+1)%4] for i in range(4)) for s in spins]
    norm=sum(weights)
    edge=[s[0]*s[1] for s in spins]
    average=sum(q*w for q,w in zip(edge,weights))/norm
    t=(r*r-1)/(r*r+1)
    for pair,pattern in [((0,3),1),((2,3),6)]:
        other=[s[pair[0]]*s[pair[1]] for s in spins]
        cov=sum(q*p*w for q,p,w in zip(edge,other,weights))/norm-average*average
        prob=sum(weights[i]*(weights[i^pattern]+weights[i^pattern^15])
                 for i in range(16))/(norm*norm)
        assert cov==2*prob*t*t


def log_interval(x):
    y=(x-1)/(x+1)
    assert 0<=y<1
    partial=sum(2*y**(2*i+1)/(2*i+1) for i in range(7))
    return partial,partial+2*y**15/(15*(1-y*y))


def check_pressure_curvature():
    start,stop=F(25,16),F(81,64)**2
    for r in (F(1),start,F(19,12),F(4)):
        free_square_check(r)
    grid=[start+(stop-start)*j/8 for j in range(9)]
    tables=[square_weights(r) for r in grid]
    lower=[]
    for i,r in enumerate(grid[:-1]):
        bend=least_agreement_event(tables[i],tables[i+1],1)
        opposite=least_agreement_event(tables[i],tables[i+1],3)
        tan=(r*r-1)/(r*r+1)
        lower.append(2*tan*tan*(8*bend+4*opposite))
    gain=F('0.15835')
    assert min(lower)>gain
    upper=F('0.477946739723')
    # Validate the original hyperbolic chord from two exact pressure endpoints.
    L5=F(293779920449747959463094921875,5981399566861883634775752704)
    U=F(22184861939,10**10)
    q=F(81,80)
    assert ((q+1/q)/2+upper*(q-1/q)/2)**15 > U**5/L5
    improved=upper-gain*(q-1)/(q+1)
    assert improved < F('0.476963200')
    small=F(10001,10000)
    C,H=(small+1/small)/2,(small-1/small)/2
    energy_lower=F('0.3170321192')
    lo=3*log_interval(C+energy_lower*H+gain*(C-1))[0]
    hi=3*log_interval(C+upper*H+gain*(C-1-H*(q-1)/(q+1)))[1]
    print('curvature improvement',float(min(lower)),'nearest energy upper',float(improved))
    print('pressure increment lower, upper', float(lo),float(hi))
    assert lo>F('0.000095120746') and hi<F('0.000143095767')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratio',type=F,default=F(25,16))
    parser.add_argument('--chi',type=int,nargs='*',default=[200,250,300,400])
    parser.add_argument('--degree',type=int,default=2)
    parser.add_argument('--steps',type=int,default=28)
    parser.add_argument('--energy',action='store_true',help='bound energy with local residuals in every phase')
    parser.add_argument('--curvature',action='store_true',help='verify the plaquette covariance and pressure interval')
    parser.add_argument('--magnetization',action='store_true',help='bound zero-mode mass with local residuals')
    args=parser.parse_args()
    if args.curvature:
        check_pressure_curvature()
        return
    check_chain()
    check_cube(args.ratio)
    check_periodic_cube(args.ratio)
    mode='magnetization' if args.magnetization else 'energy' if args.energy else 'exclude'
    for x in (args.chi if mode=='exclude' else [0]):
        class_bounds(args.ratio,x,args.degree,args.steps,mode)


if __name__=='__main__':
    main()
