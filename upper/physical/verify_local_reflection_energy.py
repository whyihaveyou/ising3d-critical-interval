"""Rational certificates for the cubic Ising energy from local reflection positivity."""
import argparse
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from math import prod

import numpy as np
from scipy.optimize import linprog

import verify_conditional_energy_star as old

T = old.TOTAL_SCALE
S = old.SCALE
ROUND = F(1, 2*S)
GRID = F(1, 2*T)


def orbit_key(mask):
    return old.orbit(mask)[0]


def reflection_bases(max_degree):
    planes = [
        (lambda x: x[0] <= 0, lambda x: (1-x[0], x[1], x[2])),
        (lambda x: x[0] < 0, lambda x: (-x[0], x[1], x[2])),
        (lambda x: x[0] < x[1], lambda x: (x[1], x[0], x[2])),
    ]
    for plane,(eligible, reflect) in enumerate(planes):
        half = [p for p in old.SITES if eligible(p) and reflect(p) in old.INDEX]
        if plane:
            fixed = [p for p in old.INTERIOR if reflect(p)==p]
            half += fixed
        for parity in (0, 1):
            patterns = [a for size in range(parity, max_degree+1, 2)
                        for a in combinations(half, size)]
            left = [old.mask(a) for a in patterns]
            right = [old.mask(reflect(p) for p in a) for a in patterns]
            yield half, [reflect(p) for p in half], left, right


def reflection_blocks(max_degree):
    return [[[orbit_key(a ^ b) for b in right] for a in left]
            for _, _, left, right in reflection_bases(max_degree)]


def shared_bonds():
    # Two translates or rotations of the same nearest-neighbour energy have
    # nonnegative covariance; one- and two-edge expectations are even.
    edges = [old.mask([v, w]) for v,w in combinations(old.SITES, 2)
             if sum(abs(a-b) for a,b in zip(v,w)) == 1]
    return sorted({orbit_key(e ^ f) for e,f in combinations(edges, 2)})


def exterior_weights(ratio, words):
    # Delete the seven interior spins.  The remaining boundary weight is the
    # original boundary marginal divided by the interior partition function.
    a, b = ratio.numerator, ratio.denominator
    w = [a**(6-i)*b**i+b**(6-i)*a**i for i in range(7)]
    fields = [sum(1 << j for j,x in enumerate(old.BOUNDARY)
                  if sum(abs(i-k) for i,k in zip(x,y))==1) for y in old.DIRECTIONS]
    den = []
    for word in words:
        minus = [bin(int(word)&field).count("1") for field in fields]
        den.append(sum(prod(w[j+c] for j in minus) for c in (0,1)))
    least = min(den)
    return np.array([(2*T*least+d)//(2*d) for d in den], dtype=np.int64)


def exterior_patterns(max_degree):
    return sorted({orbit_key(old.mask(group))
                   for n in range(2,max_degree+1,2)
                   for group in combinations(old.BOUNDARY,n)})


def shell_measure(ratio, words):
    # Keep the exterior bonds between distances two and three. After the ring
    # is fixed, all spins at distance three can be summed independently.
    shell = {old.add(v,e) for v in old.BOUNDARY for e in old.DIRECTIONS
             if sum(map(abs,old.add(v,e)))==3}
    masks = [sum(1<<j for j,v in enumerate(old.BOUNDARY)
                 if sum(abs(x-y) for x,y in zip(v,w))==1) for w in shell]
    a,b = ratio.numerator, ratio.denominator
    fields = [(mask,[a**(d-j)*b**j+a**j*b**(d-j) for j in range(d+1)])
              for mask in masks for d in [bin(mask).count('1')]]
    full = (1<<len(old.BOUNDARY))-1
    weights, size = [], 0
    for wo in words:
        word = int(wo)
        images = {k>>7 for k in old.orbit(word<<7)}
        multiplicity = len(images | {full ^ k for k in images})
        size += multiplicity
        weights.append(multiplicity*prod(w[bin(word & mask).count('1')] for mask,w in fields))
    assert size == 1<<len(old.BOUNDARY)
    return np.array(weights,dtype=object)


def pair_class_table(ratio, words):
    import verify_conditional_residual_spectrum as residual
    classes = residual.boundary_classes()
    lookup = {}
    for i,(a,b) in enumerate(classes):
        for key in ((a,b),(b,a),(-a,-b),(-b,-a)):
            lookup[key] = i
    fields = [sum(1<<j for j,v in enumerate(old.BOUNDARY)
                  if sum(abs(x-y) for x,y in zip(v,arm)) == 1) for arm in old.DIRECTIONS]
    a,b=ratio.numerator,ratio.denominator
    weights=[(a**(6-i)*b**i,b**(6-i)*a**i) for i in range(7)]
    out=np.empty((len(classes),len(words)),dtype=np.int64)
    for col,word in enumerate(words):
        counts=[bin(int(word)&mask).count('1') for mask in fields]
        totals=[0]*len(classes)
        den=0
        for center in (0,1):
            branches=[weights[n+center] for n in counts]
            den+=prod(l+r for l,r in branches)
            for i,(pos,neg) in enumerate(branches):
                p=[1]
                for j,(l,r) in enumerate(branches):
                    if i!=j:
                        p=[l*x+r*y for x,y in zip(p+[0],[0]+p)]
                for j,value in enumerate(p):
                    totals[lookup[(5-2*j,5-2*counts[i])]]+=(pos+neg)*value
        den*=6
        assert sum(totals)==den
        out[:,col]=[(2*T*v+den)//(2*den) for v in totals]
    return out, [residual.pair_data(ratio,a,b) for a,b in classes]


def round_row(values, denominator):
    return ((2*values+denominator)//(2*denominator)).astype(np.int64)


class LocalTests:
    def prepare_reflection(self, degree):
        blocks = reflection_blocks(degree)
        self.edges = shared_bonds()
        self.exterior = exterior_patterns(4)
        patterns = sorted(set(self.edges+self.exterior).union(
            key for block in blocks for row in block for key in row))
        self.lookup = {key:i for i,key in enumerate(patterns)}
        self.patterns = patterns
        self.values = np.asarray([self.ball.averaged(key) for key in patterns])
        self.blocks = [np.asarray([[self.lookup[k] for k in row] for row in block])
                       for block in blocks]

    def __init__(self, ratio, degree, residuals=False):
        self.ratio = ratio
        self.ball = old.BallMoments(ratio,4)
        self.prepare_reflection(degree)
        self.inverse = exterior_weights(ratio,self.ball.words)
        weights = shell_measure(ratio,self.ball.words)
        norm = sum(weights)
        ring = self.values[[self.lookup[k] for k in self.exterior]]
        low = np.array([sum(row*weights)//norm for row in ring],dtype=np.int64)
        assert np.all((0 <= low) & (low <= T))
        self.outside = np.asarray([round_row((row-lower)*self.inverse,T)
                                  for row,lower in zip(ring,low)])
        extra = sorted(set(old.translation_rows(8))-set(old.translation_rows(4)))
        self.extra_rows = [self.ball.averaged(a)-self.ball.averaged(b) for a,b in extra]
        self.extra = np.asarray(self.extra_rows)
        self.moments, self.returns = self.ball.walk_moments()
        self.u = (ratio-1)/(ratio+1)
        self.c0 = 1/self.u+self.u
        self.cuts = {}
        self.resolvent = old.walk_resolvent(F(0))
        self.pair_matrices=[]
        if residuals:
            from verify_conditional_residual_spectrum import class_matrices
            self.pair_table, data = pair_class_table(ratio,self.ball.words)
            self.pair_matrices=[np.stack(block) for block in zip(*[
                class_matrices(d,self.c0,0,2,False,self.resolvent,F(0)) for d in data])]
            self.pair_approx=[np.asarray(block,dtype=float) for block in self.pair_matrices]
            check=np.einsum("i,ij->j",[float(d[0]) for d in data],self.pair_table)
            assert np.max(np.abs(check-self.ball.g1))/T < float(ROUND+13*GRID)

    def register(self, key, row, error, mass=F(0)):
        if key not in self.cuts:
            self.cuts[key] = (row,error,mass)

    def add_basic_bounds(self):
        # The trace of the massless infrared bound gives 4*g >= 6-c0.
        d = self.c0.denominator
        self.register(('trace',),round_row(4*d*self.ball.g1.astype(object)+(self.c0.numerator-6*d)*T,d),
                      4*ROUND+GRID)
        lower_inputs = {
            F(25,16): {'100':3161770520,'110':1827100670,'200':1318704742},
            F(19,12): {'100':3423083784,'110':2171455219,'200':1554593759,
                       '111':1668826676,'210':1313141950,'211':1156234195,
                       '220':962241808,'300':818948573,'310':712059035,'400':466386683},
        }
        for label,lower in lower_inputs.get(self.ratio,{}).items():
            delta=tuple(map(int,label))
            x=next(v for v in old.SITES if old.add(v,delta) in old.INDEX)
            values=self.ball.averaged(old.mask([x,old.add(x,delta)]))
            self.register(('free',label),values-(T*lower//10**10),ROUND)

    def gram_cut(self, block_number, q):
        mat = self.blocks[block_number]
        weights = Counter()
        for i in range(len(q)):
            for j in range(len(q)):
                weights[int(mat[i,j])] += int(q[i])*int(q[j])
        den = sum(abs(int(x)) for x in q)**2
        if not den:
            return
        # Signed integer accumulation with an explicit int64 overflow guard.
        assert 2*T*den+den < 2**63
        raw = np.zeros(len(self.ball.words),dtype=np.int64)
        for idx,coefficient in weights.items():
            raw += coefficient*self.values[idx]
        self.register(('rp',block_number,tuple(q)),round_row(raw,den),ROUND+GRID)

    def separate(self, probs, use_mass, mass, local=True):
        scale = probs/T
        support = np.flatnonzero(probs)
        ev = np.einsum('ij,j->i',self.values[:,support],scale[support])
        if local:
            if self.pair_matrices:
                cluster=np.einsum("ij,j->i",self.pair_table[:,support],scale[support])
                for k,matrices in enumerate(self.pair_approx):
                    vals,vec=np.linalg.eigh(np.einsum("i,ijk->jk",cluster,matrices))
                    for j in np.where(vals < -0.000005)[0][:2]:
                        q=np.rint(1000*vec[:,j]).astype(int)
                        key=("residual",k,tuple(q))
                        if key in self.cuts:
                            continue
                        q=np.asarray(list(map(int,q)),dtype=object)
                        v=[sum(q*(m @ q)) for m in self.pair_matrices[k]]
                        scale_q=max(map(abs,v))
                        if not scale_q:
                            continue
                        num=np.array([round(10**6*z/scale_q) for z in v],dtype=np.int64)
                        row=round_row(num @ self.pair_table,10**6)
                        self.register(key,row,F(1,2*10**6)+13*GRID)
            for i,block in enumerate(self.blocks):
                mat = ev[block]
                # Bond reflection translates the centre of the ball. A trial
                # marginal need not yet give identical transposed entries.
                vals, vectors = np.linalg.eigh((mat+mat.T)/2)
                for idx in np.where(vals < -0.00005)[0][:3]:
                    self.gram_cut(i,np.rint(250*vectors[:,idx]).astype(int))
            pairs = [(ev[self.lookup[k]],k) for k in self.edges]
            g = np.einsum('i,i->',self.ball.g1[support],scale[support])
            t = min(1000,max(0,round(g*1000)))
            den = 1000**2
            for value,k in sorted(pairs)[:12]:
                if value >= 2*(t/1000)*g-(t/1000)**2-1e-5:
                    break
                raw = den*self.values[self.lookup[k]]-2*t*1000*self.ball.g1+t*t*T
                self.register(('gks',k,t),round_row(raw,den),(1+F(2*t,1000))*ROUND+GRID)
            outside_values = np.einsum('ij,j->i',self.outside[:,support],scale[support])
            for i in np.argsort(outside_values)[:16]:
                if outside_values[i] >= -0.00001:
                    break
                self.register(('exterior',self.exterior[i]),self.outside[i],3*GRID)
            extra = np.einsum('ij,j->i',self.extra[:,support],scale[support])
            for i in np.argsort(-np.abs(extra))[:12]:
                if abs(extra[i]) < 0.00001:
                    break
                sign = -1 if extra[i]>0 else 1
                self.register(('translate',int(i),sign),sign*self.extra[i],2*ROUND)
        if use_mass:
            m = np.array([np.einsum('i,i->',r[support],scale[support]) for r in self.moments])
            d,_,high = self.resolvent
            vals = np.array([float(d**j*high-sum(d**(j-1-i)*self.returns[i] for i in range(j)))
                             *(float(self.c0)-2*m[1])/6-m[j]+mass for j in range(5)])
            for localize, n in ((False,3),(True,2)):
                mat = np.array([[vals[i+j]-(vals[i+j+2] if localize else 0)
                                 for i in range(n)] for j in range(n)])
                eigen, vectors = np.linalg.eigh(mat)
                for i in np.where(eigen < -1e-6)[0]:
                    q=np.rint(2000*vectors[:,i]).astype(np.int64)
                    poly=np.convolve(q,q)
                    if localize:
                        poly=np.convolve(poly,(1,0,-1))
                    key=tuple(map(int,poly))
                    if ('ir',key) not in self.cuts:
                        self.register(('ir',key),*old.spectral_cut(key,self.moments,self.returns,
                                                                 self.c0,self.resolvent))

    def optimize(self, target, use_mass, objective, steps, local=True):
        # The auxiliary mass can rise to satisfy the infrared cuts. For an
        # energy supporting line its penalty is fixed; all certificates are
        # recovered independently of the objective used to discover them.
        rows = self.ball.rows/T
        for iteration in range(steps+1):
            cuts = list(self.cuts.values())
            parts = [rows,-rows]
            if cuts:
                parts.append(-np.asarray([r for r,_,_ in cuts])/T)
            matrix = np.vstack(parts)
            zeros = np.zeros(2*len(rows))
            weights = np.r_[zeros,[-float(m) for _,_,m in cuts]]
            c = -target/T
            constraints = np.column_stack((matrix,weights))
            cost = np.r_[c,objective]
            normal = np.r_[np.ones(len(c)),0.]
            ans = linprog(cost,A_eq=[normal],b_eq=[1.],A_ub=constraints,
                          b_ub=np.full(len(constraints),0.000003),bounds=(0,None),
                          method='highs-ds')
            if not ans.success:
                raise RuntimeError(ans.message)
            if iteration==steps:
                break
            count=len(cuts)
            self.separate(ans.x[:-1],use_mass,ans.x[-1],local=local)
            if len(self.cuts)==count:
                break
        upper, coeff = self.ball.certify(target,1,cuts,ans)
        dual = ans.ineqlin.marginals[2*len(rows):]
        self.cuts = {key: value for (key,value), y in zip(self.cuts.items(),dual)
                     if y < -0.0000001}
        return upper,coeff,ans,iteration


def cross_plane_target(hypotheses):
    import verify_improved_phase_bound as angular
    ratio=F(25,16)
    u=(ratio-1)/(ratio+1)
    c0=1/u+u
    ball=old.BallMoments(ratio,4)
    features=angular.reflection_features(ball)
    angular_rows=angular.angular_rows(ball)
    local=[angular.reflection_cut(q,c0,features) for kind,q in angular.CROSS_PLANE
           if kind=='extension']
    tests=[(kind,q) for kind,q in angular.CROSS_PLANE if kind!='extension']
    tests+=angular.ENERGY+angular.MASSIVE+angular.ORDER
    reduced={tuple(angular.polynomial(kind,q)) for kind,q in tests}
    bounds={'100':3170321192,'110':1895460819,'111':1400810325,
            '210':1070877973,'211':913839564,'220':742699851,'200':1318704742}
    for label,limit in bounds.items():
        z=tuple(map(int,label))
        a=next(v for v in old.SITES if old.add(v,z) in old.INDEX)
        row=ball.averaged(old.mask([a,old.add(a,z)]))-T*limit//10**10
        local.append((row,ROUND,F(0)))
    for limit in hypotheses:
        mass=F(0)
        if limit:
            # Enclose the minimal possible stiffness from G(0)=1 and the
            # monotone integrated upper kernel C/(lambda+C/X).
            lo,hi=c0-2*F('0.345826740'),c0-2*F(bounds['100'],10**10)
            for _ in range(10):
                mid=(lo+hi)/2
                d,low,high=old.walk_resolvent(mid/limit)
                if mid*high<6:
                    lo=mid
                else:
                    hi=mid
            mass=lo/limit
        green=angular.AngularGreen(mass)
        cuts=local+[angular.spectral_cut(p,angular_rows,c0,green) for p in sorted(reduced)]
        result=ball.dual(ball.g1,1,cuts,relax=True)
        a,b=ball.certify(ball.g1,1,cuts,result)
        print('cross-plane target, finite chi hypothesis',limit,
              'g1 intercept',a,float(a),'mass slope',b,float(b),
              'excludes hypothesis',bool(limit and a<F(bounds['100'],10**10)),flush=True)


def main(tests_class=LocalTests):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ratio',nargs='*',type=F,default=[F(25,16),F(19,12)])
    parser.add_argument('--degree',type=int,default=2)
    parser.add_argument('--steps',type=int,default=9)
    parser.add_argument('--only',choices=['all','upper','mass'],default='upper')
    parser.add_argument('--residuals',action='store_true')
    parser.add_argument('--cross',nargs='+',type=int,help='target-point cross-plane tests, 0 for massless')
    args=parser.parse_args()
    if args.cross:
        cross_plane_target(args.cross)
        return
    for r in args.ratio:
        tests=tests_class(r,args.degree,args.residuals)
        tests.add_basic_bounds()
        g=tests.ball.g1
        tasks=[('lower',-g,False,1),('upper',g,True,4),('mass',np.zeros_like(g),True,1)]
        for name,target,infrared,penalty in tasks:
            if args.only not in ('all',name):
                continue
            a,b,ans,n=tests.optimize(target,infrared,penalty,args.steps)
            print(r,name,'intercept',a,float(a),'mass coeff',b,float(b),
                  'orbits',len(tests.ball.words),'patterns',len(tests.patterns),'cuts',
                  len(tests.cuts),'iterations',n,flush=True)
            if name=='upper' and r==F(19,12):
                lower=F('0.3434408514')  # 27,27,26-site tilted free sections
                if b>0 and lower>a:
                    print('high-precision free-region comparison gives m^2 >',
                          (lower-a)/b,float((lower-a)/b),flush=True)
            if name=='mass' and a<0 and b>0:
                print('m^2 lower',-a/b,float(-a/b),flush=True)


if __name__ == '__main__':
    main()
