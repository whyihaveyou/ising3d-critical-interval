"""Optimize physical Ising bounds using mixed spin moments on the deleted lattice."""
import argparse
from collections import deque
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, product
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import verify_ordered_exterior_fields as base

fixed, fields, geometry = base.fixed, base.fields, base.geometry
T, DEN = fixed.TOTAL_SCALE, fixed.DENOMINATOR
FILE = Path(__file__).with_name("deleted_walsh_order_certificates.json")
WALSH_SCALE = 32


def describe_order_bound(lower):
    """Interpret the signed lower bound L=-U/Q in 0 <= U+Q*m_plus^2, Q>0."""
    if lower > 0:
        return "m_plus^2 >= L=-U/Q, L>0: positive lower bound excludes zero"
    relation = "L=0" if lower == 0 else "L<0"
    return f"m_plus^2 >= L=-U/Q, {relation}: zero compatible with this bound"


def walsh(vector):
    out = np.array(vector, copy=True)
    width = 1
    while width < len(out):
        pairs = out.reshape(-1, 2*width)
        left, right = pairs[:,:width].copy(), pairs[:,width:].copy()
        pairs[:,:width], pairs[:,width:] = left+right, left-right
        width *= 2
    return out


def bridge_shell(ratio, words, sizes, original):
    """Add all 36 radius-four sites of cubic type (3,1,0) and (2,2,0)."""
    a,b = ratio.numerator,ratio.denominator
    sites = geometry.BOUNDARY
    dirs = geometry.DIRECTIONS
    def point(u,v,cu,cv):
        return tuple(cu*x+cv*y for x,y in zip(u,v))
    axis_bits = [sites.index(point(u,u,2,0)) for u in dirs]
    free_sites = set(sites).union(*(fields.neighbours(x)-set(geometry.INTERIOR) for x in sites))
    angles = []
    zero = (0,0,0)
    for i,j in combinations(range(6),2):
        u,v = dirs[i],dirs[j]
        if geometry.add(u,v) == zero:
            continue
        angles.append((i,j,sites.index(geometry.add(u,v))))
        free_sites.update((point(u,v,3,1),point(u,v,2,2),point(u,v,1,3)))
    assert len(angles)==12 and len(free_sites)==92
    assert sum(len(fields.neighbours(p)&free_sites) for p in free_sites)==300
    corners = [tuple(sites.index(v) for v in sorted(fields.neighbours(x).intersection(sites)))
               for x in product((-1,1),repeat=3)]
    assert all(len(c)==3 for c in corners)
    corner_trace = [a**(3-j)*b**j+b**(3-j)*a**j for j in range(4)]
    all_axes = np.arange(64,dtype=np.int32)
    spins = [(all_axes>>i)&1 for i in range(6)]
    lookup = [a,b]
    def contract(bridge, selected):
        c,d = bridge
        subdivision = (c*c+d*d,2*c*d)
        table = np.empty((2,2,2,2,2),dtype=object)
        for x,y,h,i,j in product(range(2),repeat=5):
            total = sum(subdivision[i^k]*subdivision[k^l]*subdivision[l^j]*
                        lookup[k^x]*lookup[k^h]*lookup[l^y]*lookup[l^h]
                        for k,l in product(range(2),repeat=2))
            table[x,y,h,i,j] = total
            brute = 0
            for p,k,q,l,s in product(range(2),repeat=5):
                differences = (i^p)+(p^k)+(k^q)+(q^l)+(l^s)+(s^j)
                brute += c**(6-differences)*d**differences*lookup[k^x]*lookup[k^h]*lookup[l^y]*lookup[l^h]
            assert brute == total
        results = []
        bonds = np.array(lookup,dtype=object)
        for n in selected:
            w = int(words[n])
            neg = [(w>>k)&1 for k in axis_bits]
            weight = np.ones(64,dtype=object)
            for j,bit in enumerate(neg):
                weight *= bonds[spins[j]^bit]
            for i,j,mid in angles:
                weight *= table[neg[i],neg[j],(w>>mid)&1,spins[i],spins[j]]
            corner_weight = 1
            for coords in corners:
                corner_weight *= corner_trace[sum((w>>k)&1 for k in coords)]
            results.append(int(sizes[n])*int(sum(weight))*corner_weight)
        return np.asarray(results,dtype=object)
    sample = [0,7,len(words)//2,len(words)-1]
    assert np.array_equal(contract((1,1),sample),original[sample]*(1<<36))
    return contract((a,b),range(len(words)))


class DeletedMoments:
    def __init__(self, ratio, ball, z, owner, bridges=False):
        self.ball, self.z, self.owner = ball, z, owner
        self.sizes = np.bincount(owner, minlength=len(z))
        weights = fixed.local.shell_measure(ratio, ball.words)
        original = weights
        if bridges:
            weights = bridge_shell(ratio,ball.words,self.sizes,original)
        assert all(w % int(n) == 0 for w,n in zip(weights,self.sizes))
        per_word = np.array([w//int(n) for w,n in zip(weights,self.sizes)], dtype=object)
        self.free = walsh(per_word[owner])
        self.norm = int(self.free[0])
        self.lowest_z = min(z)
        powers = np.fromiter((bin(i).count("1") for i in range(len(owner))), dtype=np.int8)
        self.even = np.flatnonzero(powers%2 == 0)
        assert self.norm == sum(weights) and self.free[self.even].min() >= 0
        assert np.all(self.free[powers%2 == 1] == 0)
        if bridges:
            old = np.array([w//int(n) for w,n in zip(original,self.sizes)],dtype=object)
            before = walsh(old[owner])
            assert np.all(self.free[self.even]*before[0] >= before[self.even]*self.norm)
        self.free_float = np.array([x/self.norm for x in self.free], dtype=float)
        self.rep = np.full(len(owner), -1, dtype=np.int32)
        reps = []
        for entry in self.even:
            a = int(entry)
            if self.rep[a] >= 0:
                continue
            images = [n>>7 for n in geometry.orbit(a<<7)]
            assert images[0] == a
            self.rep[images] = a
            reps.append(a)
        self.reps = np.asarray(reps, dtype=np.int32)
        self.toggles = self.reps[(powers[self.reps] == 2) | (powers[self.reps] == 4)]
        self.path_check(ratio)
        for a in (0, 3, 15, 255, len(owner)-1):
            chars = self.ball.averaged(a<<7)
            assert int(np.dot(chars.astype(object),weights)) == T*self.free[a]

    def path_check(self, ratio):
        sites = geometry.BOUNDARY
        shell = set(sites).union(*(fields.neighbours(p) - set(geometry.INTERIOR) for p in sites))
        graph = {x:fields.neighbours(x).intersection(shell) for x in shell}
        assert len(graph) == 56 and sum(map(len,graph.values())) == 156
        t = (ratio-1)/(ratio+1)
        for i,x in enumerate(sites):
            distance = {x:0}
            q = deque([x])
            while q:
                u = q.popleft()
                for y in graph[u]:
                    if y not in distance:
                        distance[y] = distance[u]+1
                        q.append(y)
            for j in range(i):
                pair = (1<<i)|(1<<j)
                assert F(self.free[pair],self.norm) >= t**distance[sites[j]]

    @lru_cache(None)
    def row(self, a, t):
        # F_{A xor T} >= <sigma_T>_free F_A. Multiplying by min(Z_I)>0
        # retains the sign; the word-orbit multiplicity is in the orbit average.
        assert 0 <= a < len(self.owner) and 0 < t < len(self.owner)
        assert self.rep[a] >= 0 and self.rep[t] >= 0
        left = self.ball.averaged(int(self.rep[a^t])<<7).astype(object)
        right = self.ball.averaged(int(self.rep[a])<<7).astype(object)
        numerator = (self.norm*left-int(self.free[t])*right)*self.lowest_z*WALSH_SCALE
        denominator = self.norm*self.z
        return ((2*numerator+denominator)//(2*denominator)).astype(np.int64)

    def separate(self, orbit_mass, used, limit):
        density = orbit_mass/self.sizes*np.array([self.lowest_z/z for z in self.z])
        moments = walsh(density[self.owner])
        moments /= moments[0]
        assert np.isfinite(moments).all()
        candidates = []
        direct = self.free_float[self.reps] - moments[self.reps]
        for i in np.argsort(-direct)[:3*limit]:
            t = int(self.reps[i])
            if t:
                candidates.append((float(direct[i]), 0, t))
        for entry in self.toggles:
            t = int(entry)
            loss = self.free_float[t]*moments[self.even] - moments[self.even^t]
            indices = np.argpartition(loss, -96)[-96:]
            for i in indices:
                if loss[i] > 1.e-8:
                    candidates.append((float(loss[i]), int(self.even[i]), t))
        chosen, keys, per_target = [], set(used), {}
        for delta,a,t in sorted(candidates, reverse=True):
            if delta < 1.e-8:
                break
            target,source,toggle = map(int,self.rep[[a^t,a,t]])
            key = target,source,toggle
            if key in keys or per_target.get(target,0) >= 2:
                continue
            keys.add(key)
            per_target[target] = per_target.get(target,0)+1
            chosen.append((a,t))
            if len(chosen) == limit:
                break
        return chosen


class Certificate:
    def __init__(self, ratio, mode, green, discovery, bridges=False, expanded=False, corners=False):
        self.ratio, self.mode = ratio, mode
        self.corners = corners
        self.ball, self.translations, self.terms = base.rebuild(ratio,green,expanded)
        if expanded:
            self.translations = fixed.EXTENDED_PROOFS[-1][2]
            if ratio >= F(87,55):
                # Free 27/27/26 section; exact replay and outward endpoints are
                # recorded in ORDERED_EXTERIOR_FIELDS_20261121.md.
                lower = {"100":3413482793,"110":2162957223,"111":1661899605,
                         "210":1311839020,"211":1152512442,"220":962346105}
                for label,num in lower.items():
                    shift = tuple(map(int,label))
                    x = next(p for p in geometry.SITES if geometry.add(p,shift) in geometry.INDEX)
                    row = self.ball.averaged(geometry.mask([x,geometry.add(x,shift)]))
                    self.terms.append((0,row-T*num//10**10,fixed.local.ROUND,F(0)))
        self.words = self.ball.words.tolist()
        self.owners = fields.orbit_owners(self.words)
        inner, ratios = base.trace_and_field_ratios(ratio,self.words,1)
        self.deleted = DeletedMoments(ratio,self.ball,inner,self.owners,bridges)
        self.weights,self.ratios = base.knight_fields(ratio,self.words,inner,ratios)
        self.base_error, self.base_mass = F(0),F(0)
        self.baseline = np.zeros(len(self.words),dtype=object)
        if mode != "order":
            raise ValueError("this supplement implements the order certificate")
        if discovery:
            self.kernel = base.discovery_kernel(self.words,self.owners,self.weights,self.ratios)

    def exact(self, record):
        eq = record["translations"]
        assert len({i for i,_ in eq}) == len(eq)
        assert all(i in {j for j,_ in self.translations} for i,_ in eq)
        selected = record["ordinary"]
        assert len({i for i,_ in selected}) == len(selected)
        assert all(0<=i<len(self.terms) and w>0 for i,w in selected)
        used = [(w,*self.terms[i][1:]) for i,w in selected]
        extras = record["walsh"]
        assert all(w>0 for _,_,w in extras)
        used.extend((w,self.deleted.row(a,t),fixed.local.GRID,F(0)) for a,t,w in extras)
        values = self.baseline.copy()
        radius = self.base_error + F(sum(abs(w) for _,w in eq),DEN*fixed.SCALE)
        mass = self.base_mass
        for i,w in eq:
            values -= w*self.ball.rows[i].astype(object)
        for w,row,error,slope in used:
            values += w*row.astype(object)
            radius += F(w,DEN)*error
            mass += F(w,DEN)*slope
        assert mass > 0
        upper, word = fields.upper_over_fields(values,self.weights,self.words,self.owners,
                                               DEN*T,self.ratios)
        if self.corners:
            verified = None
            if "value" in record:
                previous = F(record["value"])
                verified = (previous if self.mode=="energy" else -previous*mass) - radius
            upper = base.corner_upper(self.ratio,values,self.weights,self.owners,DEN*T,
                                      self.ratios,upper,verified)
            word = None
        result = upper+radius if self.mode == "energy" else -(upper+radius)/mass
        location = f"field={word}" if word is not None else "14 corner classes"
        print(f"{self.mode} r={self.ratio}: value={float(result):.14f}, "
              f"Q={mass}, {location}, nonzero Walsh rows={len(extras)}",flush=True)
        if self.mode == "order":
            print(describe_order_bound(result), flush=True)
        return result,mass,word

    def search(self, rounds, cut_count):
        idx = [i for i,_ in self.translations]
        equality = self.ball.rows[idx].astype(float)/T
        local = np.asarray([row for _,row,_,_ in self.terms],dtype=float)/T
        with np.errstate(all="ignore"):
            eq = self.kernel @ equality.T
            physical = self.kernel @ local.T
            offset = self.kernel @ (np.asarray(self.baseline,dtype=float)/(DEN*T))
        offset += float(self.base_error)
        physical += [float(error) for _,_,error,_ in self.terms]
        cols = np.column_stack((-eq+1/fixed.SCALE,eq+1/fixed.SCALE,physical))
        assert np.isfinite(cols).all() and np.isfinite(offset).all()
        n = len(idx)
        active = np.linspace(0,len(cols)-1,300,dtype=int)
        ordinary = len(self.terms)
        mass = [0.]*(2*n) + [float(q) for _,_,_,q in self.terms]
        extras, used = [], set()
        best = None
        for step in range(rounds):
            block = np.column_stack((cols[active],-np.ones(len(active))))
            cost = np.r_[np.zeros(len(mass)),1.]
            order = self.mode == "order"
            sol = linprog(cost,A_ub=block,b_ub=-offset[active],
                          A_eq=[mass+[0.]] if order else None,b_eq=[1.] if order else None,
                          bounds=[(0,10.)]*len(mass)+[(None,None)],method="highs-ipm",
                          options={"time_limit":120.})
            if not sol.success:
                if best is None:
                    raise RuntimeError(sol.message)
                print(sol.message,flush=True)
                break
            x = sol.x[:-1]
            with np.errstate(all="ignore"):
                all_values = cols @ x + offset
                w = np.maximum(0.,-sol.ineqlin.marginals)
                orbit_mass = w @ self.kernel[active]
            assert np.isfinite(all_values).all() and np.isfinite(orbit_mass).all()
            assert abs(w.sum()-1) < 1.e-5
            attained = float(all_values.max())
            if best is None or attained < best[0]:
                best = (attained,x.copy(),list(extras))
            near = float(attained-sol.fun)
            print(f"step {step}: fields={len(active)} cuts={len(extras)} lp={sol.fun:.9f} "
                  f"full={attained:.9f}",flush=True)
            if order and attained < -0.000035 and any(x[2*n+ordinary:] > 1.e-6):
                break
            top = np.argsort(-all_values)[:120]
            new_fields = top[all_values[top] > sol.fun+1.e-8]
            if near < .001 or step>=3:
                new = self.deleted.separate(orbit_mass,used,cut_count)
            else:
                new = []
            if new:
                rows = np.asarray([self.deleted.row(a,t) for a,t in new],dtype=float)/T
                with np.errstate(all="ignore"):
                    columns = self.kernel @ rows.T
                assert np.isfinite(columns).all()
                cols = np.column_stack((cols,columns+float(fixed.local.GRID)))
                mass += [0.]*len(new)
                for a,t in new:
                    used.add(tuple(map(int,self.deleted.rep[[a^t,a,t]])))
                extras.extend(new)
            active_next = np.union1d(active,new_fields)
            if not new and len(active_next) == len(active):
                break
            active = active_next
        assert best is not None
        _,x,extras = best
        equality = [(i,round(DEN*w)) for i,w in zip(idx,x[:n]-x[n:2*n]) if round(DEN*w)]
        ordinary_weights = [(i,round(DEN*w)) for i,w in enumerate(x[2*n:2*n+ordinary])
                            if round(DEN*w)>0]
        walsh_weights = [(a,t,round(DEN*w)) for (a,t),w in zip(extras,x[2*n+ordinary:])
                         if round(DEN*w)>0]
        return dict(translations=equality,ordinary=ordinary_weights,walsh=walsh_weights)
