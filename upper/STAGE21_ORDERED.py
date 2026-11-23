"""Jointly optimize physical rows and class-probability inequalities at r=780/493."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from math import ceil, floor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import STAGE20_ORDERED as base
import STAGE20_ORDERED_corner_star as star
import STAGE20_ORDERED_mixed_caps as fields
import STAGE20_ORDERED_pricing as priced
import STAGE20_ORDERED_star_direct as spin_check

stage = base.previous
HERE = Path(__file__).resolve().parent
POINT = Fraction(780, 493)
ARCHIVE = HERE / 'STAGE21_ORDERED_780_493.json'
DUAL_GRID = 10**12
DUAL_STEP = 0.001
MAX_ROUNDS = 32


def class_geometry():
    star.geometry()
    s = 1 / POINT
    v = s**3
    outward = (s+s**5)/(1+s**6)
    message = (s+s**4*outward)/(1+s**5*outward)
    w = message**3
    odds = (w*w-v*v)/(1-v*v*w*w)
    d = odds/(1+odds)
    assert spin_check.exact_probability(POINT,[4]*6,[5]*6) == d
    assert spin_check.exact_probability(POINT,[0]*6,[0]*6) == 1-d
    assert d > star.LOW
    owners = star.local.classes()
    sizes = Counter(owners)
    edges = sorted({(owners[t],owners[t^(1<<j)]) for t in range(256) for j in range(8)
                    if owners[t] != owners[t^(1<<j)]})
    assert len(edges)==40 and len(sizes)==14
    return d, sizes, edges


def class_offsets(rows, sizes, edges):
    shift = dict.fromkeys(sizes,Fraction(0))
    seen = set()
    h = star.LOW/(1-star.LOW)
    for i,j,k in rows:
        assert type(k) is int and k>0 and (i,j) in edges and (i,j) not in seen
        seen.add((i,j))
        weight = Fraction(k,DUAL_GRID)
        shift[i] -= weight*h/sizes[i]
        shift[j] += weight/sizes[j]
    return shift


def make_model():
    green = stage.W.fixed.phase.AngularGreen(Fraction(0))
    model = stage.W.Certificate(POINT,'order',green,False,bridges=True,expanded=False,corners=True)
    base.add_new_tests(model,green)
    pool, _ = base.pool_for(model)
    # Only mask identities and integer square vectors are transported.
    # Every coefficient depending on K is computed anew at POINT.
    other = json.loads((HERE/'STAGE20_ORDERED_price_791_500.json').read_text())
    priced.add_cuts(model,pool,other['cuts'])
    fields.rays.extend_rays(model,3)
    return model,pool,stage.corner_geometry(model)


def groups_and_start(model,pool):
    groups,specs,values,masses=stage.parameters(model,pool)
    initial=priced.source_rows()
    old,_=stage.source()
    keys={key:{tuple(row[:-1]) for row in rows} for key,rows in old.items()}
    for j,(family,idx,step) in enumerate(specs,1):
        if family=='ordinary' and idx not in keys[family]:
            step=250000 if idx[0]>=28 else 25000
        elif family=='walsh' and idx not in keys[family]:
            step=500
        groups[j][family][0][-1]=step
        specs[j-1]=family,idx,step
        if family=='ordinary':
            masses[j]=float(step*model.terms[idx[0]][3]/stage.W.DEN)
        values[j]=max(0.,float(initial[family].get(idx,0)/step))
    assert abs(masses@values-float(stage.MASS))<1.e-9
    return groups,specs,values,masses


def separate(envelope, factors):
    """Add near-maximal labelled fields and return each class maximum."""
    orbit = np.einsum('ij,j->i',envelope.features,factors)[envelope.owners]
    eps = envelope.errors @ factors
    peaks = {}
    words = envelope.words
    for c,weights in envelope.weights.items():
        raw = fields.compact.average_float(weights*orbit,envelope.ratios)/envelope.den[c]
        assert np.isfinite(raw).all()
        peaks[c] = float(raw.max()+eps)
        distinct = set()
        for f in np.argpartition(-raw,24)[:24]:
            key = c,int(f)
            rounded = round(float(raw[f]),12)
            if rounded in distinct or key in envelope.constraints:
                continue
            distinct.add(rounded)
            density=weights*envelope.kernel[words^f]
            marginal=np.bincount(envelope.owners,weights=density,minlength=len(envelope.features))/density.sum()
            coeff=np.einsum('i,ij->j',marginal,envelope.features)+envelope.errors
            assert abs(coeff@factors - (raw[f]+eps)) < 1.e-9
            envelope.constraints[key]=coeff
    return peaks


def discovery(model,pool,layout,sizes,edges):
    groups,specs,factors,mass = groups_and_start(model,pool)
    envelope=stage.prior.FieldEnvelope(model,groups,layout)
    classes=sorted(sizes)
    ids={c:i for i,c in enumerate(classes)}
    dual=np.zeros((len(classes),len(edges)))
    seed=dict(((i,j),w/DUAL_GRID) for i,j,w in star.DUAL)
    class_vars=np.array([seed.get(edge,0)/DUAL_STEP for edge in edges])
    h=float(star.LOW/(1-star.LOW))
    for j,(a,b) in enumerate(edges):
        dual[ids[a],j]=-DUAL_STEP*h/sizes[a]
        dual[ids[b],j]= DUAL_STEP/sizes[b]
    P=len(specs)
    bounds=[(0,max(10.,2*f)) for f in factors[1:]]+[(0,20.)]*len(edges)+[(None,None)]
    objective=np.r_[np.zeros(P+len(edges)),1.]
    best=(np.inf,None,None)
    for no in range(MAX_ROUNDS+1):
        peaks=separate(envelope,factors)
        unshifted=np.array([peaks[c] for c in classes])
        upper=float(max(unshifted+dual@class_vars))
        if upper<best[0]:
            best=upper,factors.copy(),class_vars.copy()
        print(f'round={no} fields={len(envelope.constraints)} joint_U~{upper:+.11g} '
              f'unshifted_U~{unshifted.max():+.10g}',flush=True)
        if no==MAX_ROUNDS or best[0]<-0.000025:
            break
        rows=np.array(list(envelope.constraints.values()))
        cid=[ids[c] for c,_ in envelope.constraints]
        A=np.column_stack((1.e5*rows[:,1:],1.e5*dual[cid],-np.ones(len(rows))))
        fit=linprog(objective,A_ub=A,b_ub=-1.e5*rows[:,0],
                    A_eq=np.r_[mass[1:],np.zeros(len(edges)+1)][None,:], b_eq=[float(stage.MASS)],
                    bounds=bounds,method='highs',options={'time_limit':25.})
        if not fit.success:
            print(f'LP termination={fit.message}; taking best separated iterate',flush=True)
            break
        if upper <= fit.fun/1.e5+3.e-9:
            break
        factors=np.r_[1.,fit.x[:P]]
        class_vars=fit.x[P:-1]
    record,denominator=stage.integerise(model,specs,best[1][1:])
    drows=[]
    for (i,j),f in zip(edges,best[2]):
        number=round(float(f)*DUAL_STEP*DUAL_GRID)
        if number:
            drows.append([i,j,number])
    return record,denominator,drows


def validate_rows(record,pool):
    assert set(record)==set(pool)
    for family,rows in record.items():
        allowed={tuple(r[:-1]) for r in pool[family]}
        keys=[tuple(r[:-1]) for r in rows]
        assert set(keys)<=allowed and len(keys)==len(set(keys))
        assert all(type(x) is int for r in rows for x in r)
        assert all(r[-1]!=0 if family=='translations' else r[-1]>0 for r in rows)


def propose_caps(model,v,scale,geom):
    classes,counts,ratios,trace,_,_=geom
    max_weight=max(model.weights)
    initial=np.array([z/max_weight for z in model.weights])[model.owners]
    theta=np.array([z/trace[0] for z in trace],dtype=float)
    signed=np.array([int(z)/scale for z in v])[model.owners]
    caps={}
    for c in classes:
        x=initial.copy()
        for j,count in enumerate(counts):
            x*=theta[3-count if c & (1<<j) else count]
        values=fields.compact.average_float(x*signed,ratios)/fields.compact.average_float(x,ratios)
        assert np.isfinite(values).all()
        f=int(values.argmax())
        caps[c]=Fraction(ceil(values[f]*10**9)+5,10**9),f
    return caps


def run(mode):
    assert POINT<Fraction(1791,1132)
    prob,sizes,edges=class_geometry()
    model,pool,geom=make_model()
    if mode=='discover':
        record,den,dual=discovery(model,pool,geom,sizes,edges)
    else:
        saved=json.loads(ARCHIVE.read_text())
        assert Fraction(saved['ratio'])==POINT
        record,den,dual=saved['rows'],saved['denominator'],saved['class_dual']
    assert type(den) is int and den>0
    validate_rows(record,pool)
    shifts=class_offsets(dual,sizes,edges)
    v,eps,q=stage.build_residual(model,record)
    eps/=den
    q/=den
    scale=den*stage.W.DEN*stage.W.T
    assert eps>0 and q==stage.MASS>0
    if mode=='discover':
        caps_words=propose_caps(model,v,scale,geom)
    else:
        caps_words={int(c):(Fraction(pair[0]),pair[1]) for c,pair in saved['caps_words'].items()}
    assert set(caps_words)==set(sizes)
    c=max(caps_words,key=lambda i:caps_words[i][0]+shifts[i])
    C,f=caps_words[c]
    assert 0<=f<len(model.owners)
    w=stage.corner_weights(model,c,geom[1],geom[3])
    w=stage.direct_field(w,geom[2],geom[4],geom[5],f)
    one=Fraction(int(w@v[model.owners]),int(w.sum())*scale)
    assert 0<=C-one<Fraction(1,10**7)
    U=C+shifts[c]+eps
    L=-U/q
    one_shifted=one+shifts[c]+eps
    no_dual=max(z for z,_ in caps_words.values())+eps
    print(f'r={POINT}, physical row counts={ {k:len(rows) for k,rows in record.items()} }, '
          f'class multipliers={len(dual)}, witness={(c,f)}, '
          f'floor(1e15 L)={floor(10**15*L)}, floor(1e15 one_shifted)={floor(10**15*one_shifted)}, '
          f'without class rows U~{float(no_dual):+.11g}',flush=True)
    result={'ratio':str(POINT),'rows':record,'denominator':den,'class_dual':dual,'class_denominator':DUAL_GRID,
            'probability_floor':str(star.LOW),'actual_probability_floor':str(prob),'caps_words':{
                str(k):[str(z),n] for k,(z,n) in caps_words.items()}, 'radius':str(eps),'Q':str(q),
            'signed_control':str(no_dual),'U':str(U),'L':str(L),'witness':[c,f],
            'witness_shifted_including_radius':str(one_shifted)}
    if L<=0:
        assert one_shifted>0
    if mode!='witness':
        fields.check_fields(model,v,scale,geom,{i:z for i,(z,_) in caps_words.items()},raw=(mode=='raw'))
    else:
        assert L<=0
    if mode=='discover':
        ARCHIVE.write_text(json.dumps(result,indent=2)+'\n')
    else:
        assert result==saved
    print(f'PASS {mode}, L={L}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('discover','directed','raw','witness'))
    assert __debug__
    run(p.parse_args().mode)
