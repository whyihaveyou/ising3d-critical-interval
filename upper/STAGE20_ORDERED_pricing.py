"""Price new reflection squares, deleted inequalities and translations on the ordered-side dual measure."""

import argparse
from fractions import Fraction
import json
from math import floor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import STAGE20_ORDERED as base
import STAGE20_ORDERED_rays as rays

stage = base.previous


def source_rows():
    saved = json.loads(Path(__file__).with_name('STAGE20_ORDERED_1579_998.json').read_text())
    return {key: {tuple(row[:-1]): Fraction(row[-1], saved['denominator']) for row in rows}
            for key, rows in saved['rows'].items()}


def solve(model, pool, warm, layout, rounds):
    groups, specs, factors, masses = stage.parameters(model, pool)
    old, _ = stage.source()
    old_keys = {k: {tuple(row[:-1]) for row in rows} for k, rows in old.items()}
    for j, (family, idx, step) in enumerate(specs, 1):
        if family == 'ordinary' and idx not in old_keys[family]:
            step = 250000 if idx[0] >= 28 else 25000
            groups[j][family][0][-1] = step
            specs[j-1] = family, idx, step
            masses[j] = float(step*model.terms[idx[0]][3]/stage.W.DEN)
        if family == 'walsh' and idx not in old_keys[family]:
            step = 500
            groups[j][family][0][-1] = step
            specs[j-1] = family, idx, step
        factors[j] = max(0., float(warm[family].get(idx, 0)/step))
    assert abs(masses@factors-float(stage.MASS)) < 1.e-9
    bounds = [(0., max(10., 2.*f)) for f in factors[1:]] + [(None,None)]
    envelope = stage.prior.FieldEnvelope(model, groups, layout)
    n = len(specs)
    best = (np.inf, None)
    for i in range(rounds+1):
        cap = envelope.separate(factors)
        print(f'pricing pass round={i} physical_fields={len(envelope.constraints)} U~{cap:+.10g}', flush=True)
        if cap < best[0]:
            best = cap, factors.copy()
        rows = np.asarray(list(envelope.constraints.values()))*1e5
        fit = linprog([0.]*n+[1.], A_ub=np.column_stack((rows[:,1:], -np.ones(len(rows)))),
                      b_ub=-rows[:,0], A_eq=np.r_[masses[1:],0.][None,:], b_eq=[float(stage.MASS)],
                      bounds=bounds, method='highs')
        assert fit.success, fit.message
        predicted = fit.fun/1e5
        if cap <= predicted+5.e-9 or best[0]<-5.e-5 or i==rounds:
            break
        factors = np.r_[1., fit.x[:-1]]
    density = np.zeros(len(model.words))
    weights = -np.asarray(fit.ineqlin.marginals)
    assert abs(weights.sum()-1.)<1.e-8 and np.min(weights)>-1.e-9
    for probability, (source, field) in zip(weights, envelope.constraints):
        if probability <= 1.e-12:
            continue
        real = envelope.weights[source]*envelope.kernel[envelope.words^field]
        density += probability*np.bincount(model.owners, weights=real, minlength=len(density))/real.sum()
    assert abs(density.sum()-1.)<1.e-8
    return (*stage.integerise(model, specs, best[1][1:]), density)


def plan_cuts(model, density, pool):
    fixed = stage.W.fixed
    ordinary, values, base_ids, shared = fixed.phase.reflection_features(model.ball)
    u=(model.ratio-1)/(model.ratio+1)
    physical = np.einsum('ij,j->i',values/stage.W.T, density)
    matrix = (1-2*shared)*physical[ordinary] + float(u+1/u)*shared*physical[base_ids]
    # A dual relaxation supported on finitely many exterior fields need not be
    # invariant under the shifted reflection plane. For a real test vector
    # the antisymmetric part drops out of the quadratic form identically.
    symmetric = (matrix+matrix.T)/2
    assert np.isfinite(symmetric).all()
    probe=np.arange(1,len(matrix)+1,dtype=float)/len(matrix)
    assert abs(float(probe@matrix@probe-probe@symmetric@probe)) < 1.e-9
    print(f'dual field matrix skew max={np.max(abs(matrix-matrix.T)):.8g}',flush=True)
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    forms=[]
    for index in range(8):
        if eigenvalues[index]>=-1.e-8:
            break
        vec = np.rint(8192*vectors[:,index]).astype(np.int64)
        assert vec.shape == (112,) and vec@vec>0
        forms.append(vec.tolist())
    existing = {i for i,_ in pool['translations']}
    costs = abs(np.einsum('ij,j->i',model.ball.rows/stage.W.T, density))
    translations = [int(i) for i in np.argsort(-costs) if i not in existing and costs[i]>1/stage.W.fixed.SCALE][:32]
    used = {tuple(map(int,model.deleted.rep[[a^t,a,t]])) for a,t,_ in pool['walsh']}
    walsh = model.deleted.separate(density, used, 24)
    print(f'priced Gram eigenvalues={eigenvalues[:8].tolist()}, added squares={len(forms)}, '
          f'translations={len(translations)}, deleted={len(walsh)}', flush=True)
    return {'reflections':forms, 'translations':translations, 'walsh':[list(w) for w in walsh]}


def add_cuts(model, pool, cuts):
    fixed = stage.W.fixed
    u=(model.ratio-1)/(model.ratio+1)
    features = fixed.phase.reflection_features(model.ball)
    for vec in cuts['reflections']:
        assert len(vec)==112 and all(type(z) is int for z in vec) and any(vec)
        pool['ordinary'].append([len(model.terms),0])
        model.terms.append((0,*fixed.exact_reflection(vec,u+1/u,features)))
    for i in cuts['translations']:
        assert type(i) is int and 0<=i<len(model.ball.rows)
        pool['translations'].append([i,0])
    for a,t in cuts['walsh']:
        assert all(type(i) is int and 0<=i<len(model.owners) and bin(i).count('1')%2==0 for i in (a,t))
        assert t and model.deleted.rep[a]>=0 and model.deleted.rep[t]>=0
        pool['walsh'].append([a,t,0])
    for rows in pool.values():
        assert len({tuple(row[:-1]) for row in rows}) == len(rows)


def run(args):
    assert 1<args.ratio<Fraction(1579,998)
    g = stage.W.fixed.phase.AngularGreen(Fraction(0))
    model = stage.W.Certificate(args.ratio,'order', g, False, bridges=True, expanded=False, corners=True)
    base.add_new_tests(model, g)
    pool, _ = base.pool_for(model)
    rays.extend_rays(model, 3)
    layout=stage.corner_geometry(model)
    path=Path(__file__).with_name(f'STAGE20_ORDERED_price_{args.ratio.numerator}_{args.ratio.denominator}.json')
    if args.mode=='discover':
        rec, den, density=solve(model,pool,source_rows(),layout,args.rounds)
        cuts=plan_cuts(model,density,pool)
        warm={key:{tuple(r[:-1]):Fraction(r[-1],den) for r in rows} for key,rows in rec.items()}
        add_cuts(model,pool,cuts)
        rec,den,_=solve(model,pool,warm,layout,args.rounds+8)
    else:
        saved=json.loads(path.read_text()); cuts=saved['cuts']
        add_cuts(model,pool,cuts)
        rec,den=saved['rows'],saved['denominator']
    for key,rows in rec.items():
        allowed={tuple(r[:-1]) for r in pool[key]}
        ids=[tuple(r[:-1]) for r in rows]
        assert set(ids)<=allowed and len(ids)==len(set(ids))
        assert all(type(v) is int for r in rows for v in r)
        assert all(r[-1]!=0 if key=='translations' else r[-1]>0 for r in rows)
    assert type(den) is int and den>0
    v,eps,q=stage.build_residual(model,rec)
    eps/=den; q/=den; scale=stage.W.DEN*stage.W.T*den
    assert q==stage.MASS>0
    if args.mode=='discover':
        cap,(c,f)=stage.prior.bridge.propose(model,v,layout,scale)
    else:
        cap=Fraction(saved['cap']); c,f=saved['witness']
    weight=stage.corner_weights(model,c,layout[1],layout[3])
    weight=stage.direct_field(weight,layout[2],layout[4],layout[5],f)
    assert c in layout[0] and 0<=f<len(model.owners) and int(weight.sum())>0
    one=Fraction(int(weight@v[model.owners]), int(weight.sum())*scale)
    assert 0<=cap-one<Fraction(1,10**7)
    L=-(cap+eps)/q
    print(f'r={args.ratio} C={cap} field={(c,f)} floor(1e15L)={floor(10**15*L)} '
          f'floor(1e15*(one+eps))={floor(10**15*(one+eps))}',flush=True)
    result={'ratio':str(args.ratio),'cuts':cuts,'rows':rec,'denominator':den,'radius':str(eps),'Q':str(q),
            'cap':str(cap),'L':str(L),'witness':[c,f],'one_plus_radius':str(one+eps)}
    if L>0:
        assert args.mode in ('raw','directed','discover')
        check=stage.W.base.corner_upper if args.mode=='raw' else stage.CornerCheck(stage.W.base).all_corners
        assert check(args.ratio,v,model.weights,model.owners,scale,model.ratios,Fraction(max(map(int,v)),scale),cap)==cap
        assert len(layout[0])*len(model.owners)==3670016
        print(f'PASS {args.mode} all 3670016 fields L={L}',flush=True)
    else:
        assert args.mode in ('witness','discover') and one+eps>0
        print('PASS exact one-field obstruction to this candidate', flush=True)
    if args.mode=='discover':
        path.write_text(json.dumps(result,indent=2)+'\n')
    else:
        assert result==saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ratio',type=Fraction)
    p.add_argument('mode',choices=('discover','raw','directed','witness'))
    p.add_argument('--rounds',type=int,default=25)
    assert __debug__
    run(p.parse_args())
