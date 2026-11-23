"""Check the 14 classwise physical upper bounds and apply the corner-source law."""

import argparse
from fractions import Fraction
from itertools import product
import json
from math import ceil, floor
from pathlib import Path

import numpy as np

import STAGE20_ORDERED as base
import STAGE20_ORDERED_rays as rays
import STAGE20_ORDERED_corner_law as law
import threshold_corner_compact_20261121 as compact

stage=base.previous


def make_rows(model, green):
    source=json.loads(Path(__file__).with_name('STAGE20_ORDERED_1579_998.json').read_text())
    record=source['rows']
    denominator=source['denominator']
    assert type(denominator) is int and denominator>0
    extra=base.add_new_tests(model,green)
    pool,_=base.pool_for(model)
    assert sorted((i,extra[i]) for i,_ in record['ordinary'] if i in extra)==[(54,7),(69,27),(72,41)]
    for name,rows in record.items():
        allowed={tuple(e[:-1]) for e in pool[name]}
        used=[tuple(e[:-1]) for e in rows]
        assert set(used)<=allowed and len(used)==len(set(used))
        assert all(type(z) is int for e in rows for z in e)
        assert all(e[-1]!=0 if name=='translations' else e[-1]>0 for e in rows)
    v,eps,q=stage.build_residual(model,record)
    return v,eps/denominator,q/denominator,stage.W.T*stage.W.DEN*denominator


def propose(model, v, scale, geom):
    classes, counts, ratios, trace, _, _=geom
    peak=max(model.weights)
    initial=np.array([w/peak for w in model.weights])[model.owners]
    theta=np.array([n/trace[0] for n in trace],dtype=float)
    signed=np.array([int(z)/scale for z in v])[model.owners]
    result={}
    for c in classes:
        x=initial.copy()
        for j,count in enumerate(counts):
            x*=theta[3-count if c & (1<<j) else count]
        quot=compact.average_float(x*signed,ratios)/compact.average_float(x,ratios)
        assert np.isfinite(quot).all()
        result[c]=Fraction(ceil(float(quot.max())*10**9)+5,10**9)
    return result


def check_fields(model,v,scale,geom,caps,raw):
    classes,counts,ratios,trace,_,_=geom
    assert set(caps)==set(classes)==set(law.classes())
    compact.check_products()
    tab=[Fraction(int(n),int(trace[0])) for n in trace]
    low,high=(np.array(x,dtype=np.int64) for x in zip(*map(compact.coefficients,tab)))
    for i,c in enumerate(classes,1):
        ccap=caps[c]
        dif=ccap.numerator*scale-ccap.denominator*v
        if raw:
            weights=stage.corner_weights(model,c,counts,trace)
            assert stage.W.fields.transform(weights*dif[model.owners],ratios).min()>=0
        else:
            residual=[int(w)*int(z) for w,z in zip(model.weights,dif)]
            vec=compact.normalize_lower(residual,model.owners)
            for j,count in enumerate(counts):
                idx=3-count if c & (1<<j) else count
                vec=compact.multiply_lower(vec,low[idx],high[idx])
            assert compact.average_lower(vec,ratios).min()>=0
        if i in (1,7,14):
            print(f'classwise integer cap checked {i}/14',flush=True)
    assert len(classes)*len(model.owners)==3670016


def run(args):
    assert 1<args.ratio<=Fraction(1579,998)
    g=stage.W.fixed.phase.AngularGreen(Fraction(0))
    model=stage.W.Certificate(args.ratio,'order',g,False,bridges=True,expanded=False,corners=True)
    v,eps,q,scale=make_rows(model,g)
    assert q==stage.MASS>0
    rays.extend_rays(model,3)
    geom=stage.corner_geometry(model)
    path=Path(__file__).with_name(f'STAGE20_ORDERED_mix_{args.ratio.numerator}_{args.ratio.denominator}.json')
    if args.mode=='discover':
        caps=propose(model,v,scale,geom)
    else:
        saved=json.loads(path.read_text())
        caps={int(c):Fraction(b) for c,b in saved['caps'].items()}
    d=law.verify_local(args.ratio)
    # Replace the exact probability floor by a smaller explicit five-digit rational.
    floor_d=Fraction(floor(10**5*d),10**5)
    mixed=law.sweep(caps,floor_d)
    old=max(caps.values())
    assert mixed < old
    L=-(eps+mixed)/q
    print(f'r={args.ratio}, d_lower={floor_d}, class improvement~{float(old-mixed):.12g}, '
          f'floor(1e15*L)={floor(10**15*L)}, cap extrema={min(caps.values())},{old}',flush=True)
    check_fields(model,v,scale,geom,caps,args.mode=='raw')
    result={'ratio':str(args.ratio),'Q':str(q),'radius':str(eps),'d':str(d),'d_lower':str(floor_d),
            'caps':{str(c):str(b) for c,b in caps.items()},'mixed':str(mixed),'L':str(L),
            'previous_class_max':str(old),'floor_1e15_L':floor(10**15*L)}
    if args.mode=='discover':
        path.write_text(json.dumps(result,indent=2)+'\n')
    else:
        assert saved==result
    print(f'PASS {args.mode}, all 3670016 fields, L={L}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ratio',type=Fraction)
    parser.add_argument('mode',choices=('discover','directed','raw'))
    assert __debug__
    run(parser.parse_args())
