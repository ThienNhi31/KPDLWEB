"""An out-of-fit sparse feature source for supervised hybrid stacking."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits
from scoped_cpu import features,local_context
from cpu_ensemble import preprocess,NBLinear,choose_cuts
from phobert_absa import split_indices,predictions,to_frame
from stacked_absa import file_hash
import score_absa as scorer

# Selected by the earlier tuning search, before its validation was reported.
SPEC=[(2,.5),(2,.5),(.5,0),(2,.5),(.5,.5),(.5,.5),(.5,.5)]


def probabilities(bundle,texts):
    texts=pd.Series(list(texts)).map(preprocess)
    gx=bundle['global_vectorizer'].transform(texts)
    global_p=expit(3*bundle['global_model'].decision_function(gx))
    parts=[]
    for k,b in enumerate(bundle['aspects']):
        if not b['weight']:
            parts.append(global_p[:,k*3:k*3+3]);continue
        lx=b['local_vectorizer'].transform(texts.map(lambda s:local_context(s,b['aspect'])))
        local_p=expit(3*b['model'].decision_function(hstack([gx*.35,lx],format='csr')))
        parts.append(b['weight']*local_p+(1-b['weight'])*global_p[:,k*3:k*3+3])
    return np.hstack(parts)


def fit(data_dir,out_dir):
    started=time.perf_counter();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    path=Path(data_dir)/'train.csv';train=pd.read_csv(path)
    assert list(train.columns)==['id','text']+scorer.COLS and train.id.is_unique
    assert not train.isna().any().any() and train[scorer.COLS].isin([0,1]).all().all()
    fi,tu,va=split_indices(train,2026);y=train[scorer.COLS].to_numpy(int)
    texts=train.text.map(preprocess)
    gv=features();gx=gv.fit_transform(texts.iloc[fi])
    gm=OneVsRestClassifier(NBLinear(C=1),n_jobs=1).fit(gx,y[fi])
    selected=[]
    for k,(aspect,(c,w)) in enumerate(zip(scorer.ASPECTS,SPEC)):
        block={'aspect':aspect,'C':c,'weight':w}
        if w:
            lv=features();lx=lv.fit_transform(texts.iloc[fi].map(lambda s:local_context(s,aspect)))
            lm=OneVsRestClassifier(NBLinear(C=c),n_jobs=1).fit(hstack([gx*.35,lx],format='csr'),y[fi,k*3:k*3+3])
            block.update(local_vectorizer=lv,model=lm)
        selected.append(block)
        print('Fit-only sparse source:',aspect,flush=True)
    bundle={'global_vectorizer':gv,'global_model':gm,'aspects':selected}
    tune=probabilities(bundle,train.text.iloc[tu]);validation=probabilities(bundle,train.text.iloc[va])
    cuts=choose_cuts(y[tu],tune)
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(validation,cuts)),n_boot=1000)
    joblib.dump(bundle,out/'model.joblib',compress=3)
    for part,indices,scores in [('tune',tu,tune),('validation',va,validation)]:
        np.savez_compressed(out/(part+'_scores.npz'),scores=scores,ids=train.id.iloc[indices].to_numpy(dtype=str))
    cfg={'architecture':'classical_scoped','model_name':'NB-SVM global and aspect-scoped TF-IDF',
        'seed':2026,'columns':scorer.COLS,'production_refit':False,'fit_rows':len(fi),
        'thresholds':cuts.tolist(),'hyperparameters_selected_on':'original tuning partition',
        'spec':[{'aspect':a,'C':c,'local_weight':w} for a,(c,w) in zip(scorer.ASPECTS,SPEC)]}
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    info={'train_sha256':file_hash(path),'fit_rows':len(fi),'tune_rows':len(tu),'validation_rows':len(va),
        'validation_F1':report['ABSA_F1'],'gpu':False,'total_seconds':time.perf_counter()-started}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report)
    print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='..');p.add_argument('--out-dir',default='classical_source_results')
    a=p.parse_args()
    with threadpool_limits(limits=1):fit(a.data_dir,a.out_dir)
