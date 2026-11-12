"""Select one regularization strength for the whole train-only sparse stack.

This reduces label-by-label hyperparameter selection. The fixed candidate pools
are the original five encoders and all eight available encoders. Every choice
uses original training OOF predictions; validation does not select a parameter.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,platform
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stacked_absa import file_hash,model_identity,read_scores,feature_arrays,best_cut
from phobert_absa import split_indices,predictions,to_frame
import score_absa as scorer


def estimator(c,seed):
    return make_pipeline(StandardScaler(),LogisticRegression(C=c,penalty='l1',
        solver='liblinear',random_state=seed,max_iter=3000,intercept_scaling=100.))


def fit(selection_dir,train_path,out_dir):
    started=time.perf_counter()
    src,out=Path(selection_dir),Path(out_dir)
    cfg=json.loads((src/'config.json').read_text())
    assert cfg['architecture']=='supervised_stacking' and not cfg['production_refit']
    assert len(cfg['sources'])==8 and file_hash(train_path)==cfg['train_sha256']
    train=pd.read_csv(train_path);y=train[scorer.COLS].to_numpy(int)
    fi,tu,va=split_indices(train,cfg['seed'])
    arrays,val_arrays,folders=[],[],[]
    for source in cfg['sources']:
        folder=src/source['path'];folders.append(folder)
        assert model_identity(folder)==source['identity']
        base=json.loads((folder/'config.json').read_text())
        assert not base.get('production_refit',False) and not base.get('refit_scope')
        arrays.append(read_scores(folder,'tune',train.id.iloc[tu].to_numpy(dtype=str),y[tu]))
        val_arrays.append(read_scores(folder,'validation',train.id.iloc[va].to_numpy(dtype=str),y[va]))
    scored=[i for i,n in enumerate(scorer.COLS) if not n.endswith('_nhac')]
    best=None;comparisons=[]
    for count in [5,8]:
        for c in [.03,.1,.3,1.]:
            oof=np.zeros((len(tu),21));cuts=[];fs=[]
            for j in range(21):
                target=y[tu,j];xt=feature_arrays(arrays[:count],j,'own_aspect')
                folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=cfg['seed'])
                for fit_idx,held in folds.split(target,target):
                    m=estimator(c,cfg['seed']).fit(xt[fit_idx],target[fit_idx])
                    oof[held,j]=m.predict_proba(xt[held])[:,1]
                f1,cut=best_cut(target,oof[:,j]);fs.append(f1);cuts.append(cut)
            mean=float(np.mean(np.array(fs)[scored]))
            row={'source_count':count,'C':c,'meta_OOF_F1':mean}
            comparisons.append(row);print('CANDIDATE',json.dumps(row),flush=True)
            if best is None or mean>best[0]:best=mean,count,c,oof,np.array(cuts),fs
    mean,count,c,oof,cuts,fs=best
    models=[];scores=np.zeros((len(va),21))
    for j in range(21):
        m=estimator(c,cfg['seed']).fit(feature_arrays(arrays[:count],j,'own_aspect'),y[tu,j])
        scores[:,j]=m.predict_proba(feature_arrays(val_arrays[:count],j,'own_aspect'))[:,1]
        models.append(m)
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(scores,cuts)),n_boot=1000)
    out.mkdir(parents=True,exist_ok=True)
    sources=[{**s,'path':os.path.relpath(folder.resolve(),out.resolve())}
        for s,folder in zip(cfg['sources'][:count],folders[:count])]
    labels=[{'label':name,'feature_mode':'own_aspect','C':c,'threshold':float(cuts[j]),'oof_F1':fs[j]}
        for j,name in enumerate(scorer.COLS)]
    cfg.update(sources=sources,labels=labels,meta_estimator='l1_logistic_global_C',
        meta_selection='one global encoder pool and C selected by training OOF macro F1',
        meta_intercept_scaling=100.,production_refit=False)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    joblib.dump(models,out/'meta_models.joblib',compress=3)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    pd.DataFrame(comparisons).to_csv(out/'training_oof_candidates.csv',index=False)
    np.savez_compressed(out/'tune_oof_scores.npz',scores=oof,ids=train.id.iloc[tu].to_numpy(dtype=str))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    info={'total_seconds':time.perf_counter()-started,'python':platform.python_version(),
        'gpu':False,'meta_OOF_F1':mean,'validation_F1':report['ABSA_F1'],
        'train_sha256':cfg['train_sha256'],'selected_sources':count,'selected_global_C':c,
        'validation_feedback_seen_in_prior_development':True,'blind_file_read':False}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report);print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection-dir',required=True)
    p.add_argument('--train',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();fit(a.selection_dir,a.train,a.out_dir)
