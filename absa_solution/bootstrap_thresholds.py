"""Regularize selected meta-model cutoffs using train-only OOF bootstraps.

The same selected meta estimators remain fixed. Validation labels never choose
the cutoffs, and the blind test is not read. This is a development comparison
on the existing holdout, rather than a new independent validation experiment.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from stacked_absa import file_hash,model_identity,read_scores,feature_arrays,estimator,best_cut
from phobert_absa import split_indices,predictions,to_frame,main_score
import score_absa as scorer


def run(selection_dir,train_path,out_dir,replicates=1000):
    src,out=Path(selection_dir),Path(out_dir)
    cfg=json.loads((src/'config.json').read_text())
    assert cfg['architecture']=='supervised_stacking' and not cfg['production_refit']
    assert not cfg.get('refit_evaluation')
    assert file_hash(train_path)==cfg['train_sha256']
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS
    y=train[scorer.COLS].to_numpy(int)
    fi,tu,va=split_indices(train,cfg['seed'])
    arrays,sources=[],[]
    for source in cfg['sources']:
        folder=src/source['path']
        assert model_identity(folder)==source['identity']
        c=json.loads((folder/'config.json').read_text())
        assert not c.get('production_refit',False) and not c.get('refit_scope')
        arrays.append(read_scores(folder,'tune',train.id.iloc[tu].to_numpy(dtype=str),y[tu]))
        sources.append({**source,'path':os.path.relpath(folder.resolve(),out.resolve())})
    oof=np.zeros((len(tu),21))
    for j,label in enumerate(cfg['labels']):
        target=y[tu,j]
        folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=cfg['seed'])
        xt=feature_arrays(arrays,j,label['feature_mode'])
        for fit,held in folds.split(target,target):
            m=estimator(label['C'],cfg['seed']).fit(xt[fit],target[fit])
            oof[held,j]=m.predict_proba(xt[held])[:,1]
        f1,cut=best_cut(target,oof[:,j])
        assert abs(f1-label['oof_F1'])<1e-12 and abs(cut-label['threshold'])<1e-12
    grid=np.linspace(.1,.9,17)
    rng=np.random.default_rng(cfg['seed']+197)
    weights=rng.exponential(size=(replicates,len(tu)))
    adjusted=[];diagnostics=[]
    for j,label in enumerate(cfg['labels']):
        target=y[tu,j].astype(bool)
        predicted=oof[:,j,None]>=grid[None,:]
        tp=weights@(predicted & target[:,None])
        total=weights@predicted+weights@target[:,None]
        f1=2*tp/np.maximum(total,1e-12)
        # Break near ties toward 0.5, as in the original empirical F1 rule.
        tied=np.isclose(f1,f1.max(axis=1,keepdims=True),atol=1e-12,rtol=0)
        dist=np.where(tied,abs(grid-.5),np.inf)
        chosen=grid[dist.argmin(axis=1)]
        threshold=float(np.median(chosen))
        adjusted.append({**label,'threshold':threshold,'original_threshold':label['threshold']})
        diagnostics.append({'label':label['label'],'original_threshold':label['threshold'],
            'bootstrap_median_threshold':threshold,'p10':float(np.quantile(chosen,.1)),
            'p90':float(np.quantile(chosen,.9))})
    oldcuts=np.array([c['threshold'] for c in cfg['labels']])
    cuts=np.array([c['threshold'] for c in adjusted])
    scores=read_scores(src,'validation',train.id.iloc[va].to_numpy(dtype=str),y[va])
    oldreport=json.loads((src/'metrics.json').read_text())
    assert abs(main_score(y[va],predictions(scores,oldcuts))-oldreport['ABSA_F1'])<1e-12
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(scores,cuts)),n_boot=1000)
    out.mkdir(parents=True,exist_ok=True)
    cfg.update(sources=sources,labels=adjusted,
        threshold_selection=f'median cutoff from {replicates} seeded Bayesian bootstraps of original training OOF scores',
        thresholds_use_validation_labels=False)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    shutil.copyfile(src/'meta_models.joblib',out/'meta_models.joblib')
    shutil.copyfile(src/'metrics.json',out/'selection_metrics.json')
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'tune_oof_scores.npz',scores=oof,ids=train.id.iloc[tu].to_numpy(dtype=str))
    shutil.copyfile(src/'validation_scores.npz',out/'validation_scores.npz')
    pd.DataFrame(diagnostics).to_csv(out/'threshold_diagnostics.csv',index=False)
    info={'validation_F1':report['ABSA_F1'],'original_validation_F1':oldreport['ABSA_F1'],
        'changed_thresholds':int(np.sum(abs(cuts-oldcuts)>1e-12)),
        'bootstrap_replicates':replicates,'meta_coefficients_unchanged':True,
        'thresholds_chosen_from_original_training_oof_only':True,
        'validation_feedback_seen_in_prior_development':True,'public_score':None}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report)
    print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection-dir',required=True)
    p.add_argument('--train',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();run(a.selection_dir,a.train,a.out_dir)
