"""Compare a shallow boosted-tree meta model with the selected linear meta.

Only original training labels select the feature mode, tree size and cutoff.
The unchanged review-level validation split is evaluated once at the end.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,platform
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from stacked_absa import file_hash,model_identity,read_scores,feature_arrays,best_cut
from phobert_absa import split_indices,predictions,to_frame
import score_absa as scorer


def estimator(leaves,seed):
    return HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=120,
        learning_rate=.05,min_samples_leaf=15,l2_regularization=10.,max_bins=63,
        class_weight='balanced',early_stopping=False,random_state=seed)


def fit(selection_dir,train_path,out_dir):
    started=time.perf_counter()
    src,out=Path(selection_dir),Path(out_dir)
    cfg=json.loads((src/'config.json').read_text())
    assert cfg['architecture']=='supervised_stacking' and not cfg['production_refit']
    assert file_hash(train_path)==cfg['train_sha256']
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS
    y=train[scorer.COLS].to_numpy(int)
    fi,tu,va=split_indices(train,cfg['seed'])
    tune_sources,val_sources,sources=[],[],[]
    for source in cfg['sources']:
        folder=src/source['path']
        assert model_identity(folder)==source['identity']
        c=json.loads((folder/'config.json').read_text())
        assert not c.get('production_refit',False) and not c.get('refit_scope')
        tune_sources.append(read_scores(folder,'tune',train.id.iloc[tu].to_numpy(dtype=str),y[tu]))
        val_sources.append(read_scores(folder,'validation',train.id.iloc[va].to_numpy(dtype=str),y[va]))
        sources.append({**source,'path':os.path.relpath(folder.resolve(),out.resolve())})
    models,labels=[],[]
    scores=np.zeros((len(va),21));oof_scores=np.zeros((len(tu),21))
    for j,name in enumerate(scorer.COLS):
        target=y[tu,j]
        folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=cfg['seed']).split(target,target))
        best=None
        for mode in ['own_aspect','all_aspects']:
            xt=feature_arrays(tune_sources,j,mode)
            for leaves in [3,7]:
                oof=np.zeros(len(tu))
                for subfit,held in folds:
                    m=estimator(leaves,cfg['seed']).fit(xt[subfit],target[subfit])
                    oof[held]=m.predict_proba(xt[held])[:,1]
                f1,cut=best_cut(target,oof)
                if best is None or f1>best[0]:best=f1,cut,mode,leaves,oof
        f1,cut,mode,leaves,oof=best
        m=estimator(leaves,cfg['seed']).fit(feature_arrays(tune_sources,j,mode),target)
        scores[:,j]=m.predict_proba(feature_arrays(val_sources,j,mode))[:,1]
        oof_scores[:,j]=oof
        labels.append({'label':name,'feature_mode':mode,'max_leaf_nodes':leaves,
            'threshold':cut,'oof_F1':f1})
        models.append(m)
        print(name,'meta OOF F1',round(f1,4),mode,'leaves',leaves,flush=True)
    cuts=np.array([c['threshold'] for c in labels])
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(scores,cuts)),n_boot=1000)
    out.mkdir(parents=True,exist_ok=True)
    cfg.update(sources=sources,labels=labels,meta_estimator='hist_gradient_boosting',
        meta_selection='five-fold original training OOF predictions; four predeclared candidates per label',
        meta_tree_settings=estimator(3,cfg['seed']).get_params(),production_refit=False)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    joblib.dump(models,out/'meta_models.joblib',compress=3)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'tune_oof_scores.npz',scores=oof_scores,ids=train.id.iloc[tu].to_numpy(dtype=str))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    info={'total_seconds':time.perf_counter()-started,'python':platform.python_version(),
        'gpu':False,'train_sha256':cfg['train_sha256'],
        'meta_OOF_F1':float(np.mean([c['oof_F1'] for c in labels if not c['label'].endswith('_nhac')])),
        'validation_F1':report['ABSA_F1'],'validation_feedback_seen_in_prior_development':True,
        'public_score':None}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report)
    print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection-dir',required=True)
    p.add_argument('--train',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();fit(a.selection_dir,a.train,a.out_dir)
