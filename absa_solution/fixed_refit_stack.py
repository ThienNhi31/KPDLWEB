"""Evaluate base-model outer80 refits while retaining the selected meta exactly.

Never fit a meta model on in-sample tuning predictions of the refit encoders.
The original 938 validation reviews remain outside every training stage.
"""
import argparse,json,os,shutil
from pathlib import Path
import joblib,numpy as np,pandas as pd
from stacked_absa import model_identity,file_hash,read_scores,feature_arrays
from phobert_absa import split_indices,predictions,to_frame
import score_absa as scorer


def evaluate(selection_dir,train_path,replacements,out_dir):
    selection,out=Path(selection_dir),Path(out_dir)
    cfg=json.loads((selection/'config.json').read_text())
    assert cfg['architecture']=='supervised_stacking' and not cfg['production_refit']
    assert replacements and set(replacements)<=set(range(len(cfg['sources'])))
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS
    assert file_hash(train_path)==cfg['train_sha256']
    fi,tu,va=split_indices(train,cfg['seed'])
    ids=train.id.iloc[va].to_numpy(dtype=str);y=train[scorer.COLS].iloc[va].to_numpy(int)
    sources,arrays,fit_counts=[],[],[]
    for i,source in enumerate(cfg['sources']):
        folder=Path(replacements[i]) if i in replacements else selection/source['path']
        c=json.loads((folder/'config.json').read_text())
        assert c['seed']==cfg['seed'] and c['columns']==cfg['columns']
        assert c['model_name']==source['pretrained_model']
        assert c.get('architecture','joint_21')==source['architecture']
        identity=model_identity(folder)
        if i in replacements:
            assert c.get('refit_scope')=='outer80' and not c['production_refit']
            assert c['selection_identity']==source['identity'],'Refit initialized from a different selected source'
            assert c['fit_rows']==len(fi)+len(tu)
            info=json.loads((folder/'run_info.json').read_text())
            assert info['train_sha256']==cfg['train_sha256']
        else:assert identity==source['identity']
        fit_counts.append(c.get('fit_rows',cfg['base_fit_rows']))
        arrays.append(read_scores(folder,'validation',ids,y))
        sources.append({**source,'path':os.path.relpath(folder.resolve(),out.resolve()),
            'identity':identity,'selection_identity':source['identity'],'refit_on_outer80':i in replacements})
    models=joblib.load(selection/'meta_models.joblib')
    scores=np.column_stack([m.predict_proba(feature_arrays(arrays,j,c['feature_mode']))[:,1]
        for j,(m,c) in enumerate(zip(models,cfg['labels']))])
    cuts=np.array([c['threshold'] for c in cfg['labels']])
    report=scorer.report(to_frame(ids,y),to_frame(ids,predictions(scores,cuts)),n_boot=1000)
    out.mkdir(parents=True,exist_ok=True)
    cfg.update(sources=sources,production_refit=False,meta_refit=False,
        base_training_rows_per_source=fit_counts,validation_rows=len(va),
        refit_evaluation='fixed original meta and thresholds; base refits exclude original validation reviews')
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    shutil.copyfile(selection/'meta_models.joblib',out/'meta_models.joblib')
    shutil.copyfile(selection/'metrics.json',out/'selection_metrics.json')
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=ids)
    info={'train_sha256':cfg['train_sha256'],'validation_F1':report['ABSA_F1'],
        'validation_reviews_unseen_by_all_training_stages':len(va),'meta_refit':False,'thresholds_refit':False,
        'replaced_source_indices':sorted(replacements),'base_training_rows_per_source':fit_counts,'gpu':False}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report)
    print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection-dir',required=True)
    p.add_argument('--train',default='../train.csv');p.add_argument('--out-dir',required=True)
    p.add_argument('--replace',nargs='+',required=True,help='Source-index=refit-model-directory')
    a=p.parse_args();replacement={}
    for item in a.replace:
        index,path=item.split('=',1);replacement[int(index)]=path
    evaluate(a.selection_dir,a.train,replacement,a.out_dir)
