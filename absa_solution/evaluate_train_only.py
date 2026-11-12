"""Evaluate a selected stack on validation reviews from train.csv only."""
import train_only
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gc,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from stacked_absa import file_hash,model_identity,read_scores,feature_arrays
from phobert_absa import split_indices,predictions,to_frame
import score_absa as scorer


def evaluate(model_dir,train_path,out_dir,fresh=False,threads=2,batch_size=8):
    started=time.perf_counter()
    root,out=Path(model_dir),Path(out_dir)
    cfg=json.loads((root/'config.json').read_text())
    assert cfg['architecture']=='supervised_stacking' and not cfg['production_refit']
    assert file_hash(train_path)==cfg['train_sha256']
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    _,_,va=split_indices(train,cfg['seed'])
    ids=train.id.iloc[va].to_numpy(dtype=str)
    y=train[scorer.COLS].iloc[va].to_numpy(int)
    arrays=[]
    for i,source in enumerate(cfg['sources']):
        folder=root/source['path']
        assert model_identity(folder)==source['identity']
        base_cfg=json.loads((folder/'config.json').read_text())
        assert not base_cfg.get('production_refit',False)
        if fresh:
            from neural_evaluation import open_model,probabilities_for
            base_cfg,model,tok=open_model(folder,threads)
            scores=probabilities_for(base_cfg,model,tok,train.text.iloc[va],batch_size)
            del model,tok;gc.collect()
        else:scores=read_scores(folder,'validation',ids,y)
        arrays.append(scores)
        print('Train validation source ready:',i+1,'/',len(cfg['sources']),flush=True)
    models=joblib.load(root/'meta_models.joblib')
    scores=np.column_stack([m.predict_proba(feature_arrays(arrays,j,c['feature_mode']))[:,1]
        for j,(m,c) in enumerate(zip(models,cfg['labels']))])
    cuts=np.array([c['threshold'] for c in cfg['labels']])
    pred=predictions(scores,cuts)
    report=scorer.report(to_frame(ids,y),to_frame(ids,pred),n_boot=1000)
    expected=json.loads((root/'metrics.json').read_text())
    if not fresh:assert abs(report['ABSA_F1']-expected['ABSA_F1'])<1e-12
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    to_frame(ids,pred).to_csv(out/'validation_prediction.csv',index=False)
    examples=[]
    scored=[j for j,n in enumerate(scorer.COLS) if not n.endswith('_nhac')]
    wrong=np.sum(y[:,scored]!=pred[:,scored],axis=1)
    for i in np.argsort(-wrong,kind='stable'):
        if wrong[i]==0 or len(examples)==5:break
        examples.append({'id':str(ids[i]),'text':str(train.text.iloc[va[i]]),
            'errors':{scorer.COLS[j]:{'truth':int(y[i,j]),'prediction':int(pred[i,j])}
                for j in scored if y[i,j]!=pred[i,j]}})
    (out/'training_validation_errors.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    info={'ABSA_F1':report['ABSA_F1'],'CI95':report['ABSA_F1_CI95'],
        'validation_rows':len(va),'input_file':'train.csv','blind_file_read':False,
        'mode':'fresh local encoder inference on train validation' if fresh else 'saved out-of-fit train encoder outputs',
        'seconds':time.perf_counter()-started,'validation_feedback_seen_in_prior_development':True,
        'public_score_claimed':False}
    (out/'evaluation_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report);print('FINAL',json.dumps(info),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True)
    p.add_argument('--train',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--fresh-encoder-scores',action='store_true')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();evaluate(a.model_dir,a.train,a.out_dir,a.fresh_encoder_scores,a.threads,a.batch_size)
