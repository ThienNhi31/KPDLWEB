"""Fixed two-epoch raw-encoder refit on fit+tune; validation stays excluded."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,math,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import get_cosine_schedule_with_warmup
from neural_evaluation import open_model,probabilities_for
from phobert_absa import (split_indices,Reviews,loader,optimizer_for,train_epoch,
    seed_everything,cpu_thread_limit,predictions,to_frame)
from text_normalization import prepare_texts
from stacked_absa import file_hash,model_identity
import score_absa as scorer


def run(a):
    started=time.perf_counter()
    src,out=Path(a.init_dir),Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    cfg,model,tok=open_model(src,a.threads)
    assert not cfg.get('production_refit',False) and not cfg.get('refit_scope')
    assert cfg.get('architecture','joint_21')=='joint_21' and cfg['columns']==scorer.COLS
    identity=model_identity(src)
    train_path=Path(a.data_dir)/'train.csv'
    train_hash=file_hash(train_path)
    original=json.loads((src/'run_info.json').read_text())
    assert original.get('cache_key',{}).get('train_sha256',original.get('train_sha256'))==train_hash
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y=train[scorer.COLS].to_numpy(int);assert np.isin(y,[0,1]).all()
    fi,tu,va=split_indices(train,cfg['seed']);fit=np.concatenate([fi,tu])
    seed_everything(cfg['seed']);torch.set_num_threads(cpu_thread_limit(a.threads))
    texts=prepare_texts(cfg,train.text.iloc[fit])
    ds=Reviews(texts,tok,y[fit],cfg['max_length'],cfg['stride'],cfg.get('segment',True))
    batches=loader(ds,tok,a.batch_size,True,cfg['seed'])
    opt=optimizer_for(model,a.lr)
    steps=math.ceil(len(batches)/a.accumulation)*a.epochs
    schedule=get_cosine_schedule_with_warmup(opt,int(.08*steps),steps)
    pos=np.maximum(1,y[fit].sum(0));neg=np.maximum(1,len(fit)-y[fit].sum(0))
    pw=torch.as_tensor(np.clip(np.sqrt(neg/pos),1,3),dtype=torch.float32)
    history=[]
    for ep in range(1,a.epochs+1):
        ts=time.perf_counter()
        loss=train_epoch(model,batches,opt,schedule,torch.device('cpu'),pw,a.accumulation,100)
        row={'epoch':ep,'loss':loss,'seconds':time.perf_counter()-ts}
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print('EPOCH',json.dumps(row),flush=True)
    # Cutoffs are fixed before refit. No validation statistic selects an epoch.
    scores=probabilities_for(cfg,model,tok,train.text.iloc[va],a.batch_size)
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(scores,np.asarray(cfg['thresholds']))),n_boot=1000)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    cfg.update(production_refit=False,refit_scope='outer80',fit_rows=len(fit),
        refit_epochs=a.epochs,refit_training=vars(a),selection_identity=identity,
        threshold_source='fixed original out-of-fit tuning thresholds',refit_loss='sqrt_pos_weight')
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    names={n for n,p in model.named_parameters() if p.requires_grad}
    torch.save({n:t for n,t in model.state_dict().items() if n in names},out/'adapter.pt')
    tok.save_pretrained(out/'tokenizer')
    (out/'selection_metrics.json').write_bytes((src/'metrics.json').read_bytes())
    info={'train_sha256':train_hash,'selection_identity':identity,'fit_rows':len(fit),
        'validation_rows':len(va),'refit_scope':'outer80','production_refit':False,
        'validation_F1':report['ABSA_F1'],'total_seconds':time.perf_counter()-started,
        'epochs_fixed_before_refit':a.epochs,'thresholds_refit':False,'blind_file_read':False,
        'threads':torch.get_num_threads(),'torch':str(torch.__version__),'gpu':False}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    scorer.show(report);print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--init-dir',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=5e-6)
    a=p.parse_args()
    if min(a.epochs,a.threads,a.batch_size,a.accumulation)<1 or not 0<a.lr<1:p.error('Invalid training settings')
    run(a)
