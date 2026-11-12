"""Warm-start more trainable PhoBERT layers, using the original review split.

Newly unfrozen layers start from the pinned pretrained weights. The initial
checkpoint remains eligible as epoch zero, selected by tuning F1 only.
No blind test data is read here.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from export_phobert_adapter import construct_model
from finetune_cpu_cached import make_cache,CachedModel,cached_loader,probabilities,epoch
from phobert_absa import (MODEL,REVISION,seed_everything,cpu_thread_limit,split_indices,
    Reviews,loader,optimizer_for,tune_thresholds,predictions,main_score,to_frame)
from stacked_absa import file_hash,model_identity
from text_normalization import prepare_texts
import score_absa as scorer


def run(a):
    started=time.perf_counter()
    src,out=Path(a.init_dir),Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((src/'config.json').read_text())
    source_cfg=cfg.copy()
    assert not cfg.get('production_refit',False) and not cfg.get('refit_scope')
    assert cfg['model_name']==MODEL and cfg['revision']==REVISION
    assert cfg.get('architecture','joint_21') in ['joint_21','label_attention']
    assert cfg['columns']==scorer.COLS and cfg['unfreeze_layers']<a.layers<=12
    seed_everything(cfg['seed'])
    torch.set_num_threads(cpu_thread_limit(a.threads))
    train_path=Path(a.data_dir)/'train.csv'
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y=train[scorer.COLS].to_numpy(int)
    assert np.isin(y,[0,1]).all()
    train_hash=file_hash(train_path)
    previous=json.loads((src/'run_info.json').read_text())
    assert previous.get('cache_key',{}).get('train_sha256',previous.get('train_sha256'))==train_hash
    fi,tu,va=split_indices(train,cfg['seed'])
    cfg['unfreeze_layers']=a.layers
    model=construct_model(cfg)
    checkpoint=src/'adapter.pt' if (src/'adapter.pt').exists() else src/'model.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    base=model.state_dict()
    # Old compact exports contain every previously trained tensor, including
    # the head. Newly trainable layers omitted by that export remain pretrained.
    newly_trainable_prefixes=tuple('encoder.encoder.layer.'+str(i)+'.'
        for i in range(12-a.layers,12-source_cfg['unfreeze_layers']))
    frozen={n for n,p in model.named_parameters() if not p.requires_grad}
    assert all(torch.equal(v,base[n]) for n,v in state.items() if n in frozen)
    loaded=model.load_state_dict(state,strict=False)
    assert not loaded.unexpected_keys
    assert all(n in frozen or n.startswith(newly_trainable_prefixes) for n in loaded.missing_keys)
    del state,base
    tok=AutoTokenizer.from_pretrained(src/'tokenizer',use_fast=False)
    depth=12-a.layers
    key={'model':MODEL,'revision':REVISION,'frozen_depth':depth,
        'max_length':cfg['max_length'],'stride':cfg['stride'],'train_sha256':train_hash}
    if cfg.get('text_normalizer'):
        key.update(text_normalizer=cfg['text_normalizer'],text_normalizer_sha256=cfg['text_normalizer_sha256'])
    cp=Path(a.cache);cp.parent.mkdir(parents=True,exist_ok=True)
    texts=prepare_texts(cfg,train.text)
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        assert saved['key']==key
        cache=saved['cache'];del saved
    else:
        ds=Reviews(texts,tok,y,cfg['max_length'],cfg['stride'])
        cache=make_cache(model,ds,tok,depth,a.batch_size)
        torch.save({'key':key,'cache':cache},cp)
        del ds
    if cfg.get('architecture')=='label_attention':
        from label_attention_absa import CachedAttentionModel
        wrapped=CachedAttentionModel(model,depth)
    else:wrapped=CachedModel(model,depth)
    # Verify raw and cached inference before doing any optimizer update. This
    # also checks the expanded model kept the exact frozen prefix for its cache.
    check=np.arange(min(4,len(train)))
    raw=Reviews([texts[int(i)] if isinstance(texts,list) else texts.iloc[int(i)] for i in check],
        tok,y[check],cfg['max_length'],cfg['stride'])
    model.eval();wrapped.eval()
    with torch.no_grad():
        inputs,counts,_=next(iter(loader(raw,tok,a.batch_size)))
        c,cc,_=next(iter(cached_loader(cache,check,y,a.batch_size)))
        full,fast=model(inputs,counts),wrapped(c,cc)
        torch.testing.assert_close(full,fast,atol=2e-5,rtol=2e-5)
    print('Expanded-layer cached/full logits parity passed',flush=True)
    batches=[cached_loader(cache,idx,y,a.batch_size,j==0,cfg['seed'])
        for j,idx in enumerate([fi,tu,va])]
    positive=np.maximum(1,y[fi].sum(0));negative=np.maximum(1,len(fi)-y[fi].sum(0))
    pw=torch.as_tensor(np.clip(negative/positive,.3,30),dtype=torch.float32)
    scale=len(fi)/(2*negative)
    scale*=np.array([.5 if c.endswith('_nhac') else 1 for c in scorer.COLS])
    scale=torch.as_tensor(scale,dtype=torch.float32)
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    opt=optimizer_for(model,a.lr)
    steps=math.ceil(len(batches[0])/a.accumulation)*a.epochs
    scheduler=get_cosine_schedule_with_warmup(opt,int(.08*steps),steps)
    scores=probabilities(wrapped,batches[1])
    cuts=tune_thresholds(y[tu],scores)
    best=main_score(y[tu],predictions(scores,cuts));best_epoch=0;history=[]
    def save_selected():
        torch.save({'state_dict':{n:t for n,t in model.state_dict().items() if n in trainable},
            'thresholds':cuts.tolist()},out/'best.pt')
        np.savez_compressed(out/'tune_scores.npz',scores=scores,ids=train.id.iloc[tu].to_numpy(dtype=str))
    save_selected()
    print('Tuning F1 before deeper training:',best,'; trainable layers',a.layers,flush=True)
    for ep in range(1,a.epochs+1):
        ts=time.perf_counter()
        loss=epoch(wrapped,batches[0],opt,scheduler,pw,a.accumulation,100,scale)
        scores=probabilities(wrapped,batches[1]);cuts=tune_thresholds(y[tu],scores)
        f1=main_score(y[tu],predictions(scores,cuts))
        row={'epoch':ep,'loss':loss,'tune_F1':f1,'seconds':time.perf_counter()-ts}
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print('EPOCH',json.dumps(row),flush=True)
        if f1>best:
            best,best_epoch=f1,ep;save_selected()
        if ep-best_epoch>=a.patience:break
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    restored=model.load_state_dict(saved['state_dict'],strict=False)
    assert not restored.unexpected_keys and set(restored.missing_keys)<=frozen
    cuts=np.asarray(saved['thresholds']);del saved
    scores=probabilities(wrapped,batches[2])
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(scores,cuts)),n_boot=1000)
    scorer.show(report)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    cfg.update(thresholds=cuts.tolist(),best_epoch=best_epoch,training=vars(a),
        production_refit=False,fit_rows=len(fi),source_identity=model_identity(src),
        newly_trainable_layers=a.layers-source_cfg['unfreeze_layers'])
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    torch.save({n:t for n,t in model.state_dict().items() if n in trainable},out/'adapter.pt')
    tok.save_pretrained(out/'tokenizer')
    result={'cache_key':key,'train_sha256':train_hash,'fit_rows':len(fi),
        'tune_rows':len(tu),'validation_rows':len(va),'best_tune_F1':best,'best_epoch':best_epoch,
        'validation_F1':report['ABSA_F1'],'total_seconds':time.perf_counter()-started,
        'trainable_encoder_layers':a.layers,'source_trainable_encoder_layers':source_cfg['unfreeze_layers'],
        'threads':torch.get_num_threads(),'torch':str(torch.__version__),'gpu':False}
    (out/'run_info.json').write_text(json.dumps(result,indent=2))
    (out/'best.pt').unlink()
    print('FINAL',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='..');p.add_argument('--init-dir',required=True)
    p.add_argument('--out-dir',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--layers',type=int,default=8);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--patience',type=int,default=2);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=8e-6)
    a=p.parse_args()
    if min(a.epochs,a.patience,a.threads,a.batch_size,a.accumulation)<1 or not 0<a.lr<1:
        p.error('Invalid training settings')
    run(a)
