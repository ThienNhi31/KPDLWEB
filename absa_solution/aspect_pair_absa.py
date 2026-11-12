"""Aspect-conditioned PhoBERT: shared 3-label head, 7 explicit queries.
Split original reviews before creating pairs. No new review or label is generated.
Default train never reads test_blind.csv or exports blind-test predictions.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from pyvi import ViTokenizer
from phobert_absa import (MODEL,REVISION,PhoBERTABSA,Reviews,clean,loader,infer,
    seed_everything,split_indices,cpu_thread_limit,tune_thresholds,predictions,
    to_frame,main_score,optimizer_for)
from finetune_cpu_cached import make_cache,CachedModel,cached_loader,probabilities,epoch
import score_absa as scorer

QUERIES=['nhân viên và dịch vụ','sự sạch sẽ','vị trí và đi lại','phòng và thiết bị trong phòng',
         'giá trị so với giá tiền','ăn uống và bữa sáng','tiện nghi chung của khách sạn']

class PairModel(PhoBERTABSA):
    def __init__(self,model_name=MODEL,revision=REVISION,unfreeze_layers=4):
        super().__init__(model_name,revision,unfreeze_layers)
        self.head[-1]=nn.Linear(self.encoder.config.hidden_size,3)

class PairReviews(Reviews):
    def __init__(self,texts,tokenizer,labels=None,pairs=None,max_length=160,stride=32,queries=None):
        texts=list(texts)
        if pairs is None:pairs=[(i,a) for i in range(len(texts)) for a in range(7)]
        self.pairs=pairs;self.items=[]
        descriptions=QUERIES if queries is None else queries
        assert len(descriptions)==7
        queries=[tokenizer.encode(ViTokenizer.tokenize(q),add_special_tokens=False) for q in descriptions]
        needed={i for i,_ in pairs}
        encoded={i:tokenizer.encode(ViTokenizer.tokenize(clean(texts[i])),add_special_tokens=False) for i in needed}
        for i,a in pairs:
            query=queries[a];ids=encoded[i]
            capacity=max_length-tokenizer.num_special_tokens_to_add(pair=True)-len(query)
            if capacity<=stride:raise ValueError('Invalid query/window size')
            windows=[]
            for start in range(0,max(1,len(ids)),capacity-stride):
                seq=tokenizer.build_inputs_with_special_tokens(query,ids[start:start+capacity])
                windows.append({'input_ids':seq,'attention_mask':[1]*len(seq)})
                if start+capacity>=len(ids):break
            self.items.append(windows)
        self.labels=(None if labels is None else torch.as_tensor(
            np.array([labels[i,a*3:a*3+3] for i,a in pairs]),dtype=torch.float32))

def choose_pairs(indices,y,negatives,seed):
    rng=np.random.default_rng(seed);result=[]
    for i in indices:
        positive=[a for a in range(7) if y[i,a*3:a*3+3].any()]
        absent=[a for a in range(7) if a not in positive]
        if len(absent)>negatives:absent=rng.choice(absent,size=negatives,replace=False).tolist()
        result.extend((int(i),a) for a in sorted(positive+absent))
    return result

def run(a):
    started=time.perf_counter();seed_everything(a.seed);torch.set_num_threads(cpu_thread_limit(a.threads))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(Path(a.data_dir)/'train.csv')
    assert list(train.columns)==['id','text']+scorer.COLS and train.id.is_unique
    assert not train.isna().any().any() and train[scorer.COLS].isin([0,1]).all().all()
    y=train[scorer.COLS].to_numpy(int);fi,tu,va=split_indices(train,a.seed)
    fit_pairs=choose_pairs(fi,y,a.negatives,a.seed)
    tune_pairs=[(int(i),k) for i in tu for k in range(7)]
    val_pairs=[(int(i),k) for i in va for k in range(7)]
    pairs=fit_pairs+tune_pairs+val_pairs
    nfit,ntune,nval=len(fit_pairs),len(tune_pairs),len(val_pairs)
    ranges=[np.arange(nfit),np.arange(nfit,nfit+ntune),np.arange(nfit+ntune,len(pairs))]
    pair_y=np.array([y[i,k*3:k*3+3] for i,k in pairs])
    print('Original split:',len(fi),len(tu),len(va),'pair counts:',nfit,ntune,nval,flush=True)
    tok=AutoTokenizer.from_pretrained(a.model_name,revision=a.revision,use_fast=False)
    model=PairModel(a.model_name,a.revision,a.layers)
    depth=len(model.encoder.encoder.layer)-a.layers
    key={'architecture':'aspect_pair','queries':QUERIES,'model':a.model_name,'revision':a.revision,
         'frozen_depth':depth,'max_length':a.max_length,'stride':a.stride,'seed':a.seed,
         'negatives':a.negatives,'train_sha256':hashlib.sha256((Path(a.data_dir)/'train.csv').read_bytes()).hexdigest()}
    cp=Path(a.cache);cp.parent.mkdir(parents=True,exist_ok=True)
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        assert saved['key']==key,'Cache/data configuration mismatch'
        cache=saved['cache'];del saved
    else:
        ds=PairReviews(train.text,tok,y,pairs,a.max_length,a.stride)
        cache=make_cache(model,ds,tok,depth,a.batch_size)
        torch.save({'key':key,'cache':cache},cp);del ds
    wrapped=CachedModel(model,depth)
    check_pairs=pairs[:min(4,len(pairs))]
    check_ds=PairReviews(train.text,tok,y,check_pairs,a.max_length,a.stride)
    original=loader(check_ds,tok,len(check_pairs))
    fast=cached_loader(cache,np.arange(len(check_pairs)),pair_y,len(check_pairs))
    model.eval();wrapped.eval()
    with torch.no_grad():
        inputs,counts,_=next(iter(original));cached,cc,_=next(iter(fast))
        full=model(inputs,counts);same=wrapped(cached,cc)
        torch.testing.assert_close(full,same,atol=2e-5,rtol=2e-5)
        print('Aspect-pair cached/full logits parity passed.',flush=True)
    batches=[cached_loader(cache,idx,pair_y,a.batch_size,j==0,a.seed) for j,idx in enumerate(ranges)]
    opt=optimizer_for(model,a.lr)
    steps=math.ceil(len(batches[0])/a.accumulation)*a.epochs
    scheduler=get_cosine_schedule_with_warmup(opt,int(.08*steps),steps)
    positive=np.maximum(1,pair_y[:nfit].sum(0))
    pw=torch.as_tensor(np.clip(np.sqrt((nfit-positive)/positive),1,3),dtype=torch.float32)
    best=-1;best_epoch=0;history=[]
    for ep in range(1,a.epochs+1):
        ts=time.perf_counter()
        loss=epoch(wrapped,batches[0],opt,scheduler,pw,a.accumulation,50)
        scores=probabilities(wrapped,batches[1]).reshape(len(tu),21)
        cuts=tune_thresholds(y[tu],scores)
        f1=main_score(y[tu],predictions(scores,cuts))
        history.append({'epoch':ep,'loss':loss,'tune_F1':f1,'seconds':time.perf_counter()-ts})
        print('EPOCH',json.dumps(history[-1]),flush=True)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if f1>best:
            best,best_epoch=f1,ep
            torch.save({'state_dict':model.state_dict(),'cuts':cuts.tolist(),'epoch':ep},out/'best.pt')
            np.savez_compressed(out/'tune_scores.npz',scores=scores,ids=train.id.iloc[tu].to_numpy(dtype=str))
        if ep-best_epoch>=a.patience:break
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['state_dict']);cuts=np.asarray(selected['cuts']);del selected
    scores=probabilities(wrapped,batches[2]).reshape(len(va),21)
    vp=predictions(scores,cuts)
    gt,pred=to_frame(train.id.iloc[va],y[va]),to_frame(train.id.iloc[va],vp)
    report=scorer.report(gt,pred,n_boot=1000);scorer.show(report)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    gt.to_csv(out/'validation_truth.csv',index=False);pred.to_csv(out/'validation_prediction.csv',index=False)
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    config={'architecture':'aspect_pair','model_name':a.model_name,'revision':a.revision,'unfreeze_layers':a.layers,
            'seed':a.seed,'max_length':a.max_length,'stride':a.stride,'thresholds':cuts.tolist(),
            'columns':scorer.COLS,'queries':QUERIES,'best_epoch':best_epoch,'production_refit':False,
            'segment':True,'tokenizer_fast':False,'training':vars(a)}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    torch.save(model.state_dict(),out/'model.pt');tok.save_pretrained(out/'tokenizer')
    info={'best_tune_F1':best,'validation_F1':report['ABSA_F1'],'best_epoch':best_epoch,
          'total_seconds':time.perf_counter()-started,'torch':str(torch.__version__),
          'threads':torch.get_num_threads(),'gpu':False,'cache_key':key,'fit_pairs':nfit,
          'tune_pairs':ntune,'validation_pairs':nval}
    (out/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print('FINAL',json.dumps(info,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='..')
    p.add_argument('--out-dir',default='aspect_pair_results');p.add_argument('--cache',default='private_cache/aspect_pair.pt')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--layers',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--max-length',type=int,default=160)
    p.add_argument('--stride',type=int,default=32);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--negatives',type=int,default=2)
    p.add_argument('--model-name',default=MODEL);p.add_argument('--revision',default=REVISION)
    a=p.parse_args()
    if not 1<=a.layers<=24 or not 0<=a.negatives<=7:p.error('Invalid layers or negatives')
    run(a)

if __name__=='__main__':main()
