"""CPU PhoBERT fine-tuning with cached frozen layers. Keep the feature cache private."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler,Subset
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from phobert_absa import (MODEL,REVISION,PhoBERTABSA,Reviews,loader,seed_everything,
                          split_indices,tune_thresholds,predictions,to_frame,main_score,
                          optimizer_for,predict_saved)
import score_absa as scorer

def attention(mask,dtype):
    return (1-mask[:,None,None,:].to(dtype))*torch.finfo(dtype).min

@torch.no_grad()
def make_cache(model,ds,tokenizer,depth,batch_size=8):
    model.eval();cache=[];started=time.perf_counter()
    batches=loader(ds,tokenizer,batch_size)
    for step,(inputs,counts,_) in enumerate(batches):
        hidden=model.encoder.embeddings(input_ids=inputs['input_ids'])
        mask=attention(inputs['attention_mask'],hidden.dtype)
        for layer in model.encoder.encoder.layer[:depth]:
            hidden=layer(hidden,attention_mask=mask)[0]
        cursor=0
        for count in counts:
            windows=[]
            for j in range(cursor,cursor+count):
                n=int(inputs['attention_mask'][j].sum())
                windows.append(hidden[j,:n].clone())
            cursor+=count;cache.append(windows)
        if (step+1)%25==0 or step+1==len(batches):
            print('cache',step+1,'/',len(batches),'reviews',len(cache),'seconds',round(time.perf_counter()-started,1),flush=True)
    return cache

class CachedReviews(Dataset):
    def __init__(self,cache,indices,y):self.cache=cache;self.indices=indices;self.y=y
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i]
        return self.cache[j],torch.as_tensor(self.y[j],dtype=torch.float32)

def collate(batch):
    counts=[len(x[0]) for x in batch]
    windows=[w for x,_ in batch for w in x]
    length=max(len(w) for w in windows)
    hidden=torch.zeros(len(windows),length,windows[0].shape[-1])
    mask=torch.zeros(len(windows),length,dtype=torch.long)
    for i,w in enumerate(windows):hidden[i,:len(w)]=w;mask[i,:len(w)]=1
    return {'hidden':hidden,'attention_mask':mask},counts,torch.stack([y for _,y in batch])

class CachedModel(nn.Module):
    def __init__(self,model,depth):
        super().__init__();self.model=model;self.depth=depth
    def forward(self,inputs,counts):
        h=inputs['hidden'];mask=attention(inputs['attention_mask'],h.dtype)
        for layer in self.model.encoder.encoder.layer[self.depth:]:h=layer(h,attention_mask=mask)[0]
        logits=self.model.head(h[:,0])
        return torch.stack([part.max(0).values for part in logits.split(counts)])

class LengthBatches(Sampler):
    """Shuffle batches of similar lengths; avoid computing gradients on padding."""
    def __init__(self,ds,batch_size,generator):
        order=sorted(range(len(ds)),key=lambda i:max(len(w) for w in ds.cache[ds.indices[i]]))
        self.batches=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
        self.generator=generator
    def __len__(self):return len(self.batches)
    def __iter__(self):
        for j in torch.randperm(len(self.batches),generator=self.generator).tolist():
            group=self.batches[j]
            yield [group[i] for i in torch.randperm(len(group),generator=self.generator).tolist()]

def cached_loader(cache,indices,y,batch_size,shuffle=False,seed=2026):
    ds=CachedReviews(cache,indices,y);g=torch.Generator().manual_seed(seed)
    if shuffle:
        return DataLoader(ds,batch_sampler=LengthBatches(ds,batch_size,g),num_workers=0,
                          collate_fn=collate,generator=g)
    return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate,generator=g)

@torch.no_grad()
def probabilities(model,batches):
    model.eval();rows=[]
    ds=batches.dataset
    order=sorted(range(len(ds)),key=lambda i:max(len(w) for w in ds.cache[ds.indices[i]]))
    ordered=DataLoader(Subset(ds,order),batch_size=batches.batch_size or 8,num_workers=0,
                       collate_fn=collate,generator=torch.Generator().manual_seed(0))
    for inputs,counts,_ in ordered:rows.append(model(inputs,counts).sigmoid().numpy())
    result=np.concatenate(rows);original=np.empty_like(result);original[order]=result
    return original

def epoch(model,batches,opt,scheduler,pw,accumulation,log_steps,label_weights=None):
    model.train();opt.zero_grad(set_to_none=True);total=0;started=time.perf_counter()
    lossfn=nn.BCEWithLogitsLoss(pos_weight=pw,reduction='none' if label_weights is not None else 'mean')
    for step,(inputs,counts,y) in enumerate(batches):
        logits=model(inputs,counts);loss=lossfn(logits,y)
        if label_weights is not None:loss=(loss*label_weights).mean()
        start=(step//accumulation)*accumulation
        (loss/min(accumulation,len(batches)-start)).backward()
        if (step+1)%accumulation==0 or step+1==len(batches):
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
            opt.step();scheduler.step();opt.zero_grad(set_to_none=True)
        total+=float(loss.detach())
        if (step+1)%log_steps==0 or step+1==len(batches):
            print('train',step+1,'/',len(batches),'loss',round(total/(step+1),4),
                  'seconds',round(time.perf_counter()-started,1),flush=True)
    return total/len(batches)

def run(a):
    start=time.perf_counter();seed_everything(a.seed);torch.set_num_threads(a.threads)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(Path(a.data_dir)/'train.csv')
    assert list(train.columns)==['id','text']+scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y=train[scorer.COLS].to_numpy(int);assert np.isin(y,[0,1]).all()
    fi,tu,va=split_indices(train,a.seed)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=False)
    model=PhoBERTABSA(unfreeze_layers=a.layers)
    depth=len(model.encoder.encoder.layer)-a.layers
    key={'model':MODEL,'revision':REVISION,'frozen_depth':depth,'max_length':a.max_length,
         'stride':a.stride,'train_sha256':hashlib.sha256((Path(a.data_dir)/'train.csv').read_bytes()).hexdigest()}
    cache_path=Path(a.cache);cache_path.parent.mkdir(parents=True,exist_ok=True)
    if cache_path.exists():
        saved=torch.load(cache_path,map_location='cpu',weights_only=True)
        if saved['key']!=key:raise ValueError('Cache configuration/data mismatch; select a new cache filename')
        cache=saved['cache'];del saved
        print('Loaded cache:',len(cache),'reviews',flush=True)
    else:
        ds=Reviews(train.text,tokenizer,y,a.max_length,a.stride)
        cache=make_cache(model,ds,tokenizer,depth,a.batch_size)
        torch.save({'key':key,'cache':cache},cache_path)
        del ds
    wrapped=CachedModel(model,depth)
    check=np.arange(min(4,len(train)))
    ds_check=Reviews(train.text.iloc[check],tokenizer,y[check],a.max_length,a.stride)
    original=loader(ds_check,tokenizer,a.batch_size);cached=cached_loader(cache,check,y,a.batch_size)
    model.eval();wrapped.eval()
    with torch.no_grad():
        i,counts,_=next(iter(original));c,cc,_=next(iter(cached))
        full=model(i,counts);fast=wrapped(c,cc)
        torch.testing.assert_close(full,fast,atol=2e-5,rtol=2e-5)
        print('Cached/full logits parity passed; max error:',float((full-fast).abs().max()),flush=True)
    if a.init_dir:
        init_config=json.loads((Path(a.init_dir)/'config.json').read_text())
        assert not init_config.get('production_refit',False), 'Cannot use an all-train production checkpoint for held-out evaluation'
        assert init_config['seed']==a.seed, 'Warm-start split seed must match the original split'
        init_info_path=Path(a.init_dir)/'run_info.json'
        if init_info_path.exists():
            init_info=json.loads(init_info_path.read_text())
            init_hash=init_info.get('cache_key',{}).get('train_sha256',init_info.get('train_sha256'))
            assert init_hash in (None,key['train_sha256']), 'Warm-start training data differs'
        initial=torch.load(Path(a.init_dir)/'model.pt',map_location='cpu',weights_only=True)
        base_state=model.state_dict()
        frozen={n for n,p in model.named_parameters() if not p.requires_grad}
        assert all(torch.equal(initial[n],base_state[n]) for n in frozen), 'Frozen cache basis differs from initialization model'
        model.load_state_dict(initial);del initial,base_state
        print('Warm start:',a.init_dir,'; all frozen tensors match cache basis',flush=True)
    fit_batches=cached_loader(cache,fi,y,a.batch_size,True,a.seed)
    tune_batches=cached_loader(cache,tu,y,a.batch_size)
    val_batches=cached_loader(cache,va,y,a.batch_size)
    opt=optimizer_for(model,a.lr)
    total=math.ceil(len(fit_batches)/a.accumulation)*a.epochs
    scheduler=get_cosine_schedule_with_warmup(opt,int(.08*total),total)
    pw=torch.as_tensor(np.clip(np.sqrt((len(fi)-y[fi].sum(0))/np.maximum(1,y[fi].sum(0))),1,3),dtype=torch.float32)
    label_weights=None
    if a.balanced_loss:
        positives=np.maximum(1,y[fi].sum(0));negatives=np.maximum(1,len(fi)-y[fi].sum(0))
        pw=torch.as_tensor(np.clip(negatives/positives,.3,30),dtype=torch.float32)
        scale=len(fi)/(2*negatives)
        scale*=np.array([.5 if c.endswith('_nhac') else 1 for c in scorer.COLS])
        label_weights=torch.as_tensor(scale,dtype=torch.float32)
    best=-1;best_epoch=0;history=[];start_epoch=1
    if a.init_dir and not a.resume:
        initial_probs=probabilities(wrapped,tune_batches)
        cuts=tune_thresholds(y[tu],initial_probs)
        best=main_score(y[tu],predictions(initial_probs,cuts))
        torch.save({'state_dict':model.state_dict(),'cuts':cuts.tolist(),'epoch':0},out/'best.pt')
        np.savez_compressed(out/'best_tuning_predictions.npz',scores=initial_probs,labels=y[tu])
        np.savez_compressed(out/'tune_scores.npz',scores=initial_probs,ids=train.id.iloc[tu].to_numpy(dtype=str))
        print('Warm-start tuning F1:',best,flush=True)
    if a.resume:
        previous=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(previous['model']);opt.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler']);torch.set_rng_state(previous['torch_rng'])
        fit_batches.generator.set_state(previous['loader_rng'])
        start_epoch=previous['epoch']+1;best=previous['best_tune_F1'];best_epoch=previous['best_epoch']
        history=pd.read_csv(out/'history.csv').to_dict('records');del previous
        print('Resuming at epoch',start_epoch,flush=True)
    for ep in range(start_epoch,a.epochs+1):
        ts=time.perf_counter()
        loss=epoch(wrapped,fit_batches,opt,scheduler,pw,a.accumulation,a.log_steps,label_weights)
        probs=probabilities(wrapped,tune_batches)
        cuts=tune_thresholds(y[tu],probs);s=main_score(y[tu],predictions(probs,cuts))
        row={'epoch':ep,'loss':loss,'tune_F1':s,'seconds':time.perf_counter()-ts};history.append(row)
        print('EPOCH',json.dumps(row),flush=True)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if s>best:
            best,best_epoch=s,ep
            torch.save({'state_dict':model.state_dict(),'cuts':cuts.tolist(),'epoch':ep},out/'best.pt')
            np.savez_compressed(out/'best_tuning_predictions.npz',scores=probs,labels=y[tu])
            np.savez_compressed(out/'tune_scores.npz',scores=probs,ids=train.id.iloc[tu].to_numpy(dtype=str))
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':ep,'torch_rng':torch.get_rng_state(),'loader_rng':fit_batches.generator.get_state(),
                    'best_epoch':best_epoch,'best_tune_F1':best},out/'latest.pt')
        if ep-best_epoch>=a.patience:break
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict']);cuts=np.asarray(saved['cuts']);del saved
    val_probs=probabilities(wrapped,val_batches);vp=predictions(val_probs,cuts)
    gt,pr=to_frame(train.id.iloc[va],y[va]),to_frame(train.id.iloc[va],vp)
    r=scorer.report(gt,pr,n_boot=1000);scorer.show(r)
    gt.to_csv(out/'validation_truth.csv',index=False);pr.to_csv(out/'validation_prediction.csv',index=False)
    np.savez_compressed(out/'validation_probabilities.npz',scores=val_probs,labels=y[va])
    np.savez_compressed(out/'validation_scores.npz',scores=val_probs,ids=train.id.iloc[va].to_numpy(dtype=str))
    (out/'metrics.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    config={'model_name':MODEL,'revision':REVISION,'unfreeze_layers':a.layers,'seed':a.seed,
            'max_length':a.max_length,'stride':a.stride,'thresholds':cuts.tolist(),'columns':scorer.COLS,
            'best_epoch':best_epoch,'production_refit':False,'training':vars(a)}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    torch.save(model.state_dict(),out/'model.pt');tokenizer.save_pretrained(out/'tokenizer')
    info={'best_tune_F1':best,'best_epoch':best_epoch,'validation_F1':r['ABSA_F1'],
          'total_seconds':time.perf_counter()-start,'cache_key':key,'torch':str(torch.__version__),
          'threads':a.threads,'gpu':False,'fit_rows':len(fi),'tune_rows':len(tu),'validation_rows':len(va)}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    print('FINAL',json.dumps(info),flush=True)
    if a.predict_test:predict_saved(out,Path(a.data_dir)/'test_blind.csv',out/'submission.csv',a.batch_size,a.threads)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='..');p.add_argument('--out-dir',default='cached_phobert_results')
    p.add_argument('--cache',default='private_cache/phobert_frozen8_160.pt')
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--layers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-length',type=int,default=160);p.add_argument('--stride',type=int,default=32)
    p.add_argument('--log-steps',type=int,default=25);p.add_argument('--predict-test',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--init-dir',default=None,help='Warm start from a model with the identical frozen basis')
    p.add_argument('--balanced-loss',action='store_true',help='Equalize positive/negative loss per label; downweight mention-only labels')
    a=p.parse_args()
    if not 1<=a.layers<=12:p.error('layers must be in [1,12]')
    run(a)

if __name__=='__main__':main()
