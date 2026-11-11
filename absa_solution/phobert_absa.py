"""Offline supervised PhoBERT ABSA. No generative API or test-label creation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from pyvi import ViTokenizer
from sklearn.model_selection import GroupShuffleSplit
import score_absa as scorer

MODEL = 'vinai/phobert-base-v2'
REVISION = '86cd7fd4c148980922ac11a2cf5e257f2ba639e1'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def clean(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(text))).strip()

def split_indices(train, seed=2026):
    # Exactly the classical pipeline's grouping and split for a fair comparison.
    groups = train.text.map(lambda s: hashlib.sha256(clean(s).lower().encode()).hexdigest())
    tt, val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                   .split(train, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1)
               .split(train.iloc[tt], groups=groups.iloc[tt]))
    fit, tune = tt[a], tt[b]
    assert not set(groups.iloc[fit]) & set(groups.iloc[tune])
    assert not set(groups.iloc[tt]) & set(groups.iloc[val])
    return fit, tune, val

class Reviews(Dataset):
    def __init__(self, texts, tokenizer, labels=None, max_length=256, stride=64):
        self.items = []
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)
        content_len = max_length-tokenizer.num_special_tokens_to_add(pair=False)
        if not 0 <= stride < content_len:
            raise ValueError('stride must be smaller than available content length')
        for text in texts:
            segmented = ViTokenizer.tokenize(clean(text))
            ids = tokenizer.encode(segmented, add_special_tokens=False)
            chunks = []
            for start in range(0, max(1, len(ids)), content_len-stride):
                part = ids[start:start+content_len]
                seq = tokenizer.build_inputs_with_special_tokens(part)
                chunks.append({'input_ids': seq, 'attention_mask': [1]*len(seq)})
                if start+content_len >= len(ids):
                    break
            self.items.append(chunks)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], None if self.labels is None else self.labels[i]

class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        chunks, counts = [], []
        for windows, _ in batch:
            chunks.extend(windows)
            counts.append(len(windows))
        inputs = self.tokenizer.pad(chunks, padding=True, return_tensors='pt')
        labels = None if batch[0][1] is None else torch.stack([b[1] for b in batch])
        return inputs, counts, labels

class PhoBERTABSA(nn.Module):
    def __init__(self, model_name=MODEL, revision=REVISION, unfreeze_layers=-1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=False,
                                                 add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Dropout(.2), nn.Linear(hidden, len(scorer.COLS)))
        if unfreeze_layers >= 0:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_layers:
                for layer in self.encoder.encoder.layer[-unfreeze_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, inputs, counts):
        # Learn bag-level labels: do NOT copy a review's labels to each chunk.
        cls = self.encoder(**inputs).last_hidden_state[:, 0]
        logits = self.head(cls)
        return torch.stack([part.max(dim=0).values for part in logits.split(counts)])

def predictions(scores, thresholds):
    result = (scores >= thresholds).astype(int)
    for a in range(7):
        result[:, a*3] |= result[:, a*3+1] | result[:, a*3+2]
    return result

def to_frame(ids, pred):
    result = pd.DataFrame(pred, columns=scorer.COLS)
    result.insert(0, 'id', np.asarray(ids))
    return result

def main_score(y, pred):
    gt = to_frame(np.arange(len(y)), y)
    pr = to_frame(np.arange(len(y)), pred)
    return scorer.core_score(scorer.to_arrays(gt)[1], scorer.to_arrays(pr)[1])

def tune_thresholds(y, scores):
    cuts = []
    for j in range(y.shape[1]):
        grid = np.linspace(.15, .75, 13)
        values = np.array([scorer.f1(y[:,j].astype(bool), scores[:,j]>=c) for c in grid])
        if np.isnan(values).all():
            cuts.append(.5)
            continue
        best = np.flatnonzero(values == np.nanmax(values))
        cuts.append(float(grid[best[np.argmin(abs(grid[best]-.5))]]))
    return np.array(cuts)

def loader(dataset, tokenizer, batch_size, shuffle=False, seed=2026):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=Collate(tokenizer), generator=g)

@torch.no_grad()
def infer(model, batches, device):
    model.eval()
    outputs=[]
    for inputs, counts, _ in batches:
        logits = model({k:v.to(device) for k,v in inputs.items()}, counts)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)

def optimizer_for(model, lr):
    groups = [{'params':[p for p in model.encoder.parameters() if p.requires_grad], 'lr':lr},
              {'params':model.head.parameters(), 'lr':lr*5}]
    return torch.optim.AdamW(groups, weight_decay=.01)

def train_epoch(model, batches, optimizer, scheduler, device, pos_weight, accumulation=2, log_steps=50):
    model.train()
    if not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    optimizer.zero_grad(set_to_none=True)
    total = 0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for step, (inputs, counts, labels) in enumerate(batches):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            logits = model({k:v.to(device) for k,v in inputs.items()}, counts)
            loss = loss_fn(logits, labels.to(device))
            # Last accumulation group may be shorter; preserve its mean gradient.
            start = (step//accumulation)*accumulation
            divisor = min(accumulation, len(batches)-start)
            scaled = loss/divisor
        scaler.scale(scaled).backward()
        if (step+1)%accumulation==0 or step+1==len(batches):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach())
        if log_steps and (step+1)%log_steps==0:
            print('  batch', step+1, '/', len(batches), 'loss', round(total/(step+1),4), flush=True)
    return total/max(1,len(batches))

def setup(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=False,
                                               trust_remote_code=False)
    return device, tokenizer

def fit(args):
    started=time.perf_counter()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    device,tokenizer=setup(args)
    train=pd.read_csv(Path(args.data_dir)/'train.csv')
    assert list(train.columns)==['id','text']+scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    assert train[scorer.COLS].isin([0,1]).all().all()
    fit_idx,tune_idx,val_idx=split_indices(train,args.seed)
    y=train[scorer.COLS].to_numpy(int)
    print('Device:',device,'split:',len(fit_idx),len(tune_idx),len(val_idx),flush=True)
    datasets={name:Reviews(train.text.iloc[idx],tokenizer,y[idx],args.max_length,args.stride)
              for name,idx in [('fit',fit_idx),('tune',tune_idx),('validation',val_idx)]}
    batches={name:loader(ds,tokenizer,args.batch_size,name=='fit',args.seed) for name,ds in datasets.items()}
    print('Review windows:',{n:sum(map(len,d.items)) for n,d in datasets.items()},flush=True)
    model=PhoBERTABSA(unfreeze_layers=args.unfreeze_layers).to(device)
    optimizer=optimizer_for(model,args.lr)
    updates=math.ceil(len(batches['fit'])/args.accumulation)*args.epochs
    schedule=get_cosine_schedule_with_warmup(optimizer,int(updates*.1),updates)
    pw=torch.as_tensor(np.clip(np.sqrt((len(fit_idx)-y[fit_idx].sum(0))/np.maximum(1,y[fit_idx].sum(0))),1,3),dtype=torch.float32,device=device)
    best=-1;best_epoch=0;history=[]
    for epoch in range(1,args.epochs+1):
        ts=time.perf_counter()
        loss=train_epoch(model,batches['fit'],optimizer,schedule,device,pw,args.accumulation)
        scores=infer(model,batches['tune'],device)
        cuts=tune_thresholds(y[tune_idx],scores)
        score=main_score(y[tune_idx],predictions(scores,cuts))
        history.append({'epoch':epoch,'loss':loss,'tune_F1':score,'seconds':time.perf_counter()-ts})
        print(json.dumps(history[-1]),flush=True)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if score>best:
            best,best_epoch=score,epoch
            torch.save({'state_dict':model.cpu().state_dict(),'thresholds':cuts.tolist(),'epoch':epoch},out/'best.pt')
            model.to(device)
        if epoch-best_epoch>=args.patience:
            break
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict']);model.to(device)
    cuts=np.asarray(saved['thresholds'])
    # Validation is evaluated only once, after epoch/threshold selection on tune.
    ts=time.perf_counter()
    val_scores=infer(model,batches['validation'],device)
    inference_seconds=time.perf_counter()-ts
    vp=predictions(val_scores,cuts)
    gt,pred=to_frame(train.id.iloc[val_idx],y[val_idx]),to_frame(train.id.iloc[val_idx],vp)
    report=scorer.report(gt,pred,n_boot=1000)
    scorer.show(report)
    gt.to_csv(out/'validation_truth.csv',index=False)
    pred.to_csv(out/'validation_prediction.csv',index=False)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    pd.DataFrame({'id':train.id,'partition':['fit' if i in set(fit_idx) else 'tune' if i in set(tune_idx) else 'validation' for i in range(len(train))]}).to_csv(out/'split.csv',index=False)
    errors=[]
    for i,ix in enumerate(val_idx):
        wrong=np.flatnonzero(y[ix]!=vp[i])
        if len(wrong):
            errors.append({'id':str(train.id.iloc[ix]),'text':train.text.iloc[ix],
                           'errors':{scorer.COLS[j]:{'truth':int(y[ix,j]),'prediction':int(vp[i,j])} for j in wrong}})
    (out/'validation_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    # Production refit: fresh weights and all labeled train, selected epoch count.
    if not args.no_refit:
        del model,optimizer
        seed_everything(args.seed)
        if device.type=='cuda':torch.cuda.empty_cache()
        all_ds=Reviews(train.text,tokenizer,y,args.max_length,args.stride)
        all_batches=loader(all_ds,tokenizer,args.batch_size,True,args.seed)
        model=PhoBERTABSA(unfreeze_layers=args.unfreeze_layers).to(device)
        optimizer=optimizer_for(model,args.lr)
        updates=math.ceil(len(all_batches)/args.accumulation)*best_epoch
        schedule=get_cosine_schedule_with_warmup(optimizer,int(updates*.1),updates)
        pw=torch.as_tensor(np.clip(np.sqrt((len(y)-y.sum(0))/np.maximum(1,y.sum(0))),1,3),dtype=torch.float32,device=device)
        for epoch in range(best_epoch):
            loss=train_epoch(model,all_batches,optimizer,schedule,device,pw,args.accumulation)
            print('Refit epoch',epoch+1,'loss',loss,flush=True)
    config=vars(args).copy()
    config.update({'model_name':MODEL,'revision':REVISION,'columns':scorer.COLS,'best_epoch':best_epoch,
                   'thresholds':cuts.tolist(),'production_refit':not args.no_refit})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    torch.save(model.cpu().state_dict(),out/'model.pt');model.to(device)
    tokenizer.save_pretrained(out/'tokenizer')
    run_info={'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
              'python':platform.python_version(),'torch':torch.__version__,'os':platform.platform(),
              'cpu_count':os.cpu_count(),'threads':torch.get_num_threads(),
              'ram':next((s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemTotal:')),''),
              'best_tune_F1':best,'validation_predict_seconds':inference_seconds,'total_seconds_before_test':time.perf_counter()-started,
              'train_sha256':hashlib.sha256((Path(args.data_dir)/'train.csv').read_bytes()).hexdigest()}
    (out/'run_info.json').write_text(json.dumps(run_info,indent=2))
    predict_saved(out,Path(args.data_dir)/'test_blind.csv',out/'submission.csv',args.batch_size,args.cpu_threads)

def predict_saved(model_dir,test_path,output,batch_size=4,cpu_threads=3):
    model_dir=Path(model_dir)
    config=json.loads((model_dir/'config.json').read_text())
    seed_everything(config['seed']);torch.set_num_threads(cpu_threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(model_dir/'tokenizer',use_fast=False)
    model=PhoBERTABSA(config['model_name'],config['revision'],config['unfreeze_layers'])
    model.load_state_dict(torch.load(model_dir/'model.pt',map_location='cpu',weights_only=True));model.to(device)
    test=pd.read_csv(test_path)
    assert list(test.columns)==['id','text'] and test.id.is_unique and not test.isna().any().any()
    ds=Reviews(test.text,tokenizer,None,config['max_length'],config['stride'])
    started=time.perf_counter()
    scores=infer(model,loader(ds,tokenizer,batch_size),device)
    pred=predictions(scores,np.asarray(config['thresholds']))
    to_frame(test.id,pred).to_csv(output,index=False,encoding='utf-8')
    assert scorer.kiem_tra(output,test_path)==0
    print('Saved',len(test),'predictions; seconds:',round(time.perf_counter()-started,2),flush=True)

def smoke(args):
    """Real pretrained weights, train rows only, one backward pass; not an F1 experiment."""
    device,tokenizer=setup(args)
    train=pd.read_csv(Path(args.data_dir)/'train.csv')
    ds=Reviews(train.text.iloc[:4],tokenizer,train[scorer.COLS].iloc[:4].to_numpy(),args.max_length,args.stride)
    model=PhoBERTABSA(unfreeze_layers=args.unfreeze_layers).to(device)
    model.train();opt=optimizer_for(model,args.lr)
    inputs,counts,labels=next(iter(loader(ds,tokenizer,2)))
    ts=time.perf_counter()
    logits=model({k:v.to(device) for k,v in inputs.items()},counts)
    assert logits.shape==(2,21)
    loss=nn.BCEWithLogitsLoss()(logits,labels.to(device));loss.backward();opt.step()
    elapsed=time.perf_counter()-ts
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.head.parameters())
    print(json.dumps({'smoke_passed':True,'device':str(device),'loss':float(loss.detach()),
                      'batch_seconds':elapsed,'reviews':2,'windows':sum(counts),
                      'unfreeze_layers':args.unfreeze_layers,'cuda':torch.cuda.is_available()},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['smoke','train','predict'])
    p.add_argument('--data-dir',default='..');p.add_argument('--out-dir',default='phobert_results')
    p.add_argument('--model-dir',default='phobert_results')
    p.add_argument('--test',default='../test_blind.csv');p.add_argument('--output',default='submission_reproduced.csv')
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--cpu-threads',type=int,default=3)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--patience',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--max-length',type=int,default=256);p.add_argument('--stride',type=int,default=64)
    p.add_argument('--unfreeze-layers',type=int,default=-1,help='-1: full fine-tune; 0: head only; N: last N encoder layers')
    p.add_argument('--no-refit',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.accumulation<1:
        p.error('epochs, batch-size and accumulation must be positive')
    if args.command=='smoke':smoke(args)
    elif args.command=='train':fit(args)
    else:predict_saved(args.model_dir,args.test,args.output,args.batch_size,args.cpu_threads)

if __name__=='__main__':main()
