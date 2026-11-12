"""Reproduce a supervised encoder's held-out evaluation and local predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from export_phobert_adapter import construct_model,load_model
from phobert_absa import Reviews,loader,infer,predictions,to_frame,split_indices,cpu_thread_limit,seed_everything
import score_absa as scorer

def open_model(folder,threads=2):
    folder=Path(folder);torch.set_num_threads(cpu_thread_limit(threads))
    cfg=json.loads((folder/'config.json').read_text())
    seed_everything(cfg['seed'])
    if (folder/'adapter.pt').exists():cfg,model=load_model(folder)
    else:
        model=construct_model(cfg)
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    tok=AutoTokenizer.from_pretrained(folder/'tokenizer',use_fast=cfg.get('tokenizer_fast',False))
    return cfg,model,tok

def probabilities_for(cfg,model,tok,texts,batch_size=8):
    if cfg.get('architecture')=='aspect_pair':
        from aspect_pair_absa import PairReviews
        ds=PairReviews(texts,tok,max_length=cfg['max_length'],stride=cfg['stride'],queries=cfg.get('queries'))
    else:ds=Reviews(texts,tok,None,cfg['max_length'],cfg['stride'],cfg.get('segment',True))
    probs=infer(model,loader(ds,tok,batch_size),torch.device('cpu'))
    return probs.reshape(len(texts),21)

def evaluate(model_dir,train_path,out_dir,threads=2,batch_size=8):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    cfg,model,tok=open_model(model_dir,threads)
    if cfg.get('production_refit',False):
        raise ValueError('A production model refit on all train cannot be evaluated as a held-out model; use its pre-refit selection checkpoint.')
    train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS and train.id.is_unique
    _,_,val=split_indices(train,cfg['seed'])
    scores=probabilities_for(cfg,model,tok,train.text.iloc[val],batch_size)
    pred=predictions(scores,np.asarray(cfg['thresholds']))
    truth=to_frame(train.id.iloc[val],train[scorer.COLS].iloc[val].to_numpy(int))
    proposed=to_frame(train.id.iloc[val],pred)
    report=scorer.report(truth,proposed,n_boot=1000)
    truth.to_csv(out/'validation_truth.csv',index=False)
    proposed.to_csv(out/'validation_prediction.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    errors=[]
    y=truth[scorer.COLS].to_numpy(int)
    for i,row in enumerate(val):
        wrong=np.flatnonzero(y[i]!=pred[i])
        if len(wrong):errors.append({'id':str(train.id.iloc[row]),'text':train.text.iloc[row],
            'errors':{scorer.COLS[j]:{'truth':int(y[i,j]),'prediction':int(pred[i,j])} for j in wrong}})
    (out/'validation_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    scorer.show(report)
    return report

def predict(model_dir,test_path,output,threads=2,batch_size=8):
    cfg,model,tok=open_model(model_dir,threads)
    test=pd.read_csv(test_path)
    assert list(test.columns)==['id','text'] and test.id.is_unique and not test.isna().any().any()
    scores=probabilities_for(cfg,model,tok,test.text,batch_size)
    to_frame(test.id,predictions(scores,np.asarray(cfg['thresholds']))).to_csv(output,index=False)
    assert scorer.kiem_tra(output,test_path)==0

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['evaluate','predict'])
    p.add_argument('--model-dir',required=True);p.add_argument('--train',default='../train.csv')
    p.add_argument('--test',default='../test_blind.csv');p.add_argument('--out-dir',default='reproduced_validation')
    p.add_argument('--output',default='submission_reproduced.csv');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
    if a.command=='evaluate':evaluate(a.model_dir,a.train,a.out_dir,a.threads,a.batch_size)
    else:predict(a.model_dir,a.test,a.output,a.threads,a.batch_size)

if __name__=='__main__':main()
