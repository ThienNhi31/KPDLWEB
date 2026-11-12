"""Export trained layers only, or predict locally from that compact checkpoint."""
import argparse
import json
import shutil
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer
from phobert_absa import PhoBERTABSA,Reviews,loader,infer,predictions,to_frame,seed_everything
import numpy as np
import score_absa

def construct_model(cfg):
    if cfg.get('architecture')=='label_attention':
        from label_attention_absa import LabelAttentionModel
        return LabelAttentionModel(cfg['model_name'],cfg['revision'],cfg['unfreeze_layers'])
    if cfg.get('architecture')=='aspect_pair':
        from aspect_pair_absa import PairModel
        return PairModel(cfg['model_name'],cfg['revision'],cfg['unfreeze_layers'])
    return PhoBERTABSA(cfg['model_name'],cfg['revision'],cfg['unfreeze_layers'])

def load_model(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text())
    seed_everything(cfg['seed'])
    model=construct_model(cfg)
    state=torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    assert not trainable.intersection(result.missing_keys)
    return cfg,model

def export(folder,out,include_submission=False):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((folder/'config.json').read_text())
    assert cfg['unfreeze_layers']>=0, 'Full embedding fine-tuning requires a complete checkpoint.'
    model=construct_model(cfg)
    full=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    names={n for n,p in model.named_parameters() if p.requires_grad}
    base=model.state_dict()
    # A partial checkpoint is exact only if every omitted tensor is unchanged.
    assert all(torch.equal(v,base[n]) for n,v in full.items() if n not in names)
    adapter={n:v for n,v in full.items() if n in names}
    torch.save(adapter,out/'adapter.pt')
    shutil.copyfile(folder/'config.json',out/'config.json')
    shutil.copytree(folder/'tokenizer',out/'tokenizer',dirs_exist_ok=True)
    files=['metrics.json','selection_metrics.json','history.csv','run_info.json','tune_scores.npz','validation_scores.npz']
    if include_submission:files.append('submission.csv')
    for name in files:
        if (folder/name).exists():shutil.copyfile(folder/name,out/name)
    # No cache, training text or validation examples are included.
    print('Adapter exported; every omitted tensor exactly matches the pinned pretrained checkpoint.')

def predict(folder,test_path,out,batch_size,threads):
    torch.set_num_threads(threads)
    cfg,model=load_model(folder)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    tokenizer=AutoTokenizer.from_pretrained(Path(folder)/'tokenizer',use_fast=cfg.get('tokenizer_fast',False))
    test=pd.read_csv(test_path)
    assert list(test.columns)==['id','text'] and test.id.is_unique and not test.isna().any().any()
    paired=cfg.get('architecture')=='aspect_pair'
    if paired:
        from aspect_pair_absa import PairReviews
        ds=PairReviews(test.text,tokenizer,max_length=cfg['max_length'],stride=cfg['stride'],queries=cfg.get('queries'),segment=cfg.get('segment',True))
    else:
        ds=Reviews(test.text,tokenizer,None,cfg['max_length'],cfg['stride'],cfg.get('segment',True))
    probs=infer(model,loader(ds,tokenizer,batch_size),device)
    if paired:probs=probs.reshape(len(test),21)
    to_frame(test.id,predictions(probs,np.asarray(cfg['thresholds']))).to_csv(out,index=False)
    assert score_absa.kiem_tra(out,test_path)==0

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['export','predict'])
    p.add_argument('--model-dir',default='phobert_results');p.add_argument('--adapter-dir',default='phobert_adapter')
    p.add_argument('--test',default='../test_blind.csv');p.add_argument('--output',default='submission_reproduced.csv')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--include-submission',action='store_true',help='Explicitly opt into exporting a blind-test submission')
    a=p.parse_args()
    if a.command=='export':export(a.model_dir,a.adapter_dir,a.include_submission)
    else:predict(a.adapter_dir,a.test,a.output,a.batch_size,a.threads)

if __name__=='__main__':main()
