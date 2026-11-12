"""Controlled augmentation experiment; original split stays fixed."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from export_phobert_adapter import construct_model
from finetune_cpu_cached import make_cache, CachedModel, cached_loader, probabilities
from phobert_absa import (seed_everything, cpu_thread_limit, split_indices, Reviews,
    optimizer_for, tune_thresholds, predictions, main_score, to_frame, MODEL, REVISION)
from stacked_absa import file_hash, model_identity
from generate_training_augmentation import normalize
from text_normalization import NORMALIZER_ID,SIGNATURE,prepare_texts
import score_absa as scorer


def train_epoch(model, batches, optimizer, scheduler, pw, scale, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, started = 0, time.perf_counter()
    for step, (inputs, counts, target) in enumerate(batches):
        logits = model(inputs, counts)
        loss = (F.binary_cross_entropy_with_logits(logits, target[:, :21],
            pos_weight=pw, reduction='none') * scale * target[:, 21:22]).mean()
        first = (step // accumulation) * accumulation
        (loss / min(accumulation, len(batches)-first)).backward()
        if (step+1) % accumulation == 0 or step+1 == len(batches):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach())
        if (step+1) % 100 == 0 or step+1 == len(batches):
            print('train', step+1, '/', len(batches), 'weighted loss', round(total/(step+1), 4),
                  'seconds', round(time.perf_counter()-started, 1), flush=True)
    return total / len(batches)


def run(a):
    started = time.perf_counter()
    src, out = Path(a.init_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src/'config.json').read_text())
    assert not cfg.get('production_refit', False) and not cfg.get('refit_scope')
    assert cfg['model_name'] == MODEL and cfg['revision'] == REVISION
    assert cfg['unfreeze_layers'] == 4 and cfg['columns'] == scorer.COLS
    seed_everything(cfg['seed'])
    torch.set_num_threads(cpu_thread_limit(a.threads))
    train_path = Path(a.data_dir)/'train.csv'
    train=pd.read_csv(train_path)
    extra=pd.read_csv(a.augmentation) if a.augmentation else pd.DataFrame(columns=train.columns)
    normalizer=a.text_normalizer or cfg.get('text_normalizer')
    input_cfg={'text_normalizer':normalizer,'text_normalizer_sha256':SIGNATURE if normalizer else None}
    assert list(train.columns) == list(extra.columns) == ['id','text']+scorer.COLS
    assert not train.isna().any().any() and not extra.isna().any().any()
    assert train.id.is_unique and extra.id.is_unique and not set(train.id) & set(extra.id)
    assert not set(train.text.map(normalize)) & set(extra.text.map(normalize))
    assert extra[scorer.COLS].isin([0,1]).all().all()
    for aspect in scorer.ASPECTS:
        assert ((extra[aspect+'_khen'] | extra[aspect+'_che']) <= extra[aspect+'_nhac']).all()
    info = json.loads((src/'run_info.json').read_text())
    train_hash = file_hash(train_path)
    assert info['cache_key']['train_sha256'] == train_hash
    fi, tu, va = split_indices(train, cfg['seed'])
    full = pd.concat([train, extra], ignore_index=True)
    y = full[scorer.COLS].to_numpy(int)
    fit = np.concatenate([fi, np.arange(len(train), len(full))])
    weights = np.concatenate([np.ones(len(train)), np.full(len(extra), a.synthetic_weight)])
    augmented_target = np.column_stack([y, weights])
    model = construct_model(cfg)
    trainable = {n for n,p in model.named_parameters() if p.requires_grad}
    checkpoint = src/'adapter.pt' if (src/'adapter.pt').exists() else src/'model.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    base = model.state_dict()
    frozen = {n for n,p in model.named_parameters() if not p.requires_grad}
    assert all(torch.equal(v, base[n]) for n,v in state.items() if n not in trainable)
    loaded = model.load_state_dict(state, strict=False)
    assert not loaded.unexpected_keys and set(loaded.missing_keys) <= frozen
    del state, base
    tok = AutoTokenizer.from_pretrained(src/'tokenizer', use_fast=False)
    depth = 12-cfg['unfreeze_layers']
    key = {'model':MODEL,'revision':REVISION,'frozen_depth':depth,'max_length':cfg['max_length'],
        'stride':cfg['stride'],'train_sha256':train_hash}
    if normalizer:key.update(input_cfg)
    original_cache=Path(a.cache);original_cache.parent.mkdir(parents=True,exist_ok=True)
    if original_cache.exists():
        saved = torch.load(original_cache,map_location='cpu',weights_only=True)
        assert saved['key']==key
        cache=saved['cache'];del saved
    else:
        ds=Reviews(prepare_texts(input_cfg,train.text),tok,train[scorer.COLS].to_numpy(int),cfg['max_length'],cfg['stride'])
        cache=make_cache(model,ds,tok,depth,a.batch_size)
        torch.save({'key':key,'cache':cache},original_cache)
        del ds
    assert len(cache) == len(train)
    augmentation_hash=file_hash(a.augmentation) if a.augmentation else None
    syn_key = {**key, 'augmentation_sha256':augmentation_hash}
    syn_cache_path = Path(a.synthetic_cache)
    syn_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not len(extra):synthetic_cache=[]
    elif syn_cache_path.exists():
        saved = torch.load(syn_cache_path, map_location='cpu', weights_only=True)
        assert saved['key'] == syn_key
        synthetic_cache = saved['cache']
        del saved
    else:
        ds = Reviews(prepare_texts(input_cfg,extra.text), tok, extra[scorer.COLS].to_numpy(int), cfg['max_length'], cfg['stride'])
        synthetic_cache = make_cache(model, ds, tok, depth, a.batch_size)
        torch.save({'key':syn_key,'cache':synthetic_cache}, syn_cache_path)
        del ds
    assert len(synthetic_cache) == len(extra)
    cache.extend(synthetic_cache)
    del synthetic_cache
    if cfg.get('architecture') == 'label_attention':
        from label_attention_absa import CachedAttentionModel
        wrapped = CachedAttentionModel(model, depth)
    else:
        wrapped = CachedModel(model, depth)
    batches = [cached_loader(cache, indices, augmented_target, a.batch_size, j==0, cfg['seed'])
               for j,indices in enumerate([fit,tu,va])]
    weighted_count = weights[fit].sum()
    positive = np.maximum(1, (y[fit]*weights[fit,None]).sum(0))
    negative = np.maximum(1, weighted_count-positive)
    pw = torch.as_tensor(np.clip(negative/positive,.3,30), dtype=torch.float32)
    scale = weighted_count/(2*negative)
    scale *= np.array([.5 if c.endswith('_nhac') else 1 for c in scorer.COLS])
    scale = torch.as_tensor(scale,dtype=torch.float32)
    opt = optimizer_for(model,a.lr)
    steps = math.ceil(len(batches[0])/a.accumulation)*a.epochs
    scheduler = get_cosine_schedule_with_warmup(opt,int(.08*steps),steps)
    initial = probabilities(wrapped,batches[1])
    cuts = tune_thresholds(y[tu],initial)
    best = main_score(y[tu],predictions(initial,cuts))
    source_tune_f1 = best
    history, best_epoch = [], 0
    def save_selected(cuts):
        selected={n:t for n,t in model.state_dict().items() if n in trainable}
        torch.save({'state_dict':selected,'thresholds':cuts.tolist()},out/'best.pt')
    save_selected(cuts)
    np.savez_compressed(out/'tune_scores.npz',scores=initial,ids=train.id.iloc[tu].to_numpy(dtype=str))
    print('Control tuning F1 before augmentation:',best,'; authentic fit',len(fi),'synthetic',len(extra),flush=True)
    for ep in range(1,a.epochs+1):
        ts = time.perf_counter()
        loss = train_epoch(wrapped,batches[0],opt,scheduler,pw,scale,a.accumulation)
        scores = probabilities(wrapped,batches[1])
        cuts = tune_thresholds(y[tu],scores)
        f1 = main_score(y[tu],predictions(scores,cuts))
        row = {'epoch':ep,'loss':loss,'tune_F1':f1,'seconds':time.perf_counter()-ts}
        history.append(row)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print('EPOCH',json.dumps(row),flush=True)
        if f1 > best:
            best,best_epoch = f1,ep
            save_selected(cuts)
            np.savez_compressed(out/'tune_scores.npz',scores=scores,ids=train.id.iloc[tu].to_numpy(dtype=str))
        if ep-best_epoch >= a.patience:break
    saved = torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    restored=model.load_state_dict(saved['state_dict'],strict=False)
    assert not restored.unexpected_keys and set(restored.missing_keys)<=frozen
    cuts = np.asarray(saved['thresholds'])
    del saved
    scores = probabilities(wrapped,batches[2])
    report = scorer.report(to_frame(train.id.iloc[va],y[va]),
        to_frame(train.id.iloc[va],predictions(scores,cuts)),n_boot=1000)
    scorer.show(report)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    old_report = json.loads((src/'metrics.json').read_text())
    pd.DataFrame([{'method':'source model before this trial','tune_F1':info['best_tune_F1'],'validation_F1':old_report['ABSA_F1']},
        {'method':'trial preprocessing and supervised refinement','tune_F1':best,'validation_F1':report['ABSA_F1']}]).to_csv(out/'augmentation_ablation.csv',index=False)
    cfg.update(thresholds=cuts.tolist(),best_epoch=best_epoch,training=vars(a),
        production_refit=False,augmentation_rows=len(extra),synthetic_weight=a.synthetic_weight,
        augmentation_sha256=augmentation_hash,source_identity=model_identity(src),**input_cfg,
        augmentation_updates_in_selected_checkpoint=best_epoch>0)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    torch.save({n:t for n,t in model.state_dict().items() if n in trainable},out/'adapter.pt')
    tok.save_pretrained(out/'tokenizer')
    result = {'cache_key':key,'train_sha256':train_hash,'augmentation_sha256':augmentation_hash,
        'authentic_fit_rows':len(fi),'synthetic_rows':len(extra),'synthetic_weight':a.synthetic_weight,
        'tune_rows':len(tu),'validation_rows':len(va),'best_tune_F1':best,'best_epoch':best_epoch,
        'validation_F1':report['ABSA_F1'],'total_seconds':time.perf_counter()-started,
        'threads':torch.get_num_threads(),'torch':str(torch.__version__),'gpu':False}
    (out/'run_info.json').write_text(json.dumps(result,indent=2))
    (out/'best.pt').unlink()
    print('FINAL',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='..')
    p.add_argument('--init-dir',required=True)
    p.add_argument('--augmentation')
    p.add_argument('--text-normalizer',choices=[NORMALIZER_ID])
    p.add_argument('--out-dir',default='augmentation_results')
    p.add_argument('--cache',default='private_cache/phobert_frozen8_160.pt')
    p.add_argument('--synthetic-cache',default='private_cache/synthetic_phi.pt')
    p.add_argument('--synthetic-weight',type=float,default=.4)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--patience',type=int,default=2)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5)
    a=p.parse_args()
    if not 0<a.synthetic_weight<=1:p.error('synthetic weight must be in (0,1]')
    run(a)


if __name__=='__main__':main()
