"""Fixed-schedule aspect-pair refit with per-aspect polarity balancing.

outer80 keeps the original validation reviews out of training. all is a
production refit with no held-out claim. Neither mode reads blind test data.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
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
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup
from export_phobert_adapter import load_model
from aspect_pair_absa import PairReviews, choose_pairs
from finetune_cpu_cached import make_cache, CachedModel, cached_loader, probabilities
from phobert_absa import (seed_everything, cpu_thread_limit, split_indices,
    optimizer_for, predictions, to_frame)
from transformers import AutoTokenizer
import score_absa as scorer


def train_epoch(model, batches, opt, scheduler, pw, scale, accumulation):
    model.train()
    opt.zero_grad(set_to_none=True)
    total = 0
    started = time.perf_counter()
    for step, (inputs, counts, target) in enumerate(batches):
        logits = model(inputs, counts)
        a = target[:, 3].long()
        losses = F.binary_cross_entropy_with_logits(logits, target[:, :3],
            pos_weight=pw[a], reduction='none')
        loss = (losses * scale[a]).mean()
        first = (step // accumulation) * accumulation
        (loss / min(accumulation, len(batches)-first)).backward()
        if (step+1) % accumulation == 0 or step+1 == len(batches):
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
        total += float(loss.detach())
        if (step+1) % 100 == 0 or step+1 == len(batches):
            print('train', step+1, '/', len(batches), 'loss', round(total/(step+1), 4),
                  'seconds', round(time.perf_counter()-started, 1), flush=True)
    return total / len(batches)


def run(a):
    started = time.perf_counter()
    seed_everything(a.seed)
    torch.set_num_threads(cpu_thread_limit(a.threads))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, model = load_model(a.init_dir)
    assert cfg.get('architecture') == 'aspect_pair'
    assert not cfg.get('production_refit', False) and cfg['seed'] == a.seed
    source = json.loads((Path(a.init_dir) / 'run_info.json').read_text())
    train_path = Path(a.data_dir) / 'train.csv'
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert source['cache_key']['train_sha256'] == train_hash
    train = pd.read_csv(train_path)
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y = train[scorer.COLS].to_numpy(int)
    assert np.isin(y, [0, 1]).all()
    fi, tu, va = split_indices(train, a.seed)
    fit = np.concatenate([fi, tu]) if a.scope == 'outer80' else np.arange(len(train))
    fit_pairs = choose_pairs(fit, y, a.negatives, a.seed)
    val_pairs = [(int(i), k) for i in va for k in range(7)] if a.scope == 'outer80' else []
    pairs = fit_pairs + val_pairs
    pair_y = np.array([list(y[i, k*3:k*3+3])+[k] for i, k in pairs])
    tok = AutoTokenizer.from_pretrained(Path(a.init_dir) / 'tokenizer', use_fast=False)
    depth = 12 - cfg['unfreeze_layers']
    key = {'model': cfg['model_name'], 'revision': cfg['revision'], 'frozen_depth': depth,
        'queries': cfg['queries'], 'max_length': cfg['max_length'], 'stride': cfg['stride'],
        'train_sha256': train_hash, 'scope': a.scope, 'seed': a.seed, 'negatives': a.negatives}
    cache_path = Path(a.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        saved = torch.load(cache_path, map_location='cpu', weights_only=True)
        assert saved['key'] == key
        cache = saved['cache']
        del saved
    else:
        ds = PairReviews(train.text, tok, pairs=pairs, max_length=cfg['max_length'],
            stride=cfg['stride'], queries=cfg['queries'])
        cache = make_cache(model, ds, tok, depth, a.batch_size)
        torch.save({'key': key, 'cache': cache}, cache_path)
        del ds
    wrapped = CachedModel(model, depth)
    batches = cached_loader(cache, np.arange(len(fit_pairs)), pair_y, a.batch_size, True, a.seed)
    # Each aspect-polarity label contributes equal total positive/negative mass.
    counts = np.array([(pair_y[:len(fit_pairs), 3] == k).sum() for k in range(7)])
    positive = np.array([pair_y[:len(fit_pairs), :3][pair_y[:len(fit_pairs), 3] == k].sum(0)
                         for k in range(7)])
    negative = np.maximum(1, counts[:, None] - positive)
    pw = torch.as_tensor(np.clip(negative / np.maximum(1, positive), .3, 30), dtype=torch.float32)
    scale = len(fit_pairs) / (14 * negative)
    scale[:, 0] *= .5
    scale = torch.as_tensor(scale, dtype=torch.float32)
    opt = optimizer_for(model, a.lr)
    steps = math.ceil(len(batches) / a.accumulation) * a.epochs
    scheduler = get_cosine_schedule_with_warmup(opt, int(.08*steps), steps)
    history = []
    for ep in range(1, a.epochs+1):
        ts = time.perf_counter()
        loss = train_epoch(wrapped, batches, opt, scheduler, pw, scale, a.accumulation)
        row = {'epoch': ep, 'loss': loss, 'seconds': time.perf_counter()-ts}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print('EPOCH', json.dumps(row), flush=True)
    cuts = np.asarray(cfg['thresholds'])
    report = None
    if a.scope == 'outer80':
        vb = cached_loader(cache, np.arange(len(fit_pairs), len(pairs)), pair_y, a.batch_size)
        scores = probabilities(wrapped, vb).reshape(len(va), 21)
        report = scorer.report(to_frame(train.id.iloc[va], y[va]),
            to_frame(train.id.iloc[va], predictions(scores, cuts)), n_boot=1000)
        (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        np.savez_compressed(out / 'validation_scores.npz', scores=scores,
                            ids=train.id.iloc[va].to_numpy(dtype=str))
        scorer.show(report)
    # The selection report belongs to the PRE-refit model, kept explicitly separate.
    (out / 'selection_metrics.json').write_bytes((Path(a.init_dir) / 'metrics.json').read_bytes())
    cfg.update(production_refit=a.scope == 'all', refit_scope=a.scope,
        threshold_source='pre-refit out-of-fit tuning predictions; fixed during refit',
        refit_training=vars(a), refit_epochs=a.epochs, fit_rows=len(fit))
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    torch.save(model.state_dict(), out / 'model.pt')
    tok.save_pretrained(out / 'tokenizer')
    info = {'cache_key': key, 'source_model': source, 'threads': torch.get_num_threads(),
        'torch': str(torch.__version__), 'gpu': False, 'fit_rows': len(fit),
        'scope': a.scope, 'epochs_fixed_before_validation': a.epochs,
        'validation_F1': None if report is None else report['ABSA_F1'],
        'total_seconds': time.perf_counter()-started}
    (out / 'run_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print('FINAL', json.dumps(info, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='..')
    p.add_argument('--init-dir', required=True)
    p.add_argument('--out-dir', default='refit_pair_results')
    p.add_argument('--cache', default='private_cache/refit_pair.pt')
    p.add_argument('--scope', choices=['outer80', 'all'], default='outer80')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--negatives', type=int, default=2)
    a = p.parse_args()
    if a.epochs < 1 or a.negatives not in range(8):
        p.error('Invalid epochs or negatives')
    run(a)


if __name__ == '__main__':
    main()
