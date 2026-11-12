"""Fixed-schedule all-train refit, retaining the selection report separately."""
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from export_phobert_adapter import construct_model
from finetune_cpu_cached import make_cache, CachedModel, cached_loader, epoch
from phobert_absa import (seed_everything, cpu_thread_limit, Reviews,
                         optimizer_for, MODEL, REVISION)
from stacked_absa import file_hash, model_identity
import score_absa as scorer


def run(a):
    started = time.perf_counter()
    src, out = Path(a.init_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / 'config.json').read_text())
    assert not cfg.get('production_refit', False) and not cfg.get('refit_scope')
    assert cfg['model_name'] == MODEL and cfg['revision'] == REVISION
    assert cfg.get('architecture', 'joint_21') in ['joint_21', 'label_attention']
    assert cfg['columns'] == scorer.COLS and cfg['unfreeze_layers'] == 4
    source_identity = model_identity(src)
    info = json.loads((src / 'run_info.json').read_text())
    train_path = Path(a.data_dir) / 'train.csv'
    train_hash = file_hash(train_path)
    assert info.get('cache_key', {}).get('train_sha256', info.get('train_sha256')) == train_hash
    seed_everything(cfg['seed'])
    torch.set_num_threads(cpu_thread_limit(a.threads))
    train = pd.read_csv(train_path)
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y = train[scorer.COLS].to_numpy(int)
    assert np.isin(y, [0, 1]).all()
    model = construct_model(cfg)
    state_path = src / 'adapter.pt' if (src / 'adapter.pt').exists() else src / 'model.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    base = model.state_dict()
    assert all(torch.equal(v, base[n]) for n, v in state.items() if n in frozen)
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys and set(result.missing_keys) <= frozen
    del state, base
    tok = AutoTokenizer.from_pretrained(src / 'tokenizer', use_fast=False)
    depth = len(model.encoder.encoder.layer) - cfg['unfreeze_layers']
    key = {'model': MODEL, 'revision': REVISION, 'frozen_depth': depth,
        'max_length': cfg['max_length'], 'stride': cfg['stride'], 'train_sha256': train_hash}
    cp = Path(a.cache)
    cp.parent.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert saved['key'] == key
        cache = saved['cache']
        del saved
    else:
        ds = Reviews(train.text, tok, y, cfg['max_length'], cfg['stride'])
        cache = make_cache(model, ds, tok, depth, a.batch_size)
        torch.save({'key': key, 'cache': cache}, cp)
        del ds
    if cfg.get('architecture') == 'label_attention':
        from label_attention_absa import CachedAttentionModel
        wrapped = CachedAttentionModel(model, depth)
    else:
        wrapped = CachedModel(model, depth)
    batches = cached_loader(cache, np.arange(len(train)), y, a.batch_size, True, cfg['seed'])
    positives = np.maximum(1, y.sum(0))
    negatives = np.maximum(1, len(train) - y.sum(0))
    pw = torch.as_tensor(np.clip(negatives / positives, .3, 30), dtype=torch.float32)
    scale = len(train) / (2 * negatives)
    scale *= np.array([.5 if c.endswith('_nhac') else 1 for c in scorer.COLS])
    scale = torch.as_tensor(scale, dtype=torch.float32)
    opt = optimizer_for(model, a.lr)
    steps = math.ceil(len(batches) / a.accumulation) * a.epochs
    scheduler = get_cosine_schedule_with_warmup(opt, int(.08*steps), steps)
    history = []
    for ep in range(1, a.epochs+1):
        ts = time.perf_counter()
        loss = epoch(wrapped, batches, opt, scheduler, pw, a.accumulation, 100, scale)
        row = {'epoch': ep, 'loss': loss, 'seconds': time.perf_counter()-ts}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print('EPOCH', json.dumps(row), flush=True)
    cfg.update(production_refit=True, refit_scope='all', fit_rows=len(train),
        refit_epochs=a.epochs, refit_training=vars(a), selection_identity=source_identity,
        threshold_source='fixed pre-refit out-of-fit tuning thresholds')
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    torch.save({n: t for n, t in model.state_dict().items() if n in names}, out / 'adapter.pt')
    tok.save_pretrained(out / 'tokenizer')
    (out / 'selection_metrics.json').write_bytes((src / 'metrics.json').read_bytes())
    result_info = {'train_sha256': train_hash, 'cache_key': key, 'source_model': info,
        'selection_identity': source_identity, 'total_seconds': time.perf_counter()-started,
        'torch': str(torch.__version__), 'threads': torch.get_num_threads(), 'gpu': False,
        'fit_rows': len(train), 'validation_F1': None, 'production_refit': True,
        'epochs_fixed_before_refit': a.epochs}
    (out / 'run_info.json').write_text(json.dumps(result_info, ensure_ascii=False, indent=2))
    print('FINAL', json.dumps(result_info), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='..')
    p.add_argument('--init-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--cache', default='private_cache/phobert_frozen8_160.pt')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    a = p.parse_args()
    if min(a.epochs, a.threads, a.batch_size, a.accumulation) < 1:
        p.error('Epochs, threads, batch size and accumulation must be positive')
    run(a)


if __name__ == '__main__':
    main()
