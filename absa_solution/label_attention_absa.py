"""Train label-specific token attention on the original review-level labels.

Only train.csv is read. The lower eight frozen PhoBERT layers share the exact
cache used by the global classifier; all label attention is learned on fit.
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from phobert_absa import (MODEL, REVISION, PhoBERTABSA, Reviews, loader,
    split_indices, seed_everything, cpu_thread_limit, optimizer_for,
    tune_thresholds, predictions, main_score, to_frame)
from finetune_cpu_cached import (attention, make_cache, cached_loader,
    probabilities, epoch)
import score_absa as scorer


class LabelAttentionHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.global_head = nn.Sequential(nn.Dropout(.2), nn.Linear(hidden, hidden),
            nn.Tanh(), nn.Dropout(.2), nn.Linear(hidden, 21))
        self.queries = nn.Parameter(torch.empty(21, hidden))
        nn.init.normal_(self.queries, std=.02)
        self.query_norm = nn.LayerNorm(hidden)
        self.token_norm = nn.LayerNorm(hidden)
        self.attend = nn.MultiheadAttention(hidden, 8, dropout=.1, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden)
        self.readout = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                    nn.Dropout(.2), nn.Linear(hidden // 2, 1))
        self.mix = nn.Parameter(torch.full((21,), .1))

    def forward(self, hidden, mask):
        q = self.query_norm(hidden[:, :1] + self.queries[None])
        tokens = self.token_norm(hidden)
        context, _ = self.attend(q, tokens, tokens,
            key_padding_mask=~mask.bool(), need_weights=False)
        local = self.readout(self.output_norm(q + context)).squeeze(-1)
        return self.global_head(hidden[:, 0]) + self.mix * local


class LabelAttentionModel(PhoBERTABSA):
    def __init__(self, model_name=MODEL, revision=REVISION, unfreeze_layers=4):
        super().__init__(model_name, revision, unfreeze_layers)
        self.head = LabelAttentionHead(self.encoder.config.hidden_size)

    def forward(self, inputs, counts):
        hidden = self.encoder(**inputs).last_hidden_state
        logits = self.head(hidden, inputs['attention_mask'])
        return torch.stack([p.max(0).values for p in logits.split(counts)])


class CachedAttentionModel(nn.Module):
    def __init__(self, model, depth):
        super().__init__()
        self.model, self.depth = model, depth

    def forward(self, inputs, counts):
        h = inputs['hidden']
        mask = attention(inputs['attention_mask'], h.dtype)
        for layer in self.model.encoder.encoder.layer[self.depth:]:
            h = layer(h, attention_mask=mask)[0]
        logits = self.model.head(h, inputs['attention_mask'])
        return torch.stack([p.max(0).values for p in logits.split(counts)])


def initialize_from_global(model, folder, seed, train_hash):
    folder = Path(folder)
    cfg = json.loads((folder / 'config.json').read_text())
    assert not cfg.get('production_refit', False)
    assert cfg['seed'] == seed and cfg.get('architecture', 'joint_21') == 'joint_21'
    info = json.loads((folder / 'run_info.json').read_text())
    assert info['cache_key']['train_sha256'] == train_hash
    state = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    base = model.state_dict()
    assert all(torch.equal(state[n], base[n]) for n in frozen)
    mapped = {('head.global_head.' + n[5:] if n.startswith('head.') else n): v
              for n, v in state.items()}
    result = model.load_state_dict(mapped, strict=False)
    assert not result.unexpected_keys
    assert all(n.startswith('head.') and not n.startswith('head.global_head.')
               for n in result.missing_keys)


def run(a):
    started = time.perf_counter()
    seed_everything(a.seed)
    torch.set_num_threads(cpu_thread_limit(a.threads))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = Path(a.data_dir) / 'train.csv'
    train = pd.read_csv(train_path)
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y = train[scorer.COLS].to_numpy(int)
    assert np.isin(y, [0, 1]).all()
    fi, tu, va = split_indices(train, a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=False)
    model = LabelAttentionModel(unfreeze_layers=a.layers)
    depth = len(model.encoder.encoder.layer) - a.layers
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    key = {'model': MODEL, 'revision': REVISION, 'frozen_depth': depth,
           'max_length': a.max_length, 'stride': a.stride, 'train_sha256': train_hash}
    cp = Path(a.cache)
    cp.parent.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert saved['key'] == key
        cache = saved['cache']
        del saved
    else:
        ds = Reviews(train.text, tok, y, a.max_length, a.stride)
        cache = make_cache(model, ds, tok, depth, a.batch_size)
        torch.save({'key': key, 'cache': cache}, cp)
        del ds
    if a.init_dir:
        initialize_from_global(model, a.init_dir, a.seed, train_hash)
    wrapped = CachedAttentionModel(model, depth)
    check = np.arange(min(4, len(train)))
    ds = Reviews(train.text.iloc[check], tok, y[check], a.max_length, a.stride)
    raw, fast = loader(ds, tok, a.batch_size), cached_loader(cache, check, y, a.batch_size)
    model.eval()
    with torch.no_grad():
        i, counts, _ = next(iter(raw))
        c, cc, _ = next(iter(fast))
        torch.testing.assert_close(model(i, counts), wrapped(c, cc), atol=2e-5, rtol=2e-5)
        # Attention must ignore every padding token, even with arbitrary values.
        h = torch.randn(2, 9, model.encoder.config.hidden_size)
        mask = torch.tensor([[1]*5 + [0]*4, [1]*9])
        changed = h.clone()
        changed[0, 5:] = 100 * torch.randn_like(changed[0, 5:])
        torch.testing.assert_close(model.head(h, mask), model.head(changed, mask))
    print('Raw/cache logits and padding-mask invariance passed.', flush=True)
    fit = cached_loader(cache, fi, y, a.batch_size, True, a.seed)
    tuning = cached_loader(cache, tu, y, a.batch_size)
    validation = cached_loader(cache, va, y, a.batch_size)
    opt = optimizer_for(model, a.lr)
    steps = math.ceil(len(fit) / a.accumulation) * a.epochs
    scheduler = get_cosine_schedule_with_warmup(opt, int(.08 * steps), steps)
    positives = np.maximum(1, y[fi].sum(0))
    negatives = np.maximum(1, len(fi) - y[fi].sum(0))
    pw = torch.as_tensor(np.clip(negatives / positives, .3, 30), dtype=torch.float32)
    scale = len(fi) / (2 * negatives)
    scale *= np.array([.5 if c.endswith('_nhac') else 1 for c in scorer.COLS])
    weights = torch.as_tensor(scale, dtype=torch.float32)
    initial = probabilities(wrapped, tuning)
    cuts = tune_thresholds(y[tu], initial)
    best = main_score(y[tu], predictions(initial, cuts))
    best_epoch, history = 0, []
    torch.save({'state_dict': model.state_dict(), 'cuts': cuts.tolist()}, out / 'best.pt')
    np.savez_compressed(out / 'tune_scores.npz', scores=initial,
                        ids=train.id.iloc[tu].to_numpy(dtype=str))
    print('Initial tune F1:', best, flush=True)
    for ep in range(1, a.epochs + 1):
        ts = time.perf_counter()
        loss = epoch(wrapped, fit, opt, scheduler, pw, a.accumulation, 50, weights)
        scores = probabilities(wrapped, tuning)
        cuts = tune_thresholds(y[tu], scores)
        f1 = main_score(y[tu], predictions(scores, cuts))
        history.append({'epoch': ep, 'loss': loss, 'tune_F1': f1,
                        'seconds': time.perf_counter() - ts})
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print('EPOCH', json.dumps(history[-1]), flush=True)
        if f1 > best:
            best, best_epoch = f1, ep
            torch.save({'state_dict': model.state_dict(), 'cuts': cuts.tolist()}, out / 'best.pt')
            np.savez_compressed(out / 'tune_scores.npz', scores=scores,
                                ids=train.id.iloc[tu].to_numpy(dtype=str))
        if ep - best_epoch >= a.patience:
            break
    selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['state_dict'])
    cuts = np.asarray(selected['cuts'])
    del selected
    scores = probabilities(wrapped, validation)
    report = scorer.report(to_frame(train.id.iloc[va], y[va]),
        to_frame(train.id.iloc[va], predictions(scores, cuts)), n_boot=1000)
    scorer.show(report)
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    np.savez_compressed(out / 'validation_scores.npz', scores=scores,
                        ids=train.id.iloc[va].to_numpy(dtype=str))
    cfg = {'architecture': 'label_attention', 'model_name': MODEL, 'revision': REVISION,
        'unfreeze_layers': a.layers, 'seed': a.seed, 'max_length': a.max_length,
        'stride': a.stride, 'thresholds': cuts.tolist(), 'columns': scorer.COLS,
        'best_epoch': best_epoch, 'production_refit': False, 'training': vars(a)}
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    torch.save(model.state_dict(), out / 'model.pt')
    tok.save_pretrained(out / 'tokenizer')
    info = {'best_tune_F1': best, 'best_epoch': best_epoch, 'validation_F1': report['ABSA_F1'],
        'total_seconds': time.perf_counter() - started, 'cache_key': key,
        'torch': str(torch.__version__), 'threads': torch.get_num_threads(), 'gpu': False,
        'fit_rows': len(fi), 'tune_rows': len(tu), 'validation_rows': len(va)}
    (out / 'run_info.json').write_text(json.dumps(info, indent=2))
    (out / 'best.pt').unlink()
    print('FINAL', json.dumps(info), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='..')
    p.add_argument('--out-dir', default='label_attention_results')
    p.add_argument('--cache', default='private_cache/phobert_frozen8_160.pt')
    p.add_argument('--init-dir')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max-length', type=int, default=160)
    p.add_argument('--stride', type=int, default=32)
    run(p.parse_args())


if __name__ == '__main__':
    main()
