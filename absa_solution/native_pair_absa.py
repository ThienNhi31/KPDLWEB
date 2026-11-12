"""Aspect-conditioned mDeBERTa trained only on the labeled training partition."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from aspect_pair_absa import PairModel, PairReviews, QUERIES, choose_pairs
from phobert_absa import (seed_everything, cpu_thread_limit, split_indices,
    loader, infer, optimizer_for, train_epoch, tune_thresholds,
    predictions, main_score, to_frame)
import score_absa as scorer

NATIVE_MODEL = 'microsoft/mdeberta-v3-base'
NATIVE_REVISION = 'a0484667b22365f84929a935b5e50a51f71f159d'


def run(a):
    started = time.perf_counter()
    seed_everything(a.seed)
    torch.set_num_threads(cpu_thread_limit(a.threads))
    device = torch.device('cpu')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = Path(a.data_dir) / 'train.csv'
    train = pd.read_csv(train_path)
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y = train[scorer.COLS].to_numpy(int)
    assert np.isin(y, [0, 1]).all()
    fi, tu, va = split_indices(train, a.seed)
    pairs = [choose_pairs(fi, y, a.negatives, a.seed),
             [(int(i), k) for i in tu for k in range(7)],
             [(int(i), k) for i in va for k in range(7)]]
    tok = AutoTokenizer.from_pretrained(a.model_name, revision=a.revision,
        use_fast=True, trust_remote_code=False)
    datasets = [PairReviews(train.text, tok, y, p, a.max_length, a.stride,
                queries=QUERIES, segment=False) for p in pairs]
    batches = [loader(ds, tok, a.batch_size, j == 0, a.seed) for j, ds in enumerate(datasets)]
    model = PairModel(a.model_name, a.revision, a.layers).to(device)
    assert model.encoder.config.model_type == 'deberta-v2'
    print('Original review split:', len(fi), len(tu), len(va),
          'pair counts:', [len(p) for p in pairs], flush=True)
    if a.smoke:
        inputs, counts, labels = next(iter(batches[0]))
        logits = model(inputs, counts)
        assert logits.shape == labels.shape and logits.shape[1] == 3
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.head.parameters())
        assert not any(p.grad is not None for p in model.encoder.embeddings.parameters())
        print('Native tokenizer paired backward passed; pretrained embeddings remain frozen.', flush=True)
        return
    opt = optimizer_for(model, a.lr)
    updates = math.ceil(len(batches[0]) / a.accumulation) * a.epochs
    scheduler = get_cosine_schedule_with_warmup(opt, int(.1 * updates), updates)
    fit_y = datasets[0].labels.numpy()
    positive = np.maximum(1, fit_y.sum(0))
    pw = torch.as_tensor(np.clip(np.sqrt((len(fit_y)-positive)/positive), 1, 3), dtype=torch.float32)
    history, best, best_epoch = [], -1, 0
    for ep in range(1, a.epochs+1):
        ts = time.perf_counter()
        loss = train_epoch(model, batches[0], opt, scheduler, device, pw, a.accumulation, 100)
        scores = infer(model, batches[1], device).reshape(len(tu), 21)
        cuts = tune_thresholds(y[tu], scores)
        f1 = main_score(y[tu], predictions(scores, cuts))
        row = {'epoch': ep, 'loss': loss, 'tune_F1': f1, 'seconds': time.perf_counter()-ts}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print('EPOCH', json.dumps(row), flush=True)
        if f1 > best:
            best, best_epoch = f1, ep
            torch.save({'state_dict': model.state_dict(), 'thresholds': cuts.tolist()}, out / 'best.pt')
            np.savez_compressed(out / 'tune_scores.npz', scores=scores,
                                ids=train.id.iloc[tu].to_numpy(dtype=str))
        if ep-best_epoch >= a.patience:
            break
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    cuts = np.asarray(saved['thresholds'])
    del saved
    scores = infer(model, batches[2], device).reshape(len(va), 21)
    report = scorer.report(to_frame(train.id.iloc[va], y[va]),
        to_frame(train.id.iloc[va], predictions(scores, cuts)), n_boot=1000)
    scorer.show(report)
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    np.savez_compressed(out / 'validation_scores.npz', scores=scores,
                        ids=train.id.iloc[va].to_numpy(dtype=str))
    cfg = {'architecture': 'aspect_pair', 'model_name': a.model_name, 'revision': a.revision,
        'unfreeze_layers': a.layers, 'seed': a.seed, 'max_length': a.max_length, 'stride': a.stride,
        'queries': QUERIES, 'segment': False, 'tokenizer_fast': True, 'columns': scorer.COLS,
        'thresholds': cuts.tolist(), 'best_epoch': best_epoch, 'production_refit': False,
        'training': vars(a)}
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    torch.save(model.state_dict(), out / 'model.pt')
    tok.save_pretrained(out / 'tokenizer')
    info = {'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'total_seconds': time.perf_counter()-started, 'best_tune_F1': best, 'best_epoch': best_epoch,
        'validation_F1': report['ABSA_F1'], 'threads': torch.get_num_threads(), 'gpu': False,
        'torch': str(torch.__version__), 'python': platform.python_version(),
        'cpu_count': os.cpu_count(), 'os': platform.platform(), 'fit_rows': len(fi),
        'tune_rows': len(tu), 'validation_rows': len(va)}
    (out / 'run_info.json').write_text(json.dumps(info, indent=2))
    (out / 'best.pt').unlink()
    print('FINAL', json.dumps(info), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='..')
    p.add_argument('--out-dir', default='native_pair_results')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=160)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--negatives', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--model-name',default=NATIVE_MODEL)
    p.add_argument('--revision',default=NATIVE_REVISION)
    a = p.parse_args()
    if min(a.epochs, a.patience, a.layers, a.threads, a.batch_size, a.accumulation) < 1:
        p.error('Positive epoch/layer/thread/batch/accumulation values required')
    run(a)


if __name__ == '__main__':
    main()
