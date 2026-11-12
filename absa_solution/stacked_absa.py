"""Cross-fitted supervised stacking; train and local predict are separate commands."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from phobert_absa import split_indices, predictions, to_frame
import score_absa as scorer


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_identity(folder):
    folder = Path(folder)
    cfg_path = folder / 'config.json'
    state = folder / 'adapter.pt' if (folder / 'adapter.pt').exists() else folder / 'model.pt'
    return {'config_sha256': file_hash(cfg_path), 'weights_sha256': file_hash(state)}


def feature_arrays(sources, label, mode):
    a = label // 3
    idx = np.arange(21) if mode == 'all_aspects' else np.arange(a * 3, a * 3 + 3)
    p = np.concatenate([s[:, idx] for s in sources], axis=1)
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def best_cut(truth, scores):
    grid = np.linspace(.1, .9, 17)
    fs = np.array([scorer.f1(truth.astype(bool), scores >= c) for c in grid])
    tied = np.flatnonzero(fs == np.nanmax(fs))
    j = tied[np.argmin(abs(grid[tied] - .5))]
    return float(fs[j]), float(grid[j])


def read_scores(folder, name, expected_ids, expected_y):
    folder = Path(folder)
    candidates = ([name + '_scores.npz'] if name != 'tune' else ['tune_scores.npz', 'best_tuning_predictions.npz'])
    if name == 'validation':
        candidates.append('validation_probabilities.npz')
    path = next((folder / c for c in candidates if (folder / c).exists()), None)
    if path is None:
        raise FileNotFoundError(f'Out-of-fit {name} scores missing in {folder}')
    with np.load(path, allow_pickle=False) as obj:
        scores = obj['scores'].copy()
        if 'ids' in obj:
            assert np.array_equal(obj['ids'], expected_ids), f'{path}: review order differs'
        elif 'labels' in obj:
            assert np.array_equal(obj['labels'], expected_y), f'{path}: labels/split differ'
        else:
            raise ValueError('A score file needs review IDs or original split labels')
    assert scores.shape == expected_y.shape and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    return scores


def estimator(c, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver='liblinear',
        random_state=seed, max_iter=1000))


def fit(data_dir, model_dirs, out_dir, seed=2026):
    started = time.perf_counter()
    train_path = Path(data_dir) / 'train.csv'
    train = pd.read_csv(train_path)
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    y = train[scorer.COLS].to_numpy(int)
    assert np.isin(y, [0, 1]).all()
    fi, tu, va = split_indices(train, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tune_sources, val_sources, sources = [], [], []
    train_hash = file_hash(train_path)
    for folder in map(Path, model_dirs):
        cfg = json.loads((folder / 'config.json').read_text())
        assert cfg['seed'] == seed and cfg['columns'] == scorer.COLS
        assert not cfg.get('production_refit', False) and not cfg.get('refit_scope'), \
            'Meta tuning must remain out of every base encoder fit'
        info = json.loads((folder / 'run_info.json').read_text())
        actual_hash = info.get('cache_key', {}).get('train_sha256', info.get('train_sha256'))
        assert actual_hash == train_hash, 'Base encoder training data differs'
        tune_sources.append(read_scores(folder, 'tune', train.id.iloc[tu].to_numpy(dtype=str), y[tu]))
        val_sources.append(read_scores(folder, 'validation', train.id.iloc[va].to_numpy(dtype=str), y[va]))
        sources.append({'path': os.path.relpath(folder.resolve(), out.resolve()),
            'identity': model_identity(folder), 'architecture': cfg.get('architecture', 'joint_21'),
            'pretrained_model': cfg['model_name']})
    label_config, models = [], []
    final = np.zeros((len(va), 21))
    oof_scores = np.zeros((len(tu), 21))
    for label, name in enumerate(scorer.COLS):
        target = y[tu, label]
        assert np.bincount(target, minlength=2).min() >= 5
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(target, target))
        best = None
        for mode in ['own_aspect', 'all_aspects']:
            xt = feature_arrays(tune_sources, label, mode)
            for c in [.01, .1, 1.]:
                oof = np.zeros(len(tu))
                for subfit, held in folds:
                    m = estimator(c, seed).fit(xt[subfit], target[subfit])
                    oof[held] = m.predict_proba(xt[held])[:, 1]
                f1, cut = best_cut(target, oof)
                if best is None or f1 > best[0]:
                    best = f1, cut, mode, c, oof
        f1, cut, mode, c, oof = best
        m = estimator(c, seed).fit(feature_arrays(tune_sources, label, mode), target)
        final[:, label] = m.predict_proba(feature_arrays(val_sources, label, mode))[:, 1]
        oof_scores[:, label] = oof
        models.append(m)
        label_config.append({'label': name, 'feature_mode': mode, 'C': c,
                             'threshold': cut, 'oof_F1': f1})
        print(name, 'meta OOF F1', round(f1, 4), mode, c, flush=True)
    cuts = np.array([c['threshold'] for c in label_config])
    truth = to_frame(train.id.iloc[va], y[va])
    proposed = to_frame(train.id.iloc[va], predictions(final, cuts))
    report = scorer.report(truth, proposed, n_boot=1000)
    cfg = {'architecture': 'supervised_stacking', 'seed': seed, 'columns': scorer.COLS,
        'train_sha256': train_hash, 'sources': sources, 'labels': label_config,
        'base_fit_rows': len(fi), 'meta_fit_rows': len(tu), 'validation_rows': len(va),
        'meta_selection': 'five-fold out-of-fold tuning predictions, never validation',
        'production_refit': False}
    joblib.dump(models, out / 'meta_models.joblib', compress=3)
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    proposed.to_csv(out / 'validation_prediction.csv', index=False)
    np.savez_compressed(out / 'validation_scores.npz', scores=final,
                        ids=train.id.iloc[va].to_numpy(dtype=str))
    info = {'total_seconds': time.perf_counter() - started, 'python': platform.python_version(),
        'gpu': False, 'train_sha256': train_hash,
        'meta_OOF_F1': float(np.mean([c['oof_F1'] for c in label_config if not c['label'].endswith('_nhac')])),
        'validation_F1': report['ABSA_F1'], 'validation_feedback_seen_in_prior_development': True}
    (out / 'run_info.json').write_text(json.dumps(info, indent=2))
    scorer.show(report)
    return report


def predict(model_dir, test_path, output, model_dirs=None, threads=2, batch_size=8, score_cache_dir=None):
    from neural_evaluation import open_model, probabilities_for
    root = Path(model_dir)
    cfg = json.loads((root / 'config.json').read_text())
    assert cfg['architecture'] == 'supervised_stacking' and cfg['columns'] == scorer.COLS
    folders = list(map(Path, model_dirs)) if model_dirs else [root / s['path'] for s in cfg['sources']]
    assert len(folders) == len(cfg['sources'])
    test = pd.read_csv(test_path)
    assert list(test.columns) == ['id', 'text'] and test.id.is_unique and not test.isna().any().any()
    test_hash = file_hash(test_path)
    sources = []
    for i, (folder, source) in enumerate(zip(folders, cfg['sources'])):
        assert model_identity(folder) == source['identity'], 'A base checkpoint differs from the fitted meta model'
        cp = None if score_cache_dir is None else Path(score_cache_dir) / f'model_{i}.npz'
        scores = None
        if cp is not None and cp.exists():
            with np.load(cp, allow_pickle=False) as saved:
                if (str(saved['test_sha256']) == test_hash and
                    str(saved['weights_sha256']) == source['identity']['weights_sha256']):
                    assert np.array_equal(saved['ids'], test.id.to_numpy(dtype=str))
                    scores = saved['scores'].copy()
        if scores is None:
            base_cfg, model, tok = open_model(folder, threads)
            scores = probabilities_for(base_cfg, model, tok, test.text, batch_size)
            del model, tok
            gc.collect()
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cp, scores=scores, ids=test.id.to_numpy(dtype=str),
                    test_sha256=test_hash, weights_sha256=source['identity']['weights_sha256'])
        assert scores.shape == (len(test), 21) and np.isfinite(scores).all()
        sources.append(scores)
        print('Local encoder inference completed:', i + 1, '/', len(folders), flush=True)
    models = joblib.load(root / 'meta_models.joblib')
    assert len(models) == len(cfg['labels']) == 21
    final = np.column_stack([m.predict_proba(feature_arrays(sources, j, c['feature_mode']))[:, 1]
        for j, (m, c) in enumerate(zip(models, cfg['labels']))])
    cuts = np.array([c['threshold'] for c in cfg['labels']])
    to_frame(test.id, predictions(final, cuts)).to_csv(output, index=False)
    assert scorer.kiem_tra(output, test_path) == 0
    print('Submission SHA256:', file_hash(output), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['fit', 'predict'])
    p.add_argument('--data-dir', default='..')
    p.add_argument('--models', nargs='+')
    p.add_argument('--out-dir', default='stacked_results')
    p.add_argument('--model-dir', default='stacked_results')
    p.add_argument('--test', default='../test_blind.csv')
    p.add_argument('--output', default='submission.csv')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--score-cache-dir', help='Private local prediction cache; never exported automatically')
    a = p.parse_args()
    if a.command == 'fit':
        if not a.models:
            p.error('fit needs --models with the base model directories')
        fit(a.data_dir, a.models, a.out_dir, a.seed)
    else:
        predict(a.model_dir, a.test, a.output, a.models, a.threads, a.batch_size, a.score_cache_dir)


if __name__ == '__main__':
    main()
