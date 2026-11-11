"""Reproducible supervised ABSA; never calls an LLM or external prediction API."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import platform
import re
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from threadpoolctl import threadpool_limits
import score_absa as scorer

SEED = 2026

class NBSVM(ClassifierMixin, BaseEstimator):
    """Label-specific supervised log-count ratio, fitted on training rows only."""
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        positive = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1
        negative = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1
        self.ratio_ = np.log((positive/positive.sum())/(negative/negative.sum()))
        self.classifier_ = LinearSVC(C=self.C, dual='auto', random_state=SEED, max_iter=10000)
        self.classifier_.fit(X.multiply(self.ratio_), y)
        return self

    def decision_function(self, X):
        return self.classifier_.decision_function(X.multiply(self.ratio_))

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(text)).lower()).strip()

def vectorizer():
    return FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                max_features=70000, dtype=np.float64)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3,
                                sublinear_tf=True, max_features=100000, dtype=np.float64)),
    ])

def estimator(name):
    if name == 'nbsvm':
        return OneVsRestClassifier(NBSVM(C=1), n_jobs=1)
    base = (LogisticRegression(C=4, solver='liblinear', max_iter=2000, random_state=SEED)
            if name == 'logistic' else LinearSVC(C=1, dual='auto', random_state=SEED, max_iter=10000))
    return OneVsRestClassifier(base, n_jobs=1)

def thresholds(y, scores):
    """Select on tuning partition only; a coarse grid reduces threshold overfitting."""
    result = []
    for j in range(y.shape[1]):
        candidates = np.linspace(-1.4, 0.2, 17)
        values = np.array([scorer.f1(y[:, j].astype(bool), scores[:, j] >= t) for t in candidates])
        best = np.flatnonzero(values == np.nanmax(values))
        # Tie break closest to the natural decision boundary.
        result.append(float(candidates[best[np.argmin(abs(candidates[best]))]]))
    return np.array(result)

def constrained(scores, cuts):
    pred = (scores >= cuts).astype(int)
    for k in range(7):
        pred[:, k*3] |= pred[:, k*3+1] | pred[:, k*3+2]
    return pred

def frame(ids, pred):
    out = pd.DataFrame(pred, columns=scorer.COLS)
    out.insert(0, 'id', np.asarray(ids))
    return out

def run(data_dir, out_dir):
    started = time.perf_counter()
    data_dir, out = Path(data_dir).resolve(), Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_dir/'train.csv')
    assert list(train.columns) == ['id', 'text'] + scorer.COLS
    assert train.id.is_unique and not train.isna().any().any()
    assert train[scorer.COLS].isin([0, 1]).all().all()
    texts = train.text.map(normalize)
    y = train[scorer.COLS].to_numpy(int)
    groups = texts.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    # Normalized identical texts stay together. Fit/tune/validation ~60/20/20.
    train_tune, val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                           .split(texts, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED+1)
                .split(texts.iloc[train_tune], groups=groups.iloc[train_tune]))
    fit, tune = train_tune[a], train_tune[b]
    assert not set(groups.iloc[fit]) & set(groups.iloc[tune])
    assert not set(groups.iloc[train_tune]) & set(groups.iloc[val])
    assert (y[fit].sum(0) > 0).all() and (y[tune].sum(0) > 0).all() and (y[val].sum(0) > 0).all()
    pd.DataFrame({'id': train.id, 'partition': ['fit' if i in set(fit) else 'tune' if i in set(tune) else 'validation' for i in range(len(train))]}).to_csv(out/'split.csv', index=False)
    prevalence = pd.DataFrame({'label': scorer.COLS, 'fit_positive': y[fit].sum(0),
                               'tune_positive': y[tune].sum(0), 'val_positive': y[val].sum(0)})
    prevalence.to_csv(out/'label_counts.csv', index=False)
    print('Partitions:', len(fit), len(tune), len(val), flush=True)
    v = vectorizer()
    ts = time.perf_counter()
    xf = v.fit_transform(texts.iloc[fit])
    xt, xv = v.transform(texts.iloc[tune]), v.transform(texts.iloc[val])
    feature_seconds = time.perf_counter()-ts
    print('Features:', xf.shape[1], 'seconds:', round(feature_seconds, 2), flush=True)
    rows, reports, fitted = [], {}, {}
    for name in ['logistic', 'svm', 'nbsvm']:
        model = estimator(name)
        ts = time.perf_counter()
        model.fit(xf, y[fit])
        fit_seconds = time.perf_counter()-ts
        st = model.decision_function(xt)
        cuts = thresholds(y[tune], st)
        tuning_pred = constrained(st, cuts)
        tune_score = scorer.core_score(scorer.to_arrays(frame(train.id.iloc[tune], y[tune]))[1],
                                       scorer.to_arrays(frame(train.id.iloc[tune], tuning_pred))[1])
        ts = time.perf_counter()
        pv = constrained(model.decision_function(xv), cuts)
        infer_seconds = time.perf_counter()-ts
        gt, pred = frame(train.id.iloc[val], y[val]), frame(train.id.iloc[val], pv)
        r = scorer.report(gt, pred, n_boot=1000)
        reports[name] = r
        rows.append({'model': name, 'tune_F1': tune_score, 'validation_F1': r['ABSA_F1'],
                     'precision': r['precision_TB'], 'recall': r['recall_TB'],
                     'accuracy': r['accuracy_TB'], 'kappa': r['kappa_TB'],
                     'MCC': r['mcc_TB'], 'balanced_accuracy': r['bal_acc_TB'],
                     'CI95_low': r['ABSA_F1_CI95'][0], 'CI95_high': r['ABSA_F1_CI95'][1],
                     'fit_seconds': fit_seconds, 'validation_predict_seconds': infer_seconds})
        gt.to_csv(out/'validation_truth.csv', index=False)
        pred.to_csv(out/f'validation_{name}.csv', index=False)
        fitted[name] = (cuts, pv)
        print(name, 'tune F1:', round(tune_score, 4), 'validation F1:', round(r['ABSA_F1'], 4), flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out/'comparison.csv', index=False)
    # Choose by tuning data, not validation. Validation is only reported.
    selected = max(rows, key=lambda row: row['tune_F1'])['model']
    cuts, pv = fitted[selected]
    report = reports[selected]
    (out/'metrics.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'thresholds.json').write_text(json.dumps(dict(zip(scorer.COLS, cuts.tolist())), indent=2), encoding='utf-8')
    err = []
    for i, ix in enumerate(val):
        wrong = np.flatnonzero(pv[i] != y[ix])
        if len(wrong):
            err.append({'id': str(train.id.iloc[ix]), 'text': train.text.iloc[ix],
                        'errors': {scorer.COLS[j]: {'truth': int(y[ix,j]), 'prediction': int(pv[i,j])} for j in wrong}})
    (out/'validation_errors.json').write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Selected by tuning:', selected, flush=True)
    scorer.show(report)
    # Fresh vectorizer and classifier fit on all labeled training rows.
    ts = time.perf_counter()
    final_v = vectorizer()
    full_x = final_v.fit_transform(texts)
    final_m = estimator(selected).fit(full_x, y)
    final_fit_seconds = time.perf_counter()-ts
    bundle = {'vectorizer': final_v, 'model': final_m, 'thresholds': cuts,
              'columns': scorer.COLS, 'seed': SEED, 'selected': selected}
    joblib.dump(bundle, out/'model.joblib', compress=3)
    # Test text is consumed by classical ML locally, never printed or sent to an LLM.
    test = pd.read_csv(data_dir/'test_blind.csv')
    assert list(test.columns) == ['id', 'text'] and test.id.is_unique and not test.isna().any().any()
    ts = time.perf_counter()
    test_x = final_v.transform(test.text.map(normalize))
    test_pred = constrained(final_m.decision_function(test_x), cuts)
    test_seconds = time.perf_counter()-ts
    result = frame(test.id, test_pred)
    assert list(result.columns) == list(pd.read_csv(data_dir/'submission_demo.csv', nrows=0).columns)
    result.to_csv(out/'submission.csv', index=False, encoding='utf-8')
    assert scorer.kiem_tra(out/'submission.csv', data_dir/'test_blind.csv') == 0
    # Reload saved model and check exact prediction parity.
    reload_m = joblib.load(out/'model.joblib')
    assert np.array_equal(constrained(reload_m['model'].decision_function(
        reload_m['vectorizer'].transform(test.text.map(normalize))), reload_m['thresholds']), test_pred)
    hardware = {'python': platform.python_version(), 'os': platform.platform(),
                'cpu': next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), platform.processor()), 'logical_cpu': os.cpu_count(),
                'ram': next((s.strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemTotal:')), ''),
                'gpu': 'Not used; CPU only', 'feature_seconds': feature_seconds,
                'full_fit_seconds': final_fit_seconds, 'test_predict_seconds': test_seconds,
                'total_seconds': time.perf_counter()-started, 'selected': selected,
                'seed': SEED, 'train_rows': len(train), 'test_rows': len(test),
                'source_commit': '7c052a71010eac9361ea8803b9e8bdd6cfc99be8',
                'sha256': {name: hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
                           for name in ['train.csv', 'test_blind.csv', 'submission_demo.csv', 'score_absa.py']},
                'submission_sha256': hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()}
    (out/'run_info.json').write_text(json.dumps(hardware, indent=2), encoding='utf-8')
    print('Saved model reload: predictions match exactly.', flush=True)
    print(json.dumps(hardware, indent=2), flush=True)
    return comparison, report

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='results')
    args = p.parse_args()
    # Persist custom estimator with an importable module name, also for CLI runs.
    from pipeline import run as module_run
    with threadpool_limits(limits=1):
        module_run(args.data_dir, args.out_dir)
