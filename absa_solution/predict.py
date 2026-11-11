"""Offline inference. Test texts are never printed."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import normalize, constrained, frame
import score_absa

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='results/model.joblib')
    p.add_argument('--test', default='data/test_blind.csv')
    p.add_argument('--output', default='submission_reproduced.csv')
    args = p.parse_args()
    bundle = joblib.load(args.model)
    test = pd.read_csv(args.test)
    with threadpool_limits(limits=1):
        x = bundle['vectorizer'].transform(test.text.map(normalize))
        pred = constrained(bundle['model'].decision_function(x), bundle['thresholds'])
    frame(test.id, pred).to_csv(args.output, index=False, encoding='utf-8')
    assert score_absa.kiem_tra(args.output, args.test) == 0
    print('Prediction saved:', Path(args.output))

if __name__ == '__main__':
    main()
