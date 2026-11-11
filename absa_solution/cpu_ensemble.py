"""CPU Vietnamese segmentation + supervised NB/linear ensembles; offline test inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit
from pyvi import ViTokenizer
from threadpoolctl import threadpool_limits
from pipeline import normalize, constrained, frame, thresholds
import score_absa as scorer

SEED=2026

def preprocess(text):
    # General spelling/slang mapping, fixed before reading any test review.
    text=normalize(text)
    mapping={'nv':'nhân viên','ks':'khách sạn','ko':'không','k':'không',
             'kh':'không','hok':'không','kg':'không','dc':'được','đc':'được',
             'dthuong':'dễ thương','bf':'bữa sáng','buffe':'buffet','vs':'với'}
    for a,b in mapping.items():
        text=re.sub(r'(?<!\w)'+re.escape(a)+r'(?!\w)',b,text)
    return ViTokenizer.tokenize(text)

class NBLinear(ClassifierMixin,BaseEstimator):
    def __init__(self,C=1,alpha=1,kind='svm'):
        self.C=C;self.alpha=alpha;self.kind=kind

    def fit(self,X,y):
        self.classes_=np.array([0,1])
        p=np.asarray(X[y==1].sum(0)).ravel()+1
        n=np.asarray(X[y==0].sum(0)).ravel()+1
        ratio=np.log((p/p.sum())/(n/n.sum()))
        self.ratio_=np.sign(ratio)*np.abs(ratio)**self.alpha
        if self.kind=='logistic':
            self.model_=LogisticRegression(C=self.C,solver='liblinear',max_iter=2000,random_state=SEED)
        else:
            self.model_=LinearSVC(C=self.C,dual='auto',max_iter=10000,random_state=SEED)
        self.model_.fit(X.multiply(self.ratio_),y)
        return self

    def decision_function(self,X):
        return self.model_.decision_function(X.multiply(self.ratio_))

    def predict(self,X):return (self.decision_function(X)>=0).astype(int)

def make_features():
    return FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True,max_features=90000)),
        ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,6),min_df=3,sublinear_tf=True,max_features=140000))])

def make_model(spec):
    kind,C,alpha=spec
    if kind=='linear':
        base=LinearSVC(C=C,dual='auto',max_iter=10000,random_state=SEED)
    else:base=NBLinear(C=C,alpha=alpha,kind=kind)
    return OneVsRestClassifier(base,n_jobs=1)

def grouped_split(train):
    groups=train.text.map(lambda t:hashlib.sha256(normalize(t).encode()).hexdigest())
    tt,val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(train,groups=groups))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED+1).split(train.iloc[tt],groups=groups.iloc[tt]))
    return tt[a],tt[b],val

def score(y,p):return scorer.core_score(scorer.to_arrays(frame(range(len(y)),y))[1],scorer.to_arrays(frame(range(len(y)),p))[1])

def fit(data_dir,out_dir):
    started=time.perf_counter();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(Path(data_dir)/'train.csv')
    assert train.id.is_unique and not train.isna().any().any()
    assert list(train.columns)==['id','text']+scorer.COLS and train[scorer.COLS].isin([0,1]).all().all()
    y=train[scorer.COLS].to_numpy(int)
    texts=train.text.map(preprocess)
    fit_idx,tune_idx,val_idx=grouped_split(train)
    print('Fit/tune/validation:',len(fit_idx),len(tune_idx),len(val_idx),flush=True)
    vector=make_features();xf=vector.fit_transform(texts.iloc[fit_idx]);xt=vector.transform(texts.iloc[tune_idx])
    print('Features:',xf.shape[1],flush=True)
    specs=[(kind,C,alpha) for kind in ['svm','logistic'] for C in [.25,1,4,16] for alpha in [1]]
    specs += [('svm',1,.5),('svm',4,.5),('linear',1,1),('linear',4,1)]
    results=[];fitted=[]
    for spec in specs:
        ts=time.perf_counter();model=make_model(spec).fit(xf,y[fit_idx])
        raw=model.decision_function(xt)
        # Logistic margins have different scale: sigmoid maps them to common probabilities.
        probs=expit(raw if spec[0]=='logistic' else 3*raw)
        cut=choose_cuts(y[tune_idx],probs)
        f1=score(y[tune_idx],constrained(probs,cut))
        results.append({'spec':list(spec),'tune_F1':f1,'fit_seconds':time.perf_counter()-ts})
        fitted.append((model,probs,cut))
        print(spec,'tune',round(f1,4),flush=True)
    # Diversity: top NB-SVM, NB-logistic and original linear SVM.
    best_by_kind=[max([i for i,s in enumerate(specs) if s[0]==kind],key=lambda i:results[i]['tune_F1']) for kind in ['svm','logistic','linear']]
    single=max(range(len(specs)),key=lambda i:results[i]['tune_F1'])
    selection=[single];best=results[single]['tune_F1'];cuts=fitted[single][2]
    ensembles=[best_by_kind[:2],best_by_kind,[best_by_kind[0],best_by_kind[2]]]
    for members in ensembles:
        probs=np.mean([fitted[i][1] for i in members],axis=0)
        c=choose_cuts(y[tune_idx],probs);s=score(y[tune_idx],constrained(probs,c))
        print('Ensemble',members,'tune',round(s,4),flush=True)
        results.append({'ensemble':members,'tune_F1':s})
        if s>best:best=s;selection=members;cuts=c
    # One validation evaluation after all choices were made on tuning.
    xv=vector.transform(texts.iloc[val_idx])
    models=[fitted[i][0] for i in selection];selected_specs=[specs[i] for i in selection]
    ts=time.perf_counter()
    probs=np.mean([expit(m.decision_function(xv)*(1 if s[0]=='logistic' else 3)) for m,s in zip(models,selected_specs)],axis=0)
    vp=constrained(probs,cuts)
    predict_seconds=time.perf_counter()-ts
    gt,pred=frame(train.id.iloc[val_idx],y[val_idx]),frame(train.id.iloc[val_idx],vp)
    report=scorer.report(gt,pred,n_boot=1000);scorer.show(report)
    gt.to_csv(out/'validation_truth.csv',index=False);pred.to_csv(out/'validation_prediction.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'search.json').write_text(json.dumps(results,indent=2))
    errors=[]
    for i,ix in enumerate(val_idx):
        bad=np.flatnonzero(vp[i]!=y[ix])
        if len(bad):errors.append({'id':train.id.iloc[ix],'text':train.text.iloc[ix],
                                   'errors':{scorer.COLS[j]:{'truth':int(y[ix,j]),'prediction':int(vp[i,j])} for j in bad}})
    (out/'validation_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    # Refit from scratch with all labeled data, preserving tuning decisions.
    ts=time.perf_counter();v=make_features();x=v.fit_transform(texts)
    final_models=[make_model(s).fit(x,y) for s in selected_specs]
    bundle={'vectorizer':v,'models':final_models,'specs':selected_specs,'thresholds':cuts,'seed':SEED}
    joblib.dump(bundle,out/'model.joblib',compress=3)
    info={'selected_specs':selected_specs,'tune_F1':best,'validation_F1':report['ABSA_F1'],
          'python':platform.python_version(),'cpu_threads':1,'gpu':False,
          'cpu':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),''),
          'ram':next((s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemTotal:')),''),
          'full_fit_seconds':time.perf_counter()-ts,'validation_predict_seconds':predict_seconds,
          'total_seconds_before_test':time.perf_counter()-started,
          'train_sha256':hashlib.sha256((Path(data_dir)/'train.csv').read_bytes()).hexdigest()}
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    predict(out/'model.joblib',Path(data_dir)/'test_blind.csv',out/'submission.csv')
    return report

def choose_cuts(y,probs):
    grid=np.linspace(.15,.7,12);cuts=[]
    for j in range(y.shape[1]):
        values=np.array([scorer.f1(y[:,j].astype(bool),probs[:,j]>=t) for t in grid])
        best=np.flatnonzero(values==np.nanmax(values));cuts.append(grid[best[np.argmin(abs(grid[best]-.5))]])
    return np.array(cuts)

def predict(model_path,test_path,output):
    b=joblib.load(model_path);test=pd.read_csv(test_path)
    assert list(test.columns)==['id','text'] and test.id.is_unique and not test.isna().any().any()
    ts=time.perf_counter()
    x=b['vectorizer'].transform(test.text.map(preprocess))
    probs=np.mean([expit(m.decision_function(x)*(1 if s[0]=='logistic' else 3)) for m,s in zip(b['models'],b['specs'])],axis=0)
    frame(test.id,constrained(probs,b['thresholds'])).to_csv(output,index=False,encoding='utf-8')
    assert scorer.kiem_tra(output,test_path)==0
    print('Test inference seconds:',round(time.perf_counter()-ts,2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','predict'])
    p.add_argument('--data-dir',default='..');p.add_argument('--out-dir',default='cpu_results')
    p.add_argument('--model',default='cpu_results/model.joblib');p.add_argument('--test',default='../test_blind.csv')
    p.add_argument('--output',default='submission_reproduced.csv')
    a=p.parse_args()
    # Keep estimators importable when the script is invoked as __main__.
    from cpu_ensemble import fit as module_fit, predict as module_predict
    with threadpool_limits(limits=1):
        if a.command=='train':module_fit(a.data_dir,a.out_dir)
        else:module_predict(a.model,a.test,a.output)

if __name__=='__main__':main()
