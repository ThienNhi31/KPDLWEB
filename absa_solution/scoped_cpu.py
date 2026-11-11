"""Aspect-scoped Vietnamese clause features + learned classifiers, CPU only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from threadpoolctl import threadpool_limits
from cpu_ensemble import preprocess, grouped_split, NBLinear, choose_cuts, score
from pipeline import frame, constrained
import score_absa as scorer

# These are feature-extraction vocabularies, not labeling rules. All 21 output
# decisions come from classifiers fitted using train labels.
ANCHORS={
 'nhan_vien':['nhân viên','lễ tân','phục vụ','dịch vụ','thái độ','bảo vệ','chủ','check in','check out','checkin','checkout','tiếp tân','tiếp đón','nhiệt tình','thân thiện','niềm nở'],
 'sach_se':['sạch','bẩn','dơ','hôi','thơm','mốc','ẩm','muỗi','kiến','gián','côn trùng','vệ sinh','bụi','rác'],
 'vi_tri':['vị trí','địa điểm','biển','trung tâm','sân bay','đi lại','di chuyển','thuận tiện','xung quanh','khu vực','đường','cách xa','chợ','đi bộ','yên tĩnh','ồn ào'],
 'phong':['phòng','giường','nệm','đệm','gối','chăn','ga','toilet','nhà tắm','nhà vệ sinh','vòi','nước nóng','điều hòa','máy lạnh','máy sấy','tivi','tv','tủ','cửa','cách âm','ngủ','view','ban công','nội thất','thiết bị'],
 'gia_tri':['giá','tiền','rẻ','đắt','mắc','đáng','chi phí','phụ thu','hóa đơn','tính phí','thu phí'],
 'an_uong':['ăn','uống','buffet','bữa sáng','nhà hàng','menu','thực đơn','món','đồ ăn','đồ uống','cà phê','cafe','bar','bia','nước ép','ngon'],
 'tien_nghi':['tiện nghi','hồ bơi','bể bơi','bơi','wifi','wi fi','internet','mạng','spa','gym','thang máy','sảnh','đỗ xe','gửi xe','bãi xe','đậu xe','khuôn viên','đưa đón','xe đạp','tour','đón khách','đón sân bay']}
PATTERNS={a:re.compile(r'(?<!\w)(?:'+'|'.join(map(re.escape,words))+r')(?!\w)') for a,words in ANCHORS.items()}

def local_context(text,aspect):
    clauses=[s.strip() for s in re.split(r'[.!?;,\n]+|\b(?:nhưng|tuy_nhiên|bù_lại)\b',text) if s.strip()]
    chosen=[]
    for i,c in enumerate(clauses):
        plain=c.replace('_',' ')
        if PATTERNS[aspect].search(plain):
            chosen.append(c)
            # Carry a short adjacent description only if no other aspect anchor occurs.
            if i+1<len(clauses) and len(clauses[i+1].split())<=10:
                nxt=clauses[i+1].replace('_',' ')
                if not any(p.search(nxt) for p in PATTERNS.values()):chosen.append(clauses[i+1])
    return ' . '.join(chosen) if chosen else '__khong_co_tu_khoa__'

def features():
    return FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True,max_features=50000)),
        ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=80000))])

def fit(data_dir,out_dir):
    start=time.perf_counter();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(Path(data_dir)/'train.csv');y=train[scorer.COLS].to_numpy(int)
    texts=train.text.map(preprocess);fi,tu,va=grouped_split(train)
    global_v=features();gf=global_v.fit_transform(texts.iloc[fi]);gt=global_v.transform(texts.iloc[tu])
    global_model=OneVsRestClassifier(NBLinear(C=1),n_jobs=1).fit(gf,y[fi])
    global_t=expit(3*global_model.decision_function(gt))
    selected=[];history=[];best_tune=[]
    for k,aspect in enumerate(scorer.ASPECTS):
        local=texts.map(lambda s:local_context(s,aspect))
        v=features();lf=v.fit_transform(local.iloc[fi]);lt=v.transform(local.iloc[tu])
        xf=hstack([gf*.35,lf],format='csr');xt=hstack([gt*.35,lt],format='csr')
        best=-1;best_item=None
        for C in [.5,2,8]:
            m=OneVsRestClassifier(NBLinear(C=C),n_jobs=1).fit(xf,y[fi,k*3:k*3+3])
            local_t=expit(3*m.decision_function(xt))
            for weight in [0,.5,1]:
                probs=weight*local_t+(1-weight)*global_t[:,k*3:k*3+3]
                cuts=choose_cuts(y[tu,k*3:k*3+3],probs)
                p=(probs>=cuts).astype(int)
                f=float(np.nanmean([scorer.f1(y[tu,k*3+j].astype(bool),p[:,j].astype(bool)) for j in [1,2]]))
                history.append({'aspect':aspect,'C':C,'local_weight':weight,'tune_F1':f})
                if f>best:
                    best=f;best_item={'aspect':aspect,'C':C,'weight':weight,'cuts':cuts,'local_vectorizer':v,'model':m}
        selected.append(best_item);best_tune.append(best)
        print(aspect,'tune',round(best,4),'C',best_item['C'],'local_weight',best_item['weight'],flush=True)
    # Evaluate holdout only after all per-aspect choices are fixed on tune.
    gv=global_v.transform(texts.iloc[va]);gprobs=expit(3*global_model.decision_function(gv));parts=[]
    for k,b in enumerate(selected):
        lv=b['local_vectorizer'].transform(texts.iloc[va].map(lambda s:local_context(s,b['aspect'])))
        lp=expit(3*b['model'].decision_function(hstack([gv*.35,lv],format='csr')))
        parts.append(b['weight']*lp+(1-b['weight'])*gprobs[:,k*3:k*3+3])
    cuts=np.concatenate([b['cuts'] for b in selected]);pred=constrained(np.hstack(parts),cuts)
    truth,pr=frame(train.id.iloc[va],y[va]),frame(train.id.iloc[va],pred)
    r=scorer.report(truth,pr,n_boot=1000);scorer.show(r)
    truth.to_csv(out/'validation_truth.csv',index=False);pr.to_csv(out/'validation_prediction.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    (out/'search.json').write_text(json.dumps(history,indent=2))
    info={'method':'aspect_scoped_NB_SVM','seed':2026,'tune_F1':float(np.mean(best_tune)),
          'validation_F1':r['ABSA_F1'],'choices':[{'aspect':b['aspect'],'C':b['C'],'local_weight':b['weight']} for b in selected],
          'total_seconds_before_refit':time.perf_counter()-start,
          'train_sha256':hashlib.sha256((Path(data_dir)/'train.csv').read_bytes()).hexdigest()}
    # Refit on all labeled data; no test text used to create feature vocabularies.
    global_v=features();gx=global_v.fit_transform(texts)
    global_model=OneVsRestClassifier(NBLinear(C=1),n_jobs=1).fit(gx,y)
    for k,b in enumerate(selected):
        b['local_vectorizer']=features()
        lx=b['local_vectorizer'].fit_transform(texts.map(lambda s:local_context(s,b['aspect'])))
        b['model']=OneVsRestClassifier(NBLinear(C=b['C']),n_jobs=1).fit(hstack([gx*.35,lx],format='csr'),y[:,k*3:k*3+3])
    bundle={'global_vectorizer':global_v,'global_model':global_model,'aspects':selected,'cuts':cuts}
    joblib.dump(bundle,out/'model.joblib',compress=3)
    info['total_seconds_before_test']=time.perf_counter()-start
    (out/'run_info.json').write_text(json.dumps(info,indent=2))
    predict(out/'model.joblib',Path(data_dir)/'test_blind.csv',out/'submission.csv')
    return r

def predict(model_path,test_path,output):
    b=joblib.load(model_path);test=pd.read_csv(test_path)
    assert list(test.columns)==['id','text'] and test.id.is_unique and not test.isna().any().any()
    texts=test.text.map(preprocess);gx=b['global_vectorizer'].transform(texts)
    gp=expit(3*b['global_model'].decision_function(gx));parts=[]
    for k,a in enumerate(b['aspects']):
        lx=a['local_vectorizer'].transform(texts.map(lambda s:local_context(s,a['aspect'])))
        lp=expit(3*a['model'].decision_function(hstack([gx*.35,lx],format='csr')))
        parts.append(a['weight']*lp+(1-a['weight'])*gp[:,k*3:k*3+3])
    pred=constrained(np.hstack(parts),b['cuts'])
    frame(test.id,pred).to_csv(output,index=False)
    assert scorer.kiem_tra(output,test_path)==0

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','predict'])
    p.add_argument('--data-dir',default='..');p.add_argument('--out-dir',default='scoped_results')
    p.add_argument('--model',default='scoped_results/model.joblib');p.add_argument('--test',default='../test_blind.csv');p.add_argument('--output',default='submission_reproduced.csv')
    a=p.parse_args()
    from scoped_cpu import fit as module_fit,predict as module_predict
    with threadpool_limits(limits=1):
        if a.command=='train':module_fit(a.data_dir,a.out_dir)
        else:module_predict(a.model,a.test,a.output)

if __name__=='__main__':main()
