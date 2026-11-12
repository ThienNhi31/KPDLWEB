"""Supervised ABSA using a pretrained discriminative NLI classification head.

Fixed label descriptions are paired with reviews. The model is fine-tuned on
the original train labels; it generates no text and calls no prediction API.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,math,platform,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_cosine_schedule_with_warmup
from phobert_absa import (Reviews,clean,loader,infer,split_indices,seed_everything,cpu_thread_limit,
    optimizer_for,train_epoch,tune_thresholds,predictions,main_score,to_frame)
from stacked_absa import file_hash
import score_absa as scorer

MODEL='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
REVISION='8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c'
TOPICS=['nhân viên hoặc dịch vụ khách sạn','sự sạch sẽ',
    'vị trí khách sạn','phòng hoặc trang thiết bị trong phòng',
    'giá trị so với giá tiền','ăn uống tại khách sạn',
    'tiện nghi chung của khách sạn']
HYPOTHESES=[sentence for topic in TOPICS for sentence in [
    f'Đánh giá có nhắc đến {topic}.',
    f'Người viết khen {topic}.',
    f'Người viết chê {topic}.']]


class NLIHead(nn.Module):
    def __init__(self,pretrained,entailment_index):
        super().__init__()
        self.pooler=pretrained.pooler
        self.dropout=pretrained.dropout
        self.classifier=pretrained.classifier
        self.entailment_index=entailment_index

    def forward(self,hidden):
        logits=self.classifier(self.dropout(self.pooler(hidden)))
        other=[j for j in range(logits.shape[1]) if j!=self.entailment_index]
        # sigmoid(logit) is exactly the pretrained entailment softmax probability.
        return logits[:,self.entailment_index:self.entailment_index+1]-torch.logsumexp(logits[:,other],dim=1,keepdim=True)


class NLIQuestionModel(nn.Module):
    def __init__(self,model_name=MODEL,revision=REVISION,unfreeze_layers=4):
        super().__init__()
        pretrained=AutoModelForSequenceClassification.from_pretrained(model_name,revision=revision,trust_remote_code=False)
        matches=[int(k) for k,v in pretrained.config.id2label.items() if str(v).lower()=='entailment']
        assert len(matches)==1,'Pinned model must identify its entailment output'
        self.entailment_index=matches[0]
        self.encoder=pretrained.base_model
        self.head=NLIHead(pretrained,self.entailment_index)
        assert 0<=unfreeze_layers<=len(self.encoder.encoder.layer)
        for p in self.encoder.parameters():p.requires_grad=False
        for layer in list(self.encoder.encoder.layer)[-unfreeze_layers:] if unfreeze_layers else []:
            for p in layer.parameters():p.requires_grad=True

    def forward(self,inputs,counts):
        logits=self.head(self.encoder(**inputs).last_hidden_state)
        return torch.stack([p.max(0).values for p in logits.split(counts)])


class NLIQuestionReviews(Reviews):
    def __init__(self,texts,tokenizer,labels=None,pairs=None,max_length=192,stride=32,hypotheses=None):
        texts=list(texts)
        hypotheses=HYPOTHESES if hypotheses is None else hypotheses
        assert len(hypotheses)==21
        pairs=[(i,j) for i in range(len(texts)) for j in range(21)] if pairs is None else pairs
        self.pairs=pairs;self.items=[]
        questions=[tokenizer.encode(q,add_special_tokens=False) for q in hypotheses]
        encoded={i:tokenizer.encode(clean(texts[i]),add_special_tokens=False) for i in {i for i,_ in pairs}}
        for i,j in pairs:
            q=questions[j];tokens=encoded[i]
            capacity=max_length-tokenizer.num_special_tokens_to_add(pair=True)-len(q)
            assert capacity>stride
            windows=[]
            for start in range(0,max(1,len(tokens)),capacity-stride):
                ids=tokenizer.build_inputs_with_special_tokens(tokens[start:start+capacity],q)
                windows.append({'input_ids':ids,'attention_mask':[1]*len(ids)})
                if start+capacity>=len(tokens):break
            self.items.append(windows)
        self.labels=None if labels is None else torch.as_tensor([[labels[i,j]] for i,j in pairs],dtype=torch.float32)


def training_pairs(indices,y,negatives,seed):
    rng=np.random.default_rng(seed);pairs=[]
    for i in indices:
        selected=set(np.flatnonzero(y[i]).tolist())
        for a in range(7):
            if y[i,a*3]:selected.update([a*3+1,a*3+2])
        absent=[j for j in range(21) if j not in selected]
        if len(absent)>negatives:absent=rng.choice(absent,size=negatives,replace=False).tolist()
        pairs.extend((int(i),j) for j in sorted(selected|set(absent)))
    return pairs


def run(a):
    started=time.perf_counter();seed_everything(a.seed)
    torch.set_num_threads(cpu_thread_limit(a.threads));device=torch.device('cpu')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    train_path=Path(a.data_dir)/'train.csv';train=pd.read_csv(train_path)
    assert list(train.columns)==['id','text']+scorer.COLS and train.id.is_unique
    assert not train.isna().any().any() and train[scorer.COLS].isin([0,1]).all().all()
    y=train[scorer.COLS].to_numpy(int);fi,tu,va=split_indices(train,a.seed)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True,trust_remote_code=False)
    model=NLIQuestionModel(unfreeze_layers=a.layers)
    fp=training_pairs(fi,y,a.negatives,a.seed)
    if a.smoke:
        ds=NLIQuestionReviews(train.text,tok,y,fp[:4],a.max_length,a.stride)
        inputs,counts,target=next(iter(loader(ds,tok,2)))
        logits=model(inputs,counts)
        assert logits.shape==target.shape==(2,1)
        loss=nn.BCEWithLogitsLoss()(logits,target);loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.head.parameters())
        assert not any(p.grad is not None for p in model.encoder.embeddings.parameters())
        print('Original NLI classifier head backward passed; entailment index:',model.entailment_index,flush=True)
        return
    pairs=[fp,[(int(i),j) for i in tu for j in range(21)],[(int(i),j) for i in va for j in range(21)]]
    datasets=[NLIQuestionReviews(train.text,tok,y,p,a.max_length,a.stride) for p in pairs]
    batches=[loader(ds,tok,a.batch_size,j==0,a.seed) for j,ds in enumerate(datasets)]
    print('Original review split:',len(fi),len(tu),len(va),'question pairs:',list(map(len,pairs)),flush=True)
    opt=optimizer_for(model,a.lr)
    steps=math.ceil(len(batches[0])/a.accumulation)*a.epochs
    scheduler=get_cosine_schedule_with_warmup(opt,int(.08*steps),steps)
    positive=max(1,float(datasets[0].labels.sum()))
    pw=torch.tensor([np.clip(math.sqrt((len(fp)-positive)/positive),1,3)],dtype=torch.float32)
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    best,best_epoch,history=-1,0,[]
    for ep in range(1,a.epochs+1):
        ts=time.perf_counter()
        loss=train_epoch(model,batches[0],opt,scheduler,device,pw,a.accumulation,200)
        scores=infer(model,batches[1],device).reshape(len(tu),21)
        cuts=tune_thresholds(y[tu],scores);f1=main_score(y[tu],predictions(scores,cuts))
        row={'epoch':ep,'loss':loss,'tune_F1':f1,'seconds':time.perf_counter()-ts};history.append(row)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False);print('EPOCH',json.dumps(row),flush=True)
        if f1>best:
            best,best_epoch=f1,ep
            state={n:t for n,t in model.state_dict().items() if n in trainable}
            torch.save({'state_dict':state,'thresholds':cuts.tolist()},out/'best.pt')
            np.savez_compressed(out/'tune_scores.npz',scores=scores,ids=train.id.iloc[tu].to_numpy(dtype=str))
        if ep-best_epoch>=a.patience:break
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    restored=model.load_state_dict(saved['state_dict'],strict=False)
    assert not restored.unexpected_keys and not trainable.intersection(restored.missing_keys)
    cuts=np.asarray(saved['thresholds']);del saved
    scores=infer(model,batches[2],device).reshape(len(va),21)
    report=scorer.report(to_frame(train.id.iloc[va],y[va]),to_frame(train.id.iloc[va],predictions(scores,cuts)),n_boot=1000)
    scorer.show(report)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'validation_scores.npz',scores=scores,ids=train.id.iloc[va].to_numpy(dtype=str))
    cfg={'architecture':'nli_questions','model_name':MODEL,'revision':REVISION,'unfreeze_layers':a.layers,
        'seed':a.seed,'max_length':a.max_length,'stride':a.stride,'hypotheses':HYPOTHESES,
        'entailment_index':model.entailment_index,'columns':scorer.COLS,'segment':False,'tokenizer_fast':True,
        'thresholds':cuts.tolist(),'best_epoch':best_epoch,'production_refit':False,'training':vars(a)}
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    torch.save({n:t for n,t in model.state_dict().items() if n in trainable},out/'adapter.pt')
    tok.save_pretrained(out/'tokenizer')
    info={'train_sha256':file_hash(train_path),'best_tune_F1':best,'best_epoch':best_epoch,
        'validation_F1':report['ABSA_F1'],'total_seconds':time.perf_counter()-started,'gpu':False,
        'threads':torch.get_num_threads(),'torch':str(torch.__version__),'python':platform.python_version(),
        'fit_rows':len(fi),'tune_rows':len(tu),'validation_rows':len(va)}
    (out/'run_info.json').write_text(json.dumps(info,indent=2));(out/'best.pt').unlink()
    print('FINAL',json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='..');p.add_argument('--out-dir',default='nli_question_results')
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=2);p.add_argument('--patience',type=int,default=1)
    p.add_argument('--layers',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--max-length',type=int,default=192);p.add_argument('--stride',type=int,default=32)
    p.add_argument('--negatives',type=int,default=3);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
