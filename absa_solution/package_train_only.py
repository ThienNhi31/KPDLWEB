"""Package selected local models and train-only evaluation; no raw datasets."""
import train_only
import argparse,hashlib,json,os,zipfile
from pathlib import Path
from stacked_absa import model_identity


def pack(model_dir,solution_root,review_dir,output):
    model,root,review=map(Path,[model_dir,solution_root,review_dir])
    cfg=json.loads((model/'config.json').read_text())
    assert cfg['architecture']=='supervised_stacking' and not cfg['production_refit']
    info=json.loads((review/'evaluation_info.json').read_text())
    report=json.loads((model/'metrics.json').read_text())
    assert not info['blind_file_read'] and not info['public_score_claimed']
    assert info['input_file']=='train.csv' and abs(info['ABSA_F1']-report['ABSA_F1'])<1e-12
    entries=[]
    def add(path,name):
        path=Path(path)
        assert path.is_file()
        assert Path(name).name not in ['train.csv','test_blind.csv','submission_demo.csv']
        assert not Path(name).name.startswith('submission')
        entries.append((path,name))
    for file in sorted(root.glob('*.py')):
        if file.name!='package_best_submission.py':add(file,file.name)
    for file in sorted(root.glob('requirements*.txt')):add(file,file.name)
    for name in ['DATA_POLICY.md','TRAIN_ONLY_README.md','THU_NGHIEM_TIEP.md',
                 'train_only_report.json','train_only.ipynb','train_validation_notes.json',
                 'fresh_validation_proof.json']:
        if (root/name).exists():add(root/name,name)
    for name in ['metrics.json','evaluation_info.json','validation_prediction.csv','training_validation_errors.json']:
        add(review/name,'validation_review/'+name)
    for folder in ['model_bundle/cached_phobert','model_bundle/balanced_phobert',
                   'model_bundle/aspect_pair','model_bundle/visobert',
                   'label_attention_results','normalized_attention_results']:
        path=root/folder
        for name in ['config.json','run_info.json','history.csv','metrics.json']:
            if (path/name).is_file():add(path/name,'selection_history/'+folder+'/'+name)
    allowed=['adapter.pt','config.json','run_info.json','metrics.json',
        'selection_metrics.json','history.csv','tune_scores.npz',
        'best_tuning_predictions.npz','validation_scores.npz','validation_probabilities.npz']
    rewritten=[]
    for i,source in enumerate(cfg['sources']):
        folder=(model/source['path']).resolve()
        assert folder.is_relative_to(root.resolve())
        assert model_identity(folder)==source['identity']
        base=json.loads((folder/'config.json').read_text())
        assert not base.get('production_refit',False)
        assert base.get('augmentation_rows',0)==0
        assert (folder/'adapter.pt').exists()
        for name in allowed:
            if (folder/name).exists():add(folder/name,f'models/source_{i}/'+name)
        for file in sorted((folder/'tokenizer').rglob('*')):
            if file.is_file():add(file,f'models/source_{i}/tokenizer/'+str(file.relative_to(folder/'tokenizer')))
        rewritten.append({**source,'path':f'../source_{i}'})
    cfg['sources']=rewritten
    for name in ['meta_models.joblib','metrics.json','selection_metrics.json','run_info.json','validation_scores.npz']:
        if (model/name).exists():add(model/name,'models/stacked/'+name)
    assert len({name for _,name in entries})==len(entries)
    manifest={'train_only':True,'original_datasets_included':False,'blind_predictions_included':False,
        'validation_F1':report['ABSA_F1'],'validation_CI95':report['ABSA_F1_CI95'],
        'validation_rows':info['validation_rows'],'train_sha256':cfg['train_sha256'],
        'source_count':len(cfg['sources']),'public_score_claimed':False,
        'validation_feedback_seen_in_prior_development':True,'files':{}}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path,name in entries:
            z.write(path,name)
            manifest['files'][name]={'bytes':path.stat().st_size,
                'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
        config_bytes=json.dumps(cfg,ensure_ascii=False,indent=2).encode()
        z.writestr('models/stacked/config.json',config_bytes)
        manifest['files']['models/stacked/config.json']={'bytes':len(config_bytes),
            'sha256':hashlib.sha256(config_bytes).hexdigest()}
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    os.replace(temporary,output)
    with zipfile.ZipFile(output) as z:
        assert all(Path(n).name not in ['train.csv','test_blind.csv','submission_demo.csv']
            and not Path(n).name.startswith('submission') for n in z.namelist())
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'package':str(output),'bytes':output.stat().st_size,
        'source_count':len(cfg['sources']),'validation_F1':report['ABSA_F1'],
        'raw_datasets_included':False,'blind_predictions_included':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True)
    p.add_argument('--solution-root',default=str(Path(__file__).parent))
    p.add_argument('--review-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();pack(a.model_dir,a.solution_root,a.review_dir,a.output)
