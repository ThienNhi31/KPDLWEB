"""Package the verified 0.8840 candidate privately, without original datasets."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat
from stacked_absa import model_identity


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def build(root,output):
    root,output=Path(root),Path(output)
    evidence=json.loads((root/'public_score_report.json').read_text())
    assert sha(root/'submission.csv')==evidence['associated_candidate_sha256']
    nb=nbformat.read(root/'submission.ipynb',as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)>=7 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    assert ''.join(o.get('text','') for o in nb.cells[9].get('outputs',[])).count('ID train_')>=5
    meta=root/'model_bundle/stacked'
    cfg=json.loads((meta/'config.json').read_text())
    assert not cfg['production_refit'] and len(cfg['sources'])==4
    for source in cfg['sources']:
        assert model_identity(meta/source['path'])==source['identity']
    files={}
    for p in root.glob('*.py'):files[p.name]=p
    for p in root.glob('requirements*.txt'):files[p.name]=p
    for name in ['submission.ipynb','submission.csv','HUONG_DAN.md','HUONG_DAN_STACKING.md',
        'HUONG_DAN_CO_DIEN.md','HUAN_LUYEN_PHO_BERT.md','KET_QUA_HOC_SAU.md',
        'ASPECT_PAIR_METHOD.md','THU_NGHIEM_DU_LIEU_THEM.md','public_score_report.json','inference_cost.json']:
        if (root/name).exists():files[name]=root/name
    for p in (root/'model_bundle').rglob('*'):
        if p.is_file() and p.name not in ['validation_prediction.csv','validation_truth.csv']:
            files[p.relative_to(root).as_posix()]=p
    for name in ['metrics.json','comparison.csv','run_info.json']:
        p=root/'notebook_baseline_results'/name
        if p.exists():files['notebook_baseline_results/'+name]=p
    for source,dest in [
        ('augmentation_trial/du_lieu_them.csv','du_lieu_them.csv'),
        ('augmentation_trial/du_lieu_them.generation.json','du_lieu_them.generation.json'),
        ('augmentation_results/augmentation_ablation.csv','experiments/augmentation_ablation.csv'),
        ('normalized_attention_results/augmentation_ablation.csv','experiments/normalization_ablation.csv'),
        ('normalized_attention_results/normalization_audit.json','experiments/normalization_audit.json')]:
        if (root/source).exists():files[dest]=root/source
    forbidden={'train.csv','test_blind.csv','submission_demo.csv'}
    assert not any(Path(name).name in forbidden for name in files)
    assert not any('private_cache' in name or 'private_predictions' in name for name in files)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,path in sorted(files.items()):z.write(path,name)
    with zipfile.ZipFile(output) as z:
        assert set(z.namelist())==set(files)
        assert hashlib.sha256(z.read('submission.csv')).hexdigest()==evidence['associated_candidate_sha256']
        assert not any(Path(n).name in forbidden for n in z.namelist())
    info={'archive':str(output),'bytes':output.stat().st_size,'files':len(files),
        'public_score_reported_by_user':evidence['ABSA_F1'],'submission_sha256':evidence['associated_candidate_sha256'],
        'contains_original_datasets':False,'main_model_uses_synthetic_data':False,
        'extra_synthetic_data_is_for_documented_unselected_ablation_only':True}
    output.with_suffix('.manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--output',required=True)
    a=p.parse_args();build(a.root,a.output)
