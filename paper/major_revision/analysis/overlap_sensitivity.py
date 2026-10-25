"""Post hoc descriptive analysis of exact train/test pixel overlap. No fitting."""
from pathlib import Path
import numpy as np,json,csv
from sklearn.metrics import roc_auc_score
P=Path('paper/major_revision');F=Path('resnet50_evidence/training/resnet50_final_v1');M=Path('resnet50_evidence/training/pcam_three_backbones/resnet50/manifests');H={};Y={}
for split in ['train','valid','test']:
 with np.load(P/f'results/{split}_image_hashes.npz') as z:H[split]=z['sha256'].copy()
 with np.load(M/f'{split}.npz') as z:Y[split]=z['labels'].copy()
shared=sorted(set(H['train'])&set(H['test']));rows=[]
for h in shared:
 a=H['train']==h;b=H['test']==h;rows.append({'sha256':h,'training_rows':int(a.sum()),'test_rows':int(b.sum()),'training_positive_rows':int(Y['train'][a].sum()),'test_positive_rows':int(Y['test'][b].sum())})
with (P/'tables/train_test_exact_overlap.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
keep=~np.isin(H['test'],shared);scores={}
for path in sorted((F/'results/test_predictions').glob('*.npz')):
 with np.load(path) as z:
  np.testing.assert_array_equal(z['labels'],Y['test']);scores[path.stem]={'full_AUROC':float(roc_auc_score(z['labels'],z['scores'])),'overlap_excluded_AUROC':float(roc_auc_score(z['labels'][keep],z['scores'][keep]))}
on=[v for k,v in scores.items() if '_ry_on_' in k];ref=scores['resnet50_selected_classical']
result={'status':'Post hoc descriptive sensitivity after exact-image audit. Original locked results retained. No refit, tuning or new primary inference.','shared_unique_hashes':len(shared),'affected_training_rows':sum(r['training_rows'] for r in rows),'excluded_test_rows':int((~keep).sum()),'excluded_test_positives':int(Y['test'][~keep].sum()),'remaining_test_rows':int(keep.sum()),'remaining_positive_rows':int(Y['test'][keep].sum()),'remaining_negative_rows':int((Y['test'][keep]==0).sum()),'CNOT_on_mean_AUROC_excluding_overlap':float(np.mean([x['overlap_excluded_AUROC'] for x in on])),'selected_classical_AUROC_excluding_overlap':ref['overlap_excluded_AUROC'],'primary_difference_excluding_overlap':float(np.mean([x['overlap_excluded_AUROC'] for x in on])-ref['overlap_excluded_AUROC']),'full_primary_difference':float(np.mean([x['full_AUROC'] for x in on])-ref['full_AUROC']),'models':scores,'limits':'Exact overlap is a benchmark integrity limitation. This sensitivity does not prove the remaining images are independent or remove near duplicates. No interval or significance claim is added for the filtered cohort.'}
(P/'results/overlap_sensitivity.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))
