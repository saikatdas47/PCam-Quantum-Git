"""Verify exported checksums and recompute prediction metrics without training."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,confusion_matrix,brier_score_loss
E=Path('resnet50_evidence');F=E/'training/resnet50_final_v1';P=Path('paper/major_revision');manifest=json.loads((F/'exports/MANIFEST.json').read_text());print(type(manifest),list(manifest)[:3])
# Manifest record layout is preserved by the original export.
records=manifest.get('files',manifest)
if isinstance(records,list):items=[(r.get('path',r.get('name')),r) for r in records]
else:items=list(records.items())
verified=0
for path,r in items:
 f=E/path;h=hashlib.sha256()
 with f.open('rb') as z:
  for block in iter(lambda:z.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==r['sha256'],path
 assert f.stat().st_size==r.get('bytes',r.get('size')),path
 verified+=1
checks=[]
for split in ['valid','test']:
 table={r['tag']:r for r in csv.DictReader((F/f'reports/{split}/metrics.csv').open())}
 for path in sorted((F/f'results/{split}_predictions').glob('*.npz')):
  with np.load(path) as z:y=z['labels'];scores=z['scores'];p=z['probabilities'];threshold=z['threshold'];pred=p>=threshold
  if split=='valid':
   with np.load(E/'training/pcam_three_backbones/resnet50/manifests/roles.npz') as z:keep=~z['calibration']
   y=y[keep];scores=scores[keep];p=p[keep];pred=pred[keep]
  actual={'AUROC':roc_auc_score(y,scores),'AP':average_precision_score(y,scores),'accuracy':accuracy_score(y,pred),'Brier':brier_score_loss(y,p)}
  for key,v in actual.items():assert abs(v-float(table[path.stem][key]))<1e-12,(path,key)
  checks.append({'split':split,'tag':path.stem,**{k:float(v) for k,v in actual.items()}})
(P/'results/evidence_reverification.json').write_text(json.dumps({'manifest_entries_verified':verified,'prediction_files_verified':len(checks),'checks':checks,'status':'Same-project read-only computational recomputation. Not external replication.'},indent=2));print('Verified',verified,'manifest entries and',len(checks),'prediction files')
