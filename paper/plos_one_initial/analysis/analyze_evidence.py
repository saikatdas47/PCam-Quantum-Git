"""Post hoc manuscript analysis; reads immutable predictions, never fits models."""
from pathlib import Path
import json,csv,hashlib,os
os.environ.setdefault("MPLCONFIGDIR", str(Path("paper/plos_one_initial/analysis/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('resnet50_evidence');F=R/'training/resnet50_final_v1';O=Path('paper/plos_one_initial');D={}
for p in (F/'results/test_predictions').glob('*.npz'):
 with np.load(p,allow_pickle=False) as z:D[p.stem]={k:z[k].copy() for k in z.files}
tags=sorted(D);y=D[tags[0]]['labels'];slides=D[tags[0]]['slides'];groups=np.unique(slides);G=len(groups)
heads=['mlp_standard','mlp_atan','ry_off','ry_on','selected_classical']
def tags_for(h):return [k for k in tags if (k=='resnet50_selected_classical' if h=='selected_classical' else k.startswith('resnet50_q8_'+h+'_'))]
rows=list(csv.DictReader((F/'reports/test/metrics.csv').open()));metrics={r['tag']:r for r in rows};summary=[]
with np.load(F/'reports/test/bootstrap_auroc.npz') as b:
 for h in heads:
  ts=tags_for(h);z=np.mean([b[k] for k in ts],axis=0);row={'head':h,'fits':len(ts),'AUROC_CI_low':float(np.quantile(z,.025)),'AUROC_CI_high':float(np.quantile(z,.975))}
  for metric in ['AUROC','AP','accuracy','sensitivity','specificity','precision','F1','Brier','log_loss']:
   v=np.array([float(metrics[k][metric]) for k in ts]);row[metric+'_mean']=float(v.mean());row[metric+'_seed_SD']=float(v.std(ddof=1)) if len(v)>1 else None
  summary.append(row)
 # Independent verification of 32 actual weighted draws per model from original scores.
 with np.load(F/'reports/test/bootstrap_counts.npz') as c:counts=c['counts'][:32].copy();np.testing.assert_array_equal(c['groups'],groups)
 _,inv=np.unique(slides,return_inverse=True)
 for tag in tags:
  actual=np.array([roc_auc_score(y,D[tag]['scores'],sample_weight=c[inv]) for c in counts]);np.testing.assert_allclose(actual,b[tag][:32],rtol=0,atol=1e-12)
# Delete-one-WSI jackknife SE and approximate t test. Secondary, post hoc sensitivity.
jack={}
for tag in tags:
 jack[tag]=np.array([roc_auc_score(y[slides!=g],D[tag]['scores'][slides!=g]) for g in groups])
point={tag:roc_auc_score(y,D[tag]['scores']) for tag in tags};a=json.loads((F/'reports/test/analysis.json').read_text());contrasts=[]
for c in [a['primary'],*a['secondary']]:
 v=np.mean([jack[k] for k in c['left_tags']],axis=0)-np.mean([jack[k] for k in c['right_tags']],axis=0)
 se=float(np.sqrt((G-1)/G*np.sum((v-v.mean())**2)));stat=c['delta']/se;p=float(2*student_t.sf(abs(stat),G-1));contrasts.append(dict(c,jackknife_SE=se,t_statistic=stat,df=G-1,p_two_sided_posthoc=p))
pvals=np.array([c['p_two_sided_posthoc'] for c in contrasts[1:]]);order=np.argsort(pvals);adj=np.empty(4);adj[order]=np.minimum(1,np.maximum.accumulate((4-np.arange(4))*pvals[order]))
for i,c in enumerate(contrasts[1:]):c['p_Holm_secondary_posthoc']=float(adj[i])
# Independent data/role description and fixed-budget training diagnostics.
splits={}
for split in ['train','valid','test']:
 with np.load(R/f'training/pcam_three_backbones/resnet50/manifests/{split}.npz') as z:
  splits[split]={'rows':len(z['labels']),'positives':int(z['labels'].sum()),'negatives':int((z['labels']==0).sum()),'slides':len(np.unique(z['slides'])),'unique_images':len(np.unique(z['image_sha256'])) if len(z['image_sha256']) else None}
with np.load(R/'training/pcam_three_backbones/resnet50/manifests/roles.npz') as z:role={'selection_rows':int((z['folds']>=0).sum()),'calibration_rows':int(z['calibration'].sum()),'validation_analysis_rows':int((~z['calibration']).sum())}
timing=[]
for p in (F/'results/jobs').glob('*.json'):
 j=json.loads(p.read_text());tr=j.get('trace',[]);timing.append({'tag':p.stem,'parameters':j.get('parameters'),'epochs':len(tr),'updates':j.get('updates'),'training_compute_seconds':j.get('training_compute_seconds'),'classical_fit_seconds':j.get('fit_seconds'),'segments':len(j.get('runtime_segments',[])),'loss_initial':tr[0]['diagnostic_train_bce'] if tr else None,'loss_final':tr[-1]['diagnostic_train_bce'] if tr else None,'last20_loss_decline':tr[-20]['diagnostic_train_bce']-tr[-1]['diagnostic_train_bce'] if len(tr)>=20 else None})
def csvwrite(name,rows):
 with (O/'tables'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvwrite('test_metrics_summary.csv',summary);csvwrite('training_diagnostics.csv',timing)
result={'splits':splits,'roles':role,'model_summary':summary,'contrasts':contrasts,'weighted_bootstrap_independent_draws_per_model':32,'bootstrap_models_verified':13,'statistical_test_status':'Delete-one-WSI jackknife t tests are post hoc sensitivity; primary inference remains locked paired WSI percentile bootstrap. df=128. Holm applies only to four secondary sensitivity tests.'}
(O/'analysis/results.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.1,3.6))
for i,c in enumerate(contrasts):
 lo,hi=c['conditional95'] if i==0 else c['family95'];ax.errorbar(c['delta']*100,i,xerr=[[100*(c['delta']-lo)],[100*(hi-c['delta'])]],fmt='o',color='#0072B2' if i==0 else '#D55E00',capsize=3)
ax.axvline(0,color='.5',ls='--');ax.set(yticks=range(5),yticklabels=['CNOT on − selected classical','CNOT on − CNOT off','CNOT on − atan MLP','CNOT on − standard MLP','Atan MLP − standard MLP'],xlabel='Difference in mean per-seed AUROC (percentage points)');ax.invert_yaxis();fig.tight_layout()
for ext in ['png','pdf','tif']:fig.savefig(O/f'figures/Fig3_contrasts.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'contrasts':[{k:v for k,v in c.items() if k not in ['left_tags','right_tags']} for c in contrasts],'roles':role},indent=2))
