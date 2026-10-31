"""Reanalyse saved predictions only. Never fits models or reads new test images.
Run from project root: python paper/code/analyze_current_study.py
"""
from pathlib import Path
import json,hashlib,sys,platform,csv,time,os
import numpy as np
import scipy, sklearn
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(os.environ.get('PCAM_STUDY_ROOT',Path(__file__).resolve().parents[2]))
E=ROOT/'extensions/resnet50_extensions_evidence'; OUT=ROOT/'paper/analysis'; OUT.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return dict(np.load(p,allow_pickle=False))
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
threadpool_limits(2)
start=time.perf_counter()
manifest=json.loads((E/'MANIFEST.json').read_text())
assert all((E/k).stat().st_size==v['bytes'] and digest(E/k)==v['sha256'] for k,v in manifest.items())
lock=json.loads((E/'extension_lock.json').read_text()); assert all(digest(E/k)==v for k,v in lock['artifacts'].items())
print('Manifest and lock verified',flush=True)
metrics=[]; cohort={}; arrays={}; original={}; metric_error=0
saved={(r['tag'],r['partition']):r for r in json.loads((E/'metrics.json').read_text())}
for split in ['valid','test']:
 for path in sorted((E/'predictions'/split).glob('*.npz')):
  z=load(path); y=z['labels'];s=z['scores'];p=z['probabilities'];d=z['decisions'];
  for k in ['ids','labels','slides']:
   if split in cohort:np.testing.assert_array_equal(z[k],cohort[split][k])
  cohort[split]=z
  assert len(np.unique(z['ids']))==len(y) and np.isfinite(s).all()
  np.testing.assert_allclose(expit(s),p,rtol=0,atol=1e-14);np.testing.assert_array_equal(d,p>=z['threshold'])
  assert float(z['threshold'])==json.loads((E/'thresholds'/f'{path.stem}.json').read_text())['threshold']
  mask=~z['calibration_mask'] if split=='valid' else np.ones(len(y),bool)
  tn,fp,fn,tp=map(int,confusion_matrix(y[mask],d[mask],labels=[0,1]).ravel());n=int(mask.sum())
  r=dict(tag=path.stem,partition='test' if split=='test' else 'valid_diagnostic',rows=n,AUROC=float(roc_auc_score(y[mask],s[mask])),AP=float(average_precision_score(y[mask],s[mask])),Brier=float(np.mean((p[mask]-y[mask])**2)),log_loss=float(log_loss(y[mask],p[mask],labels=[0,1])),TN=tn,FP=fp,FN=fn,TP=tp,threshold=float(z['threshold']),accuracy=(tn+tp)/n,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp),F1=2*tp/(2*tp+fp+fn))
  prev=saved[(path.stem,r['partition'])]
  for k in prev:
   if isinstance(prev[k],(int,float)):metric_error=max(metric_error,abs(prev[k]-r[k]));assert abs(prev[k]-r[k])<1e-12,(path,k,prev[k],r[k])
  metrics.append(r)
  if split=='test':arrays[path.stem]=z
# Original study is preserved as its own stage, including its own locked inference.
for path in sorted((E/'original_evidence/results_test_predictions').glob('*.npz')):
 z=load(path)
 for k in ['ids','labels','slides']:np.testing.assert_array_equal(z[k],cohort['test'][k])
 original[path.stem]=z
write('metrics_recomputed.json',metrics)
# Verify slide-disjoint folds, transforms, and selection scores.
train=load(E/'original_evidence/manifests/train.npz'); valid=load(E/'original_evidence/manifests/valid.npz');roles=load(E/'original_evidence/manifests/roles.npz')
for split,z in [('valid',valid),('test',cohort['test'])]:assert not set(train['slides'])&set(z['slides'])
assert not set(valid['slides'])&set(cohort['test']['slides'])
for fold in range(3):
 fi=(roles['folds']>=0)&(roles['folds']!=fold);he=roles['folds']==fold
 assert not set(train['slides'][fi])&set(train['slides'][he])
 for kind in ['compression','original']:
  t=load(E/f'transforms/{kind}_cv{fold}.npz');np.testing.assert_array_equal(t['fit_ids'],train['ids'][fi])
for kind in ['compression','original']:np.testing.assert_array_equal(load(E/f'transforms/{kind}_final.npz')['fit_ids'],train['ids'])
cvchecks=[]
for sub in ['cv_predictions','original_evidence/cv_predictions']:
 for p in sorted((E/sub).glob('*.npz')):
  z=load(p);assert not set(z['fit_ids'])&set(z['ids'])
  fold=int(p.stem.split('_cv')[1][0]);he=roles['folds']==fold;fi=(roles['folds']>=0)&~he
  for k in ['ids','labels','slides']:np.testing.assert_array_equal(z[k],train[k][he])
  np.testing.assert_array_equal(z['fit_ids'],train['ids'][fi]);a=float(roc_auc_score(z['labels'],z['scores']))
  cvchecks.append(dict(tag=p.stem,AUROC=a,kind=sub))
for selection_name in ['compression','head_rates']:
 selection=json.loads((E/f'selection/{selection_name}.json').read_text())
 rows=selection['folds'] if selection_name=='compression' else selection['fold_scores']
 for r in rows:
  tag=f"compression_cv{r['fold']}_d{r['dimension']}_C{r['C']}" if selection_name=='compression' else r['tag']
  assert abs(next(v['AUROC'] for v in cvchecks if v['tag']==tag)-r['AUROC'])<1e-12
 for r in selection['means']:
  group=[x['AUROC'] for x in rows if all(x[k]==r[k] for k in (['dimension','C'] if selection_name=='compression' else ['head','lr']))]
  assert abs(np.mean(group)-r['mean_AUROC'])<1e-12
 for key,val in (selection['choices'] if selection_name=='compression' else selection['selected_rates']).items():
  rs=[r for r in selection['means'] if r['dimension']==int(key)] if selection_name=='compression' else [r for r in selection['means'] if r['head']==key]
  best=max(r['mean_AUROC'] for r in rs);field='C' if selection_name=='compression' else 'lr'
  assert val==min(r[field] for r in rs if best-r['mean_AUROC']<=1e-8)
write('cv_recomputed.json',cvchecks)
# Determine prepared matrix identities from the saved transformation signature recipe.
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sig(kind,dim,split):
 z=load(E/f'transforms/{"original" if kind=="original" else "compression"}_final.npz');m=load(E/f'original_evidence/manifests/{split}.npz')
 a={'ids':ah(m['ids']),'mean':ah(z['mean']),'components':ah(z['components']),'scale':ah(z['scale']),'dimension':dim,'kind':kind,'rows':'all','scale_mean':ah(z['scale_mean']) if 'scale_mean' in z else None,'full_scale':ah(z['full_scale']) if 'full_scale' in z else None}
 return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:24]
write('prepared_matrix_map.json',{f'{s}_{k}_{d}':sig(k,d,s) for s in ['valid','test'] for k,d in [('original',8),('compression',8),('compression',32),('compression',128),('compression',2048)]})
# AUROC pair-count matrix: A[g,h] counts positive(g)>negative(h), half for ties.
# This gives precisely the patch-weighted WSI bootstrap statistic, including ties.
slides,gi=np.unique(cohort['test']['slides'],return_inverse=True);G=len(slides);y=cohort['test']['labels'];pos=np.bincount(gi,weights=y,minlength=G);neg=np.bincount(gi,weights=1-y,minlength=G)
rng=np.random.default_rng(71913)
# Independent replicates, same weight vector for every model and optimization seed.
counts=np.empty((20000,G),np.int16)
for i in range(20000):counts[i]=np.bincount(rng.integers(0,G,size=G),minlength=G)
w=counts.astype(float);den=(w@pos)*(w@neg);assert np.all(den>0)
np.savez_compressed(OUT/'bootstrap_counts.npz',counts=counts,slides=slides,seed=np.array(71913))
boot={};max_weight_error=0;max_point_error=0
for tag,z in arrays.items():
 s=z['scores'];order=np.argsort(s,kind='stable');v=s[order];a=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];b=np.r_[a[1:],len(y)];groups=np.repeat(np.arange(len(a)),b-a)
 # For every score tie block, accumulate negative counts by slide.
 nc=np.zeros((len(a),G));np.add.at(nc,(groups,gi[order]),(1-y[order]).astype(float));less=np.cumsum(nc,axis=0)-.5*nc
 A=np.zeros((G,G));oi=order[y[order]==1];tg=groups[y[order]==1]
 np.add.at(A,gi[oi],less[tg]);point=A.sum()/(pos.sum()*neg.sum());assert abs(point-roc_auc_score(y,s))<1e-12
 max_point_error=max(max_point_error,abs(point-roc_auc_score(y,s)))
 result=np.sum((w@A)*w,axis=1)/den
 for i in range(8):
  direct=roc_auc_score(y,s,sample_weight=w[i,gi]);err=abs(direct-result[i]);assert err<1e-12;max_weight_error=max(max_weight_error,err)
 boot[tag]=result
 print('Bootstrap',tag,flush=True)
np.savez_compressed(OUT/'bootstrap_auroc.npz',**boot)
# Seed means are means of per-seed AUROCs, not score ensembles.
summary=[];means={};seed_values={}
for group in sorted(set(t if t.startswith('compression') else t.rsplit('_',1)[0] for t in arrays)):
 tags=sorted(t for t in arrays if (t if t.startswith('compression') else t.rsplit('_',1)[0])==group)
 rs=[r for r in metrics if r['partition']=='test' and r['tag'] in tags];values=np.array([r['AUROC'] for r in rs]);seed_values[group]=values;means[group]=np.mean([boot[t] for t in tags],axis=0)
 r={'group':group,'fits':len(tags),'AUROC_mean':float(values.mean()),'AUROC_seed_SD':float(values.std(ddof=1)) if len(values)>1 else None,'CI95':np.quantile(means[group],[.025,.975]).tolist()}
 for metric in ['AP','Brier','log_loss','accuracy','sensitivity','specificity','precision','F1']:
  r[metric+'_mean']=float(np.mean([x[metric] for x in rs]));r[metric+'_seed_SD']=float(np.std([x[metric] for x in rs],ddof=1)) if len(rs)>1 else None
 summary.append(r)
contrasts=[]
def contrast(name,a,b,family,m):
 d=means[a]-means[b];v=seed_values[a]-seed_values[b];r={'contrast':name,'a':a,'b':b,'family':family,'family_size':m,'delta':float(v.mean()),'CI95_nominal':np.quantile(d,[.025,.975]).tolist(),'CI95_family':np.quantile(d,[.025/m,1-.025/m]).tolist(),'seed_delta_SD':float(v.std(ddof=1)) if len(v)>1 else None,'seed_deltas':v.tolist(),'post_hoc':True};contrasts.append(r)
contrast('2048 minus 8','compression_lr_d2048','compression_lr_d8','compression_primary',1)
for d in [32,128]:contrast(f'{d} minus 8',f'compression_lr_d{d}','compression_lr_d8','compression_secondary',2)
for h in ['mlp_standard','mlp_atan','ry_off','ry_on']:contrast(h+' selected minus shared','tuned_'+h,'shared_'+h,'learning_rate_primary',4)
for a,b in [('ry_on','ry_off'),('ry_on','mlp_atan'),('ry_on','mlp_standard'),('mlp_atan','mlp_standard')]:contrast(a+' minus '+b,'tuned_'+a,'tuned_'+b,'selected_rate_secondary',4)
write('summary.json',summary);write('contrasts.json',contrasts)
# Saved convergence and runtime, without treating cumulative timers as wall-clock duration.
jobs=[]
for p in sorted((E/'jobs').glob('compression*.json')):
 j=json.loads(p.read_text());last=j['trace'][-1];jobs.append(dict(tag=p.stem,converged=j['converged'],iterations=len(j['trace']),compute_seconds=j['compute_seconds'],gradient_max=last['gradient_max'],max_parameter_change=last['max_parameter_change'],objective=last['penalized_mean_objective'],stopping_rule='gradient' if last['gradient_max']<=1e-7 else 'parameter_change' if j['converged'] else 'iteration_limit',devices=[v['device'] for v in j['runtime_segments']]))
write('logistic_optimization.json',jobs)
runtimes=[]
for p in sorted((E/'jobs').glob('pair*.json')):
 j=json.loads(p.read_text());assert len(j['trace']['shared'])==len(j['trace']['tuned'])==100
 for schedule in ['shared','tuned']:
  t=j['trace'][schedule];assert all(v['epoch']==i+1 for i,v in enumerate(t));assert t[-1]['cumulative_compute_seconds']==j['compute_seconds'][schedule]
  runtimes.append(dict(head=j['head'],seed=j['seed'],schedule=schedule,lr=j['shared_lr'] if schedule=='shared' else j['selected_lr'],compute_seconds=j['compute_seconds'][schedule],updates=204800,milliseconds_per_update=1000*j['compute_seconds'][schedule]/204800,final_diagnostic_BCE=t[-1]['diagnostic_train_BCE'],segments=j['paired_runtime_segments']))
write('neural_timing.json',runtimes)
transitions=[json.loads(p.read_text()) for p in (E/'identity_recovery').glob('*.json') if 'numerically_validated_device_transition' in p.read_text()]
write('audit.json',dict(manifest_entries=len(manifest),manifest_mismatches=0,locked_artifacts=len(lock['artifacts']),test_predictions=len(arrays),original_predictions=len(original),metric_records_verified=len(metrics),maximum_metric_error=metric_error,CV_predictions_verified=len(cvchecks),rows=len(y),slides=G,bootstrap_replicates=20000,bootstrap_seed=71913,pair_matrix_point_max_error=max_point_error,direct_weighted_bootstrap_draws_per_model=8,direct_weighted_bootstrap_max_error=max_weight_error,transitions=transitions,optimization_limit_jobs=[r['tag'] for r in jobs if not r['converged']],environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform()),analysis_elapsed_seconds=time.perf_counter()-start,limits=['No new training or encoder extraction','No external investigator replication','Full original 2048-feature caches absent','Bootstrap conditional on fitted models, selected hyperparameters and fixed seeds','Post hoc analysis after original test inspection']))
print('ANALYSIS COMPLETE',flush=True)
