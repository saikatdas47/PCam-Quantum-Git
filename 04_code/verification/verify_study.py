from pathlib import Path
import json,csv,numpy as np,hashlib
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,log_loss,brier_score_loss
from scipy.stats import t as tdist
P=Path('05_results/checks');E=Path('08_workspace/resnet50_evidence');F=E/'training/resnet50_final_v1';R=json.loads(Path('05_results/analysis/results.json').read_text());S=json.loads((F/'results/selection.json').read_text());specs=json.loads((F/'results/final_specs.json').read_text());scope=json.loads((F/'SCOPE.json').read_text());lock=json.loads((F/'results/test_lock.json').read_text())
assert len(specs)==13 and len([s for s in specs if s['type']=='neural'])==12
assert scope['dimensions']==[8] and scope['seeds']==[101,202,303] and len(scope['secondary'])==4
best=max(s['mean_auroc'] for s in S['schedules']);assert S['common_lr']==min(s['lr'] for s in S['schedules'] if best-s['mean_auroc']<=1e-8)==.003
assert S['winners']['resnet50']['definition']['id']=='fourier_nystrom_a0001'
assert abs(S['winners']['resnet50']['mean_auroc']-.8847102425890165)<1e-12
rows={r['tag']:r for r in csv.DictReader((F/'reports/test/metrics.csv').open())};metrics=[]
for spec in specs:
 tag=spec['tag']
 with np.load(F/f'results/test_predictions/{tag}.npz') as z:y=z['labels'];p=z['probabilities'];scores=z['scores'];cut=z['threshold'];ids=z['ids'];slides=z['slides']
 assert len(y)==32768 and int(y.sum())==16377 and len(np.unique(slides))==129
 tn,fp,fn,tp=map(int,confusion_matrix(y,p>=cut).ravel());calc={'AUROC':roc_auc_score(y,scores),'AP':average_precision_score(y,scores),'Brier':brier_score_loss(y,p),'log_loss':log_loss(y,p),'accuracy':(tn+tp)/len(y),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'precision':tp/(tp+fp),'F1':2*tp/(2*tp+fp+fn),'TN':tn,'FP':fp,'FN':fn,'TP':tp}
 for k,v in calc.items():assert abs(v-float(rows[tag][k]))<1e-12,(tag,k)
 if spec['type']=='neural':
  job=json.loads((F/f'results/jobs/{tag}.json').read_text());assert job['parameters']==(321 if spec['head'].startswith('mlp') else 25);assert job['identity']['shape']==[262144,8];assert job['identity']['epochs']==100 and job['identity']['lr']==.003;assert job['updates']==204800;assert len(job['trace'])==100;assert job['identity']['environment']['head_device']=='cpu' and job['identity']['environment']['head_dtype']=='float64'
 metrics.append({'tag':tag,'all_metrics_match':True})
with np.load(F/'reports/test/bootstrap_auroc.npz') as b:
 for n,c in enumerate(R['contrasts']):
  delta=np.mean([float(rows[k]['AUROC']) for k in c['left_tags']])-np.mean([float(rows[k]['AUROC']) for k in c['right_tags']]);assert abs(delta-c['delta'])<1e-12
  draws=np.mean([b[k] for k in c['left_tags']],axis=0)-np.mean([b[k] for k in c['right_tags']],axis=0);assert len(draws)==20000
  np.testing.assert_allclose(np.quantile(draws,[.025,.975]),c['conditional95'],rtol=0,atol=1e-12)
  if n:np.testing.assert_allclose(np.quantile(draws,[.00625,.99375]),c['family95'],rtol=0,atol=1e-12)
  assert abs(2*tdist.sf(abs(c['t_statistic']),128)-c['p_two_sided_posthoc'])<1e-12
  assert abs(c['delta']/c['jackknife_SE']-c['t_statistic'])<1e-12
raw=np.array([c['p_two_sided_posthoc'] for c in R['contrasts'][1:]]);order=np.argsort(raw);adj=np.empty(4);adj[order]=np.minimum(1,np.maximum.accumulate((4-np.arange(4))*raw[order]));np.testing.assert_allclose(adj,[c['p_Holm_secondary_posthoc'] for c in R['contrasts'][1:]],rtol=0,atol=1e-12)
# Check artifact lock exactly, without importing or changing experiment sources.
locked=0
for field in ['code','artifacts']:
 for path,digest in lock.get(field,{}).items():
  if isinstance(digest,str) and len(digest)==64:
   f=E/path;assert f.is_file(),path;assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,path;locked+=1
roles=E/'training/pcam_three_backbones/resnet50/manifests/roles.npz'
with np.load(roles) as z:assert int(z['calibration'].sum())==17090 and int((~z['calibration']).sum())==15678 and int((z['folds']>=0).sum())==16384
checks=json.loads(Path('05_results/checks/circuit_validation.json').read_text());assert checks['status']=='passed'
img=json.loads(Path('05_results/checks/exact_duplicate_audit.json').read_text());assert img['cross_split_exact_overlap']['train_test']['shared_unique_images']==33;assert img['splits']['test']['unique_images']==29383
report={'saved_prediction_metrics':metrics,'all_13_test_models_and_all_reported_metrics_verified':True,'selection_and_final_scope_match':True,'neural_parameter_epoch_update_and_runtime_counts_match':True,'all_5_nominal_and_4_adjusted_intervals_recomputed':True,'jackknife_p_values_and_Holm_adjustment_verified':True,'validation_roles_match':True,'existing_circuit_and_raw_image_audit_records_checked':True,'locked_hashes_verified':locked,'primary_delta':R['contrasts'][0]['delta'],'primary_conditional95':R['contrasts'][0]['conditional95'],'primary_posthoc_p':R['contrasts'][0]['p_two_sided_posthoc'],'new_fits':0,'limits':['No fresh external literature full text or external code audit in this consistency pass','No new raw pixel hash scan or complete bootstrap resampling rerun','Existing raw image and circuit validation records read; their earlier execution remains the provenance','One dataset and one frozen representation. Complete encoder feature arrays require recomputation for new training.']}
(P/'scientific_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='saved_prediction_metrics'},indent=2))
