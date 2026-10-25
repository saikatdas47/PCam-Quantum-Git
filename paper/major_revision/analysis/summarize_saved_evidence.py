"""Reporting additions from existing fitted transforms, selection records and predictions."""
from pathlib import Path
import json,csv,sys,importlib.util,hashlib,warnings
import numpy as np
import joblib,torch
from sklearn.metrics import roc_auc_score
sys.dont_write_bytecode=True
P=Path('paper/major_revision');E=Path('resnet50_evidence');F=E/'training/resnet50_final_v1';T=E/'training/pcam_three_backbones/resnet50';result={}
def csvwrite(name,rows):
 with (P/'tables'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
variance=[];warn=[]
for label,path in [('final',F/'checkpoints/resnet50_final_transform.joblib')]+[(f'cv{i}',T/f'checkpoints/resnet50_cv{i}_transform.joblib') for i in range(3)]:
 with warnings.catch_warnings(record=True) as messages:
  warnings.simplefilter('always');obj=joblib.load(path)
 warn.extend(str(m.message) for m in messages)
 ratios=np.asarray(obj['pca'].explained_variance_ratio_,dtype=float)
 variance.append({'fit':label,'training_rows':len(obj['training_ids']),'components':8,'cumulative_explained_variance':float(ratios.sum()),**{f'PC{i+1}':float(v) for i,v in enumerate(ratios)}})
csvwrite('PCA_explained_variance.csv',variance);result['PCA']=variance;result['pickle_version_warnings']=sorted(set(warn));result['PCA_method']='Stored attributes read only. No PCA fit or transform under the changed local sklearn version.'
protocol=json.loads((E/'experiments/experiment_011_train_resnet50/configs/protocol.json').read_text());rows=[]
for path in sorted((T/'results/cv').glob('*.json')):
 tag=path.stem;fold=int(tag.split('_cv')[1][0]);definition=tag.split('_cv')[1][2:];row={'tag':tag,'fold':fold,'candidate':definition,'AUROC':json.loads(path.read_text())['auroc']};rows.append(row)
csvwrite('all_78_training_fold_scores.csv',rows)
headrows=[]
for head in ['mlp_standard','mlp_atan','ry_off','ry_on']:
 for lr in [.001,.003,.01]:
  values=[r['AUROC'] for r in rows if r['candidate']==f'{head}_lr{lr}'];assert len(values)==3
  headrows.append({'head':head,'learning_rate':lr,'fold0':values[0],'fold1':values[1],'fold2':values[2],'mean_AUROC':float(np.mean(values)),'common_selected_lr':lr==.003})
csvwrite('per_head_learning_rate_results.csv',headrows);result['neural_selection']=headrows;result['candidate_grid']=protocol['classical_candidates'];(P/'results/complete_selection_configuration.json').write_text(json.dumps(protocol,indent=2))
# Reconstruct initialization only, do not fit. Record current runtime to bound cross-version claims.
source=E/'experiments/experiment_011_train_resnet50/scripts/torch_circuit.py';spec=importlib.util.spec_from_file_location('audit_circuit',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
init=[]
for seed in [71911,101,202,303]:
 for head in ['mlp_standard','mlp_atan','ry_off','ry_on']:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed)
   if head.startswith('mlp'):model=m.MLP(8,32)
   else:model=m.TwoLayerRYHead(8,True)
  for name,tensor in model.state_dict().items():
   a=tensor.detach().numpy().ravel()
   for index,value in enumerate(a):init.append({'head':head,'seed':seed,'parameter':name,'index':index,'value':float(value)})
csvwrite('reconstructed_initial_parameters.csv',init)
result['initialization']={'neural_selection_seed':71911,'final_seeds':[101,202,303],'rotation_weights':'Independent normal mean0 SD0.1','circuit_readout':'torch.nn.Linear uniform [-1/sqrt(8),1/sqrt(8)] for weights and bias','MLP':'torch.nn.Linear uses uniform bounds 1/sqrt(fan_in) for weights and biases. Hidden fan_in8, output fan_in32.','pairing':'CNOT off state copied from CNOT on within each seed in original torch_training.py','table_status':'Reconstructed using retained initialization definitions under local torch'+torch.__version__+'. Not claimed to be archived original initial tensors or a cross-version RNG proof.'}
# Slide counts from every split.
slide_rows=[];slide_summary={}
for split in ['train','valid','test']:
 with np.load(T/f'manifests/{split}.npz') as z:slides=z['slides'].copy();labels=z['labels'].copy()
 groups,counts=np.unique(slides,return_counts=True)
 for g,n in zip(groups,counts):slide_rows.append({'split':split,'slide':str(g),'patches':int(n),'positives':int(labels[slides==g].sum())})
 slide_summary[split]={'slides':len(groups),'min':int(counts.min()),'median':float(np.median(counts)),'max':int(counts.max()),'p25':float(np.quantile(counts,.25)),'p75':float(np.quantile(counts,.75)),'largest_slide_fraction':float(counts.max()/counts.sum()),'largest5_slide_fraction':float(np.sort(counts)[-5:].sum()/counts.sum())}
csvwrite('patches_per_slide.csv',slide_rows);result['slide_summary']=slide_summary
# Existing jackknife approach regenerated for primary contrast, preserving all fits.
D={}
for name in ['ry_on_101','ry_on_202','ry_on_303','selected_classical']:
 tag='resnet50_'+('q8_'+name if name!='selected_classical' else name)
 with np.load(F/f'results/test_predictions/{tag}.npz') as z:D[name]={k:z[k].copy() for k in z.files}
y=D['selected_classical']['labels'];slides=D['selected_classical']['slides'];groups=np.unique(slides)
def delta(mask):return float(np.mean([roc_auc_score(y[mask],D[k]['scores'][mask]) for k in D if k!='selected_classical'])-roc_auc_score(y[mask],D['selected_classical']['scores'][mask]))
full=delta(np.ones(len(y),dtype=bool));jack=[]
for g in groups:
 value=delta(slides!=g);jack.append({'slide':str(g),'patches':int((slides==g).sum()),'positive_patches':int(y[slides==g].sum()),'primary_difference_without_slide':value,'change_from_full':value-full})
jack.sort(key=lambda r:abs(r['change_from_full']),reverse=True);csvwrite('primary_slide_influence.csv',jack)
result['primary_influence']={'full_delta':full,'min_leave_one_delta':min(r['primary_difference_without_slide'] for r in jack),'max_leave_one_delta':max(r['primary_difference_without_slide'] for r in jack),'top5':jack[:5],'timing':'Post hoc descriptive sensitivity. No slide removed from the reported comparison.'}
result['training_diagnostics']=[]
for path in sorted((F/'results/jobs').glob('resnet50_q8_*.json')):
 obj=json.loads(path.read_text());trace=obj['trace'];decline=trace[-20]['diagnostic_train_bce']-trace[-1]['diagnostic_train_bce'];result['training_diagnostics'].append({'tag':path.stem,'last20_decline':decline,'gate_passed':decline<=.02,'first_epoch_loss':trace[0]['diagnostic_train_bce'],'final_loss':trace[-1]['diagnostic_train_bce']})
result['status']='Executed reporting additions and post hoc descriptive audits. No new training or test selection.'
(P/'results/reporting_additions.json').write_text(json.dumps(result,indent=2));print(json.dumps({'PCA':variance,'slide_summary':slide_summary,'primary_influence':result['primary_influence']},indent=2))
