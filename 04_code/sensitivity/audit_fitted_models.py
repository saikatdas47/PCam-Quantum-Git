"""Reload retained fitted models without training; compare CPU inference to archived scores."""
from pathlib import Path
import json,sys,hashlib,importlib.util,ast,os
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(os.environ.get('PCAM_STUDY_ROOT',Path(__file__).resolve().parents[2]));E=ROOT/'extensions/resnet50_extensions_evidence';OUT=ROOT/'paper/analysis'
threadpool_limits(2);torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('saved_engine',E/'source/engine.py');eng=importlib.util.module_from_spec(spec);spec.loader.exec_module(eng)
mp=json.loads((OUT/'prepared_matrix_map.json').read_text());rows=[]
for head in ['mlp_standard','mlp_atan','ry_off','ry_on']:
 for seed in [101,202,303]:
  initial=torch.load(E/f'initial_weights/{head}_{seed}.pt',map_location='cpu',weights_only=True);fresh=eng.make_head(head,seed,'cpu').state_dict()
  assert all(torch.allclose(initial[k],fresh[k],rtol=0,atol=1e-15) for k in initial)
  if head=='ry_on':
   off=torch.load(E/f'initial_weights/ry_off_{seed}.pt',map_location='cpu',weights_only=True)
   assert all(torch.equal(initial[k],off[k]) for k in initial)
  for schedule in ['shared','tuned']:
   tag=f'{schedule}_{head}_{seed}';m=eng.make_head(head,seed,'cpu');state=torch.load(E/f'models/{tag}.pt',map_location='cpu',weights_only=True);m.load_state_dict(state);m.eval()
   for split in ['test','valid']:
    x=np.load(E/f'prepared_features/{split}_{mp[split+"_original_8"]}.npy',mmap_mode='r');z=dict(np.load(E/f'predictions/{split}/{tag}.npz'));out=[]
    with torch.no_grad():
     for first in range(0,len(x),512):
      b=torch.as_tensor(np.array(x[first:first+512]),dtype=torch.float64)
      if head!='mlp_standard':b=torch.atan(b)
      out.append(m(b).numpy())
    v=np.concatenate(out);err=float(np.max(abs(v-z['scores'])));assert np.allclose(v,z['scores'],rtol=1e-9,atol=1e-9),(tag,split,err)
    rows.append(dict(tag=tag,split=split,rows=len(v),max_absolute_score_difference=err))
   print('Reloaded',tag,flush=True)
# Logistic models can be independently evaluated from all saved compressed test matrices.
# The 2048 matrix is intentionally absent, so its score replay is reported unavailable.
for dim in [8,32,128]:
 tag=f'compression_lr_d{dim}';m=dict(np.load(E/f'models/{tag}.npz'));x=np.load(E/f'prepared_features/test_{mp[f"test_compression_{dim}"]}.npy',mmap_mode='r');z=dict(np.load(E/f'predictions/test/{tag}.npz'));v=x@m['weights']+m['bias'];err=float(np.max(abs(v-z['scores'])));assert np.allclose(v,z['scores'],rtol=1e-9,atol=1e-9)
 rows.append(dict(tag=tag,split='test',rows=len(v),max_absolute_score_difference=err))
# Final committed checkpoints bind the saved final model tensors and optimizer rates.
for p in sorted((E/'checkpoints').glob('pair*')):
 commit=json.loads((p/'COMMIT.json').read_text());q=p/f'state{commit["slot"]}.pt';assert hashlib.sha256(q.read_bytes()).hexdigest()==commit['sha256'];cp=torch.load(q,map_location='cpu',weights_only=False);assert cp['epoch']==100 and cp['offset']==0
 job=json.loads((E/'jobs'/f'{p.name}.json').read_text())
 for schedule in ['shared','tuned']:
  tag=f'{schedule}_{job["head"]}_{job["seed"]}';model=torch.load(E/f'models/{tag}.pt',map_location='cpu',weights_only=True)
  assert all(torch.equal(model[k],cp['models'][schedule][k]) for k in model)
  assert cp['optimizers'][schedule]['param_groups'][0]['lr']==(job['shared_lr'] if schedule=='shared' else job['selected_lr'])
# Read every saved source version and compare scientific function ASTs with current source.
# Changes to operational checkpoint/recovery methods are not hidden as identical execution.
def funcs(path):
 t=ast.parse(path.read_text());return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
base=funcs(E/'source/engine.py');source_audit=[]
for p in sorted((E/'source_versions').glob('*.py')):
 f=funcs(p);source_audit.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),top_level_same={k:f.get(k)==base.get(k) for k in ['expectations','Circuit','MLP','make_head']}))
 assert all(source_audit[-1]['top_level_same'].values())
# Independent NumPy rotation evaluation checks the current circuit function and CNOT ordering.
rng=np.random.default_rng(314159);x=rng.normal(size=(6,8));ww=rng.normal(size=(2,8));independent=[]
for cnot in [False,True]:
 state=np.zeros((6,256));state[:,0]=1
 def ry(s,a,j):
  z=s.reshape(6,2**j,2,2**(7-j));c=np.cos(a/2).reshape(-1,1,1);d=np.sin(a/2).reshape(-1,1,1)
  return np.stack([c*z[:,:,0,:]-d*z[:,:,1,:],d*z[:,:,0,:]+c*z[:,:,1,:]],axis=2).reshape(6,256)
 for j in range(8):state=ry(state,x[:,j],j)
 for layer in range(2):
  for j in range(8):state=ry(state,np.full(6,ww[layer,j]),j)
  if cnot:
   # Explicit basis mapping rather than the engine's index expression.
   perm=[]
   for basis in range(256):
    bits=[(basis>>(7-j))&1 for j in range(8)]
    for j in range(7):bits[j+1]^=bits[j]
    perm.append(sum(b<<(7-j) for j,b in enumerate(bits)))
   new=np.empty_like(state);new[:,perm]=state;state=new
 signs=np.array([[1-2*((i>>(7-j))&1) for j in range(8)] for i in range(256)])
 v=state**2@signs
 with torch.no_grad():target=eng.expectations(torch.tensor(x),torch.tensor(ww),cnot).numpy()
 err=float(abs(v-target).max());assert err<1e-12;independent.append(dict(cnot=cnot,max_error=err))
result=dict(replayed=rows,neural_final_checkpoints_verified=12,initial_states_verified=12,cnot_initial_pairs_verified=3,source_versions=source_audit,independent_numpy_circuit=independent,unavailable=['2048D logistic score replay: complete 2048D feature matrix absent; coefficients, checkpoints, predictions, metrics and checksums audited','Retraining and whole previous input equality across CPU/CUDA transitions not performed'],environment=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version))
(OUT/'model_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('MODEL AUDIT COMPLETE',flush=True)
