"""Post hoc independent dense-matrix and finite-difference circuit validation.
No parameters are fitted and no dataset scores are used.
"""
from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np
import torch
sys.dont_write_bytecode=True
P=Path('05_results');source=Path('08_workspace/resnet50_evidence/experiments/experiment_011_train_resnet50/scripts/torch_circuit.py')
spec=importlib.util.spec_from_file_location('audited_circuit',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.set_num_threads(2);rng=np.random.default_rng(240624);checks=[]
def rotation(angle,wire,q):
 c,s=np.cos(angle/2),np.sin(angle/2);gate=np.array([[c,-s],[s,c]]);out=np.array([[1.]])
 for j in range(q):out=np.kron(out,gate if j==wire else np.eye(2))
 return out
def cnot_matrix(control,target,q):
 out=np.zeros((2**q,2**q))
 for column in range(2**q):
  bits=list(f'{column:0{q}b}')
  if bits[control]=='1':bits[target]='0' if bits[target]=='1' else '1'
  out[int(''.join(bits),2),column]=1
 return out
def dense(a,w,on):
 q=len(a);state=np.zeros(2**q);state[0]=1
 for j in range(q):state=rotation(a[j],j,q)@state
 for layer in range(2):
  for j in range(q):state=rotation(w[layer,j],j,q)@state
  if on:
   for j in range(q-1):state=cnots[q,j]@state
 probs=state**2
 return np.array([sum(p*(1 if f'{i:0{q}b}'[j]=='0' else -1) for i,p in enumerate(probs)) for j in range(q)])
cnots={(q,j):cnot_matrix(j,j+1,q) for q in [4,8] for j in range(q-1)}
for q in [4,8]:
 # Exhaustive input-basis tests explicitly establish MSB wire zero and CNOT direction.
 for j in range(q-1):
  for basis in range(2**q):
   bits=list(f'{basis:0{q}b}');expected=bits.copy()
   if bits[j]=='1':expected[j+1]='1' if bits[j+1]=='0' else '0'
   actual=int(cnots[q,j][:,basis].argmax());assert actual==int(''.join(expected),2)
 for on in [False,True]:
  errors=[];grad_errors=[];analytical=[]
  for trial in range(6):
   a=rng.uniform(-np.pi,np.pi,q);w=rng.normal(0,1,(2,q));readout=rng.normal(size=q)
   if trial==0:a[:]=0;w[:]=0
   if trial==1:a[:]=np.pi/2
   tx=torch.tensor(a[None],dtype=torch.float64,requires_grad=True);tw=torch.tensor(w,dtype=torch.float64,requires_grad=True)
   expected=dense(a,w,on);actual=mod.expectations(tx,tw,on)[0];errors.append(float(np.max(np.abs(actual.detach().numpy()-expected))))
   objective=(actual*torch.tensor(readout)).sum();ga,gw=torch.autograd.grad(objective,(tx,tw))
   vector=np.concatenate([a,w.ravel()]);finite=[];epsilon=1e-6
   for k in range(len(vector)):
    plus=vector.copy();minus=vector.copy();plus[k]+=epsilon;minus[k]-=epsilon
    finite.append((dense(plus[:q],plus[q:].reshape(2,q),on)@readout-dense(minus[:q],minus[q:].reshape(2,q),on)@readout)/(2*epsilon))
   grad_errors.append(float(np.max(np.abs(np.concatenate([ga.detach().numpy().ravel(),gw.detach().numpy().ravel()])-finite))))
   if not on:analytical.append(float(np.max(np.abs(expected-np.cos(a+w.sum(axis=0))))))
  assert max(errors)<1e-12 and max(grad_errors)<1e-8
  if analytical:assert max(analytical)<1e-12
  checks.append({'q':q,'CNOT':on,'forward_cases':6,'gradient_coordinates_per_case':3*q,'max_forward_absolute_error':max(errors),'max_gradient_absolute_error':max(grad_errors),'analytical_cosine_max_error':max(analytical) if analytical else None,'passed':True})
result={'status':'passed','timing':'Post hoc implementation validation after test inspection','independence':'Separate NumPy full Kronecker matrices and explicit computational-basis CNOT matrices. Central finite differences of this reference test PyTorch autograd. Same researcher/tool workflow, not external replication.','seed':240624,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wire_order':'Wire0 is most significant. Chains apply 0 to1 through q-2 toq-1 in that order.','checks':checks,'environment':{'numpy':np.__version__,'torch':torch.__version__},'finite_difference_step':1e-6,'forward_tolerance':1e-12,'gradient_tolerance':1e-8}
(P/'checks/circuit_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
