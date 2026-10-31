"""Figures from saved estimates only; raster exports at 400 DPI."""
from pathlib import Path
import os,json,numpy as np
ROOT=Path(os.environ.get('PCAM_STUDY_ROOT',Path(__file__).resolve().parents[2]));os.environ['MPLCONFIGDIR']=str(ROOT/'paper/analysis/.matplotlib')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve,precision_recall_curve
P=ROOT/'paper';E=ROOT/'extensions/resnet50_extensions_evidence';F=P/'figures'
F.mkdir(exist_ok=True)
# Old files are already hash-verified under Previous.
for f in F.iterdir():
 if f.is_file():f.unlink()
S={r['group']:r for r in json.loads((P/'analysis/summary.json').read_text())};C=json.loads((P/'analysis/contrasts.json').read_text());old=json.loads((P/'Previous/code/inputs/results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','axes.axisbelow':True})
blue='#2166ac';orange='#b35806';green='#237a57'
def save(fig,name):
 fig.savefig(F/f'{name}.png',dpi=400)
 with Image.open(F/f'{name}.png') as im:im.convert('RGB').save(F/f'{name}.tif',dpi=(400,400),compression='tiff_lzw')
 plt.close(fig)
# Original comparison and selected-rate secondary family, with separate intervals.
fig,axs=plt.subplots(2,1,figsize=(6.5,5.4),layout='constrained')
rows=old['contrasts'];print('old contrast type',type(rows),flush=True)
if isinstance(rows,dict):rows=[rows['primary']]+rows['secondary']
# Use authoritative locked saved contrast file.
o=json.loads((E/'original_evidence/reports_test/analysis.json').read_text());rows=[o['primary']]+o['secondary']
labels=['CNOT on − classical','CNOT on − off','CNOT on − atan MLP','CNOT on − standard MLP','Atan − standard MLP']
for ax,rs,labs,title in [(axs[0],rows,labels,'A  Original shared-schedule comparison'),(axs[1],C[7:],labels[1:],'B  Selected-rate head comparisons (post hoc)')]:
 for i,r in enumerate(rs):
  ci=r.get('family95',r.get('CI95_family'));v=r['delta'];color=blue if ax==axs[0] and i==0 else orange
  ax.plot(np.array(ci)*100,[i,i],color=color,lw=1.6);ax.plot(v*100,i,'o',color=color,ms=4)
 ax.set_yticks(range(len(rs)),labs);ax.invert_yaxis();ax.axvline(0,color='.45',ls='--',lw=.8);ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.2);ax.set_xlabel('AUROC difference (percentage points)')
save(fig,'Fig1_head_comparisons')
# Compression performance + its prespecified post hoc contrast family.
fig,axs=plt.subplots(1,2,figsize=(6.5,3.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
dims=[8,32,128,2048];v=[S[f'compression_lr_d{d}']['AUROC_mean'] for d in dims];ci=np.array([S[f'compression_lr_d{d}']['CI95'] for d in dims]);axs[0].errorbar(range(4),v,yerr=np.array([np.array(v)-ci[:,0],ci[:,1]-np.array(v)]),fmt='o-',color=blue,capsize=3);axs[0].set_xticks(range(4),list(map(str,dims)));axs[0].set_xlabel('Feature dimensions');axs[0].set_ylabel('Test AUROC');axs[0].set_title('A  Logistic regression',loc='left');axs[0].set_ylim(.82,.95);axs[0].grid(axis='y',alpha=.2)
for i,r in enumerate(C[:3]):
 lo,hi=r['CI95_family'];v=r['delta'];axs[1].plot([lo*100,hi*100],[i,i],color=green);axs[1].plot(v*100,i,'o',color=green,ms=4)
axs[1].set_yticks(range(3),['2048 − 8','32 − 8','128 − 8']);axs[1].invert_yaxis();axs[1].axvline(0,color='.45',ls='--',lw=.8);axs[1].set_xlabel('AUROC difference (percentage points)');axs[1].set_title('B  Paired compression contrasts',loc='left');axs[1].grid(axis='x',alpha=.2)
save(fig,'Fig2_compression')
# Finite LR changes with conditional intervals and paired seed differences separately.
fig,axs=plt.subplots(1,2,figsize=(6.5,3.7),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
heads=['Standard MLP','Atan MLP','CNOT off','CNOT on'];markers=['o','s','^']
for i,r in enumerate(C[3:7]):
 lo,hi=r['CI95_family'];v=r['delta'];axs[0].plot([lo*100,hi*100],[i,i],color=blue);axs[0].plot(v*100,i,'o',color=blue,ms=4)
 for k,d in enumerate(r['seed_deltas']):axs[1].plot(d*100,i+(k-1)*.15,markers[k],color=['#2166ac','#b35806','#237a57'][k],ms=4,label=str([101,202,303][k]) if i==0 else None)
for ax,title in zip(axs,['A  Conditional family intervals','B  Paired optimization seeds']):
 ax.set_yticks(range(4),heads if ax==axs[0] else []);ax.invert_yaxis();ax.axvline(0,color='.45',ls='--',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_title(title,loc='left');ax.set_xlabel('Selected − shared AUROC (pp)')
axs[1].legend(title='Seed',loc='upper right',fontsize=7,title_fontsize=8)
for ax in axs:ax.set_ylim(3.45,-.45)
save(fig,'Fig3_learning_rates')
# Necessary probability-ranking diagnostics moved to supporting information.
fig,axs=plt.subplots(2,2,figsize=(6.5,5.8),layout='constrained')
for dim,col in zip(dims,['#777777','#55a868','#2166ac','#b35806']):
 z=np.load(E/f'predictions/test/compression_lr_d{dim}.npz');y=z['labels'];s=z['scores'];fpr,tpr,_=roc_curve(y,s);pr,re,_=precision_recall_curve(y,s)
 axs[0,0].plot(fpr,tpr,color=col,lw=1,label=f'{dim}D');axs[0,1].plot(re,pr,color=col,lw=1,label=f'{dim}D')
for h,col in zip(['mlp_standard','mlp_atan','ry_off','ry_on'],['#2166ac','#9467bd','#237a57','#b35806']):
 for seed in [101,202,303]:
  z=np.load(E/f'predictions/test/tuned_{h}_{seed}.npz');y=z['labels'];s=z['scores'];fpr,tpr,_=roc_curve(y,s);pr,re,_=precision_recall_curve(y,s)
  axs[1,0].plot(fpr,tpr,color=col,alpha=.65,lw=.7,label=h.replace('mlp_standard','Standard MLP').replace('mlp_atan','Atan MLP').replace('ry_off','CNOT off').replace('ry_on','CNOT on') if seed==101 else None);axs[1,1].plot(re,pr,color=col,alpha=.65,lw=.7)
for row in range(2):
 axs[row,0].plot([0,1],[0,1],ls=':',color='.6',lw=.7);axs[row,0].set(xlabel='False positive rate',ylabel='True positive rate');axs[row,1].axhline(float(y.mean()),ls=':',color='.6',lw=.7);axs[row,1].set(xlabel='Recall',ylabel='Precision')
 for ax in axs[row]:ax.set(xlim=(0,1),ylim=(0,1.02));ax.grid(alpha=.15)
axs[0,0].set_title('A  Compression ROC',loc='left');axs[0,1].set_title('B  Compression precision–recall',loc='left');axs[1,0].set_title('C  Selected-rate head ROC',loc='left');axs[1,1].set_title('D  Selected-rate precision–recall',loc='left');axs[0,0].legend(loc='lower right');axs[1,0].legend(loc='lower right',fontsize=7)
save(fig,'S1_Fig_ranking_curves')
# Learning diagnostics and measured compute, distinct from wall time.
fig,axs=plt.subplots(2,2,figsize=(6.5,5.5),layout='constrained')
for ax,h,title in zip(axs.flat,['mlp_standard','mlp_atan','ry_off','ry_on'],heads):
 for seed in [101,202,303]:
  j=json.loads((E/f'jobs/pair_{h}_{seed}.json').read_text())
  for schedule,col,ls in [('shared',blue,'--'),('tuned',orange,'-')]:ax.plot([r['epoch'] for r in j['trace'][schedule]],[r['diagnostic_train_BCE'] for r in j['trace'][schedule]],color=col,ls=ls,lw=.8,alpha=.65,label=schedule.replace('tuned','Selected') if seed==101 else None)
 ax.set_title(title,loc='left');ax.set(xlabel='Epoch',ylabel='Diagnostic training BCE');ax.grid(alpha=.15)
axs[0,0].legend();save(fig,'S2_Fig_training_diagnostics')
verification=[]
for f in sorted(F.glob('*')):
 with Image.open(f) as im:
  assert all(abs(d-400)<.1 for d in im.info['dpi']);verification.append({'file':f.name,'pixels':list(im.size),'dpi':[float(x) for x in im.info['dpi']]})
(P/'analysis/figure_verification.json').write_text(json.dumps(verification,indent=2))
