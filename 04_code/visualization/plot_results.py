"""Scientific study figures from immutable exported predictions."""
from pathlib import Path
import os,json
P=Path('06_figures');P.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str((Path('08_workspace/cache/matplotlib')).resolve());os.environ['XDG_CACHE_HOME']=str(Path('.cache').resolve())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from PIL import Image
F=Path('08_workspace/resnet50_evidence/training/resnet50_final_v1');r=json.loads(Path('05_results/analysis/results.json').read_text())
from matplotlib import font_manager
fonts={f.name for f in font_manager.fontManager.ttflist}
figure_font='Arial' if 'Arial' in fonts else 'DejaVu Sans'
plt.rcParams.update({'font.family':figure_font,'font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,'axes.linewidth':.7,'lines.linewidth':1.15,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
heads=['mlp_standard','mlp_atan','ry_off','ry_on','selected_classical'];names=['Standard MLP','Arctangent MLP','RY CNOT off','RY CNOT on','Selected classical'];colors=['#0072B2','#56B4E9','#D89C00','#D55E00','#009E73']
def save(fig,name):
 fig.savefig(P/f'{name}.png',dpi=400,bbox_inches='tight')
 fig.savefig(P/f'{name}.pdf',bbox_inches='tight')
 im=Image.open(P/f'{name}.png').convert('RGB');im.save(P/f'{name}.tif',compression='tiff_lzw',dpi=(400,400));plt.close(fig)
# The editable workflow diagram is stored in 03_methodology.
# Each panel shows one neural head, with all seeds and the same classical reference.
D={}
for p in (F/'results/test_predictions').glob('*.npz'):
 with np.load(p) as z:D[p.stem]={k:z[k].copy() for k in z.files}
reference=D['resnet50_selected_classical']; audit=[]
from sklearn.metrics import roc_auc_score,average_precision_score,auc
for tag,d in sorted(D.items()):
 y=d['labels'];score=d['scores'];fpr,tpr,thresholds=roc_curve(y,score,drop_intermediate=False)
 precision,recall,prthresholds=precision_recall_curve(y,score)
 assert len(y)==32768 and np.isfinite(score).all()
 assert np.all(np.diff(fpr)>=0) and np.all(np.diff(tpr)>=0)
 assert np.all(np.diff(recall)<=0)
 assert np.allclose([fpr[0],tpr[0],fpr[-1],tpr[-1]],[0,0,1,1])
 assert abs(auc(fpr,tpr)-roc_auc_score(y,score))<1e-12
 for k in np.linspace(1,len(thresholds)-1,11,dtype=int):
  pred=score>=thresholds[k]
  assert abs(fpr[k]-np.mean(pred[y==0]))<1e-12
  assert abs(tpr[k]-np.mean(pred[y==1]))<1e-12
 for k in np.linspace(0,len(prthresholds)-1,11,dtype=int):
  pred=score>=prthresholds[k]
  assert abs(precision[k]-np.mean(y[pred]))<1e-12
  assert abs(recall[k]-np.mean(pred[y==1]))<1e-12
 audit.append(dict(tag=tag,AUROC=float(roc_auc_score(y,score)),AP=float(average_precision_score(y,score)),positive=int(y.sum()),negative=int((y==0).sum()),unique_scores=int(len(np.unique(score))),roc_verified=True,pr_verified=True))
for kind,filename in [('roc','01_roc'),('pr','02_precision_recall')]:
 fig,axes=plt.subplots(2,2,figsize=(6.5,5.2))
 for i,(h,name,color,ax) in enumerate(zip(heads[:-1],names[:-1],colors[:-1],axes.flat)):
  for seed,ls in zip([101,202,303],['-',':','-.']):
   d=D[f'resnet50_q8_{h}_{seed}'];y=d['labels'];score=d['scores']
   if kind=='roc':x,v,_=roc_curve(y,score);metric=roc_auc_score(y,score)
   else:v,x,_=precision_recall_curve(y,score);metric=average_precision_score(y,score)
   ax.step(x,v,where='post',color=color,ls=ls,lw=1.1,alpha=.9,label=f'Seed {seed}  {metric:.3f}')
  y=reference['labels'];score=reference['scores']
  if kind=='roc':
   x,v,_=roc_curve(y,score);metric=roc_auc_score(y,score);ax.plot([0,1],[0,1],ls=':',color='.65',lw=.7)
  else:
   v,x,_=precision_recall_curve(y,score);metric=average_precision_score(y,score);ax.axhline(y.mean(),ls=':',color='.65',lw=.7)
  ax.step(x,v,where='post',color='#222222',ls='--',lw=1,label=f'Classical  {metric:.3f}')
  ax.set(xlim=(0,1),ylim=(0,1.02),xticks=np.arange(0,1.01,.2),yticks=np.arange(0,1.01,.2),xlabel='False positive rate' if kind=='roc' else 'Recall (sensitivity)',ylabel='True positive rate (sensitivity)' if kind=='roc' else 'Precision')
  ax.set_title(f'{chr(65+i)}   {name}',loc='left',fontweight='bold',fontsize=9)
  ax.legend(loc='lower right' if kind=='roc' else 'lower left',frameon=False,fontsize=6.5)
 fig.tight_layout(w_pad=1.5,h_pad=1.5);save(fig,filename)
(Path('05_results/checks/curve_checks.json')).write_text(json.dumps(audit,indent=2))
fig,ax=plt.subplots(figsize=(6.5,3.0))
labels=['CNOT on minus selected classical','CNOT on minus CNOT off','CNOT on minus arctangent MLP','CNOT on minus standard MLP','Arctangent minus standard MLP']
for i,c in enumerate(r['contrasts']):
 lo,hi=c['conditional95'] if i==0 else c['family95'];ax.errorbar(c['delta']*100,i,xerr=[[100*(c['delta']-lo)],[100*(hi-c['delta'])]],fmt='o',markersize=4,color=colors[0] if i==0 else colors[3],capsize=3)
ax.axvline(0,ls='--',lw=.8,color='.5');ax.set(yticks=range(5),yticklabels=labels,xlabel='Difference in mean AUROC (percentage points)');ax.invert_yaxis();fig.tight_layout();save(fig,'03_paired_comparisons')
fig,axes=plt.subplots(1,2,figsize=(6.5,3.0),gridspec_kw={'width_ratios':[1.15,1]})
for i,h in enumerate(heads[:-1]):
 times=[]
 for seed in [101,202,303]:
  j=json.loads((F/f'results/jobs/resnet50_q8_{h}_{seed}.json').read_text());tr=j['trace'];axes[0].plot([v['epoch'] for v in tr],[v['diagnostic_train_bce'] for v in tr],color=colors[i],alpha=.75,label=names[i] if seed==101 else None);times.append(j['training_compute_seconds'])
 axes[1].scatter(times,np.repeat(i,3)+np.array([-.11,0,.11]),color=colors[i],s=20,alpha=.8);axes[1].plot([np.mean(times)],[i],marker='|',markersize=15,color='black')
axes[0].set(xlabel='Epoch',ylabel='Training cohort cross-entropy');axes[0].legend(frameon=False,fontsize=6.5)
axes[1].set(xscale='log',xlabel='Observed neural compute time (s)',yticks=range(4),yticklabels=['Standard\nMLP','Arctangent\nMLP','CNOT off','CNOT on']);axes[1].invert_yaxis()
for ax,label in zip(axes,['A','B']):ax.text(-.18,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
fig.tight_layout(w_pad=1.3);save(fig,'04_training_and_time')
print('Four result figures saved. The editable workflow is in 03_methodology. All 13 ROC and PR curves checked against confusion counts.')
