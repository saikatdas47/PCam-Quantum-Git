from pathlib import Path
import os
os.environ['MPLCONFIGDIR']=str(Path('paper/plos_one_initial/analysis/.matplotlib').resolve())
os.environ['XDG_CACHE_HOME']=str(Path('.cache').resolve())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.1,5.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,text,w=.8):ax.text(x,y,text,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.55',facecolor='#EDF3F7',edgecolor='#326580'),transform=ax.transAxes)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#444'),xycoords='axes fraction')
box(.5,.94,'Official PCam splits\nTrain 262,144 | Validation 32,768 | Test 32,768')
box(.5,.77,'Frozen ImageNet ResNet50 → 2,048 features\nTraining-fitted PCA (8 components) and scaling')
box(.5,.56,'Training-only 3-fold WSI selection (16,384 rows)\n4 neural heads × 3 seeds; selected classical reference\nFinal fits use all official training rows')
box(.5,.34,'Validation: 27 calibration + 27 diagnostic slides\nThresholds fixed; models, transforms and protocol locked')
box(.5,.12,'Official test: 32,768 patches, 129 WSI clusters\nPaired WSI bootstrap (20,000 draws)\nPost hoc jackknife sensitivity reported separately')
for a,b in [((.5,.885),(.5,.825)),((.5,.715),(.5,.635)),((.5,.48),(.5,.405)),((.5,.275),(.5,.205))]:arrow(a,b)
fig.tight_layout()
for ext in ['png','pdf','tif']:fig.savefig(Path('paper/plos_one_initial/figures')/f'Fig1_study_flow.{ext}',dpi=600,bbox_inches='tight')
