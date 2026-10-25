"""Conditional paired slide intervals and journal figures, from saved predictions only."""
from common import *
import argparse,csv
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,brier_score_loss,confusion_matrix,roc_curve,precision_recall_curve
from run import selected_tags,verify_lock
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import io
AVAILABLE_FONTS={f.name for f in font_manager.fontManager.ttflist}
FIGURE_FONT=next((f for f in ['Arial','Times New Roman'] if f in AVAILABLE_FONTS),'DejaVu Sans')
JOURNAL_FONT_READY=FIGURE_FONT in ['Arial','Times New Roman']

COLORS={'mlp_standard':'#0072B2','mlp_atan':'#56B4E9','ry_off':'#E69F00','ry_on':'#D55E00','selected':'#009E73'}
LABELS={'mlp_standard':'MLP standard','mlp_atan':'MLP atan','ry_off':'CNOT off','ry_on':'CNOT on','selected':'Selected classical'}
plt.rcParams.update({'font.family':FIGURE_FONT,'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300})


def weighted_auc_draws(y,scores,weights):
    order=np.argsort(scores,kind='stable');s=scores[order];y=y[order];starts=np.r_[0,np.flatnonzero(np.diff(s))+1]
    w=weights[:,order];pos=np.add.reduceat(w*y,starts,axis=1);neg=np.add.reduceat(w*(1-y),starts,axis=1)
    return (pos*(np.cumsum(neg,axis=1)-neg+.5*neg)).sum(axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))


def export(fig,folder,name):
    for extension in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight')
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=300,bbox_inches='tight',facecolor='white');buffer.seek(0)
    im=Image.open(buffer).convert('RGB');im.save(folder/f'{name}.tif',compression='tiff_lzw',dpi=(300,300))
    width,height=im.size
    atomic_json(folder/f'{name}_format.json',{'font':FIGURE_FONT,'journal_font_ready':JOURNAL_FONT_READY,'dpi':300,'pixels':[width,height],'rgb':True,'TIFF_bytes':(folder/f'{name}.tif').stat().st_size,'dimensions_within_PLOS_limits':789<=width<=2250 and height<=2625,'PLOS_source':'https://journals.plos.org/plosone/s/figures','visual_inspection_required':True})
    plt.close(fig)


def main(stage):
    if stage=='test':
        verify_lock(check_environment=False,artifacts=False);verification=json.loads((BASE/'results/test_verification.json').read_text())
        for p,h in verification['prediction_sha256'].items():assert digest(p)==h
    selection=json.loads((BASE/'results/selection.json').read_text());folder=BASE/'results'/f'{stage}_predictions'
    out=BASE/'reports'/stage;figures=BASE/'figures'/stage
    out.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    data={}
    mask=None
    if stage=='valid':
        with np.load(MAN/'roles.npz') as d:mask=~d['calibration']
    for path in sorted(folder.glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:row={k:d[k].copy() for k in d.files}
        if mask is not None:
            for key in ['ids','slides','labels','scores','probabilities']:row[key]=row[key][mask]
        data[path.stem]=row
    assert data
    bind(out/'input_identity.json',{'code':code_identity(),'predictions':{str(p):digest(p) for p in sorted(folder.glob('*.npz'))},'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest() if mask is not None else None})
    cached={}
    if (out/'bootstrap_auroc.npz').exists():
        commit=json.loads((out/'bootstrap_commit.json').read_text());assert digest(out/'bootstrap_auroc.npz')==commit['sha256']
        with np.load(out/'bootstrap_auroc.npz') as old:cached={k:old[k].copy() for k in old.files}
    reference=next(iter(data.values()));y=reference['labels'];slides=reference['slides'];groups,inverse=np.unique(slides,return_inverse=True)
    for d in data.values():
        for k in ['ids','slides','labels']:np.testing.assert_array_equal(d[k],reference[k])
    # Validate the weighting formula on real scores and nonuniform, joint slide draws.
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);draws=CONFIG['bootstrap_draws'];counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=draws)
    positive=np.bincount(inverse,weights=y);negative=np.bincount(inverse,weights=1-y);invalid=(counts@positive==0)|(counts@negative==0);redraws=0
    while invalid.any():
        redraws+=int(invalid.sum());counts[invalid]=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=int(invalid.sum()));invalid=(counts@positive==0)|(counts@negative==0)
    probe=counts[:3,inverse]
    first_score=reference['scores']
    np.testing.assert_allclose(weighted_auc_draws(y,first_score,probe),[roc_auc_score(y,first_score,sample_weight=w) for w in probe],rtol=1e-12,atol=1e-12)
    atomic_npz(out/'bootstrap_counts.npz',counts=counts,slides=groups)
    aucs={};point={};metrics=[]
    equal_slide=1/np.bincount(inverse)[inverse]
    for tag,d in data.items():
        scores=d['scores'];p=d['probabilities'];cut=float(d['threshold']);pred=p>=cut
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        point[tag]=float(roc_auc_score(y,scores));auc=cached.get(tag,np.empty(draws))
        part=out/'bootstrap_parts'/f'{tag}.npz';commit=part.with_suffix('.json')
        if tag not in cached and part.exists() and commit.exists():
            assert json.loads(commit.read_text())['sha256']==digest(part)
            with np.load(part) as old:auc=old['auroc'].copy()
        elif tag not in cached:
            for start in range(0,draws,64):auc[start:start+64]=weighted_auc_draws(y,scores,counts[start:start+64,inverse])
            atomic_npz(part,auroc=auc);atomic_json(commit,{'sha256':digest(part)})
        assert auc.shape==(draws,)
        aucs[tag]=auc
        metrics.append({'tag':tag,'AUROC':point[tag],'AP':float(average_precision_score(y,scores)),'accuracy':float(accuracy_score(y,pred)),'accuracy_fixed05':float(accuracy_score(y,p>=.5)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'Brier':float(brier_score_loss(y,p)),'threshold':cut,'equal_slide_weighted_AUROC':float(roc_auc_score(y,scores,sample_weight=equal_slide))})
        print('Analyzed',stage,tag,flush=True)
    def mean_draw(tags):return np.mean([aucs[t] for t in tags],axis=0)
    def mean_point(tags):return float(np.mean([point[t] for t in tags]))
    def core(bb,q,head):return [f'{bb}_q{q}_{head}_{seed}' for seed in CONFIG['final_seeds']]
    def interval(identifier,left,right,family=1):
        values=mean_draw(left)-mean_draw(right)
        return {'id':identifier,'delta':mean_point(left)-mean_point(right),'conditional95':np.quantile(values,[.025,.975]).tolist(),'bonferroni_nominal_family95':np.quantile(values,[.05/(2*family),1-.05/(2*family)]).tolist(),'prefix10000_conditional95':np.quantile(values[:10000],[.025,.975]).tolist(),'left_tags':left,'right_tags':right}
    primary_values=[];primary_points=[];contrasts=[]
    for bb in CONFIG['backbones']:
        left=core(bb,8,'ry_on');right=selected_tags(bb,selection)
        primary_values.append(mean_draw(left)-mean_draw(right));primary_points.append(mean_point(left)-mean_point(right))
        contrasts.append(interval(f'{bb}: on8 - selected classical',left,right,27))
        pairs=[('on8 - atan MLP8',core(bb,8,'ry_on'),core(bb,8,'mlp_atan')),('on8 - standard MLP8',core(bb,8,'ry_on'),core(bb,8,'mlp_standard')),('on8 - off8',core(bb,8,'ry_on'),core(bb,8,'ry_off')),('on4 - off4',core(bb,4,'ry_on'),core(bb,4,'ry_off')),('atan8 - standard8',core(bb,8,'mlp_atan'),core(bb,8,'mlp_standard')),('atan4 - standard4',core(bb,4,'mlp_atan'),core(bb,4,'mlp_standard')),('full - reconstructed8 MLP',[f'{bb}_full_mlp_{s}' for s in CONFIG['final_seeds']],[f'{bb}_reconstructed8_mlp_{s}' for s in CONFIG['final_seeds']]),('standard8 - standard4',core(bb,8,'mlp_standard'),core(bb,4,'mlp_standard'))]
        contrasts.extend(interval(f'{bb}: {label}',a,b,27) for label,a,b in pairs)
    assert len(contrasts)==9*len(CONFIG['backbones'])
    pooled=np.mean(primary_values,axis=0);primary={'id':'Within the fixed backbone: on8 - selected classical','delta':float(np.mean(primary_points)),'conditional95':np.quantile(pooled,[.025,.975]).tolist(),'prefix10000_conditional95':np.quantile(pooled[:10000],[.025,.975]).tolist(),'estimand':'Mean per-seed metric differences, not an ensemble; paired slide draws shared across all backbones'}
    atomic_json(out/'analysis.json',{'stage':stage,'rows':len(y),'slides':len(groups),'draws':draws,'one_class_redraws':redraws,'primary':primary,'secondary':contrasts,'scope':'Conditional on fixed training data, fitted seeds and selected configurations; no equivalence claim'})
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=metrics[0]);writer.writeheader();writer.writerows(metrics)
    if not cached:
        atomic_npz(out/'bootstrap_auroc.npz',**aucs);atomic_json(out/'bootstrap_commit.json',{'sha256':digest(out/'bootstrap_auroc.npz')})
    # Editable manuscript table: optimization-seed variation stays distinct from WSI intervals.
    metric_by_tag={r['tag']:r for r in metrics};summary=[]
    for bb in CONFIG['backbones']:
        groups_to_report=[]
        for q in CONFIG['dimensions']:
            for head in CONFIG['heads']:groups_to_report.append((f'{bb}/q{q}/{head}',core(bb,q,head)))
        for q in ['full','reconstructed8']:groups_to_report.append((f'{bb}/{q}/MLP',[f'{bb}_{q}_mlp_{seed}' for seed in CONFIG['final_seeds']]))
        groups_to_report.append((f'{bb}/selected_classical (may reuse MLP)',selected_tags(bb,selection)))
        for group,tags in groups_to_report:
            row={'group':group,'fitted_seed_count':len(tags),'selected_reference_reuses_existing_MLP':bool('selected_classical' in group and selection['winners'][bb]['type']=='neural')}
            for metric in ['AUROC','AP','accuracy','sensitivity','specificity','Brier']:
                values=[metric_by_tag[t][metric] for t in tags]
                row[metric+'_mean']=float(np.mean(values));row[metric+'_seed_SD']=float(np.std(values,ddof=1)) if len(values)>1 else None
            seconds=[]
            for tag in tags:
                job=json.loads((BASE/'results/jobs'/f'{tag}.json').read_text())
                seconds.append(job.get('training_compute_seconds',job.get('fit_seconds')))
            row['seconds_mean']=float(np.mean(seconds));row['timer_scope']='shared neural training compute' if len(tags)>1 else 'classical fit including feature-map fit'
            summary.append(row)
    with (out/'model_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    table=['# Model summary','','Seed SD is optimization variation; conditional WSI intervals appear separately. Selected-classical rows may reference existing MLP fits.','','| Group | Fits | AUROC mean | Seed SD | Accuracy | AP | Brier | Seconds | Timer scope |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in summary:
        sd='not applicable' if r['AUROC_seed_SD'] is None else f"{r['AUROC_seed_SD']:.5f}"
        table.append(f"|{r['group']}|{r['fitted_seed_count']}|{r['AUROC_mean']:.5f}|{sd}|{r['accuracy_mean']:.5f}|{r['AP_mean']:.5f}|{r['Brier_mean']:.5f}|{r['seconds_mean']:.4f}|{r['timer_scope']}|")
    (out/'MODEL_TABLE.md').write_text('\n'.join(table)+'\n')
    # Main forest: pooled primary and three backbone-specific secondary comparisons.
    main_rows=[primary]+[c for c in contrasts if 'selected classical' in c['id']]
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    for i,c in enumerate(main_rows):
        lo,hi=c['conditional95'];ax.plot([lo,hi],[i,i],color='#0072B2',lw=2);ax.scatter(c['delta'],i,color='black',s=16,zorder=3)
        if i:
            lo,hi=c['bonferroni_nominal_family95'];ax.plot([lo,hi],[i,i],color='#999999',lw=5,zorder=0)
    ax.axvline(0,color='black',ls='--',lw=.7);ax.set_yticks(range(len(main_rows)),['Primary']+list(CONFIG['backbones']));ax.invert_yaxis();ax.set_xlabel('AUROC difference: CNOT on minus selected classical');ax.set_title(f'{stage}: paired slide-bootstrap intervals');fig.tight_layout();export(fig,figures,'primary_and_backbone_contrasts')
    for bb in CONFIG['backbones']:
        rows=[c for c in contrasts if c['id'].startswith(bb+':')]
        fig,ax=plt.subplots(figsize=(7.1,4.5))
        for i,c in enumerate(rows):
            lo,hi=c['bonferroni_nominal_family95'];ax.plot([lo,hi],[i,i],color='#bbbbbb',lw=4)
            lo,hi=c['conditional95'];ax.plot([lo,hi],[i,i],color='#0072B2',lw=1.5);ax.scatter(c['delta'],i,color='black',s=10,zorder=3)
        ax.set_yticks(range(len(rows)),[c['id'].split(': ',1)[1] for c in rows],fontsize=8);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7,ls='--');ax.set_xlabel('Paired AUROC difference');ax.set_title(bb);fig.tight_layout();export(fig,figures,f'secondary_ablation_{bb}')
    # Descriptive seed curves; no test-driven choice of representative model or seed.
    for kind in ['ROC','PR']:
        fig,axes=plt.subplots(1,len(CONFIG['backbones']),figsize=(7.1,3.1),squeeze=False);axes=axes[0]
        for ax,bb in zip(axes,CONFIG['backbones']):
            for head in [*CONFIG['heads'],'selected']:
                tags=selected_tags(bb,selection) if head=='selected' else core(bb,8,head)
                for index,tag in enumerate(tags):
                    score=data[tag]['scores']
                    if kind=='ROC':xx,yy,_=roc_curve(y,score)
                    else:yy,xx,_=precision_recall_curve(y,score)
                    ax.plot(xx,yy,color=COLORS[head],alpha=.55,lw=.65,label=LABELS[head] if index==0 else None)
            if kind=='ROC':ax.plot([0,1],[0,1],color='#999999',ls='--',lw=.5)
            else:ax.axhline(y.mean(),color='#999999',ls='--',lw=.5)
            ax.set(xlim=(0,1),ylim=(0,1),title=bb,xlabel='False positive rate' if kind=='ROC' else 'Recall',ylabel='True positive rate' if kind=='ROC' else 'Precision')
        handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,fontsize=8,loc='upper center',ncol=3);fig.tight_layout(rect=(0,0,1,.79));export(fig,figures,kind.lower()+'_seed_curves')
    fig,axes=plt.subplots(2,len(CONFIG['backbones']),figsize=(7.1,4.7),squeeze=False)
    for column,bb in enumerate(CONFIG['backbones']):
        for index,head in enumerate(CONFIG['heads']):
            tags=core(bb,8,head);values=[point[t] for t in tags]
            axes[0,column].scatter(index+np.linspace(-.08,.08,len(tags)),values,color=COLORS[head],s=12)
            times=[json.loads((BASE/'results/jobs'/f'{t}.json').read_text())['training_compute_seconds'] for t in tags]
            axes[1,column].scatter(index+np.linspace(-.08,.08,len(tags)),times,color=COLORS[head],s=12)
        axes[0,column].set(title=bb,ylabel='AUROC');axes[1,column].set(yscale='log',ylabel='Full training compute (s)')
        for ax in axes[:,column]:ax.set_xticks(range(4),['MLP std','MLP atan','Off','On'],rotation=30,ha='right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();export(fig,figures,'seed_performance_and_elapsed_time')
    fig,axes=plt.subplots(1,len(CONFIG['backbones']),figsize=(7.1,3.1),squeeze=False);axes=axes[0]
    for ax,bb in zip(axes,CONFIG['backbones']):
        for head in CONFIG['heads']:
            for seed in CONFIG['final_seeds']:
                trace=json.loads((BASE/'results/jobs'/f'{bb}_q8_{head}_{seed}.json').read_text())['trace']
                ax.plot([r['epoch'] for r in trace],[r['diagnostic_train_bce'] for r in trace],color=COLORS[head],alpha=.45,lw=.6,label=LABELS[head] if seed==CONFIG['final_seeds'][0] else None)
        ax.set(title=bb,xlabel='Epoch',ylabel='Fixed training-cohort BCE')
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,fontsize=8,loc='upper center',ncol=3);fig.tight_layout(rect=(0,0,1,.8));export(fig,figures,'training_diagnostics')
    # Seed101 is prospectively designated for calibration illustration, not a best seed.
    fig,axes=plt.subplots(1,len(CONFIG['backbones']),figsize=(7.1,3.1),squeeze=False);axes=axes[0]
    for ax,bb in zip(axes,CONFIG['backbones']):
        for head in ['mlp_atan','ry_on','selected']:
            tag=selected_tags(bb,selection)[0] if head=='selected' else core(bb,8,head)[0];p=data[tag]['probabilities'];bins=np.minimum((p*10).astype(int),9);xx=[];yy=[]
            for k in range(10):
                keep=bins==k
                if keep.any():xx.append(p[keep].mean());yy.append(y[keep].mean())
            ax.plot(xx,yy,'o-',ms=2,lw=.8,color=COLORS[head],label=LABELS[head])
        ax.plot([0,1],[0,1],ls='--',color='#999999',lw=.7);ax.set(xlim=(0,1),ylim=(0,1),title=bb,xlabel='Mean predicted probability',ylabel='Observed fraction')
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,fontsize=8,loc='upper center',ncol=3);fig.tight_layout(rect=(0,0,1,.8));export(fig,figures,'reliability_seed101')
    # Preserve machine-readable summary and a factual outcome, including wide intervals.
    low,high=primary['conditional95'];decision='favors the specified circuit' if low>0 else ('favors the selected classical procedures' if high<0 else 'does not establish a difference')
    text=f'''# {stage} analysis\n\nExecuted analysis of {len(y)} patches / {len(groups)} WSI clusters, {draws} paired bootstrap draws.\n\nPrimary mean difference: {primary['delta']:+.6f}; conditional nominal 95% interval [{low:+.6f}, {high:+.6f}]. This {decision}. It is not an equivalence test.\n\nThis backbone and all declared optimization seeds are retained. The 27-comparison family is retained conservatively across the three independent runs. Secondary comparisons use a predeclared family of 27, with marginal and Bonferroni nominal intervals. The bootstrap conditions on the selected settings and fitted training sample/seeds. Seed variation is separate; uncertainty from retraining/model selection is not covered.\n\n{'Development diagnostics only; calibration slides excluded. Not final-test evidence.' if stage=='valid' else 'Locked official-test evaluation; no test-driven model selection.'}\n\nFigures export as RGB LZW TIFF, vector PDF/SVG and 300-dpi PNG, 7.1-inch width, colorblind-friendly palette, readable labels and explicit units. ROC/PR lines show each seed, not an ensemble. Reliability uses predeclared seed101. AUROC panels may have different scales. Loss uses a fixed first1024-row training diagnostic cohort. Time is whole training compute; clean timing diagnostics are separate from full-data refit durations.\n\nSee metrics.csv, analysis.json, bootstrap_counts.npz and bootstrap_auroc.npz. Source/lock identities and raw predictions are retained. Patient mapping and near-duplicate independence remain bounded. No clinical or hardware quantum claim.\n'''
    (out/'RESULTS.md').write_text(text)
    atomic_json(out/'verification.json',{'weighted_auc_real_data_matches_sklearn':True,'code':code_identity(),'conditional_half_width':float((high-low)/2),'precision_aspiration':.01,'precision_aspiration_met':bool((high-low)/2<=.01),'figures_generated_not_yet_human_visually_verified':True,'figure_font':FIGURE_FONT,'journal_font_ready':JOURNAL_FONT_READY,'analysis_environment':environment()})
    print(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['valid','test'],required=True);a=parser.parse_args();main(a.stage)
