"""Reduced ResNet-only study; reuse original CV, preserve original scientific sources."""
from pathlib import Path
import sys,json,hashlib,time,argparse,copy,runpy,csv,zipfile,io
SOURCE=Path('experiments/experiment_011_train_resnet50')
ORIGINAL=Path('training/pcam_three_backbones/resnet50')
FINAL=Path('training/resnet50_final_v1')
SCOPE={'version':1,'backbone':'resnet50','dimensions':[8],'seeds':[101,202,303],'heads':['mlp_standard','mlp_atan','ry_off','ry_on'],'selection':'Finish original three training-only WSI folds and candidates. Preserve original common LR selection.','primary':'Mean seed AUROC: CNOT-on8 minus training-selected classical reference','secondary':['on-off','on-atan','on-standard','atan-standard'],'timing':'Original common training-compute seconds from final fits. No extra timing fits.','scope_reason':'User-approved resource reduction before official test, based on available saved work, not model performance.','known_limits':['One dataset and one frozen representation','Classically simulated circuits; no hardware quantum speedup','Conditional WSI uncertainty, not patient independence or retraining uncertainty','Fixed budget is not proof of global convergence']}

def configure(stage):
    sys.path.insert(0,str(SOURCE/'scripts'))
    import common,models,run
    from stepwise_patch_compat import install
    install(common,features=stage=='features')
    # models/run may already have imported the original callable.
    models.runtime_identity=common.runtime_identity;run.runtime_identity=common.runtime_identity
    from resnet_input_cache import install as install_input_cache
    install_input_cache(common,models,run)
    original_specs=run.final_specs
    if stage not in ['features','select']:
        common.CONFIG['final_seeds']=SCOPE['seeds'];common.CONFIG['dimensions']=[8]
        for module in [common,models,run]:module.BASE=FINAL
        run.final_specs=lambda bb,selection:[s for s in original_specs(bb,selection) if (s['type']=='classical' or s['q']==8)]
    if stage not in ['features','select']:
        from resnet_prediction_replay import install as install_prediction_replay
        install_prediction_replay(common,run,stage)
    return common,models,run

def scope_bind(common):
    path=FINAL/'SCOPE.json'
    if path.exists():
        if json.loads(path.read_text())!=SCOPE:raise RuntimeError('Final scope changed. Preserved all work; no silent protocol change.')
    else:common.atomic_json(path,SCOPE)
    selection=ORIGINAL/'results/selection.json'
    if not selection.exists():raise RuntimeError('Original training-only selection is incomplete')
    target=FINAL/'results/selection.json'
    if target.exists():assert json.loads(target.read_text())==json.loads(selection.read_text())
    else:common.atomic_json(target,json.loads(selection.read_text()))

def metrics_and_analysis(common,run,stage):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,brier_score_loss,confusion_matrix,roc_curve,precision_recall_curve,f1_score,precision_score,log_loss
    from analyze import weighted_auc_draws
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist};figure_font=next((f for f in ['Arial','Times New Roman'] if f in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':figure_font,'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    folder=FINAL/'reports'/stage;folder.mkdir(parents=True,exist_ok=True);figures=FINAL/'figures'/stage;figures.mkdir(parents=True,exist_ok=True)
    paths=sorted((FINAL/'results'/f'{stage}_predictions').glob('*.npz'));assert paths
    identity={'predictions':{str(p):common.digest(p) for p in paths},'scope':SCOPE,'helper_sha256':common.digest('shared/tools/resnet_finish_study.py')};common.bind(folder/'input_identity.json',identity)
    data={}
    with np.load(common.MAN/'roles.npz') as d:mask=~d['calibration'].copy()
    for p in paths:
        with np.load(p,allow_pickle=False) as d:row={k:d[k].copy() for k in d.files}
        if stage=='valid':
            for key in ['ids','slides','labels','scores','probabilities']:row[key]=row[key][mask]
        data[p.stem]=row
    reference=next(iter(data.values()));y=reference['labels'];groups,inverse=np.unique(reference['slides'],return_inverse=True)
    for row in data.values():
        for key in ['ids','slides','labels']:np.testing.assert_array_equal(row[key],reference[key])
    draws=common.CONFIG['bootstrap_draws'];rng=np.random.default_rng(common.CONFIG['bootstrap_seed']);counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=draws)
    positive=np.bincount(inverse,weights=y);negative=np.bincount(inverse,weights=1-y);invalid=(counts@positive==0)|(counts@negative==0)
    while invalid.any():
        counts[invalid]=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=int(invalid.sum()));invalid=(counts@positive==0)|(counts@negative==0)
    common.atomic_npz(folder/'bootstrap_counts.npz',counts=counts,groups=groups,ids=reference['ids'])
    aucs={};points={};rows=[]
    for tag,row in data.items():
        score=row['scores'];prob=row['probabilities'];cut=float(row['threshold']);pred=prob>=cut;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        points[tag]=float(roc_auc_score(y,score));cache=folder/'bootstrap_parts'/f'{tag}.npz';commit=cache.with_suffix('.json')
        if cache.exists() and commit.exists():
            assert common.digest(cache)==json.loads(commit.read_text())['sha256']
            with np.load(cache) as d:auc=d['auroc'].copy()
        else:
            auc=np.empty(draws)
            for first in range(0,draws,64):auc[first:first+64]=weighted_auc_draws(y,score,counts[first:first+64,inverse])
            common.atomic_npz(cache,auroc=auc);common.atomic_json(commit,{'sha256':common.digest(cache)})
        aucs[tag]=auc
        rows.append({'tag':tag,'AUROC':points[tag],'AP':float(average_precision_score(y,score)),'accuracy':float(accuracy_score(y,pred)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'precision':float(precision_score(y,pred,zero_division=0)),'F1':float(f1_score(y,pred,zero_division=0)),'Brier':float(brier_score_loss(y,prob)),'log_loss':float(log_loss(y,prob,labels=[0,1])),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'threshold':cut})
        print('Analyzed',stage,tag,flush=True)
    def tags(head):return [f'resnet50_q8_{head}_{seed}' for seed in SCOPE['seeds']]
    selection=json.loads((FINAL/'results/selection.json').read_text());selected=run.selected_tags('resnet50',selection)
    def contrast(name,left,right,family=1):
        values=np.mean([aucs[t] for t in left],axis=0)-np.mean([aucs[t] for t in right],axis=0)
        return {'id':name,'delta':float(np.mean([points[t] for t in left])-np.mean([points[t] for t in right])),'conditional95':np.quantile(values,[.025,.975]).tolist(),'family95':np.quantile(values,[.05/(2*family),1-.05/(2*family)]).tolist(),'left_tags':left,'right_tags':right}
    primary=contrast('on-selected',tags('ry_on'),selected)
    secondary=[contrast(name,tags(left),tags(right),4) for name,left,right in [('on-off','ry_on','ry_off'),('on-atan','ry_on','mlp_atan'),('on-standard','ry_on','mlp_standard'),('atan-standard','mlp_atan','mlp_standard')]]
    common.atomic_npz(folder/'bootstrap_auroc.npz',**aucs)
    analysis={'stage':stage,'rows':len(y),'WSI_clusters':len(groups),'draws':draws,'primary':primary,'secondary':secondary,'uncertainty':'Paired slide bootstrap conditional on fitted models, settings, training sample and retained seeds. Not patient-level inference or an equivalence test.'}
    common.atomic_json(folder/'analysis.json',analysis)
    with (folder/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=[];by_tag={r['tag']:r for r in rows}
    for head,group_tags in [(h,tags(h)) for h in SCOPE['heads']]+[('selected_classical',selected)]:
        values=np.array([points[t] for t in group_tags]);seconds=[]
        for tag in group_tags:
            record=json.loads((FINAL/'results/jobs'/f'{tag}.json').read_text());seconds.append(record.get('training_compute_seconds',record.get('fit_seconds')))
        summary.append({'head':head,'fits':len(group_tags),'AUROC_mean':float(values.mean()),'AUROC_seed_SD':float(values.std(ddof=1)) if len(values)>1 else None,'seconds_mean':float(np.mean(seconds)),'timer_scope':'Common neural training compute' if len(group_tags)>1 else 'Classical fit including feature-map fit'})
    common.atomic_json(folder/'model_summary.json',summary)
    with (folder/'model_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    def export(fig,name):
        from PIL import Image
        for ext in ['png','pdf','svg']:fig.savefig(figures/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=300,bbox_inches='tight',facecolor='white');buffer.seek(0);im=Image.open(buffer).convert('RGB');im.save(figures/f'{name}.tif',compression='tiff_lzw',dpi=(300,300));common.atomic_json(figures/f'{name}_format.json',{'font':figure_font,'preferred_journal_font_available':figure_font in ['Arial','Times New Roman'],'dpi':300,'RGB':True,'pixels':list(im.size),'visual_review_required':True});plt.close(fig)
    colors={'mlp_standard':'#0072B2','mlp_atan':'#56B4E9','ry_off':'#E69F00','ry_on':'#D55E00'}
    for kind in ['ROC','PR']:
        fig,ax=plt.subplots(figsize=(7.1,4.2))
        for head in SCOPE['heads']:
            for i,tag in enumerate(tags(head)):
                a,b,_=roc_curve(y,data[tag]['scores']) if kind=='ROC' else precision_recall_curve(y,data[tag]['scores'])
                ax.plot(a if kind=='ROC' else b,b if kind=='ROC' else a,color=colors[head],alpha=.7,label=head if i==0 else None)
        if selection['winners']['resnet50']['type']=='classical':
            for i,tag in enumerate(selected):
                a,b,_=roc_curve(y,data[tag]['scores']) if kind=='ROC' else precision_recall_curve(y,data[tag]['scores']);ax.plot(a if kind=='ROC' else b,b if kind=='ROC' else a,color='#009E73',ls='--',label='Selected classical' if i==0 else None)
        ax.set(xlabel='False positive rate' if kind=='ROC' else 'Recall',ylabel='True positive rate' if kind=='ROC' else 'Precision',title=f'ResNet50 {stage}: {kind}, individual seeds');ax.legend();fig.tight_layout();export(fig,kind.lower())
    fig,ax=plt.subplots(figsize=(7.1,3.2));contrasts=[primary]+secondary
    for i,c in enumerate(contrasts):
        lo,hi=c['family95'];ax.plot([lo,hi],[i,i],color='#999999',lw=4);lo,hi=c['conditional95'];ax.plot([lo,hi],[i,i],color='#0072B2',lw=2);ax.scatter(c['delta'],i,color='black',s=18)
    ax.axvline(0,color='black',ls='--',lw=.8);ax.set_yticks(range(len(contrasts)),[c['id'] for c in contrasts]);ax.invert_yaxis();ax.set_xlabel('AUROC difference');fig.tight_layout();export(fig,'paired_contrasts')
    fig,ax=plt.subplots(figsize=(7.1,4.2))
    for head in SCOPE['heads']:
        for i,tag in enumerate(tags(head)):
            trace=json.loads((FINAL/'results/jobs'/f'{tag}.json').read_text())['trace'];ax.plot([x['epoch'] for x in trace],[x['diagnostic_train_bce'] for x in trace],color=colors[head],alpha=.65,label=head if i==0 else None)
    ax.set(xlabel='Epoch',ylabel='Fixed training-cohort BCE',title='Training diagnostics, individual seeds');ax.legend();fig.tight_layout();export(fig,'training_loss')
    text=f"# ResNet50 {stage} results\n\nPrimary AUROC difference: {primary['delta']:+.6f}; conditional95% CI {primary['conditional95']}.\n\nSeed SD is separate from WSI bootstrap uncertainty. Four secondary comparisons retain family-adjusted intervals. No equivalence, patient-independence or quantum-speedup claim.\n\nSee metrics.csv, model_summary.csv, analysis.json and bootstrap arrays. Figures require final human visual/font review.\n"
    (folder/'RESULTS.md').write_text(text);print(text)

def main(stage,split=None):
    common,models,run=configure(stage)
    if stage=='features':
        sys.argv=[str(SOURCE/'scripts/extract.py'),'--backbone','resnet50','--split',split,'--device','auto'];runpy.run_path(sys.argv[0],run_name='__main__');return
    if stage=='select':
        from resnet_cv_resume import install as install_cv_resume
        audit=install_cv_resume(common,models,run)
        try:run.select()
        finally:audit()
        return
    scope_bind(common)
    if stage=='fit':run.refit();return
    if stage=='valid':run.verify_development();metrics_and_analysis(common,run,'valid');return
    if stage=='lock':
        if (FINAL/'results/test_lock.json').exists():run.verify_lock();return
        run.verify_development();specs=json.loads((FINAL/'results/final_specs.json').read_text());assert len(specs) in [12,13]
        for spec in specs:
            if spec['type']!='neural':continue
            record=json.loads((FINAL/'results/jobs'/f"{spec['tag']}.json").read_text());trace=record['trace'];assert len(trace)==common.CONFIG['epochs']
            if len(trace)>=20 and trace[-20]['diagnostic_train_bce']-trace[-1]['diagnostic_train_bce']>.02:raise RuntimeError('Continuing training-loss decline: common budget requires review before test access. Saved fits retained.')
        paths=[Path('shared/tools/resnet_finish_study.py'),FINAL/'SCOPE.json',ORIGINAL/'results/selection.json',*sorted((FINAL/'checkpoints').glob('*')),*sorted((FINAL/'results').glob('*.json')),*sorted((FINAL/'results/valid_predictions').glob('*.npz')),*sorted((FINAL/'reports/valid').glob('*')),*sorted(common.MAN.glob('*.npz')),*sorted((ORIGINAL/'features/resnet50').glob('train_*.json')),*sorted((ORIGINAL/'features/resnet50').glob('valid_*.json'))]
        paths.extend(Path(p) for p in common.code_identity());paths.extend(Path('shared/tools')/name for name in ['stepwise_patch_compat.py','resnet_input_cache.py','resnet_cv_resume.py','resnet_prediction_replay.py'])
        common.atomic_json(FINAL/'results/test_lock.json',{'code':common.code_identity(),'environment':common.runtime_identity(common.environment()),'runtime':common.environment(),'scope':SCOPE,'artifacts':{str(p):common.digest(p) for p in sorted(set(paths)) if p.is_file()},'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print('Scope and fitted artifacts locked before official test.');return
    if stage=='test_features':
        sys.argv=[str(SOURCE/'scripts/extract.py'),'--backbone','resnet50','--split','test','--device','auto'];runpy.run_path(sys.argv[0],run_name='__main__');return
    if stage=='test':run.test();return
    if stage=='analysis':
        run.verify_lock(check_environment=False);metrics_and_analysis(common,run,'test');return
    if stage=='export':
        run.verify_lock(check_environment=False)
        destination=FINAL/'exports';destination.mkdir(parents=True,exist_ok=True)
        paths=[p for directory in [FINAL/'checkpoints',FINAL/'mapped_inputs',FINAL/'results',FINAL/'reports',FINAL/'figures',common.MAN,SOURCE/'scripts',SOURCE/'configs'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        paths.extend([Path('shared/tools/resnet_finish_study.py'),Path('shared/tools/stepwise_patch_compat.py'),Path('shared/tools/resnet_input_cache.py'),Path('shared/tools/resnet_cv_resume.py'),Path('shared/tools/resnet_prediction_replay.py'),FINAL/'SCOPE.json',ORIGINAL/'results/selection.json']);paths.extend(Path(p) for p in common.code_identity());
        for directory in [ORIGINAL/'results/jobs',ORIGINAL/'results/cv',ORIGINAL/'results/cv_predictions',ORIGINAL/'checkpoints']:
            paths.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.pt','.joblib'])
        paths.extend((ORIGINAL/'features/resnet50').glob('*.json'));paths.extend((ORIGINAL/'features/resnet50').glob('*.pt'))
        for split in ['valid','test']:
            csv_path=destination/f'{split}_predictions.csv'
            with csv_path.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['tag','id','slide','label','score','probability','threshold'])
                for p in sorted((FINAL/'results'/f'{split}_predictions').glob('*.npz')):
                    import numpy as np
                    with np.load(p) as d:writer.writerows((p.stem,str(i),str(sl),int(y),float(s),float(pr),float(d['threshold'])) for i,sl,y,s,pr in zip(d['ids'],d['slides'],d['labels'],d['scores'],d['probabilities']))
            paths.append(csv_path)
        common.atomic_json(destination/'MANIFEST.json',{'files':{str(p):{'sha256':common.digest(p),'bytes':p.stat().st_size} for p in sorted(set(paths))},'raw_and_feature_arrays':'Retain original data/raw and training/pcam_three_backbones/resnet50/features; arrays are not duplicated in ZIP','scope':SCOPE,'environment':common.environment()})
        with zipfile.ZipFile(destination/'resnet50_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(paths)):z.write(p,str(p))
            z.write(destination/'MANIFEST.json',str(destination/'MANIFEST.json'))
        print('Final evidence:',destination/'resnet50_evidence.zip');return
    raise ValueError(stage)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['features','select','fit','valid','lock','test_features','test','analysis','export']);parser.add_argument('--split',choices=['train','valid']);a=parser.parse_args();main(a.stage,a.split)
