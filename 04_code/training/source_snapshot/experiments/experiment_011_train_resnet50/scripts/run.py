"""Explicit train-only selection, full refit, calibration, immutable test lock and scoring."""
from common import *
from models import *
import argparse
from scipy.special import expit
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,brier_score_loss,average_precision_score


def record_cv(tag,score):
    path=BASE/'results/cv'/f'{tag}.json'
    payload={'auroc':float(score),'code':code_identity()}
    bind(path,payload)


def cv_score(tag):return json.loads((BASE/'results/cv'/f'{tag}.json').read_text())['auroc']


def select():
    guard_pretest()
    with np.load(MAN/'roles.npz') as d:folds=d['folds'].copy()
    for backbone in CONFIG['backbones']:
        features,manifest=load_features(backbone,'train');y=manifest['labels']
        for fold in range(CONFIG['selection_folds']):
            fit=np.flatnonzero((folds>=0)&(folds!=fold));held=np.flatnonzero(folds==fold)
            t=fit_transform(features[fit],manifest['ids'][fit],f'{backbone}_cv{fold}')
            for lr in CONFIG['learning_rates']:
                for head in CONFIG['heads']:
                    mapping='standard' if head=='mlp_standard' else 'atan'
                    tag=f'{backbone}_cv{fold}_{head}_lr{lr}'
                    model=fit_head(tag,head,CONFIG['selection_seed'],map_features(t,features[fit],8,mapping),y[fit],lr)
                    scores=head_logits(model,map_features(t,features[held],8,mapping));record_cv(tag,roc_auc_score(y[held],scores));atomic_npz(BASE/'results/cv_predictions'/f'{tag}.npz',ids=manifest['ids'][held],slides=manifest['slides'][held],labels=y[held],scores=scores,probabilities=expit(scores),fit_ids=manifest['ids'][fit])
            for candidate in CONFIG['classical_candidates']:
                tag=f'{backbone}_cv{fold}_{candidate["id"]}'
                model=fit_classical(tag,candidate,map_features(t,features[fit],8,candidate['mapping']),y[fit])
                scores=classical_probabilities(model,map_features(t,features[held],8,candidate['mapping']));record_cv(tag,roc_auc_score(y[held],scores));atomic_npz(BASE/'results/cv_predictions'/f'{tag}.npz',ids=manifest['ids'][held],slides=manifest['slides'][held],labels=y[held],scores=scores,probabilities=scores,fit_ids=manifest['ids'][fit])
    schedules=[]
    for lr in CONFIG['learning_rates']:
        scores=[cv_score(f'{bb}_cv{fold}_{head}_lr{lr}') for bb in CONFIG['backbones'] for fold in range(CONFIG['selection_folds']) for head in CONFIG['heads']]
        schedules.append({'lr':lr,'mean_auroc':float(np.mean(scores)),'scores':scores})
    best=max(r['mean_auroc'] for r in schedules);chosen=min(r['lr'] for r in schedules if best-r['mean_auroc']<=1e-8)
    winners={};all_candidates={}
    for bb in CONFIG['backbones']:
        candidates=[]
        for candidate in CONFIG['classical_candidates']:
            scores=[cv_score(f'{bb}_cv{fold}_{candidate["id"]}') for fold in range(CONFIG['selection_folds'])]
            candidates.append({'id':candidate['id'],'type':'classical','definition':candidate,'mean_auroc':float(np.mean(scores)),'fold_scores':scores})
        for head in ['mlp_standard','mlp_atan']:
            scores=[cv_score(f'{bb}_cv{fold}_{head}_lr{chosen}') for fold in range(CONFIG['selection_folds'])]
            candidates.append({'id':head,'type':'neural','mean_auroc':float(np.mean(scores)),'fold_scores':scores})
        top=max(r['mean_auroc'] for r in candidates);winner=sorted([r for r in candidates if top-r['mean_auroc']<=1e-8],key=lambda r:r['id'])[0]
        winners[bb]=winner;all_candidates[bb]=candidates
    bind(BASE/'results/selection.json',{'common_lr':chosen,'epochs':CONFIG['epochs'],'schedules':schedules,'winners':winners,'all_candidates':all_candidates,'selection_scope':'16384-row subset of official train, three WSI folds; full final refit uses all262144','code':code_identity()})
    print('Training-only selection complete',chosen,winners,flush=True)


def threshold(y,p):
    fpr,tpr,values=roc_curve(y,p,drop_intermediate=False)
    good=np.isfinite(values)&(values>=0)&(values<=1);values=values[good];j=(tpr-fpr)[good]
    best=j.max();values=values[np.abs(j-best)<=1e-12]
    return float(sorted(values,key=lambda v:(abs(v-.5),v))[0])


def save_prediction(path,manifest,score,probability,threshold_value):
    assert len(score)==len(probability)==len(manifest['labels']) and np.isfinite(score).all() and np.isfinite(probability).all()
    payload={'ids':manifest['ids'],'slides':manifest['slides'],'labels':manifest['labels'],'scores':score,'probabilities':probability,'threshold':np.array(threshold_value)}
    if path.exists():
        with np.load(path) as old:
            for k,v in payload.items():np.testing.assert_array_equal(old[k],v)
    else:atomic_npz(path,**payload)


def final_specs(bb,selection):
    specs=[]
    for index,seed in enumerate(CONFIG['final_seeds']):
        dimensions=CONFIG['dimensions'] if index%2==0 else list(reversed(CONFIG['dimensions']))
        for q in dimensions:
            for head in np.roll(CONFIG['heads'],index):
                head=str(head)
                specs.append({'tag':f'{bb}_q{q}_{head}_{seed}','backbone':bb,'head':head,'q':q,'mapping':'standard' if head=='mlp_standard' else 'atan','seed':seed,'type':'neural'})
    for q in ['full','reconstructed8']:
        for seed in CONFIG['final_seeds']:
            specs.append({'tag':f'{bb}_{q}_mlp_{seed}','backbone':bb,'head':'mlp_standard','q':q,'mapping':'standard','seed':seed,'type':'neural'})
    winner=selection['winners'][bb]
    if winner['type']=='classical':
        specs.append({'tag':f'{bb}_selected_classical','backbone':bb,'q':8,'mapping':winner['definition']['mapping'],'type':'classical','definition':winner['definition']})
    return specs


def model_for(spec,selection,train_x=None,train_y=None):
    tag=spec['tag']
    if spec['type']=='neural':
        if train_x is not None:return fit_head(tag,spec['head'],spec['seed'],train_x,train_y,selection['common_lr'])
        checkpoint=torch.load(BASE/'checkpoints'/f'{tag}.pt',weights_only=True)
        model=make_head(spec['head'],spec['seed'],checkpoint['identity']['shape'][1]);model.load_state_dict(checkpoint['model']);return model
    if train_x is not None:return fit_classical(tag,spec['definition'],train_x,train_y)
    return joblib.load(BASE/'checkpoints'/f'{tag}.joblib')


def predict(model,spec,x):
    if spec['type']=='neural':
        score=head_logits(model,x);return score,expit(score)
    prob=classical_probabilities(model,x);return prob,prob


def refit():
    guard_pretest();selection=json.loads((BASE/'results/selection.json').read_text())
    with np.load(MAN/'roles.npz') as d:calibration=d['calibration'].copy()
    specs=[];thresholds={};diagnostics=[]
    for bb in CONFIG['backbones']:
        train,manifest=load_features(bb,'train');valid,vm=load_features(bb,'valid')
        t=fit_transform(train,manifest['ids'],f'{bb}_final')
        for spec in final_specs(bb,selection):
            tx=map_features(t,train,spec['q'],spec['mapping']);vx=map_features(t,valid,spec['q'],spec['mapping'])
            model=model_for(spec,selection,tx,manifest['labels']);scores,prob=predict(model,spec,vx)
            cut=threshold(vm['labels'][calibration],prob[calibration]);thresholds[spec['tag']]=cut
            save_prediction(BASE/'results/valid_predictions'/f'{spec["tag"]}.npz',vm,scores,prob,cut)
            diagnostics.append({'tag':spec['tag'],'diagnostic_rows':int((~calibration).sum()),'auroc':float(roc_auc_score(vm['labels'][~calibration],scores[~calibration])),'accuracy':float(accuracy_score(vm['labels'][~calibration],prob[~calibration]>=cut)),'calibration_rows':int(calibration.sum())})
            specs.append(spec);print('Refit/calibrated',spec['tag'],flush=True)
            del tx,vx,model
        del train,valid
    bind(BASE/'results/final_specs.json',specs);bind(BASE/'results/thresholds.json',thresholds)
    bind(BASE/'results/development_diagnostics.json',diagnostics)
    print('Full-data fitting complete; test not scored')


def timing():
    guard_pretest()
    for bb in CONFIG['backbones']:
        for repeat in range(CONFIG['timing_repeats']):
            for head in np.roll(CONFIG['heads'],repeat):
                # Fresh process per clean diagnostic; feature-load time is excluded from compute.
                import subprocess
                subprocess.run([sys.executable,str(SOURCE_BASE/'scripts/run.py'),'timing-job','--backbone',bb,'--head',str(head),'--repeat',str(repeat)],check=True)


def timing_job(bb,head,repeat):
    guard_pretest();selection=json.loads((BASE/'results/selection.json').read_text());x,m=load_features(bb,'train',limit=CONFIG['timing_diagnostic_samples'])
    t=joblib.load(BASE/'checkpoints'/f'{bb}_final_transform.joblib');x=x[:CONFIG['timing_diagnostic_samples']]
    fit_head(f'timing_{bb}_{head}_{repeat}',head,CONFIG['final_seeds'][0],map_features(t,x,8,'standard' if head=='mlp_standard' else 'atan'),m['labels'][:len(x)],selection['common_lr'],epochs=CONFIG['timing_diagnostic_epochs'])


def selected_tags(bb,selection):
    winner=selection['winners'][bb]
    return [f'{bb}_q8_{winner["id"]}_{seed}' for seed in CONFIG['final_seeds']] if winner['type']=='neural' else [f'{bb}_selected_classical']


def verify_development():
    selection=json.loads((BASE/'results/selection.json').read_text());specs=json.loads((BASE/'results/final_specs.json').read_text());checked=[]
    for bb in CONFIG['backbones']:
        x,m=load_features(bb,'valid');t=joblib.load(BASE/'checkpoints'/f'{bb}_final_transform.joblib')
        for spec in [s for s in specs if s['backbone']==bb]:
            model=model_for(spec,selection);scores,p=predict(model,spec,map_features(t,x,spec['q'],spec['mapping']))
            with np.load(BASE/'results/valid_predictions'/f'{spec["tag"]}.npz') as d:
                np.testing.assert_array_equal(d['scores'],scores);np.testing.assert_array_equal(d['probabilities'],p);np.testing.assert_array_equal(d['ids'],m['ids'])
            checked.append(spec['tag'])
    atomic_json(BASE/'results/development_verification.json',{'all_predictions_exact':True,'models':checked,'test_scored':False,'code':code_identity()})


def lock():
    guard_pretest();verify_development()
    software=json.loads((BASE/'results/software_checks.json').read_text());assert software['passed'] and software['code']==code_identity()
    precision=json.loads((BASE/'reports/valid/verification.json').read_text());assert precision['code']==code_identity()
    specs=json.loads((BASE/'results/final_specs.json').read_text());assert len(specs)>=50
    for bb in CONFIG['backbones']:
        for repeat in range(CONFIG['timing_repeats']):
            for head in CONFIG['heads']:assert (BASE/'results/jobs'/f'timing_{bb}_{head}_{repeat}.json').exists()
    traces=[json.loads(p.read_text()) for p in (BASE/'results/jobs').glob('*.json') if 'timing_' not in p.name and '_cv' not in p.name]
    continuing=[]
    for record in traces:
        trace=record.get('trace',[])
        if len(trace)>=20 and trace[-20]['diagnostic_train_bce']-trace[-1]['diagnostic_train_bce']>.02:
            continuing.append(record['identity']['tag'])
    if continuing:raise RuntimeError(f'Substantial continuing training-loss decline; review common budget before test lock: {continuing}')
    # Hash every frozen fitted artifact and every committed train/valid feature marker.
    paths=[*sorted((BASE/'checkpoints').glob('*')),BASE/'results/selection.json',BASE/'results/final_specs.json',BASE/'results/thresholds.json',BASE/'results/data_audit.json',BASE/'results/preparation_identity.json',BASE/'reports/valid/analysis.json',BASE/'reports/valid/verification.json',BASE/'results/development_verification.json',BASE/'results/software_checks.json',*sorted(MAN.glob('*.npz'))]
    paths.extend(sorted((BASE/'reports/runtime_migrations').rglob('migration.json')))
    paths.extend([p for p in RAW.glob(f'{PREFIX}*') if 'mask' not in p.name])
    for bb in CONFIG['backbones']:
        paths.extend([CACHE/bb/'identity.json',*sorted((CACHE/bb).glob('train_*.json')),*sorted((CACHE/bb).glob('valid_*.json'))])
    payload={'stage':'frozen_before_test_predictions','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code':code_identity(),'artifacts':{str(p):digest(p) for p in paths if p.is_file()},'environment':runtime_identity(environment()),'runtime':environment(),'primary':CONFIG['primary'],'known_limits':['Development validation reused','Patient mapping unresolved','Training-source compressed checksum unavailable','Fixed budget is not proof of global convergence','Rank256 Fourier is an approximation, not exact-kernel SVM']}
    atomic_json(BASE/'results/test_lock.json',payload);print('Pre-test lock saved. All further training stages are disabled.')


def verify_lock(check_environment=True,artifacts=True):
    payload=json.loads((BASE/'results/test_lock.json').read_text());assert payload['code']==code_identity(),'Code/config changed after lock'
    if check_environment:assert payload['environment']==runtime_identity(environment()),'Environment changed after lock'
    for p,h in (payload['artifacts'].items() if artifacts else []):
        if digest(p)!=h:raise RuntimeError(f'Frozen artifact changed: {p}')
    return payload


def test():
    verify_lock();selection=json.loads((BASE/'results/selection.json').read_text());specs=json.loads((BASE/'results/final_specs.json').read_text());thresholds=json.loads((BASE/'results/thresholds.json').read_text());verified=[]
    for bb in CONFIG['backbones']:
        x,m=load_features(bb,'test');t=joblib.load(BASE/'checkpoints'/f'{bb}_final_transform.joblib')
        for spec in [s for s in specs if s['backbone']==bb]:
            path=BASE/'results/test_predictions'/f'{spec["tag"]}.npz'
            # Re-compute inference only to verify recovery/preserved predictions, never select/retrain.
            model=model_for(spec,selection);score,p=predict(model,spec,map_features(t,x,spec['q'],spec['mapping']))
            save_prediction(path,m,score,p,thresholds[spec['tag']]);verified.append(spec['tag'])
    atomic_json(BASE/'results/test_verification.json',{'all_saved_predictions_match_reloaded_models':True,'models':verified,'prediction_sha256':{str(p):digest(p) for p in sorted((BASE/'results/test_predictions').glob('*.npz'))},'test_lock_sha256':digest(BASE/'results/test_lock.json')})
    print('Locked official test scoring complete. No model winner is selected here.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','refit','timing','timing-job','verify-development','lock','test']);parser.add_argument('--backbone',choices=CONFIG['backbones']);parser.add_argument('--head',choices=CONFIG['heads']);parser.add_argument('--repeat',type=int);a=parser.parse_args()
    if a.stage=='timing-job':timing_job(a.backbone,a.head,a.repeat)
    elif a.stage=='verify-development':verify_development()
    else:globals()[a.stage]()
