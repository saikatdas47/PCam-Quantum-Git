"""Reuse audited completed CV outputs; never silently resume a different x matrix."""
from pathlib import Path
import json,hashlib

def install(common,models,run):
    import numpy as np,torch,joblib
    from sklearn.metrics import roc_auc_score
    original_fit=run.fit_head;original_classical=run.fit_classical;original_logits=run.head_logits;original_prob=run.classical_probabilities
    recovery_path=common.BASE/'workflow/CV_INPUT_RECOVERY.json';recoveries=json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    events=[]
    class Saved:
        def __init__(self,model,row,tag,neural):self.model=model;self.row=row;self.tag=tag;self.neural=neural
    def digest_array(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
    def completed(tag,neural,x,y,expected):
        cv=common.BASE/'results/cv'/f'{tag}.json';pred=common.BASE/'results/cv_predictions'/f'{tag}.npz';model_tag=recoveries.get(tag,{}).get('model_tag',tag);job=common.BASE/'results/jobs'/f'{model_tag}.json'
        if not (cv.exists() and pred.exists() and job.exists()):return None
        record=json.loads(job.read_text());identity=record['identity'];assert identity['code']==common.code_identity()
        assert common.runtime_identity(identity['environment'])==common.runtime_identity(common.environment())
        for k,v in expected.items():assert identity[k]==v,(tag,k)
        assert identity['y']==digest_array(y)
        file=common.BASE/'checkpoints'/(model_tag+('.pt' if neural else '.joblib'));assert common.digest(file)==record['checkpoint_sha256']
        fold=int(tag.split('_cv',1)[1].split('_')[0]);manifest=common.load_manifest('train')
        with np.load(common.MAN/'roles.npz') as d:folds=d['folds'].copy()
        fit=np.flatnonzero((folds>=0)&(folds!=fold));held=np.flatnonzero(folds==fold)
        with np.load(pred,allow_pickle=False) as d:row={k:d[k].copy() for k in d.files}
        for k in ['ids','slides','labels']:np.testing.assert_array_equal(row[k],manifest[k][held])
        np.testing.assert_array_equal(row['fit_ids'],manifest['ids'][fit]);entry=json.loads(cv.read_text());assert entry['code']==common.code_identity();assert abs(roc_auc_score(row['labels'],row['scores'])-entry['auroc'])<=1e-12
        if neural:
            old=torch.load(file,map_location='cpu',weights_only=True);assert old['epoch']==identity['epochs'];model=models.make_head(identity['head'],identity['seed'],identity['shape'][1]);model.load_state_dict(old['model'])
        else:model=joblib.load(file)
        events.append({'tag':tag,'action':'reuse_completed_CV','original_x_sha256':identity['x'],'current_x_sha256':digest_array(x),'checkpoint_sha256':record['checkpoint_sha256']})
        print('REUSE VERIFIED COMPLETED CV:',tag,flush=True);return Saved(model,row,tag,neural)
    def fit(tag,name,seed,x,y,lr,epochs=None):
        budget=common.CONFIG['epochs'] if epochs is None else epochs
        expected={'head':name,'seed':seed,'shape':list(x.shape),'lr':lr,'epochs':budget}
        saved=completed(tag,True,x,y,expected)
        if saved:return saved
        path=common.BASE/'checkpoints'/f'{tag}.pt';model_tag=recoveries.get(tag,{}).get('model_tag',tag)
        if model_tag==tag and path.exists():
            old=torch.load(path,map_location='cpu',weights_only=True);identity=old['identity']
            assert identity['code']==common.code_identity() and identity['y']==digest_array(y)
            for k,v in expected.items():assert identity[k]==v,(tag,k)
            if identity['x']!=digest_array(x):
                # Do not pretend these inputs are byte-identical. Preserve the
                # original partial checkpoint and refit only this unfinished job.
                model_tag=tag+'_input_recovery_'+digest_array(x)[:12]
                recoveries[tag]={'model_tag':model_tag,'original_checkpoint':str(path),'original_checkpoint_sha256':common.digest(path),'original_epoch':old['epoch'],'original_x_sha256':identity['x'],'new_x_sha256':digest_array(x),'reason':'Original transformed input bytes were not saved. Restart only this unfinished job from its fixed seed/budget; do not mix optimizer state across different x hashes.'}
                common.atomic_json(recovery_path,recoveries)
                print('PRESERVED PARTIAL CHECKPOINT. Restarting only unfinished job:',tag,'as',model_tag,flush=True)
        return original_fit(model_tag,name,seed,x,y,lr,epochs=epochs)
    def classical(tag,candidate,x,y):
        saved=completed(tag,False,x,y,{'candidate':candidate})
        return saved if saved else original_classical(tag,candidate,x,y)
    def logits(model,x):
        if not isinstance(model,Saved):return original_logits(model,x)
        actual=original_logits(model.model,x);np.testing.assert_allclose(actual,model.row['scores'],rtol=1e-10,atol=1e-12,err_msg='Completed CV prediction replay differs: '+model.tag)
        return model.row['scores'].copy()
    def probabilities(model,x):
        if not isinstance(model,Saved):return original_prob(model,x)
        actual=original_prob(model.model,x);np.testing.assert_allclose(actual,model.row['scores'],rtol=1e-10,atol=1e-12,err_msg='Completed classical CV replay differs: '+model.tag)
        return model.row['scores'].copy()
    original_npz=run.atomic_npz
    def save_prediction(path,**payload):
        if Path(path).exists():
            with np.load(path,allow_pickle=False) as old:
                for k,v in payload.items():np.testing.assert_array_equal(old[k],v)
            return
        original_npz(path,**payload)
    run.fit_head=fit;run.fit_classical=classical;run.head_logits=logits;run.classical_probabilities=probabilities;run.atomic_npz=save_prediction
    return lambda:common.atomic_json(common.BASE/'workflow/CV_REUSE_AUDIT.json',{'events':events,'prediction_replay_rtol':1e-10,'prediction_replay_atol':1e-12,'partial_recovery':recoveries})
