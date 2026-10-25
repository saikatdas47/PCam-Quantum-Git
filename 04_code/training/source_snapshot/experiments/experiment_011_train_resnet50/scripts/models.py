"""Shared CPU head fitting, fixed schedules and resumable epoch commits."""
from common import *
import copy,sys,warnings
from portable_runtime import checkpoint_segments
import joblib
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(ROOT/'shared/study_pipeline'))
sys.path.insert(0,str(ROOT/'shared/models'))
from torch_training import paired_circuits,warm_then_reset,make_optimizer,train_block
from torch_circuit import MLP
from fourier_kernel import fourier_kernel

torch.set_num_threads(CONFIG['threads']);torch.use_deterministic_algorithms(True)


def save_joblib(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');joblib.dump(value,tmp);tmp.replace(path)


def fit_transform(features,ids,tag):
    assert all(str(i).startswith('train:') for i in ids)
    path=BASE/'checkpoints'/f'{tag}_transform.joblib'
    identity={'ids':hashlib.sha256(np.asarray(ids).tobytes()).hexdigest(),'code':code_identity(),
              'feature_values':hashlib.sha256(memoryview(np.ascontiguousarray(features)).cast('B')).hexdigest()}
    bind(path.with_suffix('.json'),identity)
    if path.exists():return joblib.load(path)
    tick=time.perf_counter()
    pca=PCA(n_components=8,svd_solver='randomized',random_state=CONFIG['selection_seed'],n_oversamples=24,iterated_power=7)
    pca.fit(features)
    # Canonical float64 C-order state avoids float32 layout-dependent reload rounding.
    for attribute in ['components_','mean_','explained_variance_','explained_variance_ratio_','singular_values_']:
        setattr(pca,attribute,np.ascontiguousarray(getattr(pca,attribute),dtype=np.float64))
    z=np.concatenate([pca.transform(np.asarray(features[i:i+1024],dtype=np.float64)) for i in range(0,len(features),1024)])
    scale=StandardScaler().fit(z)
    full_scale=StandardScaler().fit(features)
    payload={'pca':pca,'scale':scale,'full_scale':full_scale,'training_ids':np.asarray(ids),'seconds':time.perf_counter()-tick}
    save_joblib(path,payload);return joblib.load(path)


def map_features(t,features,q,mapping):
    # Chunk full-width controls to avoid simultaneous full-size inverse/scaling copies.
    width=features.shape[1] if q in ('full','reconstructed8') else q
    result=np.empty((len(features),width),dtype=np.float64)
    for first in range(0,len(features),1024):
        block=np.asarray(features[first:first+1024],dtype=np.float64)
        if q=='full':z=t['full_scale'].transform(block)
        elif q=='reconstructed8':z=t['full_scale'].transform(t['pca'].inverse_transform(t['pca'].transform(block)))
        else:z=t['scale'].transform(t['pca'].transform(block))[:,:q]
        result[first:first+len(block)]=np.arctan(z) if mapping=='atan' else z
    return result


def make_head(name,seed,dimension):
    torch.manual_seed(seed)
    if name.startswith('mlp'):return MLP(dimension,32)
    off,on=paired_circuits(dimension,seed)
    return off if name=='ry_off' else on


def head_logits(model,x):
    model.eval();parts=[]
    with torch.no_grad():
        for first in range(0,len(x),CONFIG['batch_size']):parts.append(model(torch.as_tensor(x[first:first+CONFIG['batch_size']],dtype=torch.float64)).numpy())
    return np.concatenate(parts)


def fit_head(tag,name,seed,x,y,lr,epochs=None):
    """Only forward differs. Epoch orders/diagnostics/checkpoint writes are outside compute."""
    epochs=CONFIG['epochs'] if epochs is None else epochs
    identity={'code':code_identity(),'tag':tag,'head':name,'seed':seed,'shape':list(x.shape),'lr':lr,'epochs':epochs,
              'x':hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest(),'y':hashlib.sha256(memoryview(np.ascontiguousarray(y)).cast('B')).hexdigest(),
              'environment':runtime_identity(environment())}
    path=BASE/'checkpoints'/f'{tag}.pt';record=BASE/'results/jobs'/f'{tag}.json'
    setup=time.perf_counter();model=make_head(name,seed,x.shape[1]);optimizer=make_optimizer(model,lr)
    tx=torch.from_numpy(np.asarray(x,dtype=np.float64));ty=torch.from_numpy(np.asarray(y,dtype=np.float64))
    warm_then_reset(model,optimizer,tx,ty,CONFIG['batch_size'])
    begin=0;blocks=[];trace=[];writes=[];resumed=False;segments=[]
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['identity']!=identity:
            differences=[k for k in sorted(set(old['identity'])|set(identity)) if old['identity'].get(k)!=identity.get(k)]
            raise RuntimeError(f'Checkpoint identity changed: {tag}; differing fields={differences}. Checkpoint preserved; no automatic retraining.')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);torch.set_rng_state(old['rng'])
        begin=old['epoch'];blocks=old['blocks'];trace=old['trace'];writes=old['writes'];resumed=begin<epochs and begin>0
        segments=checkpoint_segments(old)
        # Reproduce the last saved diagnostic before accepting a new runtime.
        if trace:
            with torch.no_grad():
                sample=tx[:min(1024,len(tx))];target=ty[:len(sample)];losses=[]
                for first in range(0,len(sample),CONFIG['batch_size']):
                    stop=min(first+CONFIG['batch_size'],len(sample))
                    losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(model(sample[first:stop]),target[first:stop],reduction='sum')))
                replay=sum(losses)/len(sample)
            if not np.isclose(replay,trace[-1]['diagnostic_train_bce'],rtol=1e-12,atol=1e-12):raise RuntimeError('Saved CPU checkpoint diagnostic differs in this runtime; checkpoint preserved: '+tag)
    if begin<epochs:segments.append({'first_epoch':begin+1,'last_epoch':begin,'environment':environment()})
    setup=time.perf_counter()-setup;fit_tick=time.perf_counter()
    for epoch in range(begin,epochs):
        # One epoch at a time avoids storing 100 complete full-dataset permutations.
        order=np.random.default_rng(seed+epoch*10000).permutation(len(x))
        orders=torch.tensor(order[None,:],dtype=torch.int64);model.train()
        seconds,steps=train_block(model,optimizer,tx,ty,orders,0,1,CONFIG['batch_size'])
        blocks.append({'epoch':epoch+1,'seconds':seconds,'updates':steps,'runtime_segment':len(segments)-1});segments[-1]['last_epoch']=epoch+1
        # Fixed training-only diagnostic cohort; it never controls stopping or selection.
        with torch.no_grad():
            sample=tx[:min(1024,len(tx))];target=ty[:len(sample)]
            losses=[]
            for first in range(0,len(sample),CONFIG['batch_size']):
                stop=min(first+CONFIG['batch_size'],len(sample))
                losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(model(sample[first:stop]),target[first:stop],reduction='sum')))
            loss=sum(losses)/len(sample)
        if not np.isfinite(loss) or not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError(f'Nonfinite fit: {tag}')
        trace.append({'epoch':epoch+1,'diagnostic_train_bce':loss,'diagnostic_samples':len(sample)})
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tick=time.perf_counter()
        torch.save({'identity':identity,'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'blocks':blocks,'trace':trace,'writes':writes,'runtime_segments':segments},tmp);tmp.replace(path)
        writes.append(time.perf_counter()-tick)
        if (epoch+1)%10==0:print(tag,epoch+1,'/',epochs,flush=True)
    total=sum(z['seconds'] for z in blocks);updates=sum(z['updates'] for z in blocks)
    assert updates==epochs*((len(x)+CONFIG['batch_size']-1)//CONFIG['batch_size'])
    metadata={'identity':identity,'parameters':sum(p.numel() for p in model.parameters()),'training_compute_seconds':total,
              'derived_ms_update':1000*total/updates,'updates':updates,'trace':trace,'blocks':blocks,
              'setup_seconds_this_invocation':setup,'fit_window_seconds_this_invocation':time.perf_counter()-fit_tick,
              'checkpoint_write_seconds':writes,'runtime_segments':segments,'resumed_this_invocation':resumed,'checkpoint_sha256':digest(path)}
    # Completed jobs are preserved instead of overwriting durations with reload-only sessions.
    if record.exists():
        saved=json.loads(record.read_text());assert saved['identity']==identity and saved['checkpoint_sha256']==digest(path)
    else:atomic_json(record,metadata)
    return model


class FourierNystrom(BaseEstimator,TransformerMixin):
    """Batched exact kernel entries, approximate rank-m feature map; train landmarks only."""
    def __init__(self,n_components=256,seed=71911):
        self.n_components=n_components;self.seed=seed

    def fit(self,x,y=None):
        count=min(self.n_components,len(x))
        indices=np.random.default_rng(self.seed).choice(len(x),count,replace=False)
        self.landmarks_=np.asarray(x[indices],dtype=np.float64).copy()
        gram=fourier_kernel(self.landmarks_,self.landmarks_)
        eigenvalues,eigenvectors=np.linalg.eigh(gram)
        self.normalization_=eigenvectors @ np.diag(1/np.sqrt(np.maximum(eigenvalues,1e-12))) @ eigenvectors.T
        return self

    def transform(self,x):
        result=np.empty((len(x),len(self.landmarks_)),dtype=np.float64)
        for first in range(0,len(x),2048):
            result[first:first+2048]=fourier_kernel(x[first:first+2048],self.landmarks_) @ self.normalization_
        return result


def classical(candidate):
    family=candidate['family']
    if family=='logistic':return LogisticRegression(C=candidate['C'],max_iter=2000,solver='lbfgs',random_state=CONFIG['selection_seed'])
    if family=='extra':return ExtraTreesClassifier(n_estimators=300,max_leaf_nodes=8192,min_samples_leaf=candidate['min_samples_leaf'],max_features=1.0,n_jobs=1,random_state=CONFIG['selection_seed'])
    if family=='hgb':return HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,max_leaf_nodes=candidate['max_leaf_nodes'],early_stopping=False,random_state=CONFIG['selection_seed'])
    if family=='fourier_nystrom':
        return make_pipeline(FourierNystrom(n_components=CONFIG['fourier_landmarks'],seed=CONFIG['selection_seed']),SGDClassifier(loss='log_loss',alpha=candidate['alpha'],max_iter=1000,tol=1e-4,early_stopping=False,random_state=CONFIG['selection_seed'],shuffle=True))
    raise ValueError(candidate)


def fit_classical(tag,candidate,x,y):
    path=BASE/'checkpoints'/f'{tag}.joblib';record=BASE/'results/jobs'/f'{tag}.json'
    identity={'candidate':candidate,'code':code_identity(),'x':hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest(),'y':hashlib.sha256(memoryview(np.ascontiguousarray(y)).cast('B')).hexdigest(),'environment':runtime_identity(environment())}
    if record.exists():
        old=json.loads(record.read_text());assert old['identity']==identity and digest(path)==old['checkpoint_sha256'];return joblib.load(path)
    tick=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model=classical(candidate);model.fit(x,y)
    seconds=time.perf_counter()-tick;save_joblib(path,model)
    atomic_json(record,{'identity':identity,'runtime':environment(),'fit_seconds':seconds,'warnings':[str(w.message) for w in caught],'checkpoint_sha256':digest(path)})
    return model


def classical_probabilities(model,x):
    return np.concatenate([model.predict_proba(x[first:first+4096])[:,1] for first in range(0,len(x),4096)])
