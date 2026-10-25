"""Synthetic software checks only. No PCam training or scientific results."""
from common import *
import copy,shutil
import models
from models import *
from fourier_kernel import explicit_features
from analyze import weighted_auc_draws
from sklearn.metrics import roc_auc_score


def main():
    guard_pretest();checks=[];rng=np.random.default_rng(71801)
    # Tiny exact feature-map / implicit kernel / full-landmark approximation checks.
    angles=rng.normal(size=(12,4));kernel=fourier_kernel(angles,angles)
    np.testing.assert_allclose(explicit_features(angles)@explicit_features(angles).T,kernel,rtol=1e-12,atol=1e-12)
    mapper=FourierNystrom(12,71911).fit(angles);mapped=mapper.transform(angles)
    np.testing.assert_allclose(mapped@mapped.T,kernel,rtol=1e-7,atol=1e-7)
    checks.append('Fourier explicit/implicit identity and full-landmark Nyström reconstruction')
    y=np.array([0,1,0,1,1,0]);s=np.array([.1,.4,.4,.7,.7,.9]);w=np.array([[1,2,3,1,2,4],[2,1,1,4,3,1]])
    np.testing.assert_allclose(weighted_auc_draws(y,s,w),[roc_auc_score(y,s,sample_weight=row) for row in w],rtol=1e-12,atol=1e-12)
    checks.append('Weighted AUROC ties/nonuniform frequency weights')
    for q in [4,8]:
        off,on=paired_circuits(q,101)
        assert all(torch.equal(v,on.state_dict()[k]) for k,v in off.state_dict().items())
        assert off.weights.shape==on.weights.shape==(2,q)
        assert sum(p.numel() for p in off.parameters())==3*q+1
        tx=torch.tensor(rng.normal(size=(12,q)),dtype=torch.float64)
        expected=torch.cos(tx+off.weights.sum(dim=0))
        np.testing.assert_allclose(off(tx).detach().numpy(),off.readout(expected).squeeze(-1).detach().numpy(),rtol=1e-12,atol=1e-12)
        for head in CONFIG['heads']:
            x=rng.normal(size=(16,q));labels=np.array([0,1]*8)
            a=f'preflight_q{q}_{head}_interrupted';b=f'preflight_q{q}_{head}_uninterrupted'
            for tag in [a,b]:
                for path in [BASE/'checkpoints'/f'{tag}.pt',BASE/'results/jobs'/f'{tag}.json']:
                    if path.exists():path.unlink()
            original=models.train_block;calls=[0]
            def interruption(*args,**kwargs):
                calls[0]+=1
                if calls[0]==2:raise RuntimeError('synthetic interruption after committed epoch')
                return original(*args,**kwargs)
            models.train_block=interruption
            try:
                fit_head(a,head,101,x,labels,.003,epochs=2)
                raise AssertionError('Interruption did not execute')
            except RuntimeError as error:
                if str(error)!='synthetic interruption after committed epoch':raise
            finally:models.train_block=original
            resumed=fit_head(a,head,101,x,labels,.003,epochs=2);baseline=fit_head(b,head,101,x,labels,.003,epochs=2)
            for k,v in baseline.state_dict().items():assert torch.equal(v,resumed.state_dict()[k])
            np.testing.assert_array_equal(head_logits(baseline,x),head_logits(resumed,x))
            changed=False
            try:fit_head(a,head,101,x,labels,.01,epochs=2)
            except RuntimeError:changed=True
            assert changed
            checks.append(f'q{q}/{head}: exact interruption/resume, predictions and identity rejection')
            for tag in [a,b]:
                (BASE/'checkpoints'/f'{tag}.pt').unlink();(BASE/'results/jobs'/f'{tag}.json').unlink()
    features=rng.normal(size=(24,12)).astype(np.float32);ids=np.array([f'train:{i}' for i in range(24)])
    tag='preflight_transform'
    for suffix in ['joblib','json']:
        p=BASE/'checkpoints'/f'{tag}_transform.{suffix}'
        if p.exists():p.unlink()
    transform=fit_transform(features,ids,tag);before=copy.deepcopy(transform['pca'].components_)
    held=rng.normal(100,1,size=(10,12)).astype(np.float32);np.testing.assert_array_equal(map_features(transform,held,8,'standard'),map_features(joblib.load(BASE/'checkpoints'/f'{tag}_transform.joblib'),held,8,'standard'));map_features(transform,held,8,'atan');np.testing.assert_array_equal(before,transform['pca'].components_)
    np.testing.assert_allclose(map_features(transform,held,4,'atan'),np.arctan(map_features(transform,held,4,'standard')))
    rejected=False
    try:fit_transform(features,np.array([f'valid:{i}' for i in range(24)]),'preflight_bad')
    except AssertionError:rejected=True
    assert rejected
    for suffix in ['joblib','json']:(BASE/'checkpoints'/f'{tag}_transform.{suffix}').unlink()
    checks.append('Training-only transform fit, no held-out mutation, nested coordinates and atan mapping')
    # No fitting after lock, including resumed completed jobs through CLI stages.
    lock_path=BASE/'results/test_lock.json'
    try:
        atomic_json(lock_path,{'synthetic_guard_test':True})
        rejected=False
        try:guard_pretest()
        except RuntimeError:rejected=True
        assert rejected
    finally:lock_path.unlink()
    checks.append('Post-lock development guard')
    atomic_json(BASE/'results/software_checks.json',{'passed':True,'scope':'Synthetic checks; no pathology experiments or GPU verification','checks':checks,'code':code_identity(),'environment':environment()})
    print(json.dumps({'passed':True,'checks':checks},indent=2))


if __name__=='__main__':main()
