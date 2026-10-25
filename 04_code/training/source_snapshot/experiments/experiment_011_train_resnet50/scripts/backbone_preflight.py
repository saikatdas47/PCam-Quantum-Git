"""Non-pathology CPU/CUDA backbone fixtures; no dataset/test images are accessed."""
from common import *
from torch_backbones import build,WEIGHTS
import torch

def main():
    guard_pretest();torch.set_num_threads(CONFIG['threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    fixtures=np.load(SOURCE_BASE/'configs/backbone_fixtures.npz',allow_pickle=False)
    x=((torch.arange(3*224*224).reshape(1,3,224,224)%251).float()/125)-1
    records=[]
    for short,spec in CONFIG['backbones'].items():
        model=build(spec['model']);h=hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):h.update(key.encode());h.update(value.cpu().contiguous().numpy().tobytes())
        assert h.hexdigest()==WEIGHTS[spec['model']]['feature_state_sha256']
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            with torch.inference_mode():actual=model.to(device)(x.to(device)).cpu().numpy()
            np.testing.assert_allclose(actual,fixtures[short],rtol=1e-4,atol=1e-4)
            records.append({'backbone':short,'device':device,'max_absolute_error':float(np.max(np.abs(actual-fixtures[short]))),'passed':True})
        print('Backbone fixture passed:',short,flush=True)
    atomic_json(BASE/'reports/backbone_preflight.json',{'checks':records,'environment':environment(),'scope':'Deterministic synthetic image; no PCam test access'})

if __name__=='__main__':main()
