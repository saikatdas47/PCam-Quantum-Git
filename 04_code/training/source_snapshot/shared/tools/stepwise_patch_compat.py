"""Python patch-only compatibility, preserving actual provenance and strict guards."""
from pathlib import Path
import json,copy,hashlib

def same_release(a,b):
    left=copy.deepcopy(a);right=copy.deepcopy(b)
    try:
        pa=tuple(map(int,left.pop('python').split('.')));pb=tuple(map(int,right.pop('python').split('.')))
    except (ValueError,KeyError,AttributeError):return False
    return len(pa)==len(pb)==3 and pa[:2]==pb[:2] and left==right

def probe_features(common):
    import numpy as np,torch,h5py
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torch_backbones import build,preprocessing,WEIGHTS
    records=[]
    torch.set_num_threads(common.CONFIG['threads']);torch.use_deterministic_algorithms(True)
    for bb,spec in common.CONFIG['backbones'].items():
        folder=common.CACHE/bb;markers=[]
        for split in ['train','valid']:markers.extend(sorted(folder.glob(split+'_*.json')))
        markers=[p for p in markers if '_session_' not in p.name and 'sha256' in json.loads(p.read_text())]
        if not markers:continue
        model=build(spec['model']).eval().requires_grad_(False)
        h=hashlib.sha256()
        for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
        assert h.hexdigest()==WEIGHTS[spec['model']]['feature_state_sha256']
        config=preprocessing(spec['model']);transform=transforms.Compose([transforms.Resize(tuple(common.CONFIG['input_resize']),interpolation=InterpolationMode(config['interpolation']),antialias=True),transforms.ToTensor(),transforms.Normalize(config['mean'],config['std'])])
        for marker in dict.fromkeys([markers[0],markers[-1]]):
            info=json.loads(marker.read_text());path=Path(info.get('external_path',str(marker.with_suffix('.npz'))))
            assert common.digest(path)==info['sha256']
            assert info['identity']['weights_sha256']==h.hexdigest()
            with np.load(path,allow_pickle=False) as d:
                ids=d['ids'][:2].copy();expected=d['features'][:2].copy();hashes=d['image_sha256'][:2].copy()
            split=marker.stem.split('_')[0];indices=[int(str(i).split(':')[1]) for i in ids]
            with h5py.File(common.source_h5(split),'r') as f:images=[f['x'][i] for i in indices]
            assert [hashlib.sha256(im.tobytes()).hexdigest() for im in images]==list(hashes)
            with torch.inference_mode():actual=model(torch.stack([transform(Image.fromarray(im)) for im in images])).numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
            records.append({'marker':str(marker),'ids':ids.tolist(),'max_absolute_error':float(np.max(np.abs(actual-expected))),'rtol':1e-4,'atol':1e-5})
    return records

def install(common,*,features=False):
    """Use one explicit compatibility anchor, never rewrite actual environments."""
    original=common.runtime_identity;actual=original(common.environment());path=common.BASE/'workflow/PYTHON_PATCH_COMPATIBILITY.json'
    if path.exists():anchor=json.loads(path.read_text())['compatibility_anchor']
    else:
        anchor=None
        # Existing head identities take precedence, so old checkpoint dictionaries
        # remain unchanged. Their input/source/optimizer/diagnostic guards still run.
        for record in sorted((common.BASE/'results/jobs').glob('*.json')):
            candidate=json.loads(record.read_text()).get('identity',{}).get('environment')
            if candidate:anchor=candidate;break
        if anchor is None:
            for bb in common.CONFIG['backbones']:
                origin=common.CACHE/bb/'identity.json'
                if origin.exists():anchor=original(json.loads(origin.read_text())['software']);break
        if anchor is None:anchor=actual
    if not same_release(anchor,actual):raise RuntimeError('Runtime differs beyond a Python patch update. Preserved all artifacts. Saved='+json.dumps(anchor,sort_keys=True)+'; current='+json.dumps(actual,sort_keys=True))
    def compatible(environment):
        value=original(environment)
        if not same_release(anchor,value):raise RuntimeError('Recorded runtime is not compatible with this patch-only anchor')
        value['python']=anchor['python']
        return value
    probes=probe_features(common) if features and anchor['python']!=actual['python'] else []
    # New actual runtime descriptions stay in environment()/runtime_segments.
    common.runtime_identity=compatible
    if not path.exists():common.atomic_json(path,{'compatibility_anchor':anchor,'semantics':'Identity compatibility only; actual Python retained in environment and runtime_segments'})
    common.atomic_json(common.BASE/'workflow'/('patch_check_'+str(__import__('time').time_ns())+'.json'),{'anchor':anchor,'actual':actual,'feature_probes':probes,'head_resume_guards':'Original x/y/source/optimizer identity and saved diagnostic replay unchanged'})
    return original
