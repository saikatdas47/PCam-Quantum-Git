"""Project-contained I/O, identities and fixed configuration for experiment008."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if Path.cwd().resolve() != ROOT:
    raise RuntimeError('Run from the PCam-Quantum-ViT project directory')
for key, value in {'HF_HOME':'.cache/huggingface','TORCH_HOME':'.cache/torch',
                   'XDG_CACHE_HOME':'.cache','MPLCONFIGDIR':'.cache/matplotlib',
                   'PIP_CACHE_DIR':'.cache/pip'}.items():
    os.environ[key] = str(ROOT/value)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import hashlib
import json
import platform
import importlib.metadata
import tempfile
import time
import numpy as np
from portable_runtime import runtime_identity,require_feature_match
SOURCE_BASE=Path('experiments/experiment_011_train_resnet50')
BASE=Path('training/pcam_three_backbones/resnet50')
CONFIG_PATH=SOURCE_BASE/'configs/protocol.json'
CONFIG=json.loads(CONFIG_PATH.read_text())
RAW=Path('data/raw')
PREFIX='camelyonpatch_level_2_split_'
CACHE=BASE/'features'
MAN=BASE/'manifests'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def atomic_npz(path, **payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**payload)
    temporary.replace(path)


def code_identity():
    paths=[CONFIG_PATH,SOURCE_BASE/'configs/legacy_feature_sources.json',SOURCE_BASE/'configs/weights.json',SOURCE_BASE/'configs/backbone_fixtures.npz',*sorted((SOURCE_BASE/'scripts').glob('*.py')),Path('shared/models/fourier_kernel.py'),Path('shared/data_audit/results/audit.json'),Path('features/local_pilot_deit_tiny/identity.json')]
    return {str(p):digest(p) for p in paths}


def environment():
    names=['torch','torchvision','numpy','scipy','scikit-learn','h5py','Pillow','matplotlib','joblib']
    return {'platform':platform.platform(),'python':platform.python_version(),
            'versions':{name:importlib.metadata.version(name) for name in names},
            'threads':CONFIG['threads'],'machine':platform.machine(),'processor':platform.processor(),'ram_cpu_profiling':False}


def bind(path, identity):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text()) != identity:raise RuntimeError(f'Identity mismatch: {path}')
    else:atomic_json(path,identity)


def guard_pretest():
    if (BASE/'results/test_lock.json').exists():
        raise RuntimeError('Test lock exists. Training/development changes are prohibited.')


def source_h5(split):
    """Unpack once within project; a partial extraction never appears complete."""
    import gzip,shutil
    p=RAW/f'{PREFIX}{split}_x.h5'
    if p.exists() and split=='train':return p
    if p.exists() and not p.with_suffix('.h5.gz').exists():
        provenance=json.loads(Path('shared/data_audit/results/audit.json').read_text())['files']
        if p.name not in provenance or digest(p)!=provenance[p.name]['sha256']:raise RuntimeError('Unverified raw extracted images')
        return p
    out=Path('data/processed/full_pcam_three_backbones')/p.name
    marker=out.with_suffix('.source.json')
    compressed=p.with_suffix('.h5.gz')
    identity={'compressed_sha256':digest(compressed)}
    if out.exists():
        if not marker.exists() or json.loads(marker.read_text())!=identity:raise RuntimeError('Unverified unpacked image file')
        return out
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp')
    with gzip.open(compressed,'rb') as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
    tmp.replace(out);atomic_json(marker,identity)
    return out


def load_manifest(split):
    with np.load(MAN/f'{split}.npz',allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def load_features(backbone, split, limit=None):
    """Every chunk carries sample identity and a file checksum in its commit marker."""
    manifest=load_manifest(split);parts=[]
    folder=CACHE/backbone
    identity=json.loads((folder/'identity.json').read_text())
    for start in range(0,len(manifest['labels']) if limit is None else min(limit,len(manifest['labels'])),CONFIG['feature_chunk']):
        stop=min(start+CONFIG['feature_chunk'],len(manifest['labels']))
        p=folder/f'{split}_{start:06d}_{stop:06d}.npz'; marker=p.with_suffix('.json')
        info=json.loads(marker.read_text())
        if 'external_path' in info:
            p=ROOT/info['external_path']
            if ROOT not in p.resolve().parents:raise RuntimeError('External feature path escapes project')
        if info['sha256']!=digest(p):raise RuntimeError('Feature chunk checksum mismatch')
        require_feature_match(info['identity'],identity)
        with np.load(p,allow_pickle=False) as d:
            np.testing.assert_array_equal(d['ids'],manifest['ids'][start:stop])
            np.testing.assert_array_equal(d['labels'],manifest['labels'][start:stop])
            values=d['features'].copy()
        if values.shape!=(stop-start,CONFIG['backbones'][backbone]['features']) or not np.isfinite(values).all():raise RuntimeError('Invalid features')
        parts.append(values)
    values=np.concatenate(parts)
    if limit is not None:
        values=values[:limit];manifest={k:v[:limit] for k,v in manifest.items()}
    return values,manifest
