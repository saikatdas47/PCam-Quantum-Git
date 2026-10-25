"""Separate mathematical artifact identities from recorded execution environments."""
from copy import deepcopy


def runtime_identity(actual):
    """Allow CPU/CUDA wheel suffixes and host descriptions, not library upgrades."""
    versions=dict(actual['versions'])
    for name in ('torch','torchvision'):
        versions[name]=versions[name].split('+',1)[0]
    return {'python':actual['python'],'versions':versions,'threads':actual['threads'],
            'machine':actual['machine'],'head_device':'cpu','head_dtype':'float64'}


def feature_science(identity):
    return {key:value for key,value in identity.items() if key not in ('software','device','gpu')}


def require_feature_match(stored,current):
    if feature_science(stored)!=feature_science(current):
        raise RuntimeError('Feature scientific identity changed: source, weights, preprocessing or dimensions differ')


def checkpoint_segments(old):
    if 'runtime_segments' in old:return deepcopy(old['runtime_segments'])
    # Legacy checkpoints are explicitly migrated before this code is used.
    raise RuntimeError('Legacy checkpoint requires the preserved-artifact portability migration')
