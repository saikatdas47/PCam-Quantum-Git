"""Persist actual mapped inputs so later resumes do not recompute floating-point bytes."""
from pathlib import Path
import hashlib,json,weakref

def install(common,models,run):
    import numpy as np
    original=models.map_features;folder=Path('training/resnet50_final_v1/mapped_inputs');folder.mkdir(parents=True,exist_ok=True);memo={}
    def array_hash(a):
        key=id(a);old=memo.get(key)
        if old and old[0]() is a:return old[1]
        value=hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest();memo[key]=(weakref.ref(a),value);return value
    def mapped(t,features,q,mapping):
        if q not in [4,8]:return original(t,features,q,mapping)
        transform=hashlib.sha256()
        for obj,names in [(t['pca'],['components_','mean_']),(t['scale'],['mean_','scale_'])]:
            for name in names:transform.update(memoryview(np.ascontiguousarray(getattr(obj,name))).cast('B'))
        identity={'transform_state':transform.hexdigest(),'features':array_hash(features),'feature_shape':list(features.shape),'feature_dtype':str(features.dtype),'q':q,'mapping':mapping,'algorithm_source':common.code_identity()}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();path=folder/(key+'.npy');marker=path.with_suffix('.json')
        if path.exists() and marker.exists():
            info=json.loads(marker.read_text());assert info['identity']==identity and common.digest(path)==info['sha256']
            result=np.load(path,allow_pickle=False);assert result.shape==(len(features),q) and np.isfinite(result).all();return result
        result=original(t,features,q,mapping);temp=path.with_suffix('.tmp')
        with temp.open('wb') as stream:np.save(stream,result,allow_pickle=False)
        temp.replace(path);common.atomic_json(marker,{'identity':identity,'sha256':common.digest(path),'mapped_array_sha256':hashlib.sha256(memoryview(np.ascontiguousarray(result)).cast('B')).hexdigest(),'actual_environment':common.environment()})
        return result
    models.map_features=mapped;run.map_features=mapped
