"""Read-only import of verified train/valid frozen features from experiment009."""
from common import *
from portable_runtime import feature_science
from torch_backbones import WEIGHTS,preprocessing

def main():
    guard_pretest()
    oldcache=Path('features/native_torch_full_pcam')
    allowed=json.loads((SOURCE_BASE/'configs/legacy_feature_sources.json').read_text())
    report=[]
    for bb,spec in CONFIG['backbones'].items():
        origin=oldcache/bb/'identity.json'
        if not origin.exists():
            print('No legacy cache; new extraction will run:',bb);continue
        old=json.loads(origin.read_text())
        if old['code']!=allowed:raise RuntimeError('Legacy feature source provenance differs. Original files preserved; no import performed.')
        expected=preprocessing(spec['model'])
        prep={'resize':CONFIG['input_resize'],'crop':None,'antialias':True,'interpolation':expected['interpolation'],'mean':list(expected['mean']),'std':list(expected['std'])}
        assert old['model']==spec['model'] and old['weights_sha256']==WEIGHTS[spec['model']]['feature_state_sha256']
        assert old['preprocessing']==prep and old['feature_dimension']==spec['features']
        identity=dict(old);identity['code']=code_identity()
        folder=CACHE/bb;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'identity.json').exists():
            require_feature_match(json.loads((folder/'identity.json').read_text()),identity)
            identity=json.loads((folder/'identity.json').read_text())
        else:atomic_json(folder/'identity.json',identity)
        for split in ['train','valid']:
            m=load_manifest(split)
            for start in range(0,len(m['labels']),CONFIG['feature_chunk']):
                stop=min(start+CONFIG['feature_chunk'],len(m['labels']));name=f'{split}_{start:06d}_{stop:06d}'
                source=oldcache/bb/(name+'.npz');commit=source.with_suffix('.json');target=folder/(name+'.json')
                if target.exists() or (folder/(name+'.npz')).exists():continue
                if not source.exists() or not commit.exists():continue
                info=json.loads(commit.read_text());assert feature_science(info['identity'])==feature_science(old)
                assert digest(source)==info['sha256']
                with np.load(source,allow_pickle=False) as d:
                    np.testing.assert_array_equal(d['ids'],m['ids'][start:stop]);np.testing.assert_array_equal(d['labels'],m['labels'][start:stop])
                    np.testing.assert_array_equal(d['image_sha256'],m['image_sha256'][start:stop])
                    assert d['features'].shape==(stop-start,spec['features']) and np.isfinite(d['features']).all()
                updated=dict(info);updated.update(identity=identity,external_path=str(source),manifest_sha256=digest(MAN/f'{split}.npz'),origin_identity=info['identity'],origin_commit_sha256=digest(commit))
                atomic_json(target,updated);report.append({'path':str(source),'sha256':info['sha256'],'source_commit_sha256':digest(commit)})
    if report:atomic_json(BASE/'reports/imported_features.json',{'sources':report,'mode':'Read-only references; legacy arrays, checkpoints and source files unchanged'})
    print('Imported frozen feature chunks without recomputation:',len(report),flush=True)

if __name__=='__main__':main()
