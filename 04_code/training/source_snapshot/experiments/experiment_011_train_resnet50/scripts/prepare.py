"""Prepare complete native-row manifests and training-only folds; no model scoring."""
from common import *
import csv,gzip,io
from collections import Counter
import h5py
from sklearn.model_selection import StratifiedGroupKFold,train_test_split


def main():
    guard_pretest();MAN.mkdir(parents=True,exist_ok=True)
    identity={'code':code_identity(),'raw':{str(p):digest(p) for p in sorted(RAW.glob(f'{PREFIX}*')) if 'mask' not in p.name}}
    provenance=json.loads(Path('shared/data_audit/results/audit.json').read_text())['files']
    for filename,details in provenance.items():
        path=RAW/filename
        if path.exists() and 'mask' not in filename:
            assert identity['raw'][str(path)]==details['sha256'], f'Raw data changed: {filename}'
    bind(BASE/'results/preparation_identity.json',identity)
    prepared={};audit={}
    for split,count in CONFIG['samples'].items():
        with (RAW/f'{PREFIX}{split}_meta.csv').open() as f:rows=list(csv.DictReader(f))
        with gzip.open(RAW/f'{PREFIX}{split}_y.h5.gz','rb') as f:
            with h5py.File(io.BytesIO(f.read()),'r') as h:y=h['y'][:].reshape(-1)
        assert len(rows)==len(y)==count
        assert all(int(r[''])==i and int(r['center_tumor_patch'].lower()=='true')==int(y[i]) for i,r in enumerate(rows))
        slide=np.array([r['wsi'] for r in rows]);ids=np.array([f'{split}:{i}' for i in range(count)])
        # Test-content hashes are deliberately deferred to the locked extraction stage.
        hashes=np.array([],dtype='<U64')
        if split!='test':
            hashes=[]
            with h5py.File(source_h5(split),'r') as h:
                assert h['x'].shape==(count,96,96,3) and h['x'].dtype==np.uint8
                for start in range(0,count,1024):hashes.extend(hashlib.sha256(im.tobytes()).hexdigest() for im in h['x'][start:start+1024])
            hashes=np.array(hashes)
            by_hash={}
            for key,label in zip(hashes,y):
                if key in by_hash and by_hash[key]!=int(label):raise RuntimeError('Conflicting identical images')
                by_hash[key]=int(label)
        data=dict(ids=ids,labels=y,slides=slide,image_sha256=hashes,
                  coord_x=np.array([int(r['coord_x']) for r in rows]),coord_y=np.array([int(r['coord_y']) for r in rows]))
        path=MAN/f'{split}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                for k,v in data.items():np.testing.assert_array_equal(old[k],v)
        else:atomic_npz(path,**data)
        prepared[split]=data
        audit[split]={'rows':count,'slides':len(np.unique(slide)),'class_counts':dict(Counter(map(str,y))),
                      'unique_images':len(set(hashes)) if len(hashes) else None,'manifest_sha256':digest(path)}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        assert not set(prepared[a]['slides'])&set(prepared[b]['slides'])
    assert not set(prepared['train']['image_sha256'])&set(prepared['valid']['image_sha256'])
    tr=prepared['train'];all_indices=np.arange(len(tr['labels']))
    selected,_=train_test_split(all_indices,train_size=CONFIG['selection_samples'],stratify=tr['labels'],random_state=CONFIG['selection_seed'])
    selected=np.sort(selected);folds=np.full(len(all_indices),-1,dtype=np.int8)
    splitter=StratifiedGroupKFold(CONFIG['selection_folds'],shuffle=True,random_state=CONFIG['fold_seed'])
    for fold,(fit,held) in enumerate(splitter.split(selected,tr['labels'][selected],tr['slides'][selected])):
        fit_ids,held_ids=selected[fit],selected[held]
        assert len(np.unique(tr['labels'][held_ids]))==2
        assert not set(tr['slides'][fit_ids])&set(tr['slides'][held_ids])
        assert not set(tr['image_sha256'][fit_ids])&set(tr['image_sha256'][held_ids])
        folds[held_ids]=fold
    va=prepared['valid'];groups=np.unique(va['slides']);rng=np.random.default_rng(CONFIG['calibration_seed']);rng.shuffle(groups)
    calibration=np.isin(va['slides'],groups[:int(len(groups)*CONFIG['calibration_fraction_slides'])])
    assert len(np.unique(va['labels'][calibration]))==len(np.unique(va['labels'][~calibration]))==2
    roles=MAN/'roles.npz'
    if roles.exists():
        with np.load(roles) as old:np.testing.assert_array_equal(old['folds'],folds);np.testing.assert_array_equal(old['calibration'],calibration)
    else:atomic_npz(roles,folds=folds,calibration=calibration)
    audit.update(native_training_rows_retained=True,test_images_read=False,test_predictions=False,
                 train_valid_exact_overlap=0,selection_rows=len(selected),roles_sha256=digest(roles),
                 limits=['Patient IDs unavailable; WSI is not a verified patient ID','Original train_x gzip absent: extracted file provenance remains bounded','No exhaustive near-duplicate/patient-independence proof'])
    atomic_json(BASE/'results/data_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
