"""Post hoc exact-image audit. Reads all official images without fitting models."""
from pathlib import Path
import gzip,shutil,hashlib,json,csv,tempfile
import numpy as np
import h5py
P=Path('paper/major_revision');R=Path('resnet50_evidence/training/pcam_three_backbones/resnet50/manifests');records={};hashes={}
for split in ['train','valid','test']:
 dst=P/f'results/{split}_image_hashes.npz';raw=Path(f'data/raw/camelyonpatch_level_2_split_{split}_x.h5');compressed=raw.with_suffix('.h5.gz')
 with np.load(R/f'{split}.npz') as z:man={k:z[k].copy() for k in z.files}
 if dst.exists():
  with np.load(dst) as z:values=z['sha256'].copy()
 else:
  path=raw;temporary=None
  try:
   if not path.exists():
    temporary=P/f'results/temporary_{split}_images.h5';path=temporary
    with gzip.open(compressed,'rb') as src,path.open('wb') as out:shutil.copyfileobj(src,out,8*1024*1024)
   with h5py.File(path,'r') as z:
    data=z['x'];assert data.shape==(len(man['labels']),96,96,3)
    values=np.empty(len(data),dtype='U64')
    for first in range(0,len(data),512):
     block=data[first:first+512]
     for i,image in enumerate(block):values[first+i]=hashlib.sha256(image.tobytes()).hexdigest()
   np.savez_compressed(dst,sha256=values,ids=man['ids'])
  finally:
   if temporary is not None and temporary.exists():temporary.unlink()
 if len(man['image_sha256']):np.testing.assert_array_equal(values,man['image_sha256'])
 # Validate raw label ordering against the exported manifest as well.
 with gzip.open(Path(f'data/raw/camelyonpatch_level_2_split_{split}_y.h5.gz'),'rb') as g:
  with h5py.File(g,'r') as z:np.testing.assert_array_equal(z['y'][:].ravel(),man['labels'])
 hashes[split]=set(values.tolist());unique,counts=np.unique(values,return_counts=True)
 records[split]={'rows':len(values),'unique_images':len(unique),'repeated_rows_beyond_first':len(values)-len(unique),'max_identical_image_multiplicity':int(counts.max()),'labels_match_exported_manifest':True,'hashes_match_saved_manifest':True if len(man['image_sha256']) else None}
 print(split,records[split],flush=True)
overlap={}
for a,b in [('train','valid'),('train','test'),('valid','test')]:
 common=hashes[a]&hashes[b];overlap[a+'_'+b]={'shared_unique_images':len(common),'sha256':sorted(common)}
result={'status':'executed','timing':'Post hoc data integrity audit after locked test analysis. No fitted models or scores changed.','hash_definition':'SHA256 over C-order uint8 RGB pixels, all96x96x3 bytes per patch','splits':records,'cross_split_exact_overlap':overlap,'limits':['Exact pixel identity only. No proof of patient independence or absence of near duplicates.','Local extracted train HDF5 lacks the original gzip compressed-file provenance equivalence. Existing pixel hashes match the exported train and validation manifests.']}
(P/'results/exact_duplicate_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
