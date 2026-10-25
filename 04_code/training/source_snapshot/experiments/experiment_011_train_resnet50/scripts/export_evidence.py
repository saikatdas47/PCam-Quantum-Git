"""Export self-contained fitted-model evidence and prediction rows without refitting."""
from common import *
import csv,zipfile

def main():
    destination=BASE/'exports';destination.mkdir(parents=True,exist_ok=True)
    for split in ['valid','test']:
        with (destination/f'{split}_predictions.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['tag','sample_id','slide','label','score','probability','threshold','predicted_label'])
            for p in sorted((BASE/'results'/f'{split}_predictions').glob('*.npz')):
                with np.load(p,allow_pickle=False) as d:
                    cut=float(d['threshold'])
                    writer.writerows((p.stem,str(i),str(sl),int(y),float(s),float(pr),cut,int(pr>=cut)) for i,sl,y,s,pr in zip(d['ids'],d['slides'],d['labels'],d['scores'],d['probabilities']))
    # Includes all CV/final model and optimizer states, transforms, epoch traces,
    # RNG states, selection, thresholds, bootstrap draws, sample IDs and code.
    paths=[]
    for directory in [SOURCE_BASE/'scripts',SOURCE_BASE/'configs',BASE/'checkpoints',BASE/'results',BASE/'reports',BASE/'figures',MAN]:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(destination.glob('*.csv'))
    paths.extend([Path('shared/models/fourier_kernel.py'),Path('shared/data_audit/results/audit.json'),Path('features/local_pilot_deit_tiny/identity.json')])
    for bb in CONFIG['backbones']:paths.extend(p for p in (CACHE/bb).glob('*') if p.suffix in ['.json','.pt'])
    paths=sorted(set(paths));manifest={str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in paths}
    atomic_json(destination/'MANIFEST.json',{'files':manifest,'feature_arrays':'Retained on Drive at the exact paths/checksums in feature commit markers; not duplicated in this evidence ZIP','raw_inputs':'Official raw files retained in data/raw','environment':environment(),'protocol_amendment':CONFIG['selection_amendment']})
    zip_path=destination/'reproducibility_evidence.zip';temp=zip_path.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths:archive.write(p,str(p))
        archive.write(destination/'MANIFEST.json',str(destination/'MANIFEST.json'))
    temp.replace(zip_path);print('Evidence archive:',zip_path)

if __name__=='__main__':main()
