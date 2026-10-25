"""Commit and verify completed training, without accessing official test images."""
from common import *
from run import verify_development

def main():
    complete=BASE/'TRAINING_COMPLETE.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old['code']!=code_identity():raise RuntimeError('Completed training source changed; saved models preserved')
        for name,h in old['artifacts'].items():
            if not Path(name).is_file() or digest(name)!=h:raise RuntimeError('Saved training artifact changed: '+name)
        print('TRAINING ALREADY COMPLETE: verified saved models; no fitting repeated.',flush=True);return
    if (BASE/'results/test_lock.json').exists() or any((BASE/'results/test_predictions').glob('*.npz')):
        raise RuntimeError('Training-only completion refused: unexpected test artifacts')
    verify_development()
    specs=json.loads((BASE/'results/final_specs.json').read_text());assert len(specs)>=50
    assert all(s['backbone'] in CONFIG['backbones'] for s in specs)
    paths=[*sorted((SOURCE_BASE/'scripts').glob('*.py')),*sorted((SOURCE_BASE/'configs').glob('*'))]
    for directory in [BASE/'checkpoints',BASE/'results/cv',BASE/'results/cv_predictions',BASE/'results/jobs',BASE/'results/valid_predictions',MAN]:
        paths.extend(p for p in directory.rglob('*') if p.is_file())
    paths.extend(p for p in (BASE/'results').glob('*.json') if p.is_file())
    for bb in CONFIG['backbones']:
        paths.extend(p for p in (CACHE/bb).glob('*') if p.suffix in ['.pt','.json'])
        for repeat in range(CONFIG['timing_repeats']):
            for head in CONFIG['heads']:assert (BASE/'results/jobs'/f'timing_{bb}_{head}_{repeat}.json').is_file()
    atomic_json(complete,{'status':'training_complete_test_unevaluated','code':code_identity(),'artifacts':{str(p):digest(p) for p in sorted(set(paths))},'models':specs,'environment':environment(),'test_predictions':False,'frozen_encoder':True,'scope':'Saved fitted heads, pretrained state, train-only transforms and selection, validation calibration/predictions, RNG/optimizer states, sample and slide IDs, epoch traces and elapsed timing. No final statistical analysis or official test evaluation performed.'})
    print('TRAINING COMPLETE. Saved models and evidence:',BASE,flush=True)

if __name__=='__main__':main()
