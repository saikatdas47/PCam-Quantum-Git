"""CPU/CUDA feature extraction, atomic resumable chunks, no fitting or test selection."""
from common import *
import argparse
import h5py
from torch_backbones import build,preprocessing as resolve_preprocessing,WEIGHTS
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');parser.add_argument('--backbone',choices=CONFIG['backbones'],required=True);parser.add_argument('--split',choices=['train','valid','test'],required=True);args=parser.parse_args()
    if args.split=='test':
        # Import here to avoid a train/development module reading test accidentally.
        from run import verify_lock
        verify_lock()
    else:guard_pretest()
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device auto or --device cpu')
    print('Feature extraction device:',args.device,flush=True)
    torch.set_num_threads(CONFIG['threads']);torch.manual_seed(CONFIG['selection_seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    specification=CONFIG['backbones'][args.backbone]
    model=build(specification['model']).eval().requires_grad_(False)
    assert model.num_features==specification['features']
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):h.update(key.encode());h.update(value.cpu().contiguous().numpy().tobytes())
    assert h.hexdigest()==WEIGHTS[specification['model']]['feature_state_sha256'], 'Pretrained state identity changed'
    data_config=resolve_preprocessing(specification['model'])
    preprocessing={'resize':CONFIG['input_resize'],'crop':None,'antialias':True,'interpolation':data_config['interpolation'],'mean':list(data_config['mean']),'std':list(data_config['std'])}
    if args.backbone=='deit':
        historical=Path('features/local_pilot_deit_tiny/identity.json')
        if not historical.exists():raise RuntimeError('Original DeiT identity required')
        old=json.loads(historical.read_text())
        assert h.hexdigest()==old['weights_state_sha256'] and preprocessing==old['preprocessing'], 'Original transformer weights/preprocess changed'
    transform=transforms.Compose([transforms.Resize(tuple(CONFIG['input_resize']),interpolation=InterpolationMode(data_config['interpolation']),antialias=True),transforms.ToTensor(),transforms.Normalize(data_config['mean'],data_config['std'])])
    folder=CACHE/args.backbone;folder.mkdir(parents=True,exist_ok=True)
    identity={'model':specification['model'],'weights_sha256':h.hexdigest(),'preprocessing':preprocessing,'feature_dimension':model.num_features,'frozen_parameters':sum(p.numel() for p in model.parameters()),'software':environment(),'code':code_identity(),'device':args.device,'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None}
    if (folder/'identity.json').exists():
        origin=json.loads((folder/'identity.json').read_text())
        require_feature_match(origin,identity)
        if runtime_identity(origin['software'])!=runtime_identity(environment()):
            committed=list(folder.glob('train_*.npz'))+list(folder.glob('valid_*.npz'))+list(folder.glob('test_*.npz'))
            committed+= [p for p in folder.glob('*.json') if 'external_path' in json.loads(p.read_text())]
            if not committed:
                # No saved features exist. Preserve the empty-cache record and
                # initialize provenance for the runtime doing the first extraction.
                atomic_json(folder/f'empty_origin_{time.time_ns()}.json',origin)
                atomic_json(folder/'identity.json',identity)
    else:atomic_json(folder/'identity.json',identity)
    # Keep a local copy of actual pretrained state. Remote cache revisions cannot silently alter a resume.
    weights_path=folder/'pretrained_state.pt'
    if not weights_path.exists():
        tmp=weights_path.with_suffix('.tmp');torch.save(model.state_dict(),tmp);tmp.replace(weights_path)
    model=model.to(args.device);manifest=load_manifest(args.split)
    started=time.perf_counter();new_chunks=0
    with h5py.File(source_h5(args.split),'r') as images:
        assert images['x'].shape==(CONFIG['samples'][args.split],96,96,3)
        for start in range(0,len(manifest['labels']),CONFIG['feature_chunk']):
            stop=min(start+CONFIG['feature_chunk'],len(manifest['labels']))
            path=folder/f'{args.split}_{start:06d}_{stop:06d}.npz';marker=path.with_suffix('.json')
            if marker.exists() and 'external_path' in json.loads(marker.read_text()):
                info=json.loads(marker.read_text());path=ROOT/info['external_path']
                if ROOT not in path.resolve().parents:raise RuntimeError('External feature path escapes project')
            if path.exists() and marker.exists():
                info=json.loads(marker.read_text());assert info['sha256']==digest(path)
                require_feature_match(info['identity'],identity)
                with np.load(path,allow_pickle=False) as old:
                    np.testing.assert_array_equal(old['ids'],manifest['ids'][start:stop]);np.testing.assert_array_equal(old['labels'],manifest['labels'][start:stop]);assert np.isfinite(old['features']).all()
                continue
            origin=json.loads((folder/'identity.json').read_text())
            if runtime_identity(origin['software'])!=runtime_identity(environment()):raise RuntimeError('Partial feature cache runtime differs. Preserved saved chunks. Saved='+json.dumps(runtime_identity(origin['software']),sort_keys=True)+'; current='+json.dumps(runtime_identity(environment()),sort_keys=True))
            # An uncommitted chunk is safe to recompute; committed chunks are never overwritten.
            values=[];compute_seconds=0.;prepare_seconds=0.;content=[]
            for first in range(start,stop,CONFIG['feature_batch']):
                tick=time.perf_counter();block=images['x'][first:min(first+CONFIG['feature_batch'],stop)]
                hashes=[hashlib.sha256(im.tobytes()).hexdigest() for im in block];content.extend(hashes)
                if args.split!='test':np.testing.assert_array_equal(hashes,manifest['image_sha256'][first:first+len(block)])
                x=torch.stack([transform(Image.fromarray(im)) for im in block]).to(args.device)
                if args.device=='cuda':torch.cuda.synchronize()
                prepare_seconds+=time.perf_counter()-tick;tick=time.perf_counter()
                with torch.inference_mode():out=model(x).cpu().numpy()
                if args.device=='cuda':torch.cuda.synchronize()
                compute_seconds+=time.perf_counter()-tick
                assert out.shape==(len(block),model.num_features) and np.isfinite(out).all();values.append(out)
            atomic_npz(path,features=np.concatenate(values),ids=manifest['ids'][start:stop],labels=manifest['labels'][start:stop],image_sha256=np.array(content))
            atomic_json(marker,{'identity':identity,'sha256':digest(path),'manifest_sha256':digest(MAN/f'{args.split}.npz'),'compute_and_output_transfer_seconds':compute_seconds,'input_prepare_transfer_seconds':prepare_seconds})
            new_chunks+=1;print(args.backbone,args.split,stop,'/',len(manifest['labels']),flush=True)
    atomic_json(folder/f'{args.split}_session_{time.time_ns()}.json',{'elapsed_seconds':time.perf_counter()-started,'new_chunks':new_chunks,'environment':environment(),'scope':'Feature extraction; no classification','monetary_cost':'unmeasured'})


if __name__=='__main__':main()
