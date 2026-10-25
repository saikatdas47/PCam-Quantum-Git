"""Original frozen weight variants using PyTorch/Torchvision, without timm/HF packages."""
from pathlib import Path
from collections import OrderedDict
import hashlib,json,time,urllib.request
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50,efficientnet_b0

HERE=Path(__file__).resolve().parent
WEIGHTS=json.loads((HERE.parent/'configs/weights.json').read_text())

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv2d(3,192,16,16)
    def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(192,576);self.proj=nn.Linear(192,192)
    def forward(self,x):
        b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,3,64).permute(2,0,3,1,4).unbind(0)
        x=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.proj(x.transpose(1,2).reshape(b,n,c))

class Mlp(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(192,768);self.fc2=nn.Linear(768,192)
    def forward(self,x):return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(192,eps=1e-6);self.attn=Attention();self.norm2=nn.LayerNorm(192,eps=1e-6);self.mlp=Mlp()
    def forward(self,x):
        x=x+self.attn(self.norm1(x));return x+self.mlp(self.norm2(x))

class DeiTTiny(nn.Module):
    num_features=192
    def __init__(self):
        super().__init__();self.cls_token=nn.Parameter(torch.zeros(1,1,192));self.pos_embed=nn.Parameter(torch.zeros(1,197,192));self.patch_embed=PatchEmbed();self.blocks=nn.Sequential(*[Block() for _ in range(12)]);self.norm=nn.LayerNorm(192,eps=1e-6)
    def forward(self,x):
        if x.shape[-2:]!=(224,224):raise ValueError('DeiT expects full 224x224 resized input')
        x=self.patch_embed(x);x=torch.cat((self.cls_token.expand(len(x),-1,-1),x),dim=1)+self.pos_embed
        return self.norm(self.blocks(x))[:,0]


def efficient_key(key):
    """Map original timm names to Torchvision names, preserving original tensors."""
    direct={'conv_stem':'features.0.0','bn1':'features.0.1','conv_head':'features.8.0','bn2':'features.8.1'}
    parts=key.split('.')
    if parts[0] in direct:return direct[parts[0]]+'.'+'.'.join(parts[1:])
    if parts[0]!='blocks':raise KeyError(key)
    stage,block=int(parts[1]),int(parts[2]);suffix='.'.join(parts[3:])
    mapping=({'conv_dw':'0.0','bn1':'0.1','se.conv_reduce':'1.fc1','se.conv_expand':'1.fc2','conv_pw':'2.0','bn2':'2.1'} if stage==0 else {'conv_pw':'0.0','bn1':'0.1','conv_dw':'1.0','bn2':'1.1','se.conv_reduce':'2.fc1','se.conv_expand':'2.fc2','conv_pwl':'3.0','bn3':'3.1'})
    for old,new in mapping.items():
        if suffix.startswith(old+'.'):return f'features.{stage+1}.{block}.block.{new}.'+suffix[len(old)+1:]
    raise KeyError(key)

class EfficientOriginal(nn.Module):
    num_features=1280
    def __init__(self,state):
        super().__init__();self.network=efficientnet_b0(weights=None);self.network.classifier=nn.Identity()
        state={k:v for k,v in state.items() if not k.startswith('classifier.')}
        self.original_keys=list(state);self.network.load_state_dict({efficient_key(k):v for k,v in state.items()},strict=True)
    def forward(self,x):return self.network(x)
    def state_dict(self,*args,**kwargs):
        native=self.network.state_dict();return OrderedDict((k,native[efficient_key(k)]) for k in self.original_keys)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def download_weights(name):
    """Regular files and atomic renames avoid Drive's HF symlink failure."""
    spec=WEIGHTS[name];folder=Path('.cache/native_torch_weights');folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.pt')
    if path.exists():
        if digest(path)!=spec['file_sha256']:raise RuntimeError(f'Weight checksum mismatch; preserve and investigate: {path}')
        return path
    url=f"https://huggingface.co/timm/{name}/resolve/{spec['revision']}/pytorch_model.bin"
    temporary=path.with_suffix('.download')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as output:
                while True:
                    block=response.read(8*1024*1024)
                    if not block:break
                    output.write(block)
            if temporary.stat().st_size!=spec['bytes'] or digest(temporary)!=spec['file_sha256']:raise RuntimeError('Downloaded weights failed size/checksum validation')
            temporary.replace(path);return path
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)


def build(name,pretrained=True,state=None):
    if pretrained and state is None:state=torch.load(download_weights(name),map_location='cpu',weights_only=True)
    if name=='deit_tiny_patch16_224.fb_in1k':
        model=DeiTTiny()
        if state is not None:model.load_state_dict({k:v for k,v in state.items() if not k.startswith('head.')},strict=True)
    elif name=='resnet50.tv_in1k':
        model=resnet50(weights=None);model.fc=nn.Identity();model.num_features=2048
        if state is not None:model.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
    elif name=='efficientnet_b0.ra_in1k':
        if state is None:raise ValueError('EfficientNet original-key state is required')
        model=EfficientOriginal(state)
    else:raise ValueError(name)
    return model.eval().requires_grad_(False)


def preprocessing(name):
    return {'interpolation':'bilinear' if name=='resnet50.tv_in1k' else 'bicubic','mean':[.485,.456,.406],'std':[.229,.224,.225]}
