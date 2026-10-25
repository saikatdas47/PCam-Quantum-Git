"""Verify and extract the retained scientific evidence. Uses Python standard libraries only."""
from pathlib import Path,PurePosixPath
import hashlib,json,io,tarfile,shutil
ROOT=Path(__file__).resolve().parents[2]
class Parts(io.RawIOBase):
 def __init__(self,paths):self.paths=iter(paths);self.current=None
 def readable(self):return True
 def read(self,size=-1):
  if size<0:raise ValueError('Bounded reads are required')
  out=bytearray()
  while len(out)<size:
   if self.current is None:
    try:self.current=next(self.paths).open('rb')
    except StopIteration:break
   b=self.current.read(size-len(out))
   if b:out.extend(b)
   else:self.current.close();self.current=None
  return bytes(out)
 def close(self):
  if self.current:self.current.close()
  super().close()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 folder=ROOT/'07_evidence/archive';index=json.loads((folder/'INDEX.json').read_text());paths=[]
 for row in index['parts']:
  p=folder/row['file'];assert p.name==row['file'];assert p.stat().st_size==row['bytes']
  assert digest(p)==row['sha256'],p.name;paths.append(p)
 count=0
 with Parts(paths) as stream,tarfile.open(fileobj=stream,mode='r|gz') as archive:
  for member in archive:
   rel=PurePosixPath(member.name)
   if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='resnet50_evidence' or not member.isfile():raise ValueError('Unexpected archive member: '+member.name)
   dst=ROOT/'08_workspace'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
   # Existing evidence stays authoritative. Verify it against the archived bytes.
   src=archive.extractfile(member);assert src is not None
   if dst.exists():
    h=hashlib.sha256()
    for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b)
    if dst.stat().st_size!=member.size or digest(dst)!=h.hexdigest():raise RuntimeError('Existing file differs; preserved: '+str(rel))
   else:
    tmp=dst.with_name(dst.name+'.unpacking')
    try:
     with tmp.open('wb') as out:shutil.copyfileobj(src,out,8*1024*1024)
     tmp.replace(dst)
    finally:
     if tmp.exists():tmp.unlink()
   count+=1
 assert count==index['files_in_archive']
 print('Verified archive and extracted/reused',count,'evidence files. No training executed.')
if __name__=='__main__':main()
