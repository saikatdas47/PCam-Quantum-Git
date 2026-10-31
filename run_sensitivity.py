"""Verify or reanalyse the retained post hoc PCam sensitivity study.
Download pcam_sensitivity_evidence.zip from the analysis-2026-10-31 release.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('command',choices=['verify','analyze','models','figures'])
parser.add_argument('--evidence-zip',type=Path,required=True)
a=parser.parse_args();asset=a.evidence_zip.resolve();index=json.loads((ROOT/'07_evidence/sensitivity_release.json').read_text())
assert asset.is_file(),'Download the release evidence ZIP first.'
assert hashlib.sha256(asset.read_bytes()).hexdigest()==index['sha256'],'Release asset checksum mismatch'
work=ROOT/'08_workspace/sensitivity_project';work.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(asset) as z:
 assert z.testzip() is None
 for entry in z.infolist():
  p=Path(entry.filename)
  assert not p.is_absolute() and '..' not in p.parts and p.parts[:2]==('extensions','resnet50_extensions_evidence'),'Unsafe archive member'
 z.extractall(work)
e=work/'extensions/resnet50_extensions_evidence';manifest=json.loads((e/'MANIFEST.json').read_text())
for name,v in manifest.items():
 p=e/name;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
lock=json.loads((e/'extension_lock.json').read_text())
for name,h in lock['artifacts'].items():assert hashlib.sha256((e/name).read_bytes()).hexdigest()==h,name
print('Verified immutable evidence:',len(manifest),'manifest entries and',len(lock['artifacts']),'lock artifacts')
(work/'paper/analysis').mkdir(parents=True,exist_ok=True)
for p in (ROOT/'05_results/sensitivity').glob('*'):
 if p.is_file():shutil.copy2(p,work/'paper/analysis'/p.name)
baseline=work/'paper/Previous/code/inputs';baseline.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'04_code/sensitivity/inputs/original_results.json',baseline/'results.json')
if a.command!='verify':
 script={'analyze':'analyze_current_study.py','models':'audit_fitted_models.py','figures':'make_current_figures.py'}[a.command]
 env=os.environ.copy();env['PCAM_STUDY_ROOT']=str(work);env['MPLCONFIGDIR']=str(work/'paper/analysis/.matplotlib')
 subprocess.run([sys.executable,str(ROOT/'04_code/sensitivity'/script)],check=True,env=env,cwd=ROOT)
 print('Outputs:',work/'paper')
print('No new training performed.')
