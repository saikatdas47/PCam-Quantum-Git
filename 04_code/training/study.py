"""Run the ResNet50 study in a separate workspace with stage and checkpoint recovery."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'08_workspace/training';SNAPSHOT=Path(__file__).resolve().parent/'source_snapshot'
EXPERIMENT='experiments/experiment_011_train_resnet50';SCRIPT=EXPERIMENT+'/scripts/'
STAGES={
 '01_checks':[SCRIPT+'preflight.py'],
 '02_encoder_check':[SCRIPT+'backbone_preflight.py'],
 '03_prepare':[SCRIPT+'prepare.py'],
 '04_train_features':['shared/tools/resnet_finish_study.py','features','--split','train'],
 '05_valid_features':['shared/tools/resnet_finish_study.py','features','--split','valid'],
 '06_select':['shared/tools/resnet_finish_study.py','select'],
 '07_fit':['shared/tools/resnet_finish_study.py','fit'],
 '08_validation':['shared/tools/resnet_finish_study.py','valid'],
 '09_lock':['shared/tools/resnet_finish_study.py','lock'],
 '10_test_features':['shared/tools/resnet_finish_study.py','test_features'],
 '11_test_predictions':['shared/tools/resnet_finish_study.py','test'],
 '12_test_analysis':['shared/tools/resnet_finish_study.py','analysis'],
 '13_export':['shared/tools/resnet_finish_study.py','export']}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind_sources():
 for src in sorted(SNAPSHOT.rglob('*')):
  if not src.is_file():continue
  dst=WORK/src.relative_to(SNAPSHOT);dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():
   if digest(dst)!=digest(src):raise RuntimeError('Workspace source differs. Existing source preserved: '+str(dst.relative_to(WORK)))
  else:shutil.copy2(src,dst)
def save(path,record):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['plan','all',*STAGES],default='plan');a=p.parse_args()
 if a.stage=='plan':
  for name,args in STAGES.items():print(name,':',*args)
  print('Workspace:',WORK.relative_to(ROOT));return
 bind_sources()
 raw=ROOT/'02_data/raw'
 if not raw.is_dir():raise FileNotFoundError('Put official PCam files in 02_data/raw. See 02_data/README.md.')
 (WORK/'data').mkdir(exist_ok=True);dest=WORK/'data/raw'
 if not dest.exists():dest.symlink_to(os.path.relpath(raw,dest.parent),target_is_directory=True)
 if dest.resolve()!=raw.resolve():raise RuntimeError('Training workspace points to another raw dataset. Preserved existing path.')
 workflow=WORK/'workflow';workflow.mkdir(exist_ok=True);state_path=workflow/'state.json'
 state=json.loads(state_path.read_text()) if state_path.exists() else {'steps':{}}
 env=os.environ.copy();env['PYTHONPATH']=str((WORK/'shared/tools').resolve())+os.pathsep+env.get('PYTHONPATH','')
 plan=list(STAGES) if a.stage=='all' else [a.stage]
 for name in plan:
  if state['steps'].get(name,{}).get('status')=='complete':print('SKIP:',name,flush=True);continue
  record={'status':'running','command':[sys.executable,*STAGES[name]],'started_ns':time.time_ns()};state['steps'][name]=record;save(state_path,state)
  print('START/RESUME:',name,flush=True);log=workflow/(name+'_'+str(record['started_ns'])+'.log');tick=time.perf_counter()
  try:
   with log.open('w') as f:
    result=subprocess.Popen(record['command'],cwd=WORK,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
     for line in result.stdout:print(line,end='',flush=True);f.write(line);f.flush()
     code=result.wait()
    except BaseException:
     result.terminate();result.wait();raise
   record.update(status='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.perf_counter()-tick,log=str(log.relative_to(ROOT)));save(state_path,state)
   if code:raise RuntimeError('Stage stopped with progress preserved. See '+str(log.relative_to(ROOT)))
  except BaseException:
   if record['status']=='running':record.update(status='interrupted',elapsed_seconds=time.perf_counter()-tick);save(state_path,state)
   raise
 print('Selected stages complete. Existing checkpoints are preserved.')
if __name__=='__main__':main()
