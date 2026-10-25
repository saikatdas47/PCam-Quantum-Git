"""Entry point for checking, analysing and plotting the retained study results."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
COMMANDS={
 'verify':['04_code/analysis/verify_evidence.py','04_code/analysis/validate_circuit.py','04_code/verification/verify_study.py'],
 'analyze':['04_code/analysis/analyze_predictions.py','04_code/analysis/summarize_saved_evidence.py','04_code/analysis/overlap_sensitivity.py','04_code/verification/verify_study.py'],
 'figures':['04_code/visualization/plot_results.py'],
 'raw-audit':['04_code/analysis/audit_images.py','04_code/analysis/overlap_sensitivity.py']}
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['unpack',*COMMANDS]);args=parser.parse_args();os.chdir(ROOT)
 for name in ['checks','analysis','tables']:(ROOT/'05_results'/name).mkdir(parents=True,exist_ok=True)
 subprocess.run([sys.executable,'04_code/verification/unpack_evidence.py'],check=True)
 for script in COMMANDS.get(args.command,[]):
  print('RUN:',script,flush=True);subprocess.run([sys.executable,script],check=True)
 print('Finished:',args.command,flush=True)
if __name__=='__main__':main()
