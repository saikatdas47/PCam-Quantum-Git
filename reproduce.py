"""Recompute manuscript evidence checks from retained outputs, without training or test tuning."""
from pathlib import Path
import subprocess,sys,argparse,os
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('--full-analysis',action='store_true',help='Also recompute the saved-prediction slide jackknife and initial reporting tables');a=p.parse_args()
subprocess.run([sys.executable,'unpack_evidence.py'],check=True)
steps=['paper/major_revision/analysis/verify_evidence.py','paper/major_revision/analysis/validate_circuit.py','paper/plos_one_word/consistency_audit/check_science.py']
if a.full_analysis:steps+=['paper/plos_one_initial/analysis/analyze_evidence.py','paper/major_revision/analysis/summarize_saved_evidence.py']
for script in steps:
 print('CHECK:',script,flush=True);subprocess.run([sys.executable,script],check=True)
print('Completed same-project saved-evidence checks. No training or model selection. See README for scope and limitations.')
