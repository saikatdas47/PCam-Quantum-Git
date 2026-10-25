"""Run from the project root using the environment specified beside this file."""
from pathlib import Path
import subprocess,sys,argparse
p=argparse.ArgumentParser();p.add_argument('--raw-images',action='store_true');a=p.parse_args()
base=Path('paper/major_revision/analysis')
for name in ['verify_evidence.py','validate_circuit.py','summarize_saved_evidence.py']:
 subprocess.run([sys.executable,str(base/name)],check=True)
if a.raw_images:
 for name in ['audit_images.py','overlap_sensitivity.py']:subprocess.run([sys.executable,str(base/name)],check=True)
print('Completed reporting checks. No training or model selection was run.')
