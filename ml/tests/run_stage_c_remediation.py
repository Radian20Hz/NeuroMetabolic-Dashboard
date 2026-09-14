"""Sequential parent watchdog; all attempts share durable campaign counters."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT/'ml/models/stage_c_remediation_20260914'
CONFIG = ROOT/'experiments/stage_c_remediation_20260914/preregistration.json'


def main():
    campaign = json.loads((CAMPAIGN/'campaign.json').read_text())
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == campaign['config_sha256']
    assert not any(r.get('status') == 'RUNNING' for r in campaign['runs']), 'previous process state unresolved'
    run = CAMPAIGN/f"run{len(campaign['runs'])+1:02d}"
    run.mkdir()
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [*ROOT.joinpath('ml/scripts').glob('*.py'), Path(__file__), Path(__file__).with_name('test_stage_c_remediation.py')]}
    (run/'source_hashes.json').write_text(json.dumps(sources, indent=2))
    row = {'run': run.name, 'status': 'RUNNING', 'start': time.time()}
    campaign['runs'].append(row)
    (CAMPAIGN/'campaign.json').write_text(json.dumps(campaign, indent=2))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=str(run), MPLCONFIGDIR=str(run/'mpl'))
    command = [str(ROOT/'.venv/bin/python'), str(Path(__file__).with_name('test_stage_c_remediation.py')), str(run)]
    with (run/'console.log').open('w') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=600)
            status = 'EXITED'
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            status = 'TIMEOUT'
    campaign = json.loads((CAMPAIGN/'campaign.json').read_text())
    campaign['runs'][-1].update(status=status, exit_code=code, seconds=time.time()-row['start'], command=command)
    (CAMPAIGN/'campaign.json').write_text(json.dumps(campaign, indent=2))
    print(json.dumps(campaign, indent=2))
    return code if code >= 0 else 128-code


if __name__ == '__main__':
    sys.exit(main())
