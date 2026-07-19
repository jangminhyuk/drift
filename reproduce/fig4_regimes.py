#!/usr/bin/env python3
"""Fig. 4 - UWB error-regime map (Mamba). Runs the regime renderer on the
committed baseline-diagnostics + gains CSV and copies the result into figures/.
No GPU / dataset needed."""
import os
import sys
import shutil
import subprocess
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
script = REPO / 'research/dr_conclusions/scripts/fig_drift_guideline_v5_uwb.py'
env = dict(os.environ, GUIDELINE_UNITS_CSV='guideline_units_alltdoa2_v47.csv',
           FIG_SUFFIX='', FIG_STYLE='group')
subprocess.run([sys.executable, str(script)], env=env, check=True)
src = REPO / 'research/dr_conclusions/figures_guideline_v5_paper/FIG_REGIMES_UWB'
out = REPO / 'figures'
out.mkdir(exist_ok=True)
for ext in ('pdf', 'png'):
    shutil.copy2(f'{src}.{ext}', out / f'FIG_REGIMES_UWB.{ext}')
print('wrote', out / 'FIG_REGIMES_UWB.pdf')
