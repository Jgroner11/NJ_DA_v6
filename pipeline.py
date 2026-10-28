"""Run export_bins.m and then umap_plots.py for every session, one at a time.

    python pipeline.py

Run from the project root, with the conda env active so umap_plots.py runs
under this same interpreter. Each session gets its own MATLAB process, told
which .mat file to export through data_file. The first step to fail stops the
whole run.
"""

import subprocess
import sys
import time
from pathlib import Path

# the .mat files in data/raw/; each session's outputs are named for its stem
SESSIONS = [
    'reduced_np_data_17_05_2026.mat',
    'reduced_np_18_05_2025.mat',
    'reduced_np_data_25_05_2026.mat',
]

ROOT = Path(__file__).resolve().parent


def run(command):
    """Run one step from the project root, stopping the pipeline if it fails."""
    started = time.time()
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f'failed with exit code {result.returncode}: {" ".join(command)}')
    print(f'  done in {time.time() - started:.0f} s')


def main():
    for data_file in SESSIONS:
        session = Path(data_file).stem
        print(f'\n=== {session} ===')
        print('export_bins.m')
        run(['matlab', '-batch', f"data_file = '{data_file}'; export_bins"])
        print('umap_plots.py')
        run([sys.executable, 'umap_plots.py', session])


if __name__ == '__main__':
    main()
