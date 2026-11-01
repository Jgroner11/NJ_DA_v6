"""Run export_bins.m, umap_plots.py and distance_on_maze.py for every session,
one at a time.

    python pipeline.py
    python pipeline.py --skip_created

Run from the project root, with the conda env active so umap_plots.py runs
under this same interpreter. Each session gets its own MATLAB process, told
which .mat file to export through data_file. The first step to fail stops the
whole run. --skip_created is passed on to both plotting scripts, which then leave
figures that already exist as they are; the export still runs every time.
"""

import argparse
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
    parser = argparse.ArgumentParser(description='Export and plot every session.')
    parser.add_argument('--skip_created', action='store_true',
                        help='leave figures that already exist untouched')
    args = parser.parse_args()
    plot_flags = ['--skip_created'] if args.skip_created else []

    for data_file in SESSIONS:
        session = Path(data_file).stem
        print(f'\n=== {session} ===')
        print('export_bins.m')
        run(['matlab', '-batch', f"data_file = '{data_file}'; export_bins"])
        print('umap_plots.py')
        run([sys.executable, 'umap_plots.py', session, *plot_flags])
        print('distance_on_maze.py')
        run([sys.executable, 'distance_on_maze.py', session, *plot_flags])


if __name__ == '__main__':
    main()
