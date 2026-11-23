"""Run export_bins.m, umap_plots.py, distance_on_maze.py, jumps_on_graph.py,
patch_umaps.py and alternate_embeddings.py for every session at every bin size
in umap_plots.py's BIN_SIZES (100 and 10 ms), and graph_occupancy.py once per
session, one at a time.

    python pipeline.py
    python pipeline.py --skip_created

Run from the project root, with the conda env active so umap_plots.py runs
under this same interpreter. Each export gets its own MATLAB process, told
which .mat file and bin size to export through data_file and bin_size_ms. The
first step to fail stops the whole run. --skip_created is passed on to every
plotting script, which then leaves figures that already exist as they are; the
exports still run every time. graph_occupancy.py and jumps_on_graph.py connect
to DataJoint (datajoint.json) only for sessions whose NaviGraph tables are not
yet cached in .cache/.

The 10 ms bins are ten times as many as the 100 ms ones, so their first run
fits every UMAP on hundreds of thousands of bins and takes hours; later runs
reuse the cached fits.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from umap_plots import BIN_SIZES

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
        for bin_ms in BIN_SIZES:
            print(f'\n=== {session}, {bin_ms} ms bins ===')
            flags = [*plot_flags, '--bin_ms', str(bin_ms)]
            print('export_bins.m')
            run(['matlab', '-batch', f"data_file = '{data_file}'; bin_size_ms = {bin_ms}; export_bins"])
            print('umap_plots.py')
            run([sys.executable, 'umap_plots.py', session, *flags])
            print('distance_on_maze.py')
            run([sys.executable, 'distance_on_maze.py', session, *flags])
            if bin_ms == BIN_SIZES[0]:
                # time at each location does not depend on the bin size
                print('graph_occupancy.py')
                run([sys.executable, 'graph_occupancy.py', session, *plot_flags])
            print('jumps_on_graph.py')
            run([sys.executable, 'jumps_on_graph.py', session, *flags])
            print('patch_umaps.py')
            run([sys.executable, 'patch_umaps.py', session, *flags])
            print('alternate_embeddings.py')
            run([sys.executable, 'alternate_embeddings.py', session, *flags])


if __name__ == '__main__':
    main()
