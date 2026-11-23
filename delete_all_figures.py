"""Delete every figure the plotting scripts have written, i.e. everything in figures/.

    python delete_all_figures.py          # lists what it will delete and asks first
    python delete_all_figures.py --yes    # deletes without asking

Run from the project root. The figures/ folder itself is kept, empty; rerun
pipeline.py to regenerate the plots. Nothing outside figures/ is touched, so
the exports in data/ and the cached UMAP fits and DataJoint tables in .cache/
stay, and the rerun reuses them.
"""

import argparse
import shutil
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / 'figures'


def main():
    parser = argparse.ArgumentParser(description='Delete everything in figures/.')
    parser.add_argument('--yes', action='store_true', help='delete without asking first')
    args = parser.parse_args()

    if not FIGURES.is_dir():
        print(f'{FIGURES} does not exist, nothing to delete')
        return

    # everything directly inside figures/: one folder per session, plus any stray files
    entries = sorted(FIGURES.iterdir())
    n_files = sum(1 for path in FIGURES.rglob('*') if path.is_file())
    if not entries:
        print(f'{FIGURES} is already empty')
        return

    print(f'{n_files} files in {FIGURES}:')
    for entry in entries:
        print(f'  {entry.name}')
    if not args.yes and input('Delete them all? [y/N] ').strip().lower() != 'y':
        print('nothing deleted')
        return

    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f'deleted {n_files} files')


if __name__ == '__main__':
    main()
