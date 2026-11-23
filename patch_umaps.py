"""A 3D UMAP per patch for one session, each labelled by valuable patch.

Patch P's UMAP is fitted only on the bins of trials that ended at patch P
(ports 2P-1 and 2P) -- every bin of those trials, wherever the mouse is, as in
v5's patch_<P>_all -- with the same parameters as the full UMAP. Each is
coloured by the trial's valuable patch label, as umap_plots.py's whole trial
umap_valuable_patch is. Writes, into figures/<session>/patch/:

    umap_patch_<P>_<N>ms.html     patch P's UMAP, P = 1-4, at N ms bins
    umap_patch_all_<N>ms.html     the four as subplots side by side

each with a PNG beside it. A patch with fewer than MIN_BINS bins is skipped:
below that UMAP's neighbourhoods stop meaning much.

    python patch_umaps.py                               # May 17
    python patch_umaps.py reduced_np_data_25_05_2026
    python patch_umaps.py reduced_np_data_25_05_2026 --skip_created
    python patch_umaps.py reduced_np_data_25_05_2026 --bin_ms 10

pipeline.py runs it for every session. Run from the project root, after
export_bins.m (binned_<N>ms.mat must hold trial_patch_per_bin). Fits are cached in
.cache/ like the full one, keyed on the patch's rows and its name.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from umap_plots import (COMBINED_HEIGHT, COMBINED_WIDTH, PNG_HEIGHT, PNG_WIDTH, SESSION,
                        UMAP_PARAMS, VALUABLE_COLOURS, VALUABLE_NAMES, add_bin_argument, apply_layout,
                        binned_path, category_figure, combined_figure, fit_key, model_for, save)

MIN_BINS = 5 * UMAP_PARAMS['n_neighbors']       # as in v5


def plot_session(session=SESSION, skip_created=False, bin_ms=100):
    """Fit, or load, each patch's UMAP and write its valuable patch plot."""
    binned = loadmat(binned_path(session, bin_ms))
    spikes = binned['spikes'].T                         # bins x units
    trial_patches = binned['trial_patch_per_bin'].ravel().astype(int)
    valuable = binned['valuable_per_bin'].ravel().astype(int)
    settings = (f'{int(binned["bin_size_ms"].item())} ms bins, '
                f'sigma {binned["sigma_bins"].item():g} bins')

    out_dir = Path('figures') / session / 'patch'
    out_dir.mkdir(parents=True, exist_ok=True)
    patches = {}
    for patch in range(1, 5):
        in_patch = trial_patches == patch
        print(f'{session}: patch {patch}, {in_patch.sum()} bins')
        if in_patch.sum() < MIN_BINS:
            print(f'  skipped, fewer than {MIN_BINS} bins')
            continue

        rows = spikes[in_patch]
        points = model_for(rows, fit_key(rows, f'patch_{patch}')).embedding_
        figure = category_figure(points, valuable[in_patch], VALUABLE_COLOURS,
                                 VALUABLE_NAMES, 'post discovery exploration, without return')
        apply_layout(figure, f'{session}: patch {patch} UMAP by valuable patch, '
                             f'whole trial ({settings})')
        save(figure, out_dir / f'umap_patch_{patch}_{bin_ms}ms.html', PNG_WIDTH, PNG_HEIGHT, skip_created)
        patches[f'patch_{patch}'] = (f'patch {patch}', figure)

    if patches:
        combined = combined_figure(
            patches, f'{session}: each patch\'s UMAP by valuable patch, whole trial ({settings})',
            cols=len(patches))
        save(combined, out_dir / f'umap_patch_all_{bin_ms}ms.html', COMBINED_WIDTH, COMBINED_HEIGHT // 2,
             skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot each patch's own UMAP by valuable patch.")
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave figures whose HTML and PNG both already exist untouched')
    add_bin_argument(parser)
    args = parser.parse_args()
    plot_session(args.session, args.skip_created, args.bin_ms)
