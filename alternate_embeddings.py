"""UMAPs of one session fitted on a subset of its bins, each with its own labelling.

Each alternate embedding is its own UMAP, fitted only on the bins its
selection keeps, with the same parameters as the full UMAP. Writes, into
figures/<session>/alternate_embeddings/:

    umap_correct_trial_since_discovery.html
                                       fitted on every bin of the correct trials
                                       (rewarded or not). Pre discovery trials
                                       are solid orange, as in valuable patch;
                                       post discovery ones are coloured on a
                                       continuous scale by their place among the
                                       block's post discovery trials, 1 being
                                       the first. (Correct trials are never
                                       perseveration or exploration.)
    umap_correct_patch_<P>_trial_since_discovery.html
                                       the same, fitted on the correct trials at
                                       patch P alone (ports 2P-1 and 2P), so only
                                       blocks whose correct patch is P, P = 1-4
    umap_incorrect_valuable_patch.html fitted on every bin of the incorrect
                                       trials (rewarded or not), coloured by
                                       valuable patch. Post discovery, correct
                                       patch never appears; exploration that
                                       never returns is drawn pale.

    umap_value_units_specific_valuable_patch.html
                                       fitted on every bin but only the units
                                       listed in the session's value selected
                                       unit PSTH CSV, coloured by specific
                                       valuable patch over the whole trial as in
                                       umap_plots.py. The CSV is looked for at
                                       data/raw/<YYYY-MM-DD>_value_selected_unit_PSTHs.csv,
                                       the date read from the session's name;
                                       a session without one gets no plot.
    umap_value_units_specific_valuable_patch_post.html
                                       the same embedding, post discovery trials
                                       only: correct ones in their block's patch
                                       colour, exploration that returns in its
                                       lighter tint, everything else pale

each with a PNG beside it, and each name ending in the bin size, e.g.
umap_incorrect_valuable_patch_100ms.html. A selection with fewer than MIN_BINS bins is
skipped: below that UMAP's neighbourhoods stop meaning much.

    python alternate_embeddings.py                               # May 17
    python alternate_embeddings.py reduced_np_data_25_05_2026
    python alternate_embeddings.py reduced_np_data_25_05_2026 --skip_created
    python alternate_embeddings.py reduced_np_data_25_05_2026 --bin_ms 10

pipeline.py runs it for every session. Run from the project root, after
export_bins.m. Fits are cached in .cache/ like the full one, keyed on the
selected rows and the embedding's name.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

import plotly.graph_objects as go

from umap_plots import (POINT_SIZE, POST_COLOURS, POST_NAMES, PNG_HEIGHT, PNG_WIDTH,
                        SESSION, SPECIFIC_COLOURS, SPECIFIC_NAMES, TEXT_COLOUR,
                        UMAP_PARAMS, VALUABLE_COLOURS, VALUABLE_NAMES, add_bin_argument, apply_layout,
                        binned_path, category_figure, fit_key, model_for, post_codes, save)

MIN_BINS = 5 * UMAP_PARAMS['n_neighbors']       # as in v5
TRIAL_NUMBER_SCALE = 'Viridis'                  # soon after discovery dark, long after bright
VALUE_UNITS_CSV = '{date}_value_selected_unit_PSTHs.csv'   # in data/raw/, date as YYYY-MM-DD


def value_units_csv(session):
    """The session's value selected unit CSV, or None if it has none. The date
    comes from the DD_MM_YYYY at the end of the session's name."""
    day, month, year = re.search(r'(\d{2})_(\d{2})_(\d{4})$', session).groups()
    path = Path('data') / 'raw' / VALUE_UNITS_CSV.format(date=f'{year}-{month}-{day}')
    return path if path.exists() else None


def pre_post_figure(points, valuable, trial_numbers):
    """Pre discovery bins in solid orange, post discovery ones coloured by
    their trial number since discovery on a continuous scale, anything else
    pale beneath."""
    pre = valuable == 1
    post = valuable == 3
    figure = category_figure(points, np.where(pre, 1, 0), VALUABLE_COLOURS[:1],
                             VALUABLE_NAMES[:1], 'unlabelled')
    figure.add_trace(go.Scatter3d(
        x=points[post, 0], y=points[post, 1], z=points[post, 2],
        mode='markers',
        marker=dict(size=POINT_SIZE * 2, color=trial_numbers[post],
                    colorscale=TRIAL_NUMBER_SCALE,
                    colorbar=dict(title=dict(text='post discovery, trial since discovery', side='right'),
                                  outlinewidth=0, thickness=14, len=0.7)),
        name='post discovery', showlegend=False,
        hovertemplate='trial %{marker.color:.0f} since discovery<extra></extra>',
    ))
    # the legend moves left, out of the colourbar's way
    figure.update_layout(legend=dict(xanchor='left', x=0.02))
    return figure


def embeddings(binned):
    """(name, fit name, title, selected bins, figure maker) per alternate
    embedding. The fit name keys the cached UMAP, so a plot can be renamed or
    recoloured without refitting; the figure maker takes the embedding and the
    selected bins and returns the plot."""
    correctness = binned['correct_per_bin'].ravel()
    correct = np.isin(correctness, [1, 2])              # correct, rewarded or not
    incorrect = np.isin(correctness, [3, 4])            # incorrect, rewarded or not
    valuable = binned['valuable_per_bin'].ravel().astype(int)
    trial_numbers = binned['trial_since_discovery_per_bin'].ravel()
    trial_patches = binned['trial_patch_per_bin'].ravel()
    pre_post = lambda points, at: pre_post_figure(points, valuable[at], trial_numbers[at])
    return [
        ('correct_trial_since_discovery', 'correct_valuable_patch',     # the fit's first name
         'correct trials, pre discovery and post discovery by trial since discovery',
         correct, pre_post),
        ('incorrect_valuable_patch', 'incorrect_valuable_patch',
         'incorrect trials, by valuable patch', incorrect,
         lambda points, at: category_figure(points, valuable[at], VALUABLE_COLOURS,
                                            VALUABLE_NAMES,
                                            'post discovery exploration, without return')),
    ] + [
        (f'correct_patch_{patch}_trial_since_discovery', f'correct_patch_{patch}',
         f'correct trials at patch {patch}, pre discovery and post discovery by trial '
         'since discovery',
         correct & (trial_patches == patch), pre_post)
        for patch in range(1, 5)
    ]


def plot_session(session=SESSION, skip_created=False, bin_ms=100):
    """Fit, or load, each alternate embedding and write its plot."""
    binned = loadmat(binned_path(session, bin_ms))
    spikes = binned['spikes'].T                         # bins x units
    settings = (f'{int(binned["bin_size_ms"].item())} ms bins, '
                f'sigma {binned["sigma_bins"].item():g} bins')

    out_dir = Path('figures') / session / 'alternate_embeddings'
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fit_name, title, selected, make_figure in embeddings(binned):
        print(f'{session}: {name}, {selected.sum()} bins')
        if selected.sum() < MIN_BINS:
            print(f'  skipped, fewer than {MIN_BINS} bins')
            continue

        rows = spikes[selected]
        points = model_for(rows, fit_key(rows, fit_name)).embedding_
        figure = make_figure(points, selected)
        apply_layout(figure, f'{session}: UMAP of {title} ({settings})')
        save(figure, out_dir / f'umap_{name}_{bin_ms}ms.html', PNG_WIDTH, PNG_HEIGHT, skip_created)

    plot_value_units(session, binned, spikes, settings, out_dir, skip_created, bin_ms)


def plot_value_units(session, binned, spikes, settings, out_dir, skip_created, bin_ms):
    """Fit, or load, the UMAP of only the session's value selected units, if
    it has a CSV listing them, and plot it by specific valuable patch."""
    csv = value_units_csv(session)
    if csv is None:
        print(f'{session}: value_units, no value selected unit CSV, skipped')
        return
    if 'unit_ids' not in binned:
        print(f'{session}: value_units, the export has no unit_ids, rerun export_bins.m')
        return

    listed = pd.read_csv(csv, usecols=['UnitID'])['UnitID'].unique()
    unit_ids = binned['unit_ids'].ravel()
    selected = np.isin(unit_ids, listed)
    missing = np.setdiff1d(listed, unit_ids)
    print(f'{session}: value_units, {selected.sum()} of {len(listed)} units from {csv.name}')
    if len(missing):
        print(f'  not among the kept good and mua units: {missing.tolist()}')

    columns = spikes[:, selected]
    points = model_for(columns, fit_key(columns, 'value_units')).embedding_
    specific = binned['specific_per_bin'].ravel().astype(int)
    valuable = binned['valuable_per_bin'].ravel().astype(int)
    versions = {
        'specific_valuable_patch': (
            '', specific, SPECIFIC_COLOURS, SPECIFIC_NAMES,
            'between trials, or pre discovery not perseveration'),
        'specific_valuable_patch_post': (
            ', post discovery only', post_codes(specific, valuable), POST_COLOURS, POST_NAMES,
            'between trials, pre discovery, or exploration without return'),
    }
    for name, (scope, codes, colours, names, none_name) in versions.items():
        figure = category_figure(points, codes, colours, names, none_name)
        apply_layout(figure, f'{session}: UMAP of the {selected.sum()} value selected units, '
                             f'by specific valuable patch{scope}, whole trial ({settings})')
        save(figure, out_dir / f'umap_value_units_{name}_{bin_ms}ms.html',
             PNG_WIDTH, PNG_HEIGHT, skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot UMAPs fitted on subsets of a session.')
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave figures whose HTML and PNG both already exist untouched')
    add_bin_argument(parser)
    args = parser.parse_args()
    plot_session(args.session, args.skip_created, args.bin_ms)
