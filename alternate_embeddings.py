"""UMAPs of one session fitted on a subset of its bins, each with its own labelling.

Each alternate embedding is its own UMAP, fitted only on the bins its
selection keeps, with the same parameters as the full UMAP. Writes, into
figures/<session>/alternate_embeddings/:

    umap_correct_trial_number.html     fitted on every bin of the correct trials
                                       (rewarded or not). Pre discovery trials
                                       are solid orange, as in valuable patch;
                                       post discovery ones are coloured by their
                                       session trial number on a continuous
                                       scale. (Correct trials are never
                                       perseveration or exploration.)
    umap_incorrect_valuable_patch.html the same for the incorrect trials, where
                                       post discovery, correct patch never
                                       appears; exploration that never returns
                                       is drawn pale.

each with a PNG beside it. A selection with fewer than MIN_BINS bins is
skipped: below that UMAP's neighbourhoods stop meaning much.

    python alternate_embeddings.py                               # May 17
    python alternate_embeddings.py reduced_np_data_25_05_2026
    python alternate_embeddings.py reduced_np_data_25_05_2026 --skip_created

pipeline.py runs it for every session. Run from the project root, after
export_bins.m. Fits are cached in .cache/ like the full one, keyed on the
selected rows and the embedding's name.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import plotly.graph_objects as go

from umap_plots import (POINT_SIZE, PNG_HEIGHT, PNG_WIDTH, SESSION, TEXT_COLOUR,
                        UMAP_PARAMS, VALUABLE_COLOURS, VALUABLE_NAMES, apply_layout,
                        category_figure, fit_key, model_for, save)

MIN_BINS = 5 * UMAP_PARAMS['n_neighbors']       # as in v5
TRIAL_NUMBER_SCALE = 'Viridis'                  # early trials dark, late bright


def pre_post_figure(points, valuable, trial_numbers):
    """Pre discovery bins in solid orange, post discovery ones coloured by
    their trial number on a continuous scale, anything else pale beneath."""
    pre = valuable == 1
    post = valuable == 3
    figure = category_figure(points, np.where(pre, 1, 0), VALUABLE_COLOURS[:1],
                             VALUABLE_NAMES[:1], 'unlabelled')
    figure.add_trace(go.Scatter3d(
        x=points[post, 0], y=points[post, 1], z=points[post, 2],
        mode='markers',
        marker=dict(size=POINT_SIZE * 2, color=trial_numbers[post],
                    colorscale=TRIAL_NUMBER_SCALE,
                    colorbar=dict(title=dict(text='post discovery, trial number', side='right'),
                                  outlinewidth=0, thickness=14, len=0.7)),
        name='post discovery', showlegend=False,
        hovertemplate='trial %{marker.color:.0f}<extra></extra>',
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
    trial_numbers = binned['trial_number_per_bin'].ravel()
    return [
        ('correct_trial_number', 'correct_valuable_patch',     # the fit's first name
         'correct trials, pre discovery and post discovery by trial number', correct,
         lambda points, at: pre_post_figure(points, valuable[at], trial_numbers[at])),
        ('incorrect_valuable_patch', 'incorrect_valuable_patch',
         'incorrect trials, by valuable patch', incorrect,
         lambda points, at: category_figure(points, valuable[at], VALUABLE_COLOURS,
                                            VALUABLE_NAMES,
                                            'post discovery exploration, without return')),
    ]


def plot_session(session=SESSION, skip_created=False):
    """Fit, or load, each alternate embedding and write its plot."""
    binned = loadmat(Path('data') / session / 'binned.mat')
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
        save(figure, out_dir / f'umap_{name}.html', PNG_WIDTH, PNG_HEIGHT, skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot UMAPs fitted on subsets of a session.')
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave figures whose HTML and PNG both already exist untouched')
    args = parser.parse_args()
    plot_session(args.session, args.skip_created)
