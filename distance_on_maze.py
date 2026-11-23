"""Where on the maze the UMAP embedding moves, for one session.

Loads the session's cached UMAP embedding, computes the Euclidean distance
from each bin's point to the next bin's, keeps the distances at or above
MIN_DISTANCE -- the big jumps -- and plots each kept bin at the mouse's head
position on the session's maze image, coloured by its distance. Bins with no
head position -- between trials or badly tracked -- are left out.

Also histograms every bin's distance, with MIN_DISTANCE marked. Both HTML
plots have a slider stepping the threshold through THRESHOLDS; they open, and
their PNGs show, MIN_DISTANCE.

Writes figures/<session>/jump_analysis/distance_on_maze_<N>ms.html and
distance_histogram_<N>ms.html, each with a PNG beside it, N the bin size
--bin_ms picks (100 by default, or 10).

    python distance_on_maze.py                               # May 17
    python distance_on_maze.py reduced_np_data_25_05_2026
    python distance_on_maze.py reduced_np_data_25_05_2026 --skip_created

pipeline.py runs it for every session. Run from the project root, after
export_bins.m (binned_<N>ms.mat must hold head_x and head_y) and with a cached fit
from umap_plots.py, or it fits one.
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.io import loadmat

from umap_plots import (BACKGROUND, TEXT_COLOUR, add_bin_argument, binned_path, fit_key,
                        model_for, save)

SESSION = 'reduced_np_data_17_05_2026'      # the default, as in umap_plots.py

# each session's blackout frame: one still from its camera, in the same pixel
# coordinates as the tracking
MAZE_PNGS = {
    'reduced_np_data_17_05_2026': 'NPC4_2026_05_17_17_10_50_blackout.png',
    'reduced_np_18_05_2025': 'NPC4_2026_05_18_18_11_03_blackout.png',
    'reduced_np_data_25_05_2026': 'NPC4_2026_05_25_15_42_57_blackout.png',
}

MIN_DISTANCE = 2             # only distances at or above this are plotted
COLOUR_SCALE = 'Plasma'      # dark blue-purple for the shortest kept jumps, bright yellow for the longest
POINT_SIZE = 8

# the thresholds the HTML plots' slider steps through; each plot opens, and its
# PNG is drawn, at MIN_DISTANCE, which must be one of them
THRESHOLDS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
assert MIN_DISTANCE in THRESHOLDS, 'MIN_DISTANCE must be one of THRESHOLDS'

HISTOGRAM_BIN = 0.1          # width of the histogram's bars, in UMAP distance
HISTOGRAM_COLOUR = '#5b6c8f'


def frame_slider(names, active, prefix):
    """A slider whose steps show the figure's frames, one per name, with no animation."""
    return dict(
        active=active, currentvalue=dict(prefix=prefix), pad=dict(t=40), x=0.02, len=0.9,
        steps=[dict(label=name, method='animate',
                    args=[[name], dict(mode='immediate', frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
               for name in names],
    )


def load_embedding(session=SESSION, bin_ms=100):
    """The session's UMAP embedding, one row per bin in bin order."""
    binned = loadmat(binned_path(session, bin_ms), variable_names=['spikes'])
    spikes = binned['spikes'].T                         # bins x units
    return np.asarray(model_for(spikes, fit_key(spikes)).embedding_)


def load_head_positions(session=SESSION, bin_ms=100):
    """The head position of every bin, bins x 2 in maze pixels, NaN where unknown."""
    binned = loadmat(binned_path(session, bin_ms), variable_names=['head_x', 'head_y'])
    return np.column_stack([binned['head_x'].ravel(), binned['head_y'].ravel()])


def compute_distances(embedding):
    """Distance from each bin's point to the next bin's: entry i is bin i to i + 1."""
    x1 = embedding[:-1]
    x2 = embedding[1:]

    return np.sum((x2-x1)**2, axis=1) ** (1/2)


def maze_figure(positions, distances, session=SESSION, bin_ms=100):
    """The maze image with a point per position at or above the threshold,
    coloured by its distance, with a slider over THRESHOLDS."""
    # longest distances drawn last, so they sit on top of the many short ones
    order = np.argsort(distances)
    positions, distances = positions[order], distances[order]

    maze = np.asarray(Image.open(Path('data') / 'raw' / MAZE_PNGS[session]).convert('RGB'))
    height, width = maze.shape[:2]

    def points(threshold):
        """The scatter's data and the title for one threshold."""
        kept = distances >= threshold
        scatter = go.Scatter(
            x=positions[kept, 0], y=positions[kept, 1],
            marker=dict(size=POINT_SIZE, color=distances[kept], colorscale=COLOUR_SCALE,
                        cmin=threshold, cmax=distances.max(),
                        colorbar=dict(title=dict(text='distance to next bin', side='right'),
                                      outlinewidth=0, thickness=14, len=0.7)),
        )
        title = (f'{session}: UMAP distance to next bin, at the head position '
                 f'({threshold:g} and above, {kept.sum()} bins of {bin_ms} ms)')
        return scatter, dict(text=title, x=0.02, xanchor='left')

    names = [f'{threshold:g}' for threshold in THRESHOLDS]
    scatter, title = points(MIN_DISTANCE)
    scatter.update(mode='markers', hovertemplate='%{marker.color:.3f}<extra></extra>')

    figure = go.Figure([go.Image(z=maze, hoverinfo='skip'), scatter])
    figure.frames = [go.Frame(name=name, data=[step_scatter], traces=[1], layout=dict(title=step_title))
                     for name, (step_scatter, step_title) in
                     zip(names, (points(threshold) for threshold in THRESHOLDS))]

    # go.Image puts the origin at the top left with y running down, as in the
    # video, so the tracking pixels land on the image without flipping
    figure.update_xaxes(visible=False, range=[0, width])
    figure.update_yaxes(visible=False, range=[height, 0], scaleanchor='x')
    figure.update_layout(
        title=title,
        sliders=[frame_slider(names, THRESHOLDS.index(MIN_DISTANCE), 'jump threshold: ')],
        margin=dict(l=0, r=0, t=40, b=110),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOUR),
    )
    return figure, width, height


def histogram_figure(distances, session=SESSION, bin_ms=100):
    """Every bin's distance to the next, binned, on a log count axis, with the jump threshold marked."""
    figure = go.Figure(go.Histogram(
        x=distances, xbins=dict(start=0, size=HISTOGRAM_BIN),
        marker=dict(color=HISTOGRAM_COLOUR, line=dict(width=0)),
        hovertemplate='%{x}: %{y} bins<extra></extra>',
    ))
    def label(threshold):
        return (f'jump threshold {threshold:g}: {(distances >= threshold).sum()} of '
                f'{len(distances)} bins at or above')

    # the threshold line and its label, moved by the slider
    figure.add_shape(type='line', x0=MIN_DISTANCE, x1=MIN_DISTANCE, y0=0, y1=1, yref='paper',
                     line=dict(color=TEXT_COLOUR, dash='dash', width=1))
    figure.add_annotation(x=MIN_DISTANCE, y=1, yref='paper', text=label(MIN_DISTANCE),
                          xanchor='left', yanchor='bottom', showarrow=False,
                          font=dict(color=TEXT_COLOUR))
    steps = [dict(label=f'{threshold:g}', method='relayout',
                  args=[{'shapes[0].x0': threshold, 'shapes[0].x1': threshold,
                         'annotations[0].x': threshold, 'annotations[0].text': label(threshold)}])
             for threshold in THRESHOLDS]
    figure.update_layout(sliders=[dict(active=THRESHOLDS.index(MIN_DISTANCE), steps=steps,
                                       currentvalue=dict(prefix='jump threshold: '),
                                       pad=dict(t=60), x=0.02, len=0.9)])
    # most distances are short and the jumps are a thin tail, so the counts are logged
    figure.update_yaxes(type='log', title='bins')
    figure.update_xaxes(title='UMAP distance to next bin')
    figure.update_layout(
        title=dict(text=f'{session}: UMAP distance to next bin, {bin_ms} ms bins '
                        f'({HISTOGRAM_BIN:g} wide bars)',
                   x=0.02, xanchor='left'),
        bargap=0, showlegend=False,
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOUR),
    )
    return figure


def plot_session(session=SESSION, skip_created=False, bin_ms=100):
    """Compute the session's distances, filter them and write its maze plot and histogram."""
    distances = compute_distances(load_embedding(session, bin_ms))
    out_dir = Path('figures') / session / 'jump_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    save(histogram_figure(distances, session, bin_ms), out_dir / f'distance_histogram_{bin_ms}ms.html',
         900, 600, skip_created)

    positions = load_head_positions(session, bin_ms)[:-1]       # bin i's position for distance i
    has_position = ~np.isnan(positions).any(axis=1)

    keep = (distances >= MIN_DISTANCE) & has_position
    print(f'{session}, {bin_ms} ms: {keep.sum()} of {len(distances)} bins kept '
          f'({(distances < MIN_DISTANCE).sum()} below {MIN_DISTANCE:g}, '
          f'{(~has_position).sum()} without a head position)')

    # the slider's lowest threshold is the most any step shows
    shown = (distances >= min(THRESHOLDS)) & has_position
    figure, width, height = maze_figure(positions[shown], distances[shown], session, bin_ms)
    save(figure, out_dir / f'distance_on_maze_{bin_ms}ms.html', width + 120, height + 150, skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot one session's big UMAP jumps on the maze.")
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave the plot untouched if its HTML and PNG already exist')
    add_bin_argument(parser)
    args = parser.parse_args()
    plot_session(args.session, args.skip_created, args.bin_ms)
