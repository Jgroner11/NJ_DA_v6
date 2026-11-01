"""Where on the maze the UMAP embedding moves, for one session.

Loads the session's cached UMAP embedding, computes the Euclidean distance
from each bin's point to the next bin's, keeps the distances at or above
MIN_DISTANCE -- the big jumps -- and plots each kept bin at the mouse's head
position on the session's maze image, coloured by its distance. Bins with no
head position -- between trials or badly tracked -- are left out.

Writes figures/<session>/jump_analysis/distance_on_maze.html, with a PNG beside it.

    python distance_on_maze.py                               # May 17
    python distance_on_maze.py reduced_np_data_25_05_2026
    python distance_on_maze.py reduced_np_data_25_05_2026 --skip_created

pipeline.py runs it for every session. Run from the project root, after
export_bins.m (binned.mat must hold head_x and head_y) and with a cached fit
from umap_plots.py, or it fits one.
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.io import loadmat

from umap_plots import BACKGROUND, TEXT_COLOUR, fit_key, model_for, save

SESSION = 'reduced_np_data_17_05_2026'      # the default, as in umap_plots.py

# each session's blackout frame: one still from its camera, in the same pixel
# coordinates as the tracking
MAZE_PNGS = {
    'reduced_np_data_17_05_2026': 'NPC4_2026_05_17_17_10_50_blackout.png',
    'reduced_np_18_05_2025': 'NPC4_2026_05_18_18_11_03_blackout.png',
    'reduced_np_data_25_05_2026': 'NPC4_2026_05_25_15_42_57_blackout.png',
}

MIN_DISTANCE = 4             # only distances at or above this are plotted
COLOUR_SCALE = 'Plasma'      # dark blue-purple for the shortest kept jumps, bright yellow for the longest
POINT_SIZE = 8


def load_embedding(session=SESSION):
    """The session's UMAP embedding, one row per bin in bin order."""
    binned = loadmat(Path('data') / session / 'binned.mat', variable_names=['spikes'])
    spikes = binned['spikes'].T                         # bins x units
    return np.asarray(model_for(spikes, fit_key(spikes)).embedding_)


def load_head_positions(session=SESSION):
    """The head position of every bin, bins x 2 in maze pixels, NaN where unknown."""
    binned = loadmat(Path('data') / session / 'binned.mat', variable_names=['head_x', 'head_y'])
    return np.column_stack([binned['head_x'].ravel(), binned['head_y'].ravel()])


def compute_distances(embedding):
    """Distance from each bin's point to the next bin's: entry i is bin i to i + 1."""
    x1 = embedding[:-1]
    x2 = embedding[1:]

    return np.sum((x2-x1)**2, axis=1) ** (1/2)


def maze_figure(positions, distances, session=SESSION):
    """The maze image with a point per position, coloured by its distance."""
    # longest distances drawn last, so they sit on top of the many short ones
    order = np.argsort(distances)
    positions, distances = positions[order], distances[order]

    maze = np.asarray(Image.open(Path('data') / 'raw' / MAZE_PNGS[session]).convert('RGB'))
    height, width = maze.shape[:2]

    figure = go.Figure(go.Image(z=maze, hoverinfo='skip'))
    figure.add_trace(go.Scattergl(
        x=positions[:, 0], y=positions[:, 1],
        mode='markers',
        marker=dict(size=POINT_SIZE, color=distances, colorscale=COLOUR_SCALE,
                    cmin=MIN_DISTANCE, cmax=distances.max(),
                    colorbar=dict(title=dict(text='distance to next bin', side='right'),
                                  outlinewidth=0, thickness=14, len=0.7)),
        hovertemplate='%{marker.color:.3f}<extra></extra>',
    ))
    # go.Image puts the origin at the top left with y running down, as in the
    # video, so the tracking pixels land on the image without flipping
    figure.update_xaxes(visible=False, range=[0, width])
    figure.update_yaxes(visible=False, range=[height, 0], scaleanchor='x')
    figure.update_layout(
        title=dict(text=f'{session}: UMAP distance to next bin, at the head position '
                        f'({MIN_DISTANCE:g} and above)', x=0.02, xanchor='left'),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOUR),
    )
    return figure, width, height


def plot_session(session=SESSION, skip_created=False):
    """Compute the session's distances, filter them and write its maze plot."""
    distances = compute_distances(load_embedding(session))
    positions = load_head_positions(session)[:-1]       # bin i's position for distance i

    keep = (distances >= MIN_DISTANCE) & ~np.isnan(positions).any(axis=1)
    print(f'{session}: {keep.sum()} of {len(distances)} bins kept '
          f'({(distances < MIN_DISTANCE).sum()} below {MIN_DISTANCE:g}, '
          f'{np.isnan(positions).any(axis=1).sum()} without a head position)')

    figure, width, height = maze_figure(positions[keep], distances[keep], session)
    out = Path('figures') / session / 'jump_analysis' / 'distance_on_maze.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    save(figure, out, width + 120, height + 40, skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot one session's big UMAP jumps on the maze.")
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave the plot untouched if its HTML and PNG already exist')
    args = parser.parse_args()
    plot_session(args.session, args.skip_created)
