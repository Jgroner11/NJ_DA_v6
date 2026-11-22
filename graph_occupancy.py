"""How long the mouse spent at each node and edge of the maze graph, for one session.

NaviGraph splits the maze into a graph -- two binary trees of depth 6 joined at
their roots, 126 nodes and 125 edges -- and DataJoint's VideoGraphLocation
holds every visit to a node or edge: its trial, its first and last frame in
the trial, and its length in frames. This sums those visits over the session's
selected trials and paints each node's and edge's area of the maze image with
the seconds spent there, on a log scale.

DataJoint stores no node positions, so the areas are measured into a mask:
every frame of a visit is labelled with its node or edge and joined to the
headstage position DataJoint holds for that frame (camera pixels, the same
frame as the maze image). Each frame stamps a disc of MASK_RADIUS pixels around
its position into its label's count; each pixel goes to the label whose discs
cover it most, if at least MIN_FRAMES do, and each label keeps only its largest
connected patch, dropping strays from mislabelled or badly tracked frames. The
mask covers where the head went, so it can stop short of the walls, and nodes
and edges never visited have no frames and so no area.

The three DataJoint tables used -- the session's tracked trials (for the frame
rate), its visits and its headstage trajectory -- are fetched once and cached
in .cache/navigraph_<session>_<table>.pkl; later runs read the cache and do
not connect. Delete those files to fetch afresh.

Writes figures/<session>/graph/time_at_location.html, with a PNG beside it.

    python graph_occupancy.py                               # May 17
    python graph_occupancy.py reduced_np_data_25_05_2026

Run from the project root, after export_bins.m (binned.mat must hold trial_number).
"""

import argparse
import functools
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.colors import sample_colorscale, unlabel_rgb
from scipy import ndimage
from scipy.io import loadmat

from distance_on_maze import MAZE_PNGS
from umap_plots import BACKGROUND, TEXT_COLOUR, save

SESSION = 'reduced_np_data_17_05_2026'      # the default, as in umap_plots.py

# each session's key in DataJoint; all three are mouse 108104
DJ_KEYS = {
    'reduced_np_data_17_05_2026': {'subject_id': 108104, 'session': 26},
    'reduced_np_18_05_2025': {'subject_id': 108104, 'session': 27},
    'reduced_np_data_25_05_2026': {'subject_id': 108104, 'session': 31},
}

BODYPART = 'headstage'       # the part export_bins.m takes head position from
MIN_LIKELIHOOD = 0.9         # as in export_bins.m

MASK_RADIUS = 6              # px, the disc each frame stamps; corridors are ~40 px wide
MIN_FRAMES = 3               # a pixel needs this many frames' discs over it to be assigned

COLOUR_SCALE = 'Viridis'
MASK_OPACITY = 0.8           # how strongly the colour covers the maze image

# DataJoint fetches are cached here, one pickle per session and table, as
# umap_plots.py caches its fits; delete .cache/navigraph_* to fetch afresh
CACHE_DIR = Path('.cache')


@functools.cache
def tracking_schema():
    """The tracking schema, connected on first use so cached runs need no database."""
    return dj.VirtualModule('tracking', 'noamjoshua_TRACKING')


def cached_fetch(session, name, fetch):
    """The table fetch() returns, read from the cache if it holds one.

    What is cached is the whole session as DataJoint returns it, before any
    filtering, so changing the selected trials or MIN_LIKELIHOOD does not make
    it stale.
    """
    path = CACHE_DIR / f'navigraph_{session}_{name}.pkl'
    if path.exists():
        return pd.read_pickle(path)

    print(f'  fetching {name} from DataJoint...')
    table = fetch()
    CACHE_DIR.mkdir(exist_ok=True)
    table.to_pickle(path)
    print(f'  cached: {path}')
    return table


def frame_rate_for(session):
    """The tracking's frames per second for the session."""
    trials = cached_fetch(session, 'tracking_trials', lambda: (
        tracking_schema().TrackingTrial & DJ_KEYS[session]
    ).proj('tracking_sampling_rate').fetch(format='frame').reset_index())
    return trials['tracking_sampling_rate'].iloc[0]


def selected_trials(session):
    """The trial numbers export_bins.m kept for the session."""
    binned = loadmat(Path('data') / session / 'binned.mat', variable_names=['trial_number'])
    return binned['trial_number'].ravel()


def load_visits(session):
    """One row per visit to a node or edge in the session's selected trials."""
    visits = cached_fetch(session, 'visits', lambda: (
        tracking_schema().VideoGraphLocation & DJ_KEYS[session]
    ).proj('trial', 'location_type', 'location_label', 'start_frame_in_trial',
           'end_frame_in_trial', 'duration_frames').fetch(format='frame').reset_index())
    return visits[visits['trial'].isin(selected_trials(session))]


def time_at_locations(visits, frame_rate):
    """Seconds spent at each node and edge, indexed by location type and label."""
    frames = visits.groupby(['location_type', 'location_label'])['duration_frames'].sum()
    return frames / frame_rate


def visit_frames(visits):
    """One row per frame of each visit: its trial, its frame in the trial and its location."""
    # a visit's end frame is its last frame, so the range includes it
    lengths = (visits['end_frame_in_trial'] - visits['start_frame_in_trial'] + 1).to_numpy()
    starts = np.repeat(visits['start_frame_in_trial'].to_numpy(), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    frames = pd.DataFrame({
        'trial': np.repeat(visits['trial'].to_numpy(), lengths),
        'frame_idx_in_trial': starts + offsets,
        'location_type': np.repeat(visits['location_type'].to_numpy(), lengths),
        'location_label': np.repeat(visits['location_label'].to_numpy(), lengths),
    })

    # a few trials have two visits claiming their first frame or two (22 frames
    # on May 18), with different locations; with no way to tell which is right,
    # those frames are dropped, so every frame has at most one location
    return frames.drop_duplicates(['trial', 'frame_idx_in_trial'], keep=False)


def labelled_positions(visits, session):
    """Every well-tracked frame of every visit: its location and headstage position in camera pixels."""
    positions = cached_fetch(session, f'{BODYPART}_trajectory', lambda: (
        tracking_schema().VideoBodypartTrajectTrialCorrected & DJ_KEYS[session]
        & {'bodypart_name': BODYPART}
    ).proj('traj_x', 'traj_y', 'traj_likelihood').fetch(format='frame').reset_index())
    positions = positions[positions['traj_likelihood'] >= MIN_LIKELIHOOD]
    positions = positions.astype({'traj_x': float, 'traj_y': float})

    # each labelled frame gets the position tracked at that trial and frame
    return visit_frames(visits).merge(positions, on=['trial', 'frame_idx_in_trial'])


def session_mask(session):
    """Everything needed to paint the session's maze: the image, the mask, its
    locations, each location's median head position, and the visits and frame rate."""
    frame_rate = frame_rate_for(session)

    visits = load_visits(session)
    frames = labelled_positions(visits, session)
    centres = frames.groupby(['location_type', 'location_label'])[['traj_x', 'traj_y']].median()

    maze = np.asarray(Image.open(Path('data') / 'raw' / MAZE_PNGS[session]).convert('RGB'))
    height, width = maze.shape[:2]
    mask, locations = build_mask(frames, width, height)
    return maze, mask, locations, centres, visits, frame_rate


def build_mask(frames, width, height):
    """Assign the maze image's pixels to locations from where their frames were.

    Returns the mask -- height x width, each pixel the index into locations of
    the node or edge it belongs to, or -1 for none -- and locations, the
    (location_type, location_label) pairs in index order.
    """
    disc_x, disc_y = np.meshgrid(np.arange(-MASK_RADIUS, MASK_RADIUS + 1),
                                 np.arange(-MASK_RADIUS, MASK_RADIUS + 1))
    disc = (disc_x ** 2 + disc_y ** 2 <= MASK_RADIUS ** 2).astype(np.float32)

    # one location at a time, keeping per pixel the most frames any location has
    # put there so far and which location that was -- a full count image for
    # every location at once would need gigabytes
    best_count = np.zeros((height, width), np.float32)
    mask = np.full((height, width), -1, np.int32)
    locations = []
    for index, (location, group) in enumerate(frames.groupby(['location_type', 'location_label'])):
        locations.append(location)
        x = np.clip(np.round(group['traj_x'].to_numpy()).astype(int), 0, width - 1)
        y = np.clip(np.round(group['traj_y'].to_numpy()).astype(int), 0, height - 1)

        # only the box around this location's frames, padded by the disc, is counted
        left, right = max(x.min() - MASK_RADIUS, 0), min(x.max() + MASK_RADIUS + 1, width)
        top, bottom = max(y.min() - MASK_RADIUS, 0), min(y.max() + MASK_RADIUS + 1, height)
        hits = np.zeros((bottom - top, right - left), np.float32)
        np.add.at(hits, (y - top, x - left), 1)
        # every frame's disc: the count at a pixel is how many frames lie within the radius
        count = ndimage.convolve(hits, disc, mode='constant')

        box = (slice(top, bottom), slice(left, right))
        wins = count > best_count[box]
        best_count[box][wins] = count[wins]
        mask[box][wins] = index

    mask[best_count < MIN_FRAMES] = -1

    # keep each location's largest connected patch; the rest are strays
    for index in range(len(locations)):
        patches, n_patches = ndimage.label(mask == index)
        if n_patches > 1:
            sizes = ndimage.sum_labels(np.ones_like(patches), patches, range(1, n_patches + 1))
            mask[(patches > 0) & (patches != np.argmax(sizes) + 1)] = -1
    return mask, locations


def mask_figure(maze, mask, locations, values, centres, title, colourbar_title,
                hover_format, log_scale=False, colour_scale=COLOUR_SCALE):
    """The maze image with each location's area of the mask coloured by its value.

    values is a Series indexed by (location_type, location_label); locations
    missing from it are left unpainted. hover_format formats one value for the
    hover text, e.g. '{:.1f} s'.
    """
    height, width = maze.shape[:2]
    painted_locations = [location for location in locations if location in values.index]
    raw = np.array([values[location] for location in painted_locations], float)
    shown = np.log10(raw) if log_scale else raw
    low, high = shown.min(), shown.max()
    if high == low:
        high = low + 1

    # each location's colour, then painted over its pixels of the maze image;
    # locations without a value get index -1 and stay as the image
    colours = np.array([unlabel_rgb(c) for c in
                        sample_colorscale(colour_scale, list((shown - low) / (high - low)))])
    colour_index = np.array([painted_locations.index(location) if location in values.index else -1
                             for location in locations])
    pixel_colour = np.where(mask >= 0, colour_index[mask], -1)
    painted = maze.astype(float)
    covered = pixel_colour >= 0
    painted[covered] = ((1 - MASK_OPACITY) * painted[covered]
                        + MASK_OPACITY * colours[pixel_colour[covered]])

    figure = go.Figure(go.Image(z=painted.astype(np.uint8), hoverinfo='skip'))

    # invisible points at each location's median position, carrying its hover
    # text and, through their colours, the colour bar
    colourbar = dict(title=dict(text=colourbar_title, side='right'),
                     outlinewidth=0, thickness=14, len=0.7)
    if log_scale:
        ticks = [t for t in [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
                 if low <= np.log10(t) <= high]
        colourbar.update(tickvals=np.log10(ticks), ticktext=[f'{t:g}' for t in ticks])
    figure.add_trace(go.Scattergl(
        x=[centres.loc[location, 'traj_x'] for location in painted_locations],
        y=[centres.loc[location, 'traj_y'] for location in painted_locations],
        mode='markers', showlegend=False,
        marker=dict(size=12, opacity=0, color=shown, colorscale=colour_scale,
                    cmin=low, cmax=high, colorbar=colourbar),
        text=[f'{kind} {label}: {hover_format.format(value)}'
              for (kind, label), value in zip(painted_locations, raw)],
        hovertemplate='%{text}<extra></extra>',
    ))

    # go.Image puts the origin at the top left with y running down, as in the
    # video, so the tracking pixels land on the image without flipping
    figure.update_xaxes(visible=False, range=[0, width])
    figure.update_yaxes(visible=False, range=[height, 0], scaleanchor='x')
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor='left'),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOUR),
    )
    return figure


def plot_session(session=SESSION, skip_created=False):
    """Measure the session's time and area at each location and write its maze plot."""
    maze, mask, locations, centres, visits, frame_rate = session_mask(session)
    height, width = maze.shape[:2]
    seconds = time_at_locations(visits, frame_rate)

    kinds = pd.Series([kind for kind, _ in locations])
    in_mask = np.unique(mask[mask >= 0])
    print(f'{session}: {(kinds == "node").sum()} nodes and {(kinds == "edge").sum()} edges with '
          f'frames, {len(in_mask)} with area in the mask; '
          f'{seconds["node"].sum():.0f} s at nodes, {seconds["edge"].sum():.0f} s on edges')

    title = (f'{session}: time at each graph node and edge, over the area the head '
             f'covered there ({MASK_RADIUS} px discs, at least {MIN_FRAMES} frames)')
    figure = mask_figure(maze, mask, locations, seconds, centres, title,
                         'seconds spent', '{:.1f} s', log_scale=True)
    out = Path('figures') / session / 'graph' / 'time_at_location.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    save(figure, out, width + 120, height + 40, skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot one session's time at each maze graph node and edge.")
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave the plot untouched if its HTML and PNG already exist')
    args = parser.parse_args()
    plot_session(args.session, args.skip_created)
