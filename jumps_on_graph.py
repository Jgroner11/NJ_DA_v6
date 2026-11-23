"""How many big UMAP jumps happened at each node and edge of the maze graph, for one session.

A jump is a bin whose UMAP point lies MIN_DISTANCE or more from the next bin's,
as in distance_on_maze.py. Each bin is placed at a node or edge by its time:
bin_times_sec and the trial timing in binned_<N>ms.mat give its trial and its frame
in the trial (at the tracking's frame rate), and DataJoint's
VideoGraphLocation gives the node or edge the mouse was at in that frame. The
jumps are counted per node and edge and each one's area of the graph mask
(built as in graph_occupancy.py) is painted with its count; visited nodes and
edges with no jumps are painted at 0. A second figure divides each count by
the time spent there, measured as the bins placed there times the bin length.
Both HTML plots have a slider stepping the jump threshold through
distance_on_maze.py's THRESHOLDS; they open, and their PNGs show, MIN_DISTANCE.

Writes figures/<session>/jump_analysis/jumps_at_location_<N>ms.html and
jump_rate_at_location_<N>ms.html, each with a PNG beside it, N the bin size
--bin_ms picks (100 by default, or 10).

    python jumps_on_graph.py                               # May 17
    python jumps_on_graph.py reduced_np_data_25_05_2026

Run from the project root, after export_bins.m (binned_<N>ms.mat must hold
trial_number) and with a cached fit from umap_plots.py, or it fits one.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from distance_on_maze import MIN_DISTANCE, THRESHOLDS, compute_distances, load_embedding
from graph_occupancy import MASK_RADIUS, MIN_FRAMES, mask_figure, session_mask, visit_frames
from umap_plots import add_bin_argument, binned_path, save

SESSION = 'reduced_np_data_17_05_2026'      # the default, as in umap_plots.py
COLOUR_SCALE = 'Plasma'                      # as in distance_on_maze.py


def bin_locations(session, visits, frame_rate, bin_ms=100):
    """Each bin's location type and label, NaN for bins outside a trial or a visit."""
    binned = loadmat(binned_path(session, bin_ms),
                     variable_names=['bin_times_sec', 'trial_number',
                                     'trial_start_sec', 'trial_duration_sec'])
    times = binned['bin_times_sec'].ravel()
    numbers = binned['trial_number'].ravel()
    starts = binned['trial_start_sec'].ravel()
    durations = binned['trial_duration_sec'].ravel()

    # the trial each bin falls in: the last one that started before it, if it
    # had not yet ended
    trial_index = np.searchsorted(starts, times, side='right') - 1
    safe_index = np.clip(trial_index, 0, len(starts) - 1)
    in_trial = (trial_index >= 0) & (times < starts[safe_index] + durations[safe_index])

    # the bin's frame in that trial: frame f of a trial is f / frame_rate after its start
    bins = pd.DataFrame({
        'bin': np.arange(len(times)),
        'trial': np.where(in_trial, numbers[safe_index], -1),
        'frame_idx_in_trial': np.floor((times - starts[safe_index]) * frame_rate).astype(int),
    })

    # the location of that frame; a left merge keeps every bin, in bin order
    located = bins.merge(visit_frames(visits), on=['trial', 'frame_idx_in_trial'], how='left')
    return located.sort_values('bin').reset_index(drop=True)


def plot_session(session=SESSION, skip_created=False, bin_ms=100):
    """Count the session's jumps at each node and edge and write its maze plot."""
    maze, mask, locations, visits, frame_rate = session_mask(session)
    height, width = maze.shape[:2]

    distances = compute_distances(load_embedding(session, bin_ms))
    located = bin_locations(session, visits, frame_rate, bin_ms).iloc[:-1]    # bin i for distance i
    index = pd.MultiIndex.from_tuples(locations)

    def jump_counts(threshold):
        """Jumps at or above threshold at every node and edge with area in the
        mask, zero where none happened there."""
        jumps = located[distances >= threshold].groupby(['location_type', 'location_label']).size()
        return pd.Series([jumps.get(location, 0) for location in locations], index=index)

    is_jump = distances >= MIN_DISTANCE
    kind = located['location_type']
    print(f'{session}, {bin_ms} ms: {is_jump.sum()} jumps of {len(distances)} bins at {MIN_DISTANCE:g}; '
          f'{(is_jump & (kind == "node")).sum()} at nodes, '
          f'{(is_jump & (kind == "edge")).sum()} on edges, '
          f'{(is_jump & kind.isna()).sum()} outside a trial or visit; '
          f'{(jump_counts(MIN_DISTANCE) > 0).sum()} of {len(locations)} nodes and edges have one or more')

    # the time at each location is the bins placed there times the bin length,
    # so jumps and time come from the same bins
    bin_seconds = loadmat(binned_path(session, bin_ms),
                          variable_names=['bin_size_ms'])['bin_size_ms'].item() / 1000
    bins_at = located.groupby(['location_type', 'location_label']).size()
    seconds = pd.Series([bins_at.get(location, 0) * bin_seconds for location in locations], index=index)

    # one slider step per threshold, for the counts and for the counts per second
    count_steps, rate_steps = [], []
    for threshold in THRESHOLDS:
        counts = jump_counts(threshold)
        name = f'{threshold:g}'
        count_steps.append((name, f'{session}: UMAP jumps ({bin_ms} ms bins, distance to next bin {threshold:g} and '
                                  f'above, {counts.sum()} in all) at each graph node and edge, over '
                                  f'the area the head covered there ({MASK_RADIUS} px discs, at '
                                  f'least {MIN_FRAMES} frames)', counts))
        rate_steps.append((name, f'{session}: UMAP jumps per second spent at each graph node and '
                                 f'edge (distance to next bin {threshold:g} and above, '
                                 f'{bin_seconds * 1000:g} ms bins)',
                           (counts / seconds)[seconds > 0]))

    active = THRESHOLDS.index(MIN_DISTANCE)
    out_dir = Path('figures') / session / 'jump_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    figure, slider_height = mask_figure(maze, mask, locations, count_steps, 'jumps', '%{z:.0f} jumps',
                                        colour_scale=COLOUR_SCALE, slider_prefix='jump threshold: ',
                                        active=active)
    save(figure, out_dir / f'jumps_at_location_{bin_ms}ms.html', width + 120, height + 40 + slider_height,
         skip_created)

    figure, slider_height = mask_figure(maze, mask, locations, rate_steps, 'jumps per second',
                                        '%{z:.2f} jumps/s', colour_scale=COLOUR_SCALE,
                                        slider_prefix='jump threshold: ', active=active)
    save(figure, out_dir / f'jump_rate_at_location_{bin_ms}ms.html', width + 120, height + 40 + slider_height,
         skip_created)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot one session's UMAP jumps at each maze graph node and edge.")
    parser.add_argument('session', nargs='?', default=SESSION,
                        help="the session's .mat file stem, as export_bins.m names it")
    parser.add_argument('--skip_created', action='store_true',
                        help='leave the plot untouched if its HTML and PNG already exist')
    add_bin_argument(parser)
    args = parser.parse_args()
    plot_session(args.session, args.skip_created, args.bin_ms)
