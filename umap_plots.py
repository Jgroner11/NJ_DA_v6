"""A 3D UMAP of every bin in one session, plotted under each labeling.

Reads data/<session>/binned.mat, written by export_bins.m, fits one UMAP on
every bin, and writes one interactive plot per labeling, each with a PNG of
its starting view beside it, into two folders under figures/<session>/:

    whole_trial/   the trial labels (reward cycle, correctness, valuable patch,
                   specific valuable patch) on every bin of the trial
    at_port/       the same labels only on the bins where the mouse is at the
                   trial's own port, every other bin pale

Session time, port and port patch location are the same in both folders. Each
folder holds:

    umap_session_time.html          coloured by time in session
    umap_port.html                  coloured by the port the mouse is at
    umap_port_patch_location.html   coloured by the patch of the port the mouse is at
                                    (ports 1-2, 3-4, 5-6, 7-8)
    umap_reward_cycle.html          coloured by each trial's reward: 30 ms, 15 ms,
                                    unrewarded right after a 15 ms trial at the
                                    same port, or other unrewarded
    umap_correctness.html           coloured by each trial's patch (correct or not)
                                    and whether it was rewarded
    umap_valuable_patch.html        coloured by pre discovery, perseveration, post
                                    discovery at the correct patch, or post
                                    discovery exploration that returns to it
    umap_specific_valuable_patch.html  post discovery trials by their block's
                                    correct patch, and perseveration trials by
                                    the previous block's, in a lighter tint
    umap_all.html                   every plot above as a subplot, each with its
                                    own legend

Run from the project root, after export_bins.m:

    python umap_plots.py
    python umap_plots.py reduced_np_data_25_05_2026

Fits are cached by content in .cache/, as in v5: the key covers the spike
matrix and every UMAP parameter, so a change to the binning or an edited
parameter refits, while a re-export with new labels reuses the cached model.
UMAP is not seeded, so the cache is also what keeps the embedding stable from
run to run -- delete .cache/umap_*.joblib to force a fresh fit.
"""

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots
from scipy.io import loadmat

SESSION = 'reduced_np_data_17_05_2026'      # the .mat file's stem, as export_bins.m names it
CACHE_DIR = Path('.cache')

UMAP_PARAMS = dict(n_neighbors=40, n_components=3, min_dist=0.1,
                   metric='correlation', n_jobs=-1)

POINT_SIZE = 1.5
BACKGROUND = '#fcfcfb'                      # the surface a figure sits on, as in v5
TEXT_COLOUR = '#52514e'
UNLABELLED = '#d8d7d2'                      # bins with no label, drawn pale underneath
PNG_WIDTH, PNG_HEIGHT, PNG_SCALE = 1000, 800, 2   # the saved PNG, in px before scaling
COMBINED_COLS = 4                                 # subplots per row in umap_all
COMBINED_WIDTH, COMBINED_HEIGHT = 2400, 1600      # its PNG, in px before scaling
LEGEND_STRIP = 0.3                                # share of each cell kept below the scene for its legend

# one colour per port, port 1 first, in dark/light pairs so the two ports of a
# rewarded pair share a hue -- v5's palette
PORT_COLOURS = ['#CC4054', '#FF8C80', '#3373CC', '#8CBFFF',
                '#409959', '#A6CC8C', '#8C66B3', '#E699D9']


def mean_colour(*colours):
    """The RGB mean of hex colours."""
    rgb = np.mean([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colours], axis=0)
    return '#' + ''.join(f'{round(v):02X}' for v in rgb)


# patch P is ports 2P-1 and 2P, coloured as the mean of their two colours
PATCH_COLOURS = [mean_colour(*PORT_COLOURS[i:i + 2]) for i in range(0, len(PORT_COLOURS), 2)]

# reward_per_bin codes 1-4, as export_bins.m defines them
REWARD_COLOURS = ['#1B7837', '#A6DBA0', '#F28E2B', '#C2A5CF']
REWARD_NAMES = ['rewarded 30 ms', 'rewarded 15 ms',
                'unrewarded after 15 ms at same port', 'other unrewarded']

# correct_per_bin codes 1-4, as export_bins.m defines them
CORRECT_COLOURS = ['#1B7837', '#A6DBA0', '#762A83', '#C2A5CF']
CORRECT_NAMES = ['rewarded, correct patch', 'unrewarded, correct patch',
                 'rewarded, incorrect patch', 'unrewarded, incorrect patch']

# valuable_per_bin codes 1-4, as export_bins.m defines them
VALUABLE_COLOURS = ['#F28E2B', '#FDC48A', '#33A02C', '#7B3294']
VALUABLE_NAMES = ['pre discovery', 'perseveration', 'post discovery, correct patch',
                  'post discovery exploration, with return']


def tint(colour, amount=0.5):
    """colour mixed with white, amount 0 leaving it as is and 1 giving white."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(v + (255 - v) * amount):02X}' for v in rgb)


# specific_per_bin codes, as export_bins.m defines them: 1-4 post discovery in
# a block whose correct patch is P, 5-8 perseveration at patch P in a tint
SPECIFIC_COLOURS = PATCH_COLOURS + [tint(c) for c in PATCH_COLOURS]
SPECIFIC_NAMES = ([f'patch {p}, post discovery' for p in range(1, 5)]
                  + [f'patch {p}, perseveration' for p in range(1, 5)])


def fit_key(spikes):
    """The cache key: the spike matrix's contents plus every UMAP parameter."""
    digest = hashlib.sha256(np.ascontiguousarray(spikes).tobytes()).hexdigest()
    source = digest + repr(spikes.shape) + 'full' + repr(sorted(UMAP_PARAMS.items()))
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def model_for(rows, key):
    """The UMAP fitted on these rows, reused from the cache if it holds one.

    The whole fitted object is cached, not just its coordinates, so a cached
    fit can still place new bins with UMAP.transform.
    """
    cache_path = CACHE_DIR / f'umap_{key}.joblib'

    if cache_path.exists():
        print(f'  reusing cached fit: {cache_path}')
        return joblib.load(cache_path)

    print(f'  no cached fit for these inputs, fitting UMAP on {rows.shape[0]} bins...')
    model = umap.UMAP(**UMAP_PARAMS).fit(rows)

    CACHE_DIR.mkdir(exist_ok=True)
    joblib.dump(model, cache_path)
    print(f'  cached fit: {cache_path}')
    return model


def session_time_figure(points, minutes):
    """The cloud coloured by when in the session each bin was recorded."""
    return go.Figure(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=POINT_SIZE, color=minutes, colorscale='Viridis',
                    colorbar=dict(title=dict(text='Time in session (min)', side='right'),
                                  outlinewidth=0, thickness=14, len=0.7)),
        hovertemplate='%{marker.color:.1f} min<extra></extra>',
    ))


def category_figure(points, codes, colours, names, none_name):
    """The cloud coloured by a category code per bin, 0 meaning none.

    Code c is drawn in colours[c - 1] and named names[c - 1]. One trace per
    code, for a legend entry each, with the bins at code 0 drawn pale
    underneath and the labelled bins larger on top.
    """
    figure = go.Figure()
    none = codes == 0
    figure.add_trace(go.Scatter3d(
        x=points[none, 0], y=points[none, 1], z=points[none, 2],
        mode='markers', marker=dict(size=POINT_SIZE, color=UNLABELLED, opacity=0.35),
        name=none_name, hoverinfo='skip',
    ))
    for code in range(1, len(colours) + 1):
        at = codes == code
        if not at.any():
            continue
        figure.add_trace(go.Scatter3d(
            x=points[at, 0], y=points[at, 1], z=points[at, 2],
            mode='markers', marker=dict(size=POINT_SIZE * 2, color=colours[code - 1]),
            name=names[code - 1], hoverinfo='skip',
        ))
    figure.update_layout(legend=dict(
        yanchor='top', y=0.98, xanchor='right', x=0.98, itemsizing='constant',
        font=dict(size=11, color=TEXT_COLOUR), bgcolor='rgba(252,252,251,0.7)'))
    return figure


def apply_layout(figure, title):
    """The layout every figure shares, as in v5: a plain off-white surface with
    the axes hidden, so no grid, grey panes, ticks or axis titles."""
    hidden = dict(visible=False, showbackground=False)
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor='left'),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=BACKGROUND,
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden),
    )
    return figure


def combined_figure(labelings, title):
    """Every labeling's figure as one subplot each, COMBINED_COLS to a row.

    Each subplot keeps its own legend (or colorbar), so a legend entry only
    toggles the points of its own subplot. The 3D scene takes the top of its
    cell and the legend sits in the strip below it, so it never covers points.
    """
    n_rows = -(-len(labelings) // COMBINED_COLS)
    specs = [[{'type': 'scene'} if r * COMBINED_COLS + c < len(labelings) else None
              for c in range(COMBINED_COLS)] for r in range(n_rows)]
    combined = make_subplots(rows=n_rows, cols=COMBINED_COLS, specs=specs,
                             subplot_titles=[label for label, _ in labelings.values()],
                             horizontal_spacing=0.01, vertical_spacing=0.04)

    hidden = dict(visible=False, showbackground=False)
    for i, (_, figure) in enumerate(labelings.values()):
        scene = 'scene' if i == 0 else f'scene{i + 1}'
        legend = 'legend' if i == 0 else f'legend{i + 1}'
        combined.add_traces(list(figure.data), rows=i // COMBINED_COLS + 1,
                            cols=i % COMBINED_COLS + 1)
        (x0, x1), (y0, y1) = combined.layout[scene].domain.x, combined.layout[scene].domain.y
        scene_y0 = y0 + LEGEND_STRIP * (y1 - y0)      # the scene's bottom edge, above the strip
        combined.layout[scene].update(domain=dict(x=[x0, x1], y=[scene_y0, y1]),
                                      xaxis=hidden, yaxis=hidden, zaxis=hidden)
        for trace in combined.data:
            if trace.scene == scene:
                trace.legend = legend
                if trace.marker.colorbar.title.text:     # the session time colourbar
                    trace.marker.colorbar.update(x=x1 - 0.02, y=(scene_y0 + y1) / 2,
                                                 len=0.7 * (y1 - scene_y0))
        combined.layout[legend] = dict(
            x=x0 + 0.01, y=scene_y0, xanchor='left', yanchor='top',
            itemsizing='constant', font=dict(size=10, color=TEXT_COLOUR),
            bgcolor='rgba(0,0,0,0)')

    combined.update_layout(
        title=dict(text=title, x=0.01, xanchor='left'),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor=BACKGROUND,
    )
    return combined


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else SESSION
    binned_path = Path('data') / session / 'binned.mat'
    if not binned_path.exists():
        raise SystemExit(f'{binned_path} not found, run export_bins.m first')
    binned = loadmat(binned_path)
    spikes = binned['spikes'].T                         # bins x units
    minutes = binned['bin_times_sec'].ravel() / 60
    ports = binned['port_per_bin'].ravel().astype(int)
    port_patches = (ports + 1) // 2                     # ports 1-2 -> patch 1, ..., 0 stays 0
    at_trial_port = binned['at_trial_port'].ravel().astype(bool)
    trial_labels = {name: binned[f'{name}_per_bin'].ravel().astype(int)
                    for name in ('reward', 'correct', 'valuable', 'specific')}
    settings = (f'{int(binned["bin_size_ms"].item())} ms bins, '
                f'sigma {binned["sigma_bins"].item():g} bins')

    print(f'{session}: {spikes.shape[0]} bins x {spikes.shape[1]} units')
    points = model_for(spikes, fit_key(spikes)).embedding_

    # Each version is written to its own folder. The trial labels are whole
    # trials in one and only the bins at the trial's own port in the other;
    # the plots not labelled by trial are the same in both.
    versions = {
        'whole_trial': (np.ones_like(at_trial_port), 'between trials', 'whole trial'),
        'at_port': (at_trial_port, "away from the trial's port", "at the trial's port"),
    }
    for folder, (keep, unlabelled, scope) in versions.items():
        rewards, correctness, valuable, specific = (
            np.where(keep, trial_labels[name], 0)
            for name in ('reward', 'correct', 'valuable', 'specific'))

        labelings = {
            'session_time': ('time in session', session_time_figure(points, minutes)),
            'port': ('port', category_figure(
                points, ports, PORT_COLOURS,
                [f'port {p}' for p in range(1, 9)], 'not at a port')),
            'port_patch_location': ('patch of port', category_figure(
                points, port_patches, PATCH_COLOURS,
                [f'patch {p}' for p in range(1, 5)], 'not at a port')),
            'reward_cycle': (f'reward cycle, {scope}', category_figure(
                points, rewards, REWARD_COLOURS, REWARD_NAMES, unlabelled)),
            'correctness': (f'correctness, {scope}', category_figure(
                points, correctness, CORRECT_COLOURS, CORRECT_NAMES, unlabelled)),
            'valuable_patch': (f'valuable patch, {scope}', category_figure(
                points, valuable, VALUABLE_COLOURS, VALUABLE_NAMES,
                f'{unlabelled}, or exploration without return')),
            'specific_valuable_patch': (f'specific valuable patch, {scope}', category_figure(
                points, specific, SPECIFIC_COLOURS, SPECIFIC_NAMES,
                f'{unlabelled}, or pre discovery not perseveration')),
        }

        out_dir = Path('figures') / session / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (label, figure) in labelings.items():
            apply_layout(figure, f'{session}: full UMAP by {label} ({settings})')
            out = out_dir / f'umap_{name}.html'
            figure.write_html(out)
            # the PNG is the plot's starting view, rendered by kaleido
            figure.write_image(out.with_suffix('.png'), width=PNG_WIDTH, height=PNG_HEIGHT,
                               scale=PNG_SCALE)
            print(f'wrote {out} and .png')

        combined = combined_figure(
            labelings, f'{session}: full UMAP by every labeling, trial labels {scope} ({settings})')
        out = out_dir / 'umap_all.html'
        combined.write_html(out)
        combined.write_image(out.with_suffix('.png'), width=COMBINED_WIDTH,
                             height=COMBINED_HEIGHT, scale=PNG_SCALE)
        print(f'wrote {out} and .png')


if __name__ == '__main__':
    main()
