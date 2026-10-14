from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .palette import distinct_palette, natural_order


def sample_indices(n, maximum=300_000, seed=42):
    rng = np.random.default_rng(seed)
    if maximum is None or n <= maximum:
        return rng.permutation(n)
    return rng.choice(n, int(maximum), replace=False)


def finish(fig, path, dpi=300):
    import matplotlib.pyplot as plt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def histogram(values, title, path, label=None, log1p=False, dpi=300):
    import matplotlib.pyplot as plt
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if log1p:
        x = np.log1p(x)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(x, bins=70)
    ax.set(xlabel=(f'ln(1 + {label or title})' if log1p else label or title),
           ylabel='Number of observations', title=title)
    finish(fig, path, dpi)


def scatter_numeric(xy, values, title, path, *, thumb=None, image_size=None,
                    coordinate_labels=('x', 'y'), max_points=300_000, dpi=300, log1p=False):
    import matplotlib.pyplot as plt
    xy = np.asarray(xy)
    values = np.asarray(values, dtype=float)
    idx = sample_indices(len(xy), max_points)
    fig, ax = plt.subplots(figsize=(10, 8))
    if thumb is not None:
        w, h = image_size
        ax.imshow(thumb, extent=(-0.5, w-0.5, h-0.5, -0.5))
    c = np.log1p(values[idx]) if log1p else values[idx]
    sc = ax.scatter(xy[idx, 0], xy[idx, 1], c=c, s=0.6, rasterized=True, linewidths=0)
    fig.colorbar(sc, ax=ax, label='ln(1 + value)' if log1p else 'value')
    ax.set(title=f'{title}\n{len(idx):,} / {len(xy):,} observations drawn',
           xlabel=coordinate_labels[0], ylabel=coordinate_labels[1])
    if thumb is not None:
        ax.set_xlim(-.5, image_size[0]-.5); ax.set_ylim(image_size[1]-.5, -.5)
        ax.set_aspect('equal')
    finish(fig, path, dpi)


def scatter_categorical(xy, values, title, path, *, colors=None, thumb=None, image_size=None,
                        max_points=300_000, dpi=300, point_size=0.8,
                        labels=('x', 'y'), allow_extension=True):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    xy = np.asarray(xy)
    vals = pd.Series(np.asarray(values)).astype('string').fillna('missing').astype(str)
    categories = natural_order(vals.unique())
    colors = colors or dict(zip(categories, distinct_palette(len(categories), allow_extension)))
    if set(categories)-set(colors):
        raise ValueError('Palette does not cover observed categories.')
    idx = sample_indices(len(xy), max_points)
    fig, ax = plt.subplots(figsize=(11, 8))
    if thumb is not None:
        w, h = image_size
        ax.imshow(thumb, extent=(-.5, w-.5, h-.5, -.5))
    ax.scatter(xy[idx, 0], xy[idx, 1], c=vals.iloc[idx].map(colors).tolist(),
               s=point_size, linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', linestyle='', color=colors[k], label=k, markersize=6)
               for k in categories]
    if len(categories) <= 60:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=12, ncol=max(1, int(np.ceil(len(categories)/30))))
    else:
        ax.text(1.02, 1, f'{len(categories)} categories\nFull legend omitted to keep the plot readable.',
                transform=ax.transAxes, va='top', fontsize=11)
    ax.set(title=f'{title}\n{len(idx):,} / {len(xy):,} observations drawn',
           xlabel=labels[0], ylabel=labels[1])
    ax.tick_params(labelsize=12)
    if thumb is not None:
        ax.set_xlim(-.5, image_size[0]-.5); ax.set_ylim(image_size[1]-.5, -.5)
        ax.set_aspect('equal')
    finish(fig, path, dpi)
    return colors


def standard_qc_plots(obs, output_dir, prefix, dpi=300):
    output_dir = Path(output_dir)
    for key in ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo',
                'cell_area_um2', 'counts_per_um2', 'he_mean_od', 'geometric_qc_keep_fraction']:
        if key in obs:
            histogram(obs[key], f'{prefix}: {key}', output_dir/f'{prefix}_{key}_hist.png',
                      label=key, log1p=key in ['total_counts', 'n_genes_by_counts', 'cell_area_um2', 'counts_per_um2'], dpi=dpi)
    if {'total_counts', 'n_genes_by_counts'} <= set(obs):
        scatter_numeric(np.log1p(obs[['total_counts', 'n_genes_by_counts']].to_numpy()),
                        obs.get('pct_counts_mt', np.zeros(len(obs))), f'{prefix}: counts vs genes; color=%mt',
                        output_dir/f'{prefix}_counts_vs_genes.png',
                        coordinate_labels=('ln(1 + UMI counts)', 'ln(1 + detected genes)'), dpi=dpi)
    keys = [k for k in obs if pd.api.types.is_numeric_dtype(obs[k])]
    obs[keys].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_csv(output_dir/f'{prefix}_qc_summary.csv')


def boundary_overlay(polygons_image, info, thumb, path, bounds=None, dpi=300, max_polygons=75_000):
    import matplotlib.pyplot as plt
    from .images import read_region
    g = polygons_image
    fig, ax = plt.subplots(figsize=(11, 9))
    if bounds is None:
        ax.imshow(thumb, extent=(-.5, info['width']-.5, info['height']-.5, -.5))
        count = len(g)
        if count > max_polygons:
            g = g.iloc[np.sort(sample_indices(count, max_polygons))]
        limits = (-.5, -.5, info['width']-.5, info['height']-.5)
        title = f'Whole field: {len(g):,} / {count:,} polygons drawn'
    else:
        x0, y0, x1, y1 = [int(x) for x in bounds]
        if x1<=x0 or y1<=y0 or x0<0 or y0<0 or x1>info['width'] or y1>info['height']:
            raise ValueError('ROI must be within original-image pixel bounds.')
        from shapely.geometry import box
        g = g[g.intersects(box(x0, y0, x1, y1))]
        rgb = read_region(info['path'], x0, y0, x1-x0, y1-y0)
        ax.imshow(rgb, extent=(x0-.5, x1-.5, y1-.5, y0-.5))
        limits = (x0-.5, y0-.5, x1-.5, y1-.5)
        title = f'ROI: {len(g):,} polygons; original-image pixels'
    if len(g):
        # Local microscopy coordinates deliberately have no geographic CRS.
        g.boundary.plot(ax=ax, linewidth=.35, aspect='equal')
    ax.set_xlim(limits[0], limits[2]); ax.set_ylim(limits[3], limits[1]); ax.set_aspect('equal')
    ax.set_title(title); ax.set_xlabel('image x (full-resolution pixels)'); ax.set_ylabel('image y (pixels)')
    finish(fig, path, dpi)
