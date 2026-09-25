"""Exact first 41 microViz brewerPlus colors; optional explicitly labeled extensions.
Source: https://github.com/david-barnett/microViz/blob/main/R/distinct_palette.R
Color values are reproduced, not the R implementation. No extra 'other'/lightgrey category.
"""
from __future__ import annotations
import warnings
import numpy as np

BREWER_PLUS = [
    '#A6CEE3','#1F78B4','#B2DF8A','#33A02C','#FB9A99','#E31A1C',
    '#FDBF6F','#FF7F00','#CAB2D6','#6A3D9A','#FFFF99','#B15928',
    '#1FF8FF',
    '#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02','#A6761D','#666666',
    '#4B6A53','#B249D5','#7EDC45','#5C47B8','#CFD251','#FF69B4','#69C86C','#CD3E50',
    '#83D5AF','#DA6130','#5E79B2','#C29545','#532A5A','#5F7B35','#C497CF','#773A27',
    '#7CB9CB','#594E50','#D3C4A8','#C17E7F',
]


def natural_order(values):
    import re
    def key(x):
        return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r'(\d+)', str(x))]
    return sorted([str(x) for x in values], key=key)


def distinct_palette(n: int, allow_extension=True):
    if n < 0:
        raise ValueError('n must be non-negative.')
    if n <= len(BREWER_PLUS):
        return BREWER_PLUS[:n]
    if not allow_extension:
        raise ValueError(f'{n} categories exceeds microViz brewerPlus maximum 41.')
    if n > 200:
        warnings.warn('More than 200 categories: a single UMAP cannot make every category visually '
                      'distinguishable. The plot is still produced; use the label/color tables or focused views.')
    warnings.warn(f'{n} categories: colors 1-41 are microViz brewerPlus; later colors are '
                  'deterministic LAB-distance extensions, NOT microViz colors. '
                  'Distinct hex values do not guarantee perceptual or color-vision accessibility.')
    from matplotlib.colors import to_rgb, to_hex
    from skimage.color import rgb2lab
    side = max(19, int(np.ceil((n * 4) ** (1/3))))
    grid = np.linspace(0.05, 0.95, side)
    candidates = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
    lab = rgb2lab(candidates[None, :, :])[0]
    keep = (lab[:, 0] >= 32) & (lab[:, 0] <= 82)
    candidates, lab = candidates[keep], lab[keep]
    if len(candidates) < n:
        raise ValueError('Insufficient color candidates; provide a reviewed custom palette for this extreme category count.')
    chosen = list(BREWER_PLUS)
    chosen_lab = rgb2lab(np.array([to_rgb(c) for c in chosen])[None, :, :])[0]
    distance = ((lab[:, None, :] - chosen_lab[None, :, :])**2).sum(axis=2).min(axis=1)
    while len(chosen) < n:
        idx = int(np.argmax(distance))
        chosen.append(to_hex(candidates[idx]).upper())
        distance = np.minimum(distance, ((lab-lab[idx])**2).sum(axis=1))
        distance[idx] = -np.inf
    if len(set(chosen)) != n:
        raise RuntimeError('Unexpected duplicated colors.')
    return chosen
