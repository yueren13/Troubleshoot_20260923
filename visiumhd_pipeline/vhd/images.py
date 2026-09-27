from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from .core import affine, crop_mask_transform, transform_xy


def image_info(path, mpp_override=None):
    """OpenSlide first; ordinary RGB TIFF fallback requires tifffile + zarr."""
    import openslide
    path = str(path)
    if openslide.OpenSlide.detect_format(path):
        with openslide.OpenSlide(path) as slide:
            mpp = mpp_override or [slide.properties.get('openslide.mpp-x'),
                                   slide.properties.get('openslide.mpp-y')]
            width, height = slide.dimensions
            backend = 'openslide'
    else:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            s = tif.series[0]
            axes, shape = s.axes, s.shape
            if axes not in ('YXS', 'YXC', 'CYX', 'SYX'):
                raise ValueError(f'Unsupported RGB TIFF axes {axes}; no silent channel/axis guessing.')
            width, height = shape[axes.index('X')], shape[axes.index('Y')]
            if shape[axes.index('S' if 'S' in axes else 'C')] not in (3, 4):
                raise ValueError('H&E input must have 3 RGB or 4 RGBA channels.')
            mpp = mpp_override
            backend = 'tifffile'
    if mpp is None or len(mpp) != 2 or any(v is None for v in mpp):
        raise ValueError('Supply image_mpp_override=[mpp_x,mpp_y]; do not guess from objective power.')
    mpp = [float(x) for x in mpp]
    if not np.isfinite(mpp).all() or min(mpp) <= 0:
        raise ValueError(f'Invalid image microns per pixel: {mpp}')
    return {'path': path, 'width': int(width), 'height': int(height), 'mpp': mpp, 'backend': backend}


def _white_rgb(array):
    a = np.asarray(array)
    if a.dtype != np.uint8:
        raise ValueError('H&E input must be uint8 RGB; explicitly rescale higher bit-depth images first.')
    if a.ndim != 3 or a.shape[-1] not in (3, 4):
        raise ValueError(f'Expected HxWx3/4 RGB, got {a.shape}.')
    if a.shape[-1] == 4:
        alpha = a[..., 3:4].astype(np.float32)/255
        return np.round(a[..., :3]*alpha+255*(1-alpha)).astype(np.uint8)
    return a[..., :3]


def read_region(path, left, top, width, height):
    """Read only a rectangular level-0 RGB region. Reopens file; safe inside delayed tasks."""
    import openslide
    path = str(path)
    left, top, width, height = map(int, (left, top, width, height))
    if openslide.OpenSlide.detect_format(path):
        with openslide.OpenSlide(path) as slide:
            return _white_rgb(np.asarray(slide.read_region((left, top), 0, (width, height))))
    import tifffile
    import zarr
    with tifffile.TiffFile(path) as tif:
        s = tif.series[0]
        store = s.aszarr()
        try:
            arr = zarr.open(store, mode='r')
            # aszarr() can expose a pyramid group instead of the scale-0 array.
            if not hasattr(arr, 'shape'):
                arr = arr['0']
            axes = s.axes
            sl = [slice(None)]*3
            sl[axes.index('Y')] = slice(top, top+height)
            sl[axes.index('X')] = slice(left, left+width)
            a = np.asarray(arr[tuple(sl)])
            a = np.moveaxis(a, axes.index('S' if 'S' in axes else 'C'), -1)
            return _white_rgb(a)
        finally:
            store.close()


def thumbnail(info, max_side=3000):
    import openslide
    if info['backend'] == 'openslide':
        with openslide.OpenSlide(info['path']) as slide:
            return _white_rgb(np.asarray(slide.get_thumbnail((max_side, max_side)).convert('RGB')))
    # Tile sampling bounds RAM for a non-pyramidal TIFF; reading compressed tiles may still be slow.
    from scipy.ndimage import map_coordinates
    step = max(1, int(np.ceil(max(info['width'], info['height'])/max_side)))
    ys = np.arange(0, info['height'], step)
    xs = np.arange(0, info['width'], step)
    out = np.empty((len(ys), len(xs), 3), dtype=np.uint8)
    tile = 2048
    for y in range(0, info['height'], tile):
        yy = np.flatnonzero((ys>=y)&(ys<y+tile))
        if not len(yy):
            continue
        for x in range(0, info['width'], tile):
            xx = np.flatnonzero((xs>=x)&(xs<x+tile))
            if not len(xx):
                continue
            a = read_region(info['path'], x, y, min(tile, info['width']-x), min(tile, info['height']-y))
            out[np.ix_(yy, xx)] = a[np.ix_(ys[yy]-y, xs[xx]-x)]
    return out


def sample_rgb(info, xy, thumb):
    """Approximate image intensity at bin centers using a thumbnail; never changes coordinates."""
    xy = np.asarray(xy)
    col = np.floor((xy[:, 0]+0.5)*thumb.shape[1]/info['width']).astype(np.int64)
    row = np.floor((xy[:, 1]+0.5)*thumb.shape[0]/info['height']).astype(np.int64)
    valid = (xy[:, 0]>=0)&(xy[:, 0]<info['width'])&(xy[:, 1]>=0)&(xy[:, 1]<info['height'])
    rgb = np.full((len(xy), 3), 255, dtype=np.uint8)
    rgb[valid] = thumb[np.clip(row[valid], 0, thumb.shape[0]-1), np.clip(col[valid], 0, thumb.shape[1]-1)]
    return rgb, valid


def lab_stats(rgb, od_threshold=0.08):
    from skimage.color import rgb2lab
    rgb = np.asarray(rgb)
    mask = (-np.log((rgb.astype(float)+1)/256)).mean(axis=-1) > od_threshold
    values = rgb2lab(rgb/255.)[mask]
    if len(values)<100:
        raise ValueError('Not enough non-white pixels for H&E color normalization.')
    return {'mean': values.mean(axis=0).tolist(), 'std': np.maximum(values.std(axis=0), 1e-6).tolist(),
            'od_threshold': float(od_threshold)}


def reinhard(rgb, source_stats, target_stats):
    """Optional LAB mean/std color matching; not stain deconvolution and not the missing legacy helper."""
    from skimage.color import rgb2lab, lab2rgb
    rgb = np.asarray(rgb)
    source = rgb2lab(rgb/255.)
    matched = (source-np.array(source_stats['mean']))/np.array(source_stats['std'])
    matched = matched*np.array(target_stats['std'])+np.array(target_stats['mean'])
    out = np.round(np.clip(lab2rgb(matched), 0, 1)*255).astype(np.uint8)
    mask = (-np.log((rgb.astype(float)+1)/256)).mean(axis=-1) > source_stats['od_threshold']
    out[~mask] = rgb[~mask]
    return out


def _image_tile(path, x, y, w, h, normalization):
    a = read_region(path, x, y, w, h)
    if normalization:
        a = reinhard(a, normalization['source'], normalization['target'])
    return np.moveaxis(a, -1, 0)


def lazy_rgb(info, tile_size=1024, normalization=None):
    import dask.array as da
    from dask import delayed
    blocks = []
    for y in range(0, info['height'], tile_size):
        row = []
        for x in range(0, info['width'], tile_size):
            w, h = min(tile_size, info['width']-x), min(tile_size, info['height']-y)
            task = delayed(_image_tile)(info['path'], x, y, w, h, normalization)
            row.append(da.from_delayed(task, shape=(3, h, w), dtype=np.uint8))
        blocks.append(da.concatenate(row, axis=2))
    return da.concatenate(blocks, axis=1).rechunk((3, tile_size, tile_size))


def resized_crop(info, bounds, target_mpp, output_path, normalization=None, tile_size=1024):
    """Tile-by-tile resampling to a memory-mapped RGB crop; actual dimensions define the affine."""
    from scipy.ndimage import map_coordinates
    left, top, right, bottom = map(int, bounds)
    source_w, source_h = right-left, bottom-top
    width = max(1, int(round(source_w*info['mpp'][0]/target_mpp)))
    height = max(1, int(round(source_h*info['mpp'][1]/target_mpp)))
    matrix = crop_mask_transform(left, top, source_w, source_h, width, height)
    out = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.uint8, shape=(height, width, 3))
    # Linear interpolation of original pixels. No per-tile percentile normalization or seam-wise fit.
    for oy in range(0, height, tile_size):
        for ox in range(0, width, tile_size):
            oh, ow = min(tile_size, height-oy), min(tile_size, width-ox)
            xsrc = matrix[0, 0]*np.arange(ox, ox+ow)+matrix[0, 2]
            ysrc = matrix[1, 1]*np.arange(oy, oy+oh)+matrix[1, 2]
            x0, x1 = max(0, int(np.floor(xsrc.min()))-1), min(info['width'], int(np.ceil(xsrc.max()))+2)
            y0, y1 = max(0, int(np.floor(ysrc.min()))-1), min(info['height'], int(np.ceil(ysrc.max()))+2)
            tile = read_region(info['path'], x0, y0, x1-x0, y1-y0)
            yy, xx = np.meshgrid(ysrc-y0, xsrc-x0, indexing='ij')
            a = np.stack([map_coordinates(tile[..., c], [yy, xx], order=1, mode='nearest')
                          for c in range(3)], axis=-1)
            if normalization:
                a = reinhard(a, normalization['source'], normalization['target'])
            out[oy:oy+oh, ox:ox+ow] = a
    out.flush()
    return matrix, [height, width]
