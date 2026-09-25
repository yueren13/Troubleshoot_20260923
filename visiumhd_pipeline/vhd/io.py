from __future__ import annotations

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .core import (VERSION, affine, completed, digest, dump_json, file_stamp, fit_grid_affine,
                   integer_csr, qc_metrics, reusable, safe_frame, sample_paths,
                   stage_lock, transform_xy, write_h5ad)
from .images import image_info


def resolve_spaceranger(spec):
    base = Path(spec['outs'])/'binned_outputs'/'square_002um'
    h5 = base/'raw_feature_bc_matrix.h5'
    mtx = base/'raw_feature_bc_matrix'
    matrix = h5 if h5.exists() else mtx
    positions = base/'spatial'/'tissue_positions.parquet'
    scales = base/'spatial'/'scalefactors_json.json'
    for p in [matrix, positions, scales]:
        if not p.exists():
            raise FileNotFoundError(f'Required unfiltered 2-um Space Ranger output: {p}')
    return {'base': base, 'matrix': matrix, 'positions': positions, 'scales': scales}


def read_raw_counts(path):
    import scanpy as sc
    path = Path(path)
    if path.is_dir():
        a = sc.read_10x_mtx(path, var_names='gene_ids', make_unique=False, cache=False, gex_only=True)
        a.var['gene_symbol'] = a.var['gene_symbols'].astype(str)
    else:
        a = sc.read_10x_h5(path, gex_only=True)
        a.var['gene_symbol'] = a.var_names.astype(str)
        if 'gene_ids' not in a.var:
            raise KeyError('Space Ranger H5 lacks stable gene_ids.')
        a.var_names = a.var['gene_ids'].astype(str)
    if not a.var_names.is_unique or not a.obs_names.is_unique:
        raise ValueError('Raw gene IDs or barcodes are duplicated. Do not suffix stable gene IDs silently.')
    a.var['gene_id'] = a.var_names.astype(str)
    # Historical sc.read_10x_mtx(var_names='gene_symbols', make_unique=True) mapping.
    from anndata.utils import make_index_unique
    a.var['legacy_unique_symbol'] = make_index_unique(pd.Index(a.var['gene_symbol'].astype(str))).to_numpy()
    a.X = integer_csr(a.X)
    a.obs_names = a.obs_names.astype(str)
    return a


def ingest(cfg, spec):
    p = sample_paths(cfg, spec)
    sr = resolve_spaceranger(spec)
    input_files = [sr['positions'], sr['scales'], Path(spec['image'])]
    input_files += sorted(x for x in sr['matrix'].iterdir() if x.is_file()) if sr['matrix'].is_dir() else [sr['matrix']]
    sig = digest({'version': VERSION, 'files': [file_stamp(x) for x in input_files],
                  'sample': spec['sample'], 'affine': spec['tenx_to_image_affine'],
                  'mpp_override': spec.get('image_mpp_override'),
                  'source_mpp_override': spec.get('source_mpp_override'),
                  'grid_tolerance': cfg['coordinates']['grid_max_residual_px']})
    marker = p['work']/'01_ingest_complete.json'
    if reusable(marker, sig, [p['bins'], p['geometry']]):
        return json.loads(p['geometry'].read_text())
    with stage_lock(p['work'], 'ingest'):
        a = read_raw_counts(sr['matrix'])
        pos = pd.read_parquet(sr['positions'])
        if 'barcode' not in pos or pos['barcode'].duplicated().any():
            raise ValueError('tissue_positions.parquet needs unique barcode column.')
        pos['barcode'] = pos['barcode'].astype(str)
        pos = pos.set_index('barcode').reindex(a.obs_names)
        required = ['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']
        if pos[required].isna().any().any():
            raise ValueError('At least one raw barcode lacks original full-resolution coordinates.')
        for key in ('array_row', 'array_col'):
            values = pos[key].to_numpy(float)
            if np.any(values<0) or not np.equal(values, np.rint(values)).all():
                raise ValueError(f'Invalid integer grid positions in {key}.')
            pos[key] = values.astype(np.int64)
        if pos[['array_row', 'array_col']].duplicated().any():
            raise ValueError('Duplicate 2-um array grid positions.')
        xy = pos[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(np.float64)
        grid_affine, residual = fit_grid_affine(pos.array_row, pos.array_col, xy,
                                                cfg['coordinates']['grid_max_residual_px'])
        info = image_info(spec['image'], spec.get('image_mpp_override'))
        scale = json.loads(sr['scales'].read_text())
        source_mpp = spec.get('source_mpp_override') or scale.get('microns_per_pixel')
        if source_mpp is None or not np.isfinite(source_mpp) or float(source_mpp)<=0:
            raise ValueError('Need source_mpp_override or Space Ranger microns_per_pixel. '
                             'Do not substitute hires scale or microscope objective.')
        a.obs = pos.copy()
        a.obs['barcode'] = a.obs_names
        a.obs['sample'] = spec['sample']
        metrics = qc_metrics(a.X, a.var['gene_symbol'])
        metrics.index = a.obs_names
        for col in metrics:
            a.obs[col] = metrics[col]
        a.obsm['spatial'] = xy  # ALWAYS x=original fullres column, y=original fullres row.
        a.obsm['spatial_image_px'] = transform_xy(xy, spec['tenx_to_image_affine'])
        a.uns['vhd'] = {'counts_location': 'X', 'counts_kind': 'Space Ranger raw integer GEX',
                        'coordinate_system': 'fullres', 'xy_order': 'column,row', 'version': VERSION}
        geom = {'sample': spec['sample'], 'image': info, 'image_mpp': info['mpp'],
                'source_mpp': float(source_mpp), 'tenx_to_image': affine(spec['tenx_to_image_affine']).tolist(),
                'grid_index_to_fullres': grid_affine.tolist(), 'grid_fit': residual,
                'grid_shape': [int(pos.array_row.max())+1, int(pos.array_col.max())+1],
                'positions_path': str(sr['positions']), 'source_scalefactors': scale,
                'n_bins': a.n_obs, 'n_genes': a.n_vars,
                'input_total_umis': int(a.X.sum(dtype=np.int64)),
                'mpp_source_ratio_image_x_to_sr': info['mpp'][0]/float(source_mpp)}
        write_h5ad(a, p['bins'])
        dump_json(geom, p['geometry'])
        completed(marker, sig, [p['bins'], p['geometry']], n_bins=a.n_obs, n_genes=a.n_vars)
        del a, pos; gc.collect()
    return geom


def h5_metadata(path):
    import anndata as ad
    a = ad.read_h5ad(path, backed='r')
    try:
        return a.obs.copy(), a.var.copy()
    finally:
        a.file.close()


def table_metadata(store, table):
    """Read ONLY obs/var; do not instantiate the huge X matrix or unrelated tables."""
    import zarr
    from anndata.io import read_elem
    path = Path(store)/'tables'/table
    if not path.exists():
        raise FileNotFoundError(path)
    root = zarr.open_group(str(path), mode='r')
    return read_elem(root['obs']), read_elem(root['var'])


def load_table(store, table='cells', *, merge_scvi=False):
    """Eagerly load one named AnnData, not all tables. This is NOT a row-backed reader."""
    import anndata as ad
    path = Path(store)/'tables'/table
    a = ad.read_zarr(str(path))
    if merge_scvi:
        if table != 'cells':
            raise ValueError('scVI annotations are defined for the cells table only.')
        extra = Path(store)/'tables'/'cells_scvi'
        if not extra.exists():
            raise FileNotFoundError('Run notebook 11 to write the zero-gene cells_scvi annotation table.')
        b = ad.read_zarr(str(extra))
        if not a.obs_names.equals(b.obs_names):
            raise ValueError('Integration annotation IDs/order do not exactly match cells.')
        for key in b.obs:
            if key.startswith('leiden_') or key=='included_in_scvi':
                a.obs[key] = b.obs[key]
        for key in ('X_scVI', 'X_umap'):
            a.obsm[key] = b.obsm[key]
        for key, value in b.uns.items():
            if key.endswith('_colors') or key.startswith('vhd_scvi'):
                a.uns[key] = value
    return a


def load_spatial_with_table(store, table='cells', *, merge_scvi=False,
                            elements=('images', 'labels', 'shapes', 'points')):
    import spatialdata as sd
    s = sd.read_zarr(str(store), selection=elements)
    s.tables[table] = load_table(store, table, merge_scvi=merge_scvi)
    return s


def selected_bins(cfg, spec):
    """Materialize only QC-kept rows of the raw-bin staging H5AD."""
    import anndata as ad
    p = sample_paths(cfg, spec)
    obs = pd.read_parquet(p['bin_obs'])
    a = ad.read_h5ad(p['bins'], backed='r')
    try:
        if not a.obs_names.equals(obs.index):
            raise ValueError('Bin-QC metadata does not match source barcode order.')
        keep = obs['qc_keep'].to_numpy(bool)
        if not keep.any():
            raise ValueError('No QC-kept bins.')
        out = a[keep].to_memory()
        out.obs = obs.loc[out.obs_names].copy()
        return out
    finally:
        a.file.close()
