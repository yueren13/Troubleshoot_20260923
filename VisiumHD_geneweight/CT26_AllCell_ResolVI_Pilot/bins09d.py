"""Recover the 00 native grid and 03 polygon assignments for the completed 09B pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from raw09d import read_square_counts


STUDY_ROOT = Path('/stash/data/nonclin/TBIO-8110_VisiumHD-Adagrasib-mouseTumor')
TEMP_PROJECT_ROOT = Path('/host_root/nethome/reny28/Projects/Visium_projects/TBIO-8110/tmp')
PIPELINE = 'tbio8110_stardist_proseg_resolvi_v1'
PREDICATE = 'within_then_intersects_boundary_fallback'
VERSION_03 = 'reusable-cell-qc-binobs-transfer-v1'


def _json(path):
    return json.loads(Path(path).read_text())


def locate_paths(sample='Ada-14R', study_root=STUDY_ROOT, temp_project_root=TEMP_PROJECT_ROOT,
                 overrides=None):
    """Resolve documented project paths, consulting the 00 manifest and sample preflight."""
    study = Path(study_root).expanduser()
    derived = study / 'derived_files' / PIPELINE
    temp = Path(temp_project_root).expanduser() / PIPELINE
    config_dir = derived / '00_config'
    manual = {key: Path(value).expanduser() for key, value in (overrides or {}).items()}

    def get(key, default, recorded=None):
        if key in manual:
            return manual[key]
        default = Path(default)
        if default.exists() or recorded is None or pd.isna(recorded):
            return default
        return Path(recorded) if Path(recorded).exists() else default

    paths = {'config': get('config', config_dir / 'pipeline_config.json'),
             'manifest': get('manifest', config_dir / 'sample_manifest.csv')}
    cfg = _json(paths['config']) if paths['config'].is_file() else {}
    if cfg.get('sample_ids') and sample not in cfg['sample_ids']:
        raise ValueError(f'{sample!r} is absent from the 00 configuration.')
    cfgpaths = cfg.get('paths', {})
    record = {}
    if paths['manifest'].is_file():
        manifest = pd.read_csv(paths['manifest'], dtype=str)
        matches = manifest.loc[manifest['sample'] == sample]
        if len(matches) != 1:
            raise ValueError(f'Expected one row for {sample} in {paths["manifest"]}.')
        record = matches.iloc[0].dropna().to_dict()

    def projected(root, suffix, recorded_root):
        preferred = root / suffix
        if not preferred.exists() and recorded_root and (Path(recorded_root) / suffix).exists():
            return Path(recorded_root) / suffix
        return preferred

    config_rel = Path(sample) / f'{sample}_preflight.json'
    paths['preflight_json'] = get('preflight_json', projected(config_dir, config_rel, cfgpaths.get('config_root')),
                                  record.get('preflight_json'))
    if paths['preflight_json'].is_file():
        preflight = _json(paths['preflight_json'])
        if preflight.get('sample', sample) != sample:
            raise ValueError('00 preflight sample does not match the 09B section.')
        record.update(preflight)
    paths['zarr'] = get('zarr', study / 'zarr_files' / f'{sample}.zarr', record.get('zarr_path'))
    paths['aligned_qc'] = get('aligned_qc', config_dir / sample / f'{sample}_aligned_qc.parquet',
                              record.get('aligned_qc_parquet'))
    paths['square_dir'] = get('square_dir', study / 'source_files' / sample / 'outs' / 'binned_outputs' / 'square_002um',
                              record.get('square_2um'))
    spatial = paths['square_dir'] / 'spatial'
    paths['positions'] = get('positions', spatial / 'tissue_positions.parquet')
    paths['scalefactors'] = get('scalefactors', spatial / 'scalefactors_json.json')
    rawdir = Path('02_proseg') / sample
    qcdir = Path('03_cell_qc_obs_transfer') / sample
    for key, suffix in [('raw_cells', rawdir / f'{sample}_proseg_qcclass_cdata.h5ad'),
                        ('boundaries', rawdir / f'{sample}_proseg_qcclass_cell_boundaries.parquet'),
                        ('assignment_summary', qcdir / f'{sample}_bin_to_cell_assignment_summary.json')]:
        paths[key] = get(key, projected(derived, suffix, cfgpaths.get('derived_root')))
    paths['assignment'] = get('assignment', projected(temp, qcdir / f'{sample}_bin_to_proseg_cell_index.npy',
                                                       cfgpaths.get('temp_root')))
    unknown = set(manual) - set(paths)
    if unknown:
        raise ValueError(f'Unknown path override(s): {sorted(unknown)}. Available: {sorted(paths)}')
    return paths


def _rows(path, columns, size=100_000):
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    stored_index = (schema.pandas_metadata or {}).get('index_columns', [])
    read_names = list(columns)
    rename = {}
    # Some exports persist the barcode/cell identifier as a named parquet index.
    for i, name in enumerate(columns):
        if name not in schema.names:
            if (name in {'barcode', 'cell'} and len(stored_index) == 1
                    and isinstance(stored_index[0], str) and stored_index[0] in schema.names):
                read_names[i] = stored_index[0]
                rename[stored_index[0]] = name
            else:
                raise ValueError(f'{path} lacks {name!r}.')
    for batch in parquet.iter_batches(batch_size=size, columns=read_names):
        yield batch.to_pandas(ignore_metadata=True).rename(columns=rename)


def _tile_grid(paths, center, inner, outer, domain):
    # Enumerate native positions first so the all-tissue denominator includes low-QC
    # rows, measured zeros and any native barcodes without a prior QC annotation.
    domains = {'all_in_tissue', 'qc_retained_in_tissue'}
    if domain not in domains:
        raise ValueError(f'domain must be one of {sorted(domains)}.')
    mpp = float(_json(paths['scalefactors'])['microns_per_pixel'])
    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError('Invalid microns_per_pixel for the native square grid.')
    positions = []
    columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    for chunk in _rows(paths['positions'], columns):
        if not chunk.in_tissue.isin([0, 1]).all():
            raise ValueError('Unexpected in_tissue values in Space Ranger positions.')
        # 00 Proseg x is full-resolution image row, y is full-resolution image column.
        xy = chunk[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy(dtype=float) * mpp
        if not np.isfinite(xy[chunk.in_tissue.to_numpy(dtype=bool)]).all():
            raise ValueError('Missing coordinates for a native in-tissue square.')
        dist = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
        hit = dist <= outer
        if hit.any():
            rows = chunk.loc[hit].copy()
            rows['x_um'], rows['y_um'] = xy[hit].T
            rows['in_core'] = dist[hit] <= inner
            positions.append(rows)
    if not positions:
        raise ValueError('No native positions overlap the 09B tile. Check units and section.')
    spots = pd.concat(positions, ignore_index=True)
    del positions
    if spots.barcode.isna().any() or not spots.barcode.is_unique:
        raise ValueError('Missing or duplicate native barcode in the tile.')
    wanted = pd.Index(spots.barcode)
    all_offset = kept_offset = 0
    annotated = []
    for chunk in _rows(paths['aligned_qc'], ['barcode', 'qc_keep', 'proseg_x_um', 'proseg_y_um']):
        if chunk.qc_keep.isna().any() or not chunk.qc_keep.isin([0, 1]).all():
            raise ValueError('Unexpected qc_keep values in 00 aligned QC.')
        keep = chunk.qc_keep.to_numpy(dtype=bool)
        loc = np.flatnonzero(wanted.get_indexer(chunk.barcode) >= 0)
        if len(loc):
            rows = chunk.iloc[loc].copy()
            rows['all_ordinal'] = loc + all_offset
            rows['kept_ordinal'] = np.where(keep[loc], np.cumsum(keep, dtype=np.int64)[loc] - 1 + kept_offset, -1)
            annotated.append(rows)
        all_offset += len(chunk)
        kept_offset += int(keep.sum())
    if not annotated:
        raise ValueError('No ROI barcode matches notebook 00 aligned QC. Verify the section and registration.')
    qc = pd.concat(annotated, ignore_index=True)
    del annotated
    if not qc.barcode.is_unique:
        raise ValueError('Duplicate ROI barcode in 00 aligned QC.')
    spots = spots.merge(qc, on='barcode', how='left', validate='one_to_one', sort=False)
    spots['qc_record_present'] = spots.all_ordinal.notna()
    tissue = spots.in_tissue.to_numpy(dtype=bool)
    matched = tissue & spots.qc_record_present.to_numpy(dtype=bool)
    if not matched.any():
        raise ValueError('No in-tissue native square matches the 00 coordinate reference.')
    expected = spots.loc[matched, ['x_um', 'y_um']].to_numpy(dtype=float)
    original = spots.loc[matched, ['proseg_x_um', 'proseg_y_um']].to_numpy(dtype=float)
    if not np.isfinite(original).all() or not np.allclose(expected, original, atol=0.05, rtol=0):
        raise ValueError('Native squares do not reproduce the notebook 00 Proseg (row, column) coordinates.')
    # Use the exact archived coordinates for polygon tests where available.
    spots.loc[matched, ['x_um', 'y_um']] = original
    spots['qc_keep'] = spots.qc_keep.eq(True)
    keep = spots.qc_keep.to_numpy(dtype=bool)
    stats = {'domain': domain, 'n_aligned_all': all_offset, 'n_aligned_qc_keep': kept_offset,
             'n_in_tissue_in_roi': int(tissue.sum()), 'n_core_in_tissue': int((tissue & spots.in_core).sum()),
             'n_in_tissue_with_qc_keep': int((tissue & keep).sum()),
             'n_in_tissue_with_qc_drop': int((matched & ~keep).sum()),
             'n_in_tissue_without_qc_record': int((tissue & ~matched).sum()),
             'n_qc_retained_off_tissue_skipped_in_roi': int((~tissue & keep).sum()), 'microns_per_pixel': mpp}
    take = tissue if domain == 'all_in_tissue' else (tissue & keep)
    spots = spots.loc[take].drop(columns=['proseg_x_um', 'proseg_y_um'])
    if not spots.in_core.any():
        raise ValueError(f'The original 09B core has no native squares in domain {domain}.')
    spots = spots.sort_values('all_ordinal', kind='stable', na_position='last').reset_index(drop=True)
    for key in ('all_ordinal', 'kept_ordinal'):
        spots[key] = spots[key].fillna(-1).astype(np.int64)
    grid = spots[['array_row', 'array_col']].to_numpy(float)
    if not np.isfinite(grid).all() or (grid < 0).any() or (grid != np.floor(grid)).any() or spots.duplicated(['array_row', 'array_col']).any():
        raise ValueError('Invalid or repeated native array coordinates in the selected tile.')
    spots[['array_row', 'array_col']] = grid.astype(np.int64)
    return spots, stats


def _raw_cell_order(path):
    with h5py.File(path, 'r') as source:
        obs = source['obs']
        key = obs.attrs.get('_index', '_index')
        obj = obs[key.decode() if isinstance(key, bytes) else key]
        if isinstance(obj, h5py.Dataset):
            ids = obj.asstr()[:] if h5py.check_string_dtype(obj.dtype) else obj[:].astype(str)
        else:
            from anndata.io import read_elem
            ids = np.asarray(read_elem(obj)).astype(str)
        areas = np.asarray(obs['cell_area_um2'], dtype=float) if 'cell_area_um2' in obs and isinstance(obs['cell_area_um2'], h5py.Dataset) else None
    index = pd.Index(ids, dtype=str)
    if len(index) == 0 or index.hasnans or not index.is_unique or (index == '').any():
        raise ValueError('Original 02 cell identifiers are empty or nonunique.')
    return index, areas


def _cached_assignment(paths, sample, spots, stats, n_cells):
    summary = _json(paths['assignment_summary'])
    scope = summary.get('bin_scope')
    ordinals = {'all_bins': ('all_ordinal', 'n_aligned_all'), 'qc_keep_only': ('kept_ordinal', 'n_aligned_qc_keep')}
    if scope not in ordinals:
        raise ValueError(f'Unsupported 03 bin_scope {scope!r}.')
    ordinal, n_key = ordinals[scope]
    n_bins = stats[n_key]
    if (spots[ordinal] < 0).any():
        raise ValueError(f'03 {scope} cache does not cover every square in the requested domain.')
    fields = ['pipeline_version', 'sample', 'bin_scope', 'n_bins', 'n_cells', 'boundary_path', 'boundary_size',
              'aligned_qc_path', 'aligned_qc_size']
    signature = {k: summary[k] for k in fields}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    expected = {'pipeline_version': VERSION_03, 'sample': sample, 'n_bins': n_bins, 'n_bins_considered': n_bins,
                'n_cells': n_cells, 'signature_hash': digest, 'predicate': PREDICATE}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'03 receipt mismatch in {key}.')
    for key, path in [('boundary', paths['boundaries']), ('aligned_qc', paths['aligned_qc'])]:
        if Path(summary[key + '_path']).name != path.name or summary[key + '_size'] != path.stat().st_size:
            raise ValueError(f'03 receipt does not identify current {key} input.')
    mapping = np.load(paths['assignment'], mmap_mode='r', allow_pickle=False)
    if mapping.shape != (n_bins,) or mapping.dtype != np.dtype('int32'):
        raise ValueError('03 assignment array shape/dtype does not match its receipt.')
    assigned = 0
    for start in range(0, n_bins, 1_000_000):
        part = mapping[start:start + 1_000_000]
        if (part < -1).any() or (part >= n_cells).any():
            raise ValueError('03 array contains cell indices outside the original 02 order.')
        assigned += int(np.count_nonzero(part >= 0))
    if assigned != summary['n_bins_assigned']:
        raise ValueError('03 assigned bin count does not match its receipt.')
    return np.asarray(mapping[spots[ordinal].to_numpy()], dtype=np.int32), {
        'assignment_method': 'notebook_03_verified_cache', 'assignment_scope': scope,
        'assignment_signature_hash': digest, 'assignment_predicate': PREDICATE}


def _local_assignment(path, spots, cell_ids):
    """Assign ROI bin centers against every overlapping original 02 geometry, as in 03."""
    import shapely

    if not hasattr(shapely, 'from_wkb'):
        raise ImportError('Local boundary recovery requires shapely >=2 in the Python kernel.')
    xy = spots[['x_um', 'y_um']].to_numpy(dtype=float)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    geometry_chunks, cell_chunks = [], []
    seen = np.zeros(len(cell_ids), dtype=bool)
    for chunk in _rows(path, ['cell', 'geometry'], size=10_000):
        ids = pd.to_numeric(chunk.cell, errors='raise')
        if ids.isna().any() or (ids != ids.astype(np.int64)).any():
            raise ValueError('Noninteger polygon cell ID in notebook 02 boundaries.')
        ix = cell_ids.get_indexer(ids.astype(np.int64).astype(str))
        use = ix >= 0  # notebook 03 reindexes boundaries to the 02 expression-cell list
        ix = ix[use]
        if seen[ix].any() or len(np.unique(ix)) != len(ix):
            raise ValueError('Duplicate cell IDs in notebook 02 boundaries.')
        seen[ix] = True
        geom = shapely.from_wkb(chunk.loc[use, 'geometry'].to_numpy())
        if shapely.is_missing(geom).any() or shapely.is_empty(geom).any():
            raise ValueError('Missing or empty geometry for a notebook 02 cell.')
        bounds = shapely.bounds(geom)
        if not np.isfinite(bounds).all():
            raise ValueError('Nonfinite polygon bounds in notebook 02.')
        hit = (bounds[:, 0] <= xmax) & (bounds[:, 2] >= xmin) & (bounds[:, 1] <= ymax) & (bounds[:, 3] >= ymin)
        if hit.any():
            geometry_chunks.append(geom[hit])
            cell_chunks.append(ix[hit])
    if not seen.all():
        raise ValueError(f'{(~seen).sum()} original 02 expression cells lack boundaries.')
    mapping = np.full(len(xy), -1, dtype=np.int32)
    if not geometry_chunks:
        return mapping
    idx = np.concatenate(cell_chunks)
    order = np.argsort(idx)
    idx = idx[order]
    geom = np.concatenate(geometry_chunks)[order]
    tree = shapely.STRtree(geom)
    areas = shapely.area(geom)
    for start in range(0, len(xy), 100_000):
        points = shapely.points(xy[start:start + 100_000])
        available = np.arange(len(points))
        for predicate in ('within', 'intersects'):
            if not len(available):
                break
            pt, poly = tree.query(points[available], predicate=predicate)
            if len(pt):
                # Like 03, stable (point, area) ordering preserves STRtree order for exact ties.
                rank = np.lexsort((areas[poly], pt))
                p, g = pt[rank], poly[rank]
                first = np.r_[True, p[1:] != p[:-1]]
                mapping[start + available[p[first]]] = idx[g[first]]
            available = np.flatnonzero(mapping[start:start + len(points)] < 0)
    return mapping


def prepare_tile(run_dir, study_root=STUDY_ROOT, temp_project_root=TEMP_PROJECT_ROOT,
                 padding_um=200., overrides=None, domain='all_in_tissue'):
    """Load native squares in the 09B core plus halo, including measured zeros.

    domain='all_in_tissue' includes the native tissue mask independently of QC;
    'qc_retained_in_tissue' restricts to the notebook 00 retained classes.
    padding_um is added to the core radius, preserving at least the 09B halo.
    """
    run_dir = Path(run_dir).expanduser()
    geometry = _json(run_dir / 'complete.json').get('geometry', {})
    if 'section' not in geometry or 'center_um' not in geometry:
        raise ValueError('Supply the completed 09B single-section tile run with geometry in complete.json.')
    sample = str(geometry['section'])
    center = np.asarray(geometry['center_um'], dtype=float)
    inner = float(geometry['inner_radius_um'])
    original_outer = float(geometry['outer_radius_um'])
    padding_um = float(padding_um)
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite([inner, original_outer, padding_um]).all() or inner <= 0 or original_outer < inner or padding_um < 0:
        raise ValueError('Invalid 09B tile geometry/padding.')
    settings = _json(run_dir / 'settings.json')
    # The archived 00/02 centroids and polygons already have units of microns.
    if float(settings['settings']['microns_per_unit']) != 1.0:
        raise ValueError('09B used a non-unit coordinate scale. Verify the registration to native Proseg µm before use.')
    outer = max(original_outer, inner + padding_um)
    paths = locate_paths(sample, study_root, temp_project_root, overrides)
    for key in ('aligned_qc', 'positions', 'scalefactors', 'raw_cells', 'boundaries'):
        if not paths[key].is_file():
            raise FileNotFoundError(f'Missing {key}: {paths[key]}. Update study_root or the matching path override.')
    spots, stats = _tile_grid(paths, center, inner, outer, domain)
    preflight = _json(paths['preflight_json']) if paths['preflight_json'].is_file() else {}
    for prior, measured in [('n_qc_bins', 'n_aligned_all'), ('n_qc_keep_bins', 'n_aligned_qc_keep')]:
        if prior in preflight and preflight[prior] != stats[measured]:
            raise ValueError(f'00 preflight {prior} does not match the supplied aligned QC table.')
    if ('source_microns_per_pixel' in preflight and
            not np.isclose(preflight['source_microns_per_pixel'], stats['microns_per_pixel'])):
        raise ValueError('Space Ranger micron scale differs from the original notebook 00 preflight.')
    cell_ids, cell_areas = _raw_cell_order(paths['raw_cells'])
    try:
        assignment, proof = _cached_assignment(paths, sample, spots, stats, len(cell_ids))
    except (OSError, EOFError, ValueError, KeyError, TypeError) as error:
        assignment = _local_assignment(paths['boundaries'], spots, cell_ids)
        proof = {'assignment_method': 'recomputed_from_02_polygons', 'assignment_scope': 'all_original_cells_in_roi',
                 'assignment_predicate': PREDICATE, 'cache_fallback_reason': str(error)}
    spots['cell_index'] = assignment
    spots.drop(columns=['all_ordinal', 'kept_ordinal'], inplace=True)
    counts, genes, raw_source = read_square_counts(paths['square_dir'], spots.barcode.to_numpy())
    meta = {**stats, **proof, 'sample': sample, 'geometry_09b': geometry, 'center_um': center.tolist(),
            'inner_radius_um': inner, 'outer_radius_um': outer, 'padding_um': outer - inner,
            'n_original_cells': len(cell_ids), 'n_tile_bins': len(spots), 'n_core_bins': int(spots.in_core.sum()),
            'n_tile_assigned': int((assignment >= 0).sum()), 'n_tile_unassigned': int((assignment == -1).sum()),
            'n_tile_zero_UMI': int((np.asarray(counts.sum(axis=0)).ravel() == 0).sum()),
            'inputs': {k: str(v) for k, v in paths.items()}, 'zarr_present': paths['zarr'].is_dir(),
            'zarr_table': 'square_002um', 'raw_source': raw_source, 'units': 'micrometers; Proseg x=image row and y=image column'}
    return {'counts': counts, 'genes': genes, 'spots': spots, 'cell_ids': cell_ids.to_numpy(),
            'cell_areas': cell_areas, 'meta': meta, 'run_dir': run_dir}
