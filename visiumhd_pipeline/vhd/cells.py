from __future__ import annotations

import gc
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (VERSION, affine, completed, digest, dump_json, file_stamp, integer_csr,
                   proseg_to_fullres, qc_metrics, reusable, sample_paths, stage_lock,
                   transform_xy, write_h5ad)
from .io import h5_metadata
from .plots import standard_qc_plots, scatter_categorical, scatter_numeric, boundary_overlay
from .images import thumbnail


def clean_local_geometry(gdf):
    """Never interpret microscopy microns or pixels as longitude/latitude."""
    gdf = gdf.copy()
    if gdf.crs is not None:
        gdf = gdf.set_crs(None, allow_override=True)
    return gdf


def transform_polygons(gdf, matrix):
    from shapely.affinity import affine_transform
    a = affine(matrix)
    out = clean_local_geometry(gdf)
    params = [a[0, 0], a[0, 1], a[1, 0], a[1, 1], a[0, 2], a[1, 2]]
    out.geometry = out.geometry.map(lambda g: affine_transform(g, params))
    return out


def read_polygons(path, id_column='cell'):
    import geopandas as gpd
    p = Path(path)
    if p.suffix == '.parquet':
        g = gpd.read_parquet(p)
    elif p.name.endswith('.gz'):
        with gzip.open(p, 'rt') as f:
            data = json.load(f)  # Strict CRC validation; never discard gzip trailing content silently.
        g = gpd.GeoDataFrame.from_features(data['features'])
    else:
        g = gpd.read_file(p)
    g = clean_local_geometry(g)
    if id_column is not None:
        if id_column not in g:
            raise KeyError(f'Polygon ID column {id_column!r} is absent.')
        ids = g[id_column].astype(str)
    else:
        ids = g.index.astype(str)
    g.index = pd.Index(ids, name=None)
    if g.index.duplicated().any():
        raise ValueError('Duplicate polygon IDs. Review layer/fragment semantics before dissolving them.')
    valid = g.geometry.notna() & ~g.geometry.is_empty & g.geometry.is_valid
    valid &= g.geometry.geom_type.isin(['Polygon', 'MultiPolygon']) & (g.geometry.area>0)
    if not valid.all():
        raise ValueError(f'{int((~valid).sum())} missing/invalid/nonpolygon geometries; '
                         'no silent repair, removal, or artificial replacement boundaries.')
    return g


def read_durable_proseg(files, orientation='cells_by_genes'):
    import anndata as ad
    from scipy.io import mmread
    for key in ['counts', 'cell_metadata', 'gene_metadata', 'polygons']:
        if not Path(files[key]).is_file() or Path(files[key]).stat().st_size==0:
            raise FileNotFoundError(f'Missing/empty Proseg output: {files[key]}')
    with gzip.open(files['counts'], 'rb') as f:
        x = mmread(f).tocsr()
        # Ensure gzip CRC/trailer is consumed even when mmread stops after the declared entries.
        if f.read().strip():
            raise ValueError('Unexpected non-whitespace after Matrix Market matrix.')
    cells = pd.read_csv(files['cell_metadata'])
    genes = pd.read_csv(files['gene_metadata'])
    if 'cell' not in cells or 'gene' not in genes:
        raise KeyError('Proseg metadata must have cell and gene columns.')
    if orientation=='genes_by_cells':
        x = x.T.tocsr()
    elif orientation!='cells_by_genes':
        raise ValueError('Explicit counts orientation must be cells_by_genes or genes_by_cells.')
    if x.shape != (len(cells), len(genes)):
        raise ValueError(f'{x.shape} does not match cell/gene metadata {(len(cells),len(genes))}. '
                         'No auto-transpose inference, especially for square matrices.')
    cell_id = pd.to_numeric(cells['cell'], errors='raise').to_numpy()
    if not np.equal(cell_id, np.rint(cell_id)).all() or set(cell_id)!=set(range(len(cells))):
        raise ValueError('Expected unique complete numeric Proseg cell IDs 0..N-1; review changed output schema.')
    cells = cells.assign(cell=cell_id.astype(np.int64)).set_index('cell', drop=False).reindex(range(len(cells)))
    cells.index = cells.index.astype(str); cells.index.name = None
    genes.index = genes['gene'].astype(str); genes.index.name = None
    if not genes.index.is_unique:
        raise ValueError('Proseg gene identifiers are not unique.')
    a = ad.AnnData(X=integer_csr(x), obs=cells, var=genes)
    a.obsm['spatial'] = a.obs[['centroid_x','centroid_y']].to_numpy(np.float64)
    g = read_polygons(files['polygons'], 'cell')
    if set(g.index)!=set(a.obs_names):
        raise ValueError('Proseg polygons and count-matrix cell IDs do not match exactly.')
    return a, g.loc[a.obs_names].copy()


def canonical_features(a, reference_var, names_are='auto_exact'):
    """Map by exact stable ID or historical Scanpy-unique symbols; ambiguity is an error."""
    ref = reference_var.copy()
    names = a.var_names.astype(str)
    if names_are in ('gene_id', 'auto_exact') and names.isin(ref.index).all():
        ids = names
    elif names_are in ('legacy_unique_symbol', 'auto_exact'):
        key = 'legacy_unique_symbol'
        if ref[key].duplicated().any():
            raise ValueError('Reference legacy symbols are not unique.')
        mapping = pd.Series(ref.index.astype(str), index=ref[key].astype(str))
        ids = mapping.reindex(names)
        if ids.isna().any():
            raise ValueError(f'Cannot match {int(ids.isna().sum())} cell features to raw stable gene IDs. '
                             'Supply an explicit reviewed mapping; suffixes are not guessed.')
        ids = pd.Index(ids.astype(str))
    else:
        raise ValueError('Feature names are not exact stable IDs under the requested mapping mode.')
    if not ids.is_unique:
        raise ValueError('Multiple cell features map to the same stable gene ID.')
    a.var['input_feature_name'] = names.to_numpy()
    a.var_names = ids
    for key in ('gene_id', 'gene_symbol', 'legacy_unique_symbol'):
        a.var[key] = ref.loc[ids, key].to_numpy()
    return a


def geometric_assign(xy, polygons, chunk_size=250_000):
    """Bin-center containment. -1=unassigned, -2=ambiguous overlap; never nearest-neighbor fill."""
    import shapely
    tree = shapely.STRtree(polygons.geometry.to_numpy())
    assignments = np.full(len(xy), -1, dtype=np.int64)
    for start in range(0, len(xy), chunk_size):
        pts = shapely.points(np.asarray(xy)[start:start+chunk_size])
        pairs = tree.query(pts, predicate='intersects')  # Includes boundaries explicitly.
        if pairs.size==0:
            continue
        query_ids, polygon_ids = pairs
        query_unique, counts = np.unique(query_ids, return_counts=True)
        assignments[start+query_unique[counts>1]] = -2
        good = query_unique[counts==1]
        choose = np.isin(query_ids, good)
        assignments[start+query_ids[choose]] = polygon_ids[choose]
    return assignments


def aggregate_annotations(obs, assignment, n_cells, spec):
    """No expression aggregation. Fractions use only nonmissing values; report each denominator."""
    if len(obs)!=len(assignment):
        raise ValueError('Annotation/assignment length mismatch.')
    assigned = assignment>=0
    counts = np.bincount(assignment[assigned], minlength=n_cells)
    out = pd.DataFrame({'geometric_n_bins': counts})
    for col in spec.get('categorical', []):
        if col not in obs:
            raise KeyError(f'Bin categorical annotation missing: {col}')
        series = obs[col].astype('string')
        values = series[assigned].to_numpy()
        groups = assignment[assigned]
        cats = sorted(series.dropna().unique().tolist())
        if len(cats)>spec.get('max_categories', 30):
            raise ValueError(f'Too many categories in {col}: {len(cats)}')
        valid_counts = np.zeros(n_cells, dtype=np.int64)
        matrix = np.zeros((n_cells, len(cats)), dtype=np.int64)
        for i, category in enumerate(cats):
            mask = series.eq(category).fillna(False).to_numpy(bool) & assigned
            matrix[:, i] = np.bincount(assignment[mask], minlength=n_cells)
            valid_counts += matrix[:, i]
        out[f'geometric__{col}__n_valid_bins'] = valid_counts
        for i, category in enumerate(cats):
            safe = ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in str(category))
            key = f'geometric__{col}__fraction__{safe}'
            if key in out:
                raise ValueError('Two category names collide after filename-safe normalization.')
            out[key] = np.divide(matrix[:, i], valid_counts, out=np.full(n_cells,np.nan), where=valid_counts>0)
        dominant = np.full(n_cells, 'missing', dtype=object)
        if cats:
            max_count = matrix.max(axis=1)
            ties = (matrix==max_count[:, None]).sum(axis=1)>1
            has = valid_counts>0
            dominant[has] = np.asarray(cats, dtype=object)[matrix.argmax(axis=1)[has]]
            dominant[has & ties] = 'mixed'
            fractions = np.divide(matrix, valid_counts[:, None], out=np.zeros_like(matrix,dtype=float),
                                  where=valid_counts[:, None]>0)
            if not np.allclose(fractions[has].sum(axis=1), 1):
                raise ValueError('Categorical fractions do not sum to one.')
        out[f'geometric__{col}__dominant'] = pd.Categorical(dominant)
    for col in spec.get('boolean', []):
        if col not in obs:
            raise KeyError(col)
        s = obs[col]
        if not pd.api.types.is_bool_dtype(s):
            raise ValueError(f'{col} must be a real boolean dtype; strings are not cast with bool().')
        valid = s.notna().to_numpy() & assigned
        num = np.bincount(assignment[valid], weights=s[valid].astype(float), minlength=n_cells)
        den = np.bincount(assignment[valid], minlength=n_cells)
        out[f'geometric__{col}__fraction_true'] = np.divide(num, den, out=np.full(n_cells,np.nan), where=den>0)
        out[f'geometric__{col}__n_valid_bins'] = den
    for col, modes in spec.get('continuous', {}).items():
        if col not in obs:
            raise KeyError(col)
        if set(modes)-{'sum','mean','median'}:
            raise ValueError('Continuous aggregations: sum, mean, median only.')
        frame = pd.DataFrame({'group': assignment[assigned],
                              'value': pd.to_numeric(obs.loc[assigned,col],errors='raise').to_numpy()})
        group = frame.groupby('group')['value']
        den = group.count().reindex(range(n_cells), fill_value=0)
        out[f'geometric__{col}__n_valid_bins'] = den.to_numpy()
        for mode in modes:
            values = group.sum(min_count=1) if mode=='sum' else getattr(group,mode)()
            out[f'geometric__{col}__{mode}'] = values.reindex(range(n_cells)).to_numpy()
    return out


def cycle_scores(a, settings):
    """Optional species-specific, user-supplied gene-symbol lists. Counts X stays unchanged."""
    if not settings['enabled']:
        return
    import anndata as ad
    import scanpy as sc
    genes = json.loads(Path(settings['gene_lists_json']).read_text())
    symbols = a.var['gene_symbol'].astype(str)
    if not symbols.is_unique:
        raise ValueError('Cell-cycle scoring needs unique symbols; provide a reviewed disambiguation.')
    lists = {key: [x for x in genes[key] if x in set(symbols)] for key in ('S','G2M')}
    if min(map(len, lists.values())) < settings['min_matched_genes_per_phase']:
        raise ValueError('Too few matched species-specific cell-cycle genes.')
    work = ad.AnnData(X=a.X.astype(np.float32, copy=True), var=pd.DataFrame(index=symbols))
    sc.pp.normalize_total(work, target_sum=1e4); sc.pp.log1p(work)
    sc.tl.score_genes_cell_cycle(work, s_genes=lists['S'], g2m_genes=lists['G2M'], random_state=42)
    for key in ('S_score','G2M_score','phase'):
        a.obs[key] = work.obs[key].to_numpy()
    a.uns['cell_cycle'] = {'method': 'Scanpy score_genes_cell_cycle, log1p normalization on separate copy',
                           'gene_lists_json': str(settings['gene_lists_json']),
                           'n_S_genes': len(lists['S']), 'n_G2M_genes': len(lists['G2M'])}


def build_cells(cfg, spec):
    import anndata as ad
    p = sample_paths(cfg, spec); mode = cfg['cells']['mode']
    if mode=='none':
        return None
    geom = json.loads(p['geometry'].read_text())
    if mode=='legacy':
        source_files = [Path(spec['legacy_cells_h5ad']), Path(spec['legacy_cell_polygons'])]
    else:
        from .segmentation import proseg_paths
        files = proseg_paths(p['proseg'])
        source_files = [files[k] for k in ('counts','cell_metadata','gene_metadata','polygons')]
    sig = digest({'version': VERSION, 'sources': [file_stamp(x) for x in source_files],
                  'bin_qc': file_stamp(p['bin_obs']), 'geometry': file_stamp(p['geometry']),
                  'settings': cfg['cells'], 'transfer': cfg['metadata_transfer'],
                  'cycle': cfg['cell_cycle'],
                  'coordinate_mode': spec.get('legacy_cell_coordinate_mode'),
                  'explicit_affine': spec.get('legacy_cell_to_fullres_affine'),
                  'overrides': spec.get('cell_qc_overrides', {}),
                  'cycle_lists': file_stamp(cfg['cell_cycle']['gene_lists_json']) if cfg['cell_cycle']['enabled'] else None})
    marker = p['work']/'05_cells_complete.json'; required = [p['cells'],p['polygons']]
    if reusable(marker, sig, required):
        return pd.read_csv(p['report']/'cell_qc_summary.csv')
    with stage_lock(p['work'], 'cells'):
        if mode=='legacy':
            a = ad.read_h5ad(spec['legacy_cells_h5ad'])
            layer = cfg['cells']['legacy_counts_layer']
            a.X = integer_csr(a.layers[layer] if layer else a.X)
            # .raw and expression layers are not assumed to have the same count semantics.
            a.raw = None
            for key in list(a.layers):
                del a.layers[key]
            idcol = spec.get('legacy_cell_id_column', 'cell')
            ids = a.obs[idcol].astype(str) if idcol else pd.Series(a.obs_names, index=a.obs_names)
            a.obs['source_obs_name'] = a.obs_names.astype(str)
            a.obs_names = pd.Index(ids.astype(str)); a.obs_names.name = None
            g = read_polygons(spec['legacy_cell_polygons'], spec.get('legacy_polygon_id_column','cell'))
            cell_mode = spec['legacy_cell_coordinate_mode']
        else:
            a, g = read_durable_proseg(files, cfg['cells']['proseg_count_orientation'])
            cell_mode = 'image_xy_um'
        if a.n_obs == 0 or a.n_vars == 0:
            raise ValueError('Cell input is empty; no nonempty cell-by-gene matrix was supplied.')
        if not a.obs_names.is_unique or set(g.index)!=set(a.obs_names):
            raise ValueError('Cell table and polygons must have a one-to-one identical ID set.')
        g = g.loc[a.obs_names].copy()
        _, refvar = h5_metadata(p['bins'])
        a = canonical_features(a, refvar, cfg['cells']['feature_mapping'])
        matrix = proseg_to_fullres(cell_mode, geom, spec.get('legacy_cell_to_fullres_affine'))
        physical_area = g.geometry.area.to_numpy() if cell_mode in ('legacy_sr_row_col_um','image_xy_um') else None
        if 'spatial' not in a.obsm:
            raise KeyError('Cell table lacks obsm[spatial]; specify/recover centroids explicitly.')
        a.obsm['spatial_input'] = np.asarray(a.obsm['spatial'],dtype=np.float64).copy()
        a.obsm['spatial'] = transform_xy(a.obsm['spatial_input'], matrix)
        # Coordinate-like obs columns must not contradict the canonical obsm coordinates.
        for axis, key in enumerate(('centroid_x', 'centroid_y')):
            if key in a.obs:
                a.obs['input_' + key] = a.obs[key].to_numpy()
            a.obs[key] = a.obsm['spatial'][:, axis]
        image_xy_for_qc = transform_xy(a.obsm['spatial'], geom['tenx_to_image'])
        a.obsm['spatial_image_px'] = image_xy_for_qc
        a.obs['image_in_bounds'] = ((image_xy_for_qc[:, 0] >= 0)
            & (image_xy_for_qc[:, 0] < geom['image']['width'])
            & (image_xy_for_qc[:, 1] >= 0)
            & (image_xy_for_qc[:, 1] < geom['image']['height']))
        g = transform_polygons(g, matrix)
        if physical_area is None:
            to_um = np.diag([*geom['image_mpp'],1.]) @ affine(geom['tenx_to_image'])
            physical_area = transform_polygons(g,to_um).geometry.area.to_numpy()
        a.obs['local_cell_id'] = a.obs_names.astype(str)
        global_ids = pd.Index([f"{spec['sample']}::{x}" for x in a.obs_names], name=None)
        a.obs_names = global_ids; g.index = global_ids
        a.obs['sample'] = spec['sample']
        a.obs['dataset_id']=spec.get('dataset_id','')
        a.obs['sample_id']=spec.get('sample_id',spec['sample'])
        a.obs['cell_boundary_available']=True
        for key,value in spec.get('metadata',{}).items():
            if key in ['sample','local_cell_id','total_counts','cell_area_um2','qc_pass']:
                raise ValueError(f'Reserved sample metadata key: {key}')
            a.obs[key] = value
        metrics = qc_metrics(a.X,a.var['gene_symbol']); metrics.index = a.obs_names
        for key in metrics:
            a.obs[key] = metrics[key]
        a.obs['cell_area_um2'] = physical_area
        a.obs['counts_per_um2'] = a.obs.total_counts.to_numpy()/physical_area
        a.obs['equivalent_2um_bins'] = physical_area/4.
        bin_qc_totals = pd.read_parquet(p['bin_obs'], columns=['qc_keep', 'total_counts'])
        available_umis = int(bin_qc_totals.loc[bin_qc_totals.qc_keep, 'total_counts'].sum())
        if int(a.X.sum(dtype=np.int64)) > available_umis:
            raise ValueError('Cell counts exceed all QC-kept source-bin UMIs. Check source sample, '
                             'counts layer, retained-bin scope, or duplicated counts.')
        if mode == 'proseg' and 'original_cell_id' in a.obs:
            prior_audit = pd.read_csv(p['report']/'stardist_prior_audit.csv.gz')
            prior_audit = prior_audit.loc[prior_audit.clean_label_id>0].set_index('clean_label_id')
            prior_ids = pd.to_numeric(a.obs['original_cell_id'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
            for output_key, source_key in {'prior_qc_keep_bins':'n_qc_keep_bins',
                                           'prior_qc_drop_bins':'n_qc_drop_bins',
                                           'prior_raw_label_id':'raw_label_id'}.items():
                a.obs[output_key] = prior_ids.map(prior_audit[source_key]).to_numpy()
        if cfg['metadata_transfer']['enabled']:
            bins = pd.read_parquet(p['bin_obs'])
            if cfg['metadata_transfer']['bin_scope']=='qc_keep_only':
                bins = bins.loc[bins.qc_keep].copy()
            elif cfg['metadata_transfer']['bin_scope']!='all_bins':
                raise ValueError('metadata_transfer.bin_scope: all_bins or qc_keep_only.')
            xy = bins[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(float)
            mapping = geometric_assign(xy, g, cfg['metadata_transfer']['chunk_size'])
            summary = aggregate_annotations(bins, mapping, a.n_obs, cfg['metadata_transfer'])
            summary.index = a.obs_names
            for key in summary:
                a.obs[key] = summary[key]
            if 'geometric__qc_keep__fraction_true' in a.obs:
                a.obs['geometric_qc_keep_fraction'] = a.obs['geometric__qc_keep__fraction_true']
            pd.DataFrame({'barcode':bins.index, 'cell_row_index':mapping}).to_parquet(p['work']/'bin_to_cell_geometric.parquet',index=False)
            dump_json({'n_bins':len(bins),'n_assigned':int((mapping>=0).sum()),
                       'n_ambiguous':int((mapping==-2).sum()),'n_unassigned':int((mapping==-1).sum()),
                       'cell_order_file':str(p['cells']), 'cell_order_sha256':digest(a.obs_names.tolist()),
                       'method':'bin-center intersects; unique only; NOT molecule assignment'},
                      p['report']/'geometric_mapping_summary.json')
        settings = {**cfg['cells']['qc'], **spec.get('cell_qc_overrides',{})}
        tests = {'low_counts': a.obs.total_counts < settings['min_counts'],
                 'low_genes': a.obs.n_genes_by_counts < settings['min_genes']}
        for key, column, comparator in [('max_pct_mt','pct_counts_mt','gt'),
                                       ('min_area_um2','cell_area_um2','lt'),
                                       ('max_area_um2','cell_area_um2','gt')]:
            if settings.get(key) is not None:
                tests[key] = getattr(a.obs[column],comparator)(settings[key])
        fail = np.zeros(a.n_obs, dtype=bool)
        reasons = np.full(a.n_obs, '', dtype=object)
        for key,mask in tests.items():
            vals = np.asarray(mask,bool); a.obs[f'qc_fail__{key}'] = vals; fail |= vals
            reasons[vals] = [f'{v};{key}' if v else key for v in reasons[vals]]
        a.obs['qc_pass'] = ~fail
        a.obs['qc_fail_reason'] = pd.Categorical(np.where(fail,reasons,'pass'))
        cycle_scores(a,cfg['cell_cycle'])
        a.uns.pop('spatialdata_attrs',None)
        a.uns['vhd'] = {'version':VERSION,'counts_location':'X','counts_kind':'integer point-estimate counts',
                        'coordinate_system':'fullres','input_coordinate_mode':cell_mode,
                        'metadata_transfer_is_not_molecule_assignment':True}
        a.obs['region'] = pd.Categorical(['cell_boundaries']*a.n_obs)
        a.obs['instance_id'] = a.obs_names.astype(str)
        write_h5ad(a,p['cells']); g.to_parquet(p['polygons'])
        standard_qc_plots(a.obs,p['report'],'cells',cfg['plots']['dpi'])
        info = geom['image']; thumb = thumbnail(info,cfg['plots']['thumbnail_max_side'])
        image_xy = transform_xy(a.obsm['spatial'],geom['tenx_to_image'])
        scatter_categorical(image_xy,a.obs['qc_pass'],f"{spec['sample']}: cell QC pass",
                            p['report']/'cells_qc_pass_on_HE.png',thumb=thumb,
                            image_size=(info['width'],info['height']),dpi=cfg['plots']['dpi'])
        for column in ('total_counts','n_genes_by_counts','pct_counts_mt','cell_area_um2'):
            scatter_numeric(image_xy,a.obs[column],f"{spec['sample']}: {column}",
                            p['report']/f'cells_{column}_on_HE.png',thumb=thumb,
                            image_size=(info['width'],info['height']),dpi=cfg['plots']['dpi'],
                            log1p=column in ('total_counts','n_genes_by_counts'))
        polygons_image = transform_polygons(g,geom['tenx_to_image'])
        boundary_overlay(polygons_image,info,thumb,p['report']/'cells_boundaries_whole.png',dpi=cfg['plots']['dpi'])
        # User ROIs are original-image level-0 pixels, NOT hires pixels or microns.
        rois = spec.get('rois_image_px',{})
        if not rois:
            cx,cy = np.median(image_xy,axis=0); size=cfg['plots']['roi_size_px']
            x0=max(0,int(cx-size/2)); y0=max(0,int(cy-size/2))
            rois={'center':[x0,y0,min(info['width'],x0+size),min(info['height'],y0+size)]}
        for name,bounds in rois.items():
            boundary_overlay(polygons_image,info,thumb,p['report']/f'cells_boundaries_ROI_{name}.png',
                             bounds=bounds,dpi=cfg['plots']['dpi'])
        summary={'sample':spec['sample'],'n_cells':a.n_obs,'n_qc_pass':int((~fail).sum()),
                 'assigned_umis':int(a.X.sum(dtype=np.int64)),
                 'median_umis':float(a.obs.total_counts.median()),
                 'median_genes':float(a.obs.n_genes_by_counts.median()),
                 'median_area_um2':float(np.median(physical_area))}
        pd.DataFrame([summary]).to_csv(p['report']/'cell_qc_summary.csv',index=False)
        completed(marker,sig,required,**summary)
        del a,g; gc.collect()
    return pd.DataFrame([summary])
