from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from .core import (VERSION, KEEP_CLASSES, canonical_qc, completed, digest, dump_json,
                   file_stamp, dataframe_digest, reusable, sample_paths, stage_lock, transform_xy)
from .images import thumbnail, sample_rgb, lab_stats, reinhard
from .io import h5_metadata, table_metadata
from .plots import standard_qc_plots, scatter_categorical, scatter_numeric, finish


def build_bin_qc(cfg, spec):
    p = sample_paths(cfg, spec)
    geom = json.loads(p['geometry'].read_text())
    legacy_store = Path(spec['legacy_bin_store']) if cfg['bin_qc']['mode']=='legacy' else None
    legacy_table = spec.get('legacy_bin_table', 'square_002um')
    files = [file_stamp(p['bins']), file_stamp(p['geometry'])]
    if legacy_store:
        # Hash the cheap metadata serialization below as well, not just directory mtime.
        legacy, _ = table_metadata(legacy_store, legacy_table)
        col = cfg['bin_qc']['legacy_class_column']
        if col not in legacy:
            raise KeyError(f'Missing {col!r} in {legacy_store}/tables/{legacy_table}')
        legacy.index = legacy.index.astype(str)
        barcode_col = spec.get('legacy_barcode_column')
        if barcode_col:
            legacy.index = legacy[barcode_col].astype(str)
        if not legacy.index.is_unique:
            raise ValueError('Legacy QC barcode identifiers are not unique.')
        legacy_hash = dataframe_digest(legacy)
    else:
        legacy = None; legacy_hash = None
    settings = {**cfg['bin_qc'], **spec.get('bin_qc_overrides', {})}
    normal = cfg['he_normalization']
    reference_stamp = file_stamp(normal['reference_image']) if normal.get('enabled') else None
    sig = digest({'version': VERSION, 'inputs': files, 'qc': settings,
                  'legacy_hash': legacy_hash, 'normalization': normal, 'reference': reference_stamp})
    marker = p['work']/'02_qc_complete.json'
    norm_path = p['work']/'he_normalization.json'
    required = [p['bin_obs'], norm_path]
    if reusable(marker, sig, required):
        return pd.read_parquet(p['bin_obs'])
    with stage_lock(p['work'], 'binqc'):
        obs, _ = h5_metadata(p['bins'])
        xy = obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(float)
        image_xy = transform_xy(xy, geom['tenx_to_image'])
        info = geom['image']
        thumb = thumbnail(info, cfg['plots']['thumbnail_max_side'])
        Image.fromarray(thumb).save(p['report']/'he_original_thumbnail.png')
        normalization = {}
        if normal['enabled']:
            with Image.open(normal['reference_image']) as ref:
                ref = ref.convert('RGB'); ref.thumbnail((3000, 3000))
                target = np.asarray(ref)
            normalization = {'method': 'Reinhard LAB mean/std on foreground',
                             'source': lab_stats(thumb, normal['foreground_od']),
                             'target': lab_stats(target, normal['foreground_od']),
                             'reference': str(normal['reference_image'])}
            norm_thumb = reinhard(thumb, normalization['source'], normalization['target'])
            Image.fromarray(norm_thumb).save(p['report']/'he_reinhard_thumbnail.png')
        # QC uses ORIGINAL H&E, not a normalized image that could change threshold semantics.
        rgb, valid = sample_rgb(info, image_xy, thumb)
        obs['image_in_bounds'] = valid
        obs['he_mean_od'] = (-np.log((rgb.astype(np.float32)+1)/256)).mean(axis=1)
        if legacy is not None:
            aligned = legacy.reindex(obs.index)
            matched = obs.index.isin(legacy.index)
            if not matched.any():
                raise ValueError('No exact barcode overlap with legacy QC; no suffix-stripping heuristics are used.')
            obs['qc_class_raw'] = aligned[settings['legacy_class_column']].astype('string')
            obs['qc_class'] = canonical_qc(obs['qc_class_raw']).to_numpy()
            obs['legacy_qc_available'] = matched
            # Preserve additional original annotations under a namespace; never overwrite coordinates or counts.
            selected_columns = settings.get('legacy_extra_obs_columns', [])
            if selected_columns == 'all':
                selected_columns = [x for x in aligned.columns if x not in obs.columns and x!=settings['legacy_class_column']]
            for column in selected_columns:
                if column not in aligned:
                    raise KeyError(f'Requested legacy annotation missing: {column}')
                obs[f'legacy__{column}'] = aligned[column].to_numpy()
            obs['qc_method'] = 'imported_legacy_classes'
        else:
            od_cut = settings['tissue_od_threshold']
            umi_cut = settings['transcript_umi_threshold']
            if od_cut is None or umi_cut is None:
                raise ValueError('Fresh mode requires explicit tissue_od_threshold and transcript_umi_threshold.')
            tissue = valid & (obs['he_mean_od'].to_numpy() >= float(od_cut))
            tx = obs['total_counts'].to_numpy() >= float(umi_cut)
            obs['qc_class'] = np.char.add(np.where(tissue, 'tissue-high_', 'tissue-low_'),
                                         np.where(tx, 'transcript-high', 'transcript-low'))
            obs['qc_class_raw'] = obs['qc_class']
            obs['qc_method'] = 'NEW_thumbnail_OD_and_UMI_thresholds'
        obs['qc_class'] = pd.Categorical(obs['qc_class'])
        # Keep both transcript-high and transcript-low tissue-high classes; no hidden count cutoff.
        obs['qc_keep'] = obs['qc_class'].isin(KEEP_CLASSES)
        obs['qc_exclusion_reason'] = np.where(obs['qc_keep'], 'retained',
                                             np.where(obs['qc_class'].eq('unassessed'), 'unassessed', 'tissue_low'))
        retained = obs['qc_keep'].to_numpy(bool)
        invalid_retained = int((retained & ~valid).sum())
        if invalid_retained:
            raise ValueError(f'{invalid_retained} QC-kept bins fall outside the original image. '
                             'Fix registration/coordinates instead of silently dropping them.')
        obs.to_parquet(p['bin_obs'])
        dump_json(normalization, norm_path)
        standard_qc_plots(obs, p['report'], 'bins', cfg['plots']['dpi'])
        for key in ['qc_class', 'qc_keep']:
            scatter_categorical(image_xy, obs[key], f"{spec['sample']}: {key}",
                                p['report']/f'bins_{key}_on_HE.png', thumb=thumb,
                                image_size=(info['width'], info['height']), dpi=cfg['plots']['dpi'],
                                max_points=cfg['plots']['max_spatial_points'])
        scatter_numeric(image_xy, obs['total_counts'], f"{spec['sample']}: 2-um UMI counts",
                        p['report']/'bins_counts_on_HE.png', thumb=thumb,
                        image_size=(info['width'], info['height']), log1p=True, dpi=cfg['plots']['dpi'])
        scatter_numeric(np.column_stack([obs['he_mean_od'], np.log1p(obs['total_counts'])]),
                        obs['n_genes_by_counts'], 'Original H&E foreground score vs bin UMI count',
                        p['report']/'bins_HEscore_vs_counts.png',
                        coordinate_labels=('mean optical density (thumbnail approximation)', 'ln(1 + UMI count)'),
                        dpi=cfg['plots']['dpi'])
        summary = {'sample': spec['sample'], 'method': str(obs['qc_method'].iloc[0]),
                   'n_raw_bins': len(obs), 'n_qc_keep': int(retained.sum()),
                   'n_unassessed': int(obs['qc_class'].eq('unassessed').sum()),
                   'n_zero_umi_kept': int((retained & obs['total_counts'].eq(0)).sum()),
                   'qc_kept_umis': int(obs.loc[retained, 'total_counts'].sum()),
                   'classes': {str(k): int(v) for k, v in obs['qc_class'].value_counts().items()},
                   'fresh_method_is_not_legacy_replica': legacy is None}
        dump_json(summary, p['report']/'bin_qc_report.json')
        completed(marker, sig, required, **summary)
    return obs
