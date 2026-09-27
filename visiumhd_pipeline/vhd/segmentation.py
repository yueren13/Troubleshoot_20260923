from __future__ import annotations

import gc
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (VERSION, affine, completed, digest, dump_json, file_stamp, integer_csr,
                   require_review, reusable, sample_paths, stage_lock, transform_xy)
from .images import resized_crop
from .io import selected_bins


def gate_prior(mask, bin_mask_xy, qc_keep, output, min_keep_bins=1, chunk_rows=256):
    """Retain whole nuclei with >=N QC-kept bin CENTERS; this is not area-overlap integration."""
    if mask.ndim!=2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Nuclear mask must be a 2D integer array.')
    if min_keep_bins < 1:
        raise ValueError('At least one retained bin center must support a nucleus.')
    max_label = int(mask.max())
    if max_label > 20_000_000:
        raise ValueError('Unexpected huge label IDs; relabel sparsely numbered external masks explicitly.')
    area = np.zeros(max_label+1, np.int64)
    for y in range(0, mask.shape[0], chunk_rows):
        a = np.asarray(mask[y:y+chunk_rows]).ravel()
        area += np.bincount(a.astype(np.int64), minlength=max_label+1)
    rc = np.rint(np.asarray(bin_mask_xy)[:, ::-1]).astype(np.int64)
    valid = (rc[:, 0]>=0)&(rc[:, 0]<mask.shape[0])&(rc[:, 1]>=0)&(rc[:, 1]<mask.shape[1])
    labels = np.zeros(len(rc), np.int64)
    labels[valid] = mask[rc[valid, 0], rc[valid, 1]]
    keep = np.bincount(labels[np.asarray(qc_keep, bool)], minlength=max_label+1)
    all_ = np.bincount(labels, minlength=max_label+1)
    retained = (keep>=int(min_keep_bins)) & (area>0)
    retained[0] = False
    lut = np.zeros(max_label+1, dtype=np.uint32)
    lut[retained] = np.arange(1, retained.sum()+1, dtype=np.uint32)
    cleaned = np.lib.format.open_memmap(output, mode='w+', dtype=np.uint32, shape=mask.shape)
    for y in range(0, mask.shape[0], chunk_rows):
        cleaned[y:y+chunk_rows] = lut[mask[y:y+chunk_rows]]
    cleaned.flush()
    audit = pd.DataFrame({'raw_label_id': np.arange(1, max_label+1),
                          'clean_label_id': lut[1:], 'area_px': area[1:],
                          'n_qc_keep_bins': keep[1:], 'n_qc_drop_bins': all_[1:]-keep[1:],
                          'n_qc_sampled_bins': all_[1:], 'retain_for_proseg': retained[1:]})
    return audit


def stardist_prior(cfg, spec):
    require_review(cfg, spec)
    import tensorflow as tf
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D
    p = sample_paths(cfg, spec)
    geom = json.loads(p['geometry'].read_text())
    settings = cfg['stardist']
    sig = digest({'version': VERSION, 'qc': file_stamp(p['bin_obs']), 'geometry': file_stamp(p['geometry']),
                  'settings': settings, 'normalize_input': cfg['he_normalization']['use_for_stardist'],
                  'normalization': file_stamp(p['work']/'he_normalization.json')})
    marker = p['work']/'03_prior_complete.json'
    required = [p['prior'], p['raw_prior'], p['prior_meta'], p['crop']]
    if reusable(marker, sig, required):
        return json.loads(p['prior_meta'].read_text())
    if settings['require_gpu'] and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('This kernel does not expose a TensorFlow GPU. Select the StarDist kernel.')
    for device in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass  # Already initialized; no claim that the allocator was changed.
    with stage_lock(p['work'], 'stardist'):
        obs = pd.read_parquet(p['bin_obs'])
        xy = transform_xy(obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(), geom['tenx_to_image'])
        keep = obs['qc_keep'].to_numpy(bool)
        if not keep.any():
            raise ValueError('No retained bins for segmentation.')
        mppx, mppy = geom['image_mpp']
        margin = settings['crop_buffer_um']
        left = max(0, int(np.floor(xy[keep, 0].min()-margin/mppx)))
        top = max(0, int(np.floor(xy[keep, 1].min()-margin/mppy)))
        right = min(geom['image']['width'], int(np.ceil(xy[keep, 0].max()+margin/mppx))+1)
        bottom = min(geom['image']['height'], int(np.ceil(xy[keep, 1].max()+margin/mppy))+1)
        target = settings['target_mpp']
        pixels = (right-left)*mppx/target * (bottom-top)*mppy/target
        estimated_gib = pixels*3*4*3/2**30  # image, normalization, inference work; lower-bound heuristic.
        if estimated_gib > settings['max_image_work_gib']:
            raise MemoryError(f'Estimated crop/inference working memory {estimated_gib:.1f} GiB exceeds configured '
                              'budget. Review crop/segmentation strategy; do not silently increase target_mpp.')
        norm = json.loads((p['work']/'he_normalization.json').read_text())
        if not cfg['he_normalization']['use_for_stardist']:
            norm = None
        matrix, shape = resized_crop(geom['image'], (left, top, right, bottom), target, p['crop'], norm or None)
        image = np.load(p['crop'], mmap_mode='r')
        normalized = normalize(image, 1, 99.8, axis=(0, 1)).astype(np.float32, copy=False)
        model = StarDist2D.from_pretrained(settings['model'])
        attempts = []
        for block_size, tiles in settings['attempts']:
            try:
                labels, _ = model.predict_instances_big(
                    normalized, axes='YXC', block_size=int(block_size),
                    min_overlap=settings['min_overlap'], context=settings['context'],
                    n_tiles=tuple(tiles), prob_thresh=settings['prob_thresh'], nms_thresh=settings['nms_thresh'])
                break
            except tf.errors.ResourceExhaustedError as exc:
                attempts.append({'block_size': block_size, 'tiles': tiles, 'error': str(exc)[:1000]})
                gc.collect()
        else:
            raise MemoryError(f'All StarDist inference attempts ran out of memory: {attempts}')
        np.save(p['raw_prior'], np.asarray(labels, dtype=np.uint32))
        del labels, normalized, model; gc.collect()
        mask = np.load(p['raw_prior'], mmap_mode='r')
        mask_xy = transform_xy(xy, np.linalg.inv(matrix))
        audit = gate_prior(mask, mask_xy, keep, p['prior'], settings['min_qc_keep_bins_per_nucleus'])
        if not audit['retain_for_proseg'].any():
            raise ValueError('All nuclei were removed by QC support gating.')
        audit.to_csv(p['report']/'stardist_prior_audit.csv.gz', index=False)
        mask_to_um = np.diag([mppx, mppy, 1.]) @ matrix
        meta = {'sample': spec['sample'], 'mask_index_to_image_px': matrix.tolist(),
                'mask_index_to_proseg_um': mask_to_um.tolist(), 'shape': shape,
                'proseg_coordinate_mode': 'image_xy_um', 'image_bounds': [left, top, right, bottom],
                'n_raw_nuclei': int(len(audit)), 'n_retained_nuclei': int(audit.retain_for_proseg.sum()),
                'inference_oom_attempts': attempts, 'estimated_work_gib': estimated_gib,
                'nominal_target_mpp': target,
                'actual_mask_mpp_xy': [float(mask_to_um[0, 0]), float(mask_to_um[1, 1])]}
        dump_json(meta, p['prior_meta'])
        # A high-resolution center preview for checking seeds against the exact resampled image.
        from skimage.segmentation import find_boundaries
        from .plots import finish
        import matplotlib.pyplot as plt
        center = np.median(mask_xy[keep], axis=0).astype(int)
        size = cfg['plots']['roi_size_px']
        x0, y0 = max(0, center[0]-size//2), max(0, center[1]-size//2)
        rgb = np.asarray(image[y0:y0+size, x0:x0+size])
        clean = np.load(p['prior'], mmap_mode='r')[y0:y0+size, x0:x0+size]
        boundary = np.ma.masked_where(~find_boundaries(clean), find_boundaries(clean).astype(float))
        fig, ax = plt.subplots(figsize=(10, 10)); ax.imshow(rgb); ax.imshow(boundary, alpha=.65)
        ax.set_title(f"{spec['sample']}: QC-retained StarDist prior, crop-center preview")
        finish(fig, p['report']/'stardist_prior_ROI.png', cfg['plots']['dpi'])
        completed(marker, sig, required, n_retained_nuclei=meta['n_retained_nuclei'])
    return meta


def proseg_paths(directory):
    d = Path(directory)
    return {'counts': d/'counts.mtx.gz', 'cell_metadata': d/'cell_metadata.csv.gz',
            'gene_metadata': d/'gene_metadata.csv.gz', 'polygons': d/'cell_polygons.geojson.gz',
            'native': d/'native_proseg.zarr', 'input': d/'input_bins.zarr'}


def check_proseg(executable, directory):
    executable = str(executable)
    version = subprocess.check_output([executable, '--version'], text=True, stderr=subprocess.STDOUT).strip()
    help_text = subprocess.check_output([executable, '--help'], text=True, stderr=subprocess.STDOUT)
    required = ['--anndata', '--anndata-coordinate-key', '--cellpose-masks', '--cellpose-x-transform',
                '--cellpose-y-transform', '--output-counts', '--output-cell-polygons', '--output-cell-metadata',
                '--output-gene-metadata', '--output-spatialdata', '--exclude-spatialdata-transcripts',
                '--nthreads', '--ignore-z-coord', '--voxel-layers', '--diffusion-sigma-near', '--diffusion-sigma-far']
    missing = [x for x in required if x not in help_text]
    if missing:
        raise RuntimeError(f'{version}: required CLI flags missing: {missing}')
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory)/'proseg_help.txt').write_text(help_text)
    return version


def run_proseg(cfg, spec):
    require_review(cfg, spec)
    import anndata as ad
    from .cells import read_durable_proseg
    p = sample_paths(cfg, spec); files = proseg_paths(p['proseg'])
    settings = cfg['proseg']
    version = check_proseg(settings['executable'], p['report'])
    meta = json.loads(p['prior_meta'].read_text()); geom = json.loads(p['geometry'].read_text())
    sig = digest({'version': VERSION, 'proseg_version': version, 'settings': settings,
                  'bins': file_stamp(p['bins']), 'qc': file_stamp(p['bin_obs']),
                  'prior': file_stamp(p['prior']), 'meta': meta})
    marker = p['work']/'04_proseg_complete.json'
    required = [files[x] for x in ('counts', 'cell_metadata', 'gene_metadata', 'polygons')]
    if reusable(marker, sig, required):
        return {'sample': spec['sample'], 'status': 'reused', 'proseg_version': version}
    if not settings['execute']:
        raise RuntimeError('Proseg launch disabled. Review config/priors, then set proseg.execute=true.')
    with stage_lock(p['work'], 'proseg'):
        if files['input'].exists() or files['native'].exists():
            raise FileExistsError('Partial Proseg input/native output exists. Inspect/archive it before retrying.')
        bins = selected_bins(cfg, spec)
        bins.X = integer_csr(bins.X)
        input_umis = int(bins.X.sum(dtype=np.int64))
        physical = np.diag([*geom['image_mpp'], 1.]) @ affine(geom['tenx_to_image'])
        bins.obsm['spatial'] = transform_xy(bins.obsm['spatial'], physical)
        # Minimal portable AnnData for the Rust reader: no SpatialData metadata, no layers.
        input_data = ad.AnnData(X=bins.X, obs=pd.DataFrame(index=bins.obs_names.copy()),
                               var=pd.DataFrame(index=bins.var_names.copy()),
                               obsm={'spatial': bins.obsm['spatial'].astype(np.float32)})
        if not hasattr(ad.settings, 'zarr_write_format'):
            raise RuntimeError('Use an AnnData build exposing settings.zarr_write_format; '
                               'the Proseg interchange is explicitly written as Zarr format 2.')
        with ad.settings.override(zarr_write_format=2):
            input_data.write_zarr(str(files['input']))
        del bins, input_data; gc.collect()
        matrix = np.asarray(meta['mask_index_to_proseg_um'])
        if np.any(matrix[:2].ravel() < -1e-6):
            raise ValueError('Negative mask->Proseg affine coefficient. Review CLI negative-number parsing '
                             'and image orientation explicitly; meaningful negatives are not clipped.')
        xcoef = [f'{float(x):.12g}' for x in matrix[0]]
        ycoef = [f'{float(x):.12g}' for x in matrix[1]]
        cmd = [str(settings['executable']), '--anndata', '--anndata-coordinate-key', 'spatial',
               '--nthreads', str(settings['threads_per_sample']), '--burnin-voxel-size', '2',
               '--voxel-size', '2', '--voxel-layers', '1', '--ignore-z-coord',
               '--diffusion-sigma-near', str(settings['diffusion_sigma_near']),
               '--diffusion-sigma-far', str(settings['diffusion_sigma_far']),
               '--cellpose-masks', str(p['prior']), '--cellpose-x-transform', *xcoef,
               '--cellpose-y-transform', *ycoef, '--output-spatialdata', str(files['native']),
               '--exclude-spatialdata-transcripts', '--output-counts', str(files['counts']),
               '--output-cell-metadata', str(files['cell_metadata']),
               '--output-gene-metadata', str(files['gene_metadata']),
               '--output-cell-polygons', str(files['polygons'])]
        # No arbitrary extra-args list that could accidentally override authoritative output paths.
        for key in ('samples', 'burnin_samples', 'ncomponents'):
            if settings.get(key) is not None:
                cmd += ['--'+key.replace('_', '-'), str(settings[key])]
        cmd.append(str(files['input']))
        (p['report']/'proseg_command.txt').write_text(shlex.join(cmd)+'\n')
        start = time.time()
        with (p['report']/'proseg_run.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        report = {'sample': spec['sample'], 'returncode': result.returncode,
                  'elapsed_seconds': time.time()-start, 'proseg_version': version,
                  'input_umis': input_umis}
        dump_json(report, p['report']/'proseg_exit.json')
        if result.returncode and not settings['accept_valid_outputs_after_nonzero_exit']:
            raise RuntimeError(f'Proseg exited {result.returncode}; see {p["report"]}/proseg_run.log. '
                               'No automatic rerun or silent success.')
        a, _ = read_durable_proseg(files, orientation='cells_by_genes')
        output_umis = int(a.X.sum(dtype=np.int64))
        if output_umis > input_umis:
            raise ValueError('Proseg assigned more integer counts than provided input UMIs.')
        report['assigned_umis'] = output_umis
        report['assigned_fraction'] = output_umis/max(1, input_umis)
        completed(marker, sig, required, **report)
        del a; gc.collect()
        return report


def run_proseg_samples(cfg, specs):
    """Parallel independent CPU processes. Memory estimates are USER-SUPPLIED, not guarantees."""
    workers = int(cfg['proseg']['parallel_samples'])
    threads = int(cfg['proseg']['threads_per_sample'])
    if workers<1 or threads<1 or workers*threads > (os.cpu_count() or 1):
        raise ValueError('parallel_samples * threads_per_sample exceeds detected CPU budget.')
    import psutil
    assumed = cfg['proseg']['estimated_gib_per_sample']
    available = psutil.virtual_memory().available/2**30
    if workers>1 and (assumed is None or workers*float(assumed) > available*0.8):
        raise MemoryError('Set a measured conservative estimated_gib_per_sample before parallel runs, '
                          'or run one sample at a time. Use <=80% currently available RAM.')
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_proseg, cfg, s): s['sample'] for s in specs}
        for future in as_completed(futures):
            result = future.result()  # Propagate errors; a failed sample is never omitted silently.
            print(result); results.append(result)
    return pd.DataFrame(results)
