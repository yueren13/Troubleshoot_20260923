from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

VERSION = '0.2.0-csv-review'
QC_CLASSES = [
    'tissue-high_transcript-high', 'tissue-high_transcript-low',
    'tissue-low_transcript-high', 'tissue-low_transcript-low',
]
KEEP_CLASSES = QC_CLASSES[:2]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False))
    tmp.replace(path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()



def dataframe_digest(df, chunk_rows=250_000):
    """Hash a large metadata frame without expanding millions of uint64 hashes into Python ints."""
    h = hashlib.sha256()
    h.update(str(list(df.columns)).encode())
    h.update(str(list(map(str, df.dtypes))).encode())
    for start in range(0, len(df), chunk_rows):
        values = pd.util.hash_pandas_object(df.iloc[start:start+chunk_rows], index=True)
        h.update(values.to_numpy(np.uint64).tobytes())
    return h.hexdigest()


def file_stamp(path: str | Path, full_hash: bool = False) -> dict:
    """Default size/mtime fingerprint is cheap; full_hash=True reads the entire file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out = {'path': str(p.resolve()), 'size': p.stat().st_size,
           'mtime_ns': p.stat().st_mtime_ns}
    if p.is_dir():
        records=[(f.relative_to(p).as_posix(),f.stat().st_size,f.stat().st_mtime_ns)
                 for f in sorted(p.rglob('*')) if f.is_file()]
        out['directory_inventory_sha256']=digest(records)
        out['file_count']=len(records)
        out['payload_bytes']=sum(r[1] for r in records)
        # This is a recursive size/mtime fingerprint, not a payload checksum.
    if full_hash and p.is_file():
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(chunk)
        out['sha256'] = h.hexdigest()
    return out


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    ids = [s['sample'] for s in cfg['samples']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Sample identifiers must be nonempty and unique.')
    for sample in ids:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', sample):
            raise ValueError(f'Use a filesystem-safe sample ID: {sample!r}')
    for s in cfg['samples']:
        affine(s['tenx_to_image_affine'])
    if cfg['bin_qc']['mode'] not in ('legacy', 'fresh'):
        raise ValueError('bin_qc.mode must be legacy or fresh.')
    if cfg['cells']['mode'] not in ('legacy', 'proseg', 'none'):
        raise ValueError('cells.mode must be legacy, proseg, or none.')
    if cfg['bin_qc']['keep_classes'] != KEEP_CLASSES:
        raise ValueError('This reviewed implementation retains both tissue-high classes. '
                         'Change code and review explicitly to use different rules.')
    return cfg


def sample_paths(cfg: dict, spec: dict) -> dict[str, Path]:
    if '_paths' in spec:
        q = {k: Path(v) for k, v in spec['_paths'].items()}
        work, report, proseg = q['work'], q['report'], q['proseg']
        for p in (work, report, proseg): p.mkdir(parents=True, exist_ok=True)
        return dict(q, bins=work/'bins_raw.h5ad', bin_obs=work/'bins_qc_obs.parquet',
                    geometry=work/'geometry.json', prior=work/'prior_qc.npy', raw_prior=work/'prior_raw.npy',
                    crop=work/'he_crop.npy', prior_meta=work/'prior_metadata.json',
                    cells=work/'cells_qc.h5ad', polygons=work/'cells_fullres.parquet')
    name = spec['sample']
    work = Path(cfg['work_root']) / name
    report = Path(cfg['output_root']) / 'reports' / name
    proseg = work / 'proseg'
    for p in (work, report, proseg): p.mkdir(parents=True, exist_ok=True)
    return dict(work=work, report=report, proseg=proseg,
                bins=work/'bins_raw.h5ad', bin_obs=work/'bins_qc_obs.parquet',
                geometry=work/'geometry.json', prior=work/'prior_qc.npy', raw_prior=work/'prior_raw.npy',
                crop=work/'he_crop.npy', prior_meta=work/'prior_metadata.json',
                cells=work/'cells_qc.h5ad', polygons=work/'cells_fullres.parquet',
                store=Path(cfg['output_root'])/'samples'/f'{name}.zarr')


def require_review(cfg: dict, spec: dict, fresh_qc: bool = True) -> None:
    if not spec.get('coordinates_reviewed', False):
        raise RuntimeError(f"{spec['sample']}: inspect full-resolution alignment plots, "
                           "then set coordinates_reviewed=true in config.")
    if fresh_qc and cfg['bin_qc']['mode'] == 'fresh' and not spec.get('fresh_qc_reviewed', False):
        raise RuntimeError(f"{spec['sample']}: inspect NEW bin QC thresholds/plots, "
                           "then set fresh_qc_reviewed=true. This is not a replica of Step03.")


@contextmanager
def stage_lock(directory: str | Path, stage: str):
    """One writer per stage/sample. A stale lock requires deliberate manual review."""
    Path(directory).mkdir(parents=True,exist_ok=True)
    path = Path(directory)/f'.{stage}.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def reusable(marker: Path, signature: str, required: list[Path]) -> bool:
    if not marker.exists():
        if any(p.exists() for p in required):
            raise RuntimeError(f'Partial/untracked outputs exist near {marker}. '
                               'Move them aside after inspection; automatic overwrites are disabled.')
        return False
    old = json.loads(marker.read_text())
    if old.get('signature') != signature or not all(p.exists() for p in required):
        raise RuntimeError(f'Stale/incomplete stage: {marker}. Use a new work/output root '
                           'or explicitly archive and remove only this stage and its dependents.')
    # Catch later modifications to completed files; size/mtime is not a cryptographic checksum.
    if old.get('outputs') != [file_stamp(p) for p in required]:
        raise RuntimeError(f'Completed output changed: {marker}')
    return True


def completed(marker: Path, signature: str, required: list[Path], **details) -> None:
    dump_json({'version': VERSION, 'signature': signature,
               'outputs': [file_stamp(p) for p in required], **details}, marker)


def integer_csr(x, tolerance: float = 1e-6):
    x = sparse.csr_matrix(x, copy=False)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    for start in range(0, x.nnz, 5_000_000):
        a = x.data[start:start+5_000_000]
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Counts contain negative or non-finite entries.')
        if a.size and np.max(np.abs(a - np.rint(a))) > tolerance:
            raise ValueError('Counts are fractional. Supply raw point counts, not log-normalized '
                             'values or Proseg expected counts. No silent rounding is performed.')
    # Only floating-point roundoff within the explicit tolerance is rounded.
    # Meaningfully fractional expected/normalized counts were rejected above.
    if np.issubdtype(x.data.dtype, np.floating):
        if x.nnz and not np.equal(x.data, np.rint(x.data)).all():
            x = x.copy()
            x.data = np.rint(x.data)
    # int64 sums below avoid row-sum overflow; stored values need only int32 unless very large.
    dtype = np.int64 if (x.nnz and x.data.max() > np.iinfo(np.int32).max) else np.int32
    return x.astype(dtype, copy=False)


def qc_metrics(x, gene_symbols) -> pd.DataFrame:
    x = integer_csr(x)
    symbols = pd.Series(list(gene_symbols), dtype='string').str.upper().fillna('')
    total = np.asarray(x.sum(axis=1, dtype=np.int64)).ravel()
    genes = np.diff(x.indptr).astype(np.int64)
    out = pd.DataFrame({'total_counts': total, 'n_genes_by_counts': genes})
    for label, mask in {'mt': symbols.str.startswith(('MT-', 'MT_')),
                        'ribo': symbols.str.startswith(('RPS', 'RPL'))}.items():
        values = np.asarray(x[:, np.asarray(mask, dtype=bool)].sum(axis=1, dtype=np.int64)).ravel()
        out[f'total_counts_{label}'] = values
        out[f'pct_counts_{label}'] = np.divide(100.0*values, total,
                                                  out=np.zeros(len(total)), where=total>0)
    return out


def canonical_qc(values) -> pd.Series:
    s = pd.Series(values, copy=True).astype('string')
    s = s.str.strip().str.replace('trancript', 'transcript', regex=False)
    bad = sorted(set(s.dropna()) - set(QC_CLASSES) - {'unassessed'})
    if bad:
        raise ValueError(f'Unknown nonmissing QC classes: {bad}. Supply an explicit mapping.')
    return s.fillna('unassessed')


def affine(values) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.shape == (6,):
        a = np.array([[a[0], a[1], a[2]], [a[3], a[4], a[5]], [0, 0, 1.]])
    if a.shape != (3, 3) or not np.isfinite(a).all() or not np.allclose(a[2], [0, 0, 1]):
        raise ValueError('Expected a finite 2D homogeneous affine (six coefficients or 3x3).')
    if abs(np.linalg.det(a[:2, :2])) < 1e-12:
        raise ValueError('Singular coordinate transform.')
    return a


def translate(x, y):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def transform_xy(xy, matrix):
    a = affine(matrix)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Coordinates must be finite N-by-2 x/y values.')
    return xy @ a[:2, :2].T + a[:2, 2]


def fit_grid_affine(rows, cols, fullres_xy, tolerance_px=0.1, seed=42):
    rows, cols = np.asarray(rows), np.asarray(cols)
    xy = np.asarray(fullres_xy, dtype=float)
    if len(xy) < 3:
        raise ValueError('Need at least three non-collinear bin centers.')
    select = np.random.default_rng(seed).choice(len(rows), min(100_000, len(rows)), replace=False)
    design = np.column_stack([cols[select], rows[select], np.ones(len(select))])
    coef, _, rank, _ = np.linalg.lstsq(design, xy[select], rcond=None)
    if rank < 3:
        raise ValueError('Grid positions are collinear; cannot fit an affine.')
    a = np.eye(3); a[:2, :] = coef.T
    max_residual = 0.
    ss = 0.
    for i in range(0, len(rows), 500_000):
        pred = transform_xy(np.column_stack([cols[i:i+500_000], rows[i:i+500_000]]), a)
        err = np.linalg.norm(pred - xy[i:i+500_000], axis=1)
        max_residual = max(max_residual, float(err.max(initial=0)))
        ss += float(np.sum(err**2))
    if max_residual > tolerance_px:
        raise ValueError(f'Grid is not sufficiently affine: maximum residual {max_residual:.6g} px '
                         f'> {tolerance_px}. Do not distort coordinates to fit. '
                         'Use an explicitly reviewed per-bin polygon representation for this sample.')
    return affine(a), {'max_residual_px': max_residual, 'rms_residual_px': float(np.sqrt(ss/len(rows)))}


def proseg_to_fullres(mode: str, geom: dict, explicit=None):
    if mode == 'legacy_sr_row_col_um':
        mpp = float(geom['source_mpp'])
        # legacy Proseg x = SR row*mpp; Proseg y = SR column*mpp.
        return affine([0, 1/mpp, 0, 1/mpp, 0, 0])
    if mode == 'image_xy_um':
        p = np.diag([*geom['image_mpp'], 1.]) @ affine(geom['tenx_to_image'])
        return np.linalg.inv(p)
    if mode == 'fullres_xy_pixels':
        return np.eye(3)
    if mode in ('explicit_affine', 'affine'):
        return affine(explicit)
    raise ValueError(f'Unknown cell coordinate mode: {mode}')


def crop_mask_transform(left, top, source_width, source_height, output_width, output_height):
    """Mask INDEX centers -> original-image INDEX centers, including resize half-pixel offsets."""
    sx, sy = source_width/output_width, source_height/output_height
    return affine([sx, 0, left+(sx-1)/2, 0, sy, top+(sy-1)/2])


def safe_uns(value):
    """Keep arbitrary legacy .uns portable without heterogeneous arrays of dictionaries."""
    if value is None:
        return ''
    if isinstance(value, dict):
        return {str(k): safe_uns(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, (dict, list, tuple)) or v is None for v in value):
            return {f'item_{i:04d}': safe_uns(v) for i, v in enumerate(value)}
        return np.asarray(value) if value else np.array([], dtype=str)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.asarray([str(v) for v in value])
    if isinstance(value, (str, int, float, bool, np.ndarray, pd.DataFrame)):
        return value
    return str(value)


def safe_frame(df):
    df = df.copy()
    df.index = df.index.astype(str)
    df.index.name = None
    for key in df:
        if isinstance(df[key].dtype, pd.StringDtype):
            df[key] = pd.Categorical(df[key].fillna('missing').astype(str))
        elif df[key].dtype == object:
            vals = df[key].dropna()
            if not vals.map(lambda v: isinstance(v, str)).all():
                df[key] = df[key].map(lambda v: '' if v is None else str(v))
            df[key] = pd.Categorical(df[key].fillna('missing').astype(str))
    return df


def write_h5ad(adata, path: Path):
    import anndata as ad
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Automatic overwrite disabled: {path}')
    temp = path.with_name(path.stem + '.partial.h5ad')
    if temp.exists():
        raise FileExistsError(f'Inspect partial output before removing it: {temp}')
    adata.obs, adata.var = safe_frame(adata.obs), safe_frame(adata.var)
    adata.uns = safe_uns(dict(adata.uns))
    adata.write_h5ad(temp, compression='lzf')
    check = ad.read_h5ad(temp, backed='r')
    try:
        if check.shape != adata.shape or not check.obs_names.equals(adata.obs_names):
            raise ValueError('H5AD read-back validation failed.')
    finally:
        check.file.close()
    temp.replace(path)


def environment_report():
    packages = ['numpy', 'pandas', 'scipy', 'anndata', 'zarr', 'spatialdata', 'spatialdata-io',
                'scanpy', 'scvi-tools', 'rapids-singlecell', 'dask-cuda', 'dask', 'distributed', 'fsspec', 's3fs', 'torch', 'tensorflow', 'stardist', 'geopandas', 'shapely',
                'openslide-python', 'xarray', 'igraph', 'leidenalg']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed in this kernel'
    return {'python': sys.version, 'executable': sys.executable, 'packages': versions,
            'pipeline_version': VERSION}
