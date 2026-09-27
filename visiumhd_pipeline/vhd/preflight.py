from __future__ import annotations
import inspect
import json
from pathlib import Path
import pandas as pd
from .core import dump_json, environment_report, sample_paths
from .io import resolve_spaceranger
from .images import image_info


def preflight(cfg):
    out=Path(cfg['output_root']); out.mkdir(parents=True,exist_ok=True)
    report=environment_report()
    dump_json(report,out/'environment_current_kernel.json')
    rows=[]
    roots=[Path(cfg['work_root']).resolve(),out.resolve()]
    if roots[0]==roots[1]:
        raise ValueError('Use distinct work_root and output_root.')
    for spec in cfg['samples']:
        source=Path(spec['outs']).resolve()
        if any(source==r or r in source.parents or source in r.parents for r in roots):
            raise ValueError('Work/output roots must be independent of Space Ranger source directories.')
        for key in ('legacy_bin_store','legacy_cells_h5ad','legacy_cell_polygons'):
            if spec.get(key):
                old=Path(spec[key]).resolve()
                if any(old==r or old in r.parents for r in roots):
                    raise ValueError('Output must not overwrite or nest within a legacy source file/store.')
        sr=resolve_spaceranger(spec)
        info=image_info(spec['image'],spec.get('image_mpp_override'))
        required=[]
        if cfg['bin_qc']['mode']=='legacy':
            required.append(Path(spec['legacy_bin_store'])/'tables'/spec.get('legacy_bin_table','square_002um'))
        if cfg['cells']['mode']=='legacy':
            required.extend([Path(spec['legacy_cells_h5ad']),Path(spec['legacy_cell_polygons'])])
        missing=[str(p) for p in required if not p.exists()]
        rows.append({'sample':spec['sample'],'raw_matrix':str(sr['matrix']),'width':info['width'],
                     'height':info['height'],'mpp_x':info['mpp'][0],'mpp_y':info['mpp'][1],
                     'coordinates_reviewed':spec.get('coordinates_reviewed',False),
                     'missing_legacy_inputs':' | '.join(missing)})
    frame=pd.DataFrame(rows); frame.to_csv(out/'preflight_samples.csv',index=False)
    if frame.missing_legacy_inputs.ne('').any():
        raise FileNotFoundError(frame[['sample','missing_legacy_inputs']].to_string(index=False))
    return frame,report


def check_spatial_apis():
    import anndata as ad
    import spatialdata as sd
    from anndata.io import read_elem
    from spatialdata.models import TableModel
    required={'spatialdata.read_zarr.selection':'selection' in inspect.signature(sd.read_zarr).parameters,
              'SpatialData.write_element':hasattr(sd.SpatialData,'write_element'),
              'AnnData.to_memory':hasattr(ad.AnnData,'to_memory'),
              'TableModel.parse.region':'region' in inspect.signature(TableModel.parse).parameters}
    if not all(required.values()):
        raise RuntimeError(f'Unsupported API combination: {required}')
    return required
