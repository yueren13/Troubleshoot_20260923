from __future__ import annotations
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (VERSION, affine, digest, dump_json, file_stamp, require_review,
                   safe_frame, safe_uns, sample_paths, stage_lock, transform_xy, translate)
from .images import lazy_rgb
from .io import table_metadata


def scale0(element):
    if hasattr(element,'data') and hasattr(element,'dims') and 'x' in element.dims and 'y' in element.dims:
        return element
    node = element['scale0']
    dataset = node.to_dataset() if hasattr(node,'to_dataset') else node.ds
    if len(dataset.data_vars)!=1:
        raise ValueError('Image scale0 does not have exactly one data variable.')
    return next(iter(dataset.data_vars.values()))


def index_to_intrinsic_transform(element, index_to_fullres):
    """Fit against the PARSED raster's real coordinates, rather than assuming 0 vs 0.5 origins."""
    e = scale0(element)
    x = np.asarray(e.coords['x'],dtype=float)
    y = np.asarray(e.coords['y'],dtype=float)
    sx = float(x[1]-x[0]) if len(x)>1 else 1.
    sy = float(y[1]-y[0]) if len(y)>1 else 1.
    if (len(x)>1 and not np.allclose(np.diff(x),sx)) or (len(y)>1 and not np.allclose(np.diff(y),sy)):
        raise ValueError('Parsed raster has nonuniform intrinsic coordinates.')
    native_to_index = affine([1/sx,0,-x[0]/sx,0,1/sy,-y[0]/sy])
    return affine(index_to_fullres) @ native_to_index


def set_fullres_transform(element, index_to_fullres):
    from spatialdata.transformations import Affine, set_transformation
    a = index_to_intrinsic_transform(element,index_to_fullres)
    set_transformation(element, {'fullres':Affine(a,input_axes=('x','y'),output_axes=('x','y'))}, set_all=True)
    return element


def make_bin_labels(obs, geom):
    import dask.array as da
    from spatialdata.models import Labels2DModel
    height,width = geom['grid_shape']
    if height*width >= np.iinfo(np.uint32).max:
        raise ValueError('Capture grid cannot be represented with unique uint32 bin labels.')
    row = obs.array_row.to_numpy(np.int64); col=obs.array_col.to_numpy(np.int64)
    ids = (row*width+col+1).astype(np.uint32)
    if len(np.unique(ids))!=len(ids):
        raise ValueError('Nonunique bin instance IDs.')
    arr = np.zeros((height,width),dtype=np.uint32); arr[row,col] = ids
    labels = Labels2DModel.parse(da.from_array(arr,chunks=(1024,1024)),dims=('y','x'))
    labels = set_fullres_transform(labels,geom['grid_index_to_fullres'])
    return labels, ids


def parse_table(a, region, instance_ids):
    from spatialdata.models import TableModel
    a.uns = safe_uns(dict(a.uns)); a.uns.pop('spatialdata_attrs',None)
    a.obs, a.var = safe_frame(a.obs),safe_frame(a.var)
    a.obs['region'] = pd.Categorical([region]*a.n_obs)
    # Preserve string object dtype for shapes, integer dtype for raster instance labels.
    a.obs['instance_id'] = np.asarray(instance_ids)
    return TableModel.parse(a,region=region,region_key='region',instance_key='instance_id')


def optional_nuclei(cfg,spec,p,geom):
    import dask.array as da
    from spatialdata.models import Labels2DModel
    from .core import proseg_to_fullres
    if not cfg['storage']['include_nuclei_prior']:
        return None
    if cfg['cells']['mode']=='proseg' and p['prior'].exists():
        meta=json.loads(p['prior_meta'].read_text())
        path=p['prior']
        matrix=np.linalg.inv(affine(geom['tenx_to_image']))@affine(meta['mask_index_to_image_px'])
    elif spec.get('legacy_prior_mask') and spec.get('legacy_prior_metadata'):
        path=Path(spec['legacy_prior_mask'])
        meta=json.loads(Path(spec['legacy_prior_metadata']).read_text())
        matrix=np.eye(3)
        matrix[0]=meta['x_transform']; matrix[1]=meta['y_transform']
        matrix=proseg_to_fullres('legacy_sr_row_col_um',geom)@matrix
    else:
        raise FileNotFoundError('include_nuclei_prior requested, but no prior mask/transform metadata was supplied.')
    array=np.load(path,mmap_mode='r')
    if array.dtype!=np.uint32 or array.ndim!=2:
        raise ValueError('Prior mask must be uint32 2D, 0=background.')
    labels=Labels2DModel.parse(da.from_array(array,chunks=(1024,1024)),dims=('y','x'))
    return set_fullres_transform(labels,matrix)


def assemble_sample(cfg,spec):
    """Create a NEW local store. Bin and cell expression matrices are never in RAM together."""
    require_review(cfg,spec)
    import anndata as ad
    import dask
    import geopandas as gpd
    import spatialdata as sd
    from spatialdata.models import Image2DModel,ShapesModel
    from spatialdata.transformations import Identity
    p=sample_paths(cfg,spec); geom=json.loads(p['geometry'].read_text())
    inputs=[p['bins'],p['bin_obs'],p['geometry']]
    if cfg['cells']['mode']!='none':
        inputs += [p['cells'],p['polygons']]
    if cfg['storage']['include_nuclei_prior']:
        if cfg['cells']['mode']=='proseg':
            inputs += [p['prior'],p['prior_meta']]
        else:
            inputs += [Path(spec['legacy_prior_mask']),Path(spec['legacy_prior_metadata'])]
    sig=digest({'version':VERSION,'files':[file_stamp(x) for x in inputs], 'settings':cfg['storage'],
                'normalization':file_stamp(p['work']/'he_normalization.json')})
    marker=p['report']/'storage_manifest.json'
    if p['store'].exists():
        if not marker.exists() or json.loads(marker.read_text()).get('signature')!=sig:
            raise FileExistsError(f'Output store exists without matching provenance: {p["store"]}')
        return validate_store(p['store'])
    temporary=p['store'].with_name(p['store'].name+'.partial')
    if temporary.exists():
        raise FileExistsError(f'Partial store must be inspected before retry: {temporary}')
    p['store'].parent.mkdir(parents=True,exist_ok=True)
    with stage_lock(p['work'],'assemble'):
        bins=ad.read_h5ad(p['bins'])
        obs=pd.read_parquet(p['bin_obs'])
        if not bins.obs_names.equals(obs.index):
            raise ValueError('Bin QC observation order changed.')
        bins.obs=obs
        bin_labels,ids=make_bin_labels(obs,geom)
        bins=parse_table(bins,'bins_2um_labels',ids)
        data=lazy_rgb(geom['image'],cfg['storage']['image_tile_size'])
        image=Image2DModel.parse(data,dims=('c','y','x'),c_coords=['r','g','b'],
                                 scale_factors=cfg['storage']['image_scale_factors'])
        image=set_fullres_transform(image,np.linalg.inv(affine(geom['tenx_to_image'])))
        images={'he_original':image}; labels={'bins_2um_labels':bin_labels}
        norm=json.loads((p['work']/'he_normalization.json').read_text())
        if cfg['storage']['include_normalized_he']:
            if not norm:
                raise ValueError('Normalized image storage requested but H&E normalization is disabled.')
            data2=lazy_rgb(geom['image'],cfg['storage']['image_tile_size'],norm)
            img2=Image2DModel.parse(data2,dims=('c','y','x'),c_coords=['r','g','b'],
                                    scale_factors=cfg['storage']['image_scale_factors'])
            images['he_reinhard']=set_fullres_transform(img2,np.linalg.inv(affine(geom['tenx_to_image'])))
        nuclei=optional_nuclei(cfg,spec,p,geom)
        if nuclei is not None:
            labels['nuclei_prior']=nuclei
        s=sd.SpatialData(images=images,labels=labels,tables={'bins_2um':bins},
                         attrs={'vhd_version':VERSION,'sample':spec['sample'],'coordinate_system':'fullres',
                                'counts_policy':'raw integer counts in each expression table X',
                                'geometry':geom,'source_signature':sig})
        with dask.config.set(scheduler='threads',num_workers=cfg['storage']['io_workers']):
            s.write(temporary,overwrite=False,consolidate_metadata=False)
        del s,bins,obs,images,labels,data,image,bin_labels; gc.collect()
        if cfg['cells']['mode']!='none':
            # This reads only lazy raster objects; no bin counts, no bin obs.
            s=sd.read_zarr(temporary,selection=('images','labels'))
            cells=ad.read_h5ad(p['cells'])
            g=gpd.read_parquet(p['polygons'])
            if not g.index.astype(str).equals(cells.obs_names):
                raise ValueError('Final cell count/polygon order mismatch.')
            s.shapes['cell_boundaries']=ShapesModel.parse(g,transformations={'fullres':Identity()})
            s.tables['cells']=parse_table(cells,'cell_boundaries',cells.obs_names.astype(str))
            s.write_element(['cell_boundaries','cells'],overwrite=False)
            del s,cells,g; gc.collect()
        validation=validate_store(temporary)
        temporary.rename(p['store'])
        # Re-read after rename: no in-memory Dask graph is retained pointing to the old partial path.
        final_validation=validate_store(p['store'])
        dump_json({'signature':sig,'path':str(p['store']),'validation':final_validation,
                   'version':VERSION,'source_paths':[str(x) for x in inputs]},marker)
    return final_validation


def validate_store(store):
    """Validate both table relations and sampled raster centers WITHOUT reading expression matrices."""
    import spatialdata as sd
    import zarr
    from anndata.io import read_elem
    from spatialdata.transformations import get_transformation
    from spatialdata.models import TableModel
    store=Path(store)
    s=sd.read_zarr(store,selection=('images','labels','shapes'))
    if 'bins_2um_labels' not in s.labels or 'he_original' not in s.images:
        raise ValueError('Missing canonical bins/image elements.')
    obs,var=table_metadata(store,'bins_2um')
    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError('Nonunique bin barcodes/features.')
    if not obs.region.astype(str).eq('bins_2um_labels').all():
        raise ValueError('Bins table region references the wrong element.')
    grid=scale0(s.labels['bins_2um_labels'])
    idx=np.random.default_rng(42).choice(len(obs),min(4096,len(obs)),replace=False)
    r=obs.array_row.to_numpy(np.int64)[idx]; c=obs.array_col.to_numpy(np.int64)[idx]
    arr=grid.data
    labels=arr.vindex[r,c].compute() if hasattr(arr,'vindex') else np.asarray(arr)[r,c]
    if not np.array_equal(labels,obs.instance_id.to_numpy()[idx]):
        raise ValueError('Bin table instance IDs do not identify the correct grid labels.')
    native=np.column_stack([np.asarray(grid.coords['x'])[c],np.asarray(grid.coords['y'])[r]])
    trans=get_transformation(s.labels['bins_2um_labels'],to_coordinate_system='fullres')
    matrix=trans.to_affine_matrix(input_axes=('x','y'),output_axes=('x','y'))
    fullres=transform_xy(native,matrix)
    target=obs.iloc[idx][['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(float)
    error=float(np.linalg.norm(fullres-target,axis=1).max(initial=0))
    geom=s.attrs['geometry']
    if error>max(1e-7,geom['grid_fit']['max_residual_px']+1e-5):
        raise ValueError('Stored raster transformation changed bin positions.')
    root=zarr.open_group(str(store/'tables'/'bins_2um'),mode='r')
    xy=read_elem(root['obsm']['spatial'])
    if not np.array_equal(np.asarray(xy)[idx],target):
        raise ValueError('Bin obsm[spatial] differs from original full-resolution Space Ranger coordinates.')
    # Verify stored image index centers map through the explicitly configured original-image affine.
    img=scale0(s.images['he_original'])
    native_corners=np.array([[float(img.coords['x'][0]),float(img.coords['y'][0])],
                             [float(img.coords['x'][-1]),float(img.coords['y'][-1])]])
    tr=get_transformation(s.images['he_original'],to_coordinate_system='fullres')
    image_to_fullres=tr.to_affine_matrix(input_axes=('x','y'),output_axes=('x','y'))
    mapped=transform_xy(native_corners,image_to_fullres)
    expected=transform_xy([[0,0],[geom['image']['width']-1,geom['image']['height']-1]],
                           np.linalg.inv(affine(geom['tenx_to_image'])))
    if not np.allclose(mapped,expected,atol=1e-6):
        raise ValueError('Stored original-image transformation is inconsistent with level-0 pixels.')
    result={'n_bins':len(obs),'n_bin_genes':len(var),'sampled_bin_center_max_error_px':error,
             'tables':['bins_2um'],'coordinate_system':'fullres',
             'expression_X_not_loaded_by_validation':True}
    del obs,var,xy; gc.collect()
    if (store/'tables'/'cells').exists():
        obs,var=table_metadata(store,'cells')
        region = str(obs.region.astype(str).iloc[0])
        g=s.shapes[region]
        if not obs.index.is_unique or set(obs.instance_id.astype(str))!=set(g.index.astype(str)):
            raise ValueError('Cell table instances do not match cell polygons.')
        if not obs.region.astype(str).eq(region).all():
            raise ValueError('Incorrect cell table region.')
        if len(obs)!=len(g):
            raise ValueError('Cell-table/polygon row count mismatch.')
        result.update(n_cells=len(obs),n_cell_genes=len(var),n_cell_qc_pass=int(obs.qc_pass.sum()))
        result['tables'].append('cells')
    return result
