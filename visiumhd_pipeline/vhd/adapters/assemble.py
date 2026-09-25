"""Assemble bin+cell, bin-only, or cell-only SpatialData without fabricating geometry."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..core import VERSION,affine,digest,dump_json,file_stamp,sample_paths,stage_lock
from ..storage import make_bin_labels,parse_table,set_fullres_transform,optional_nuclei
from ..images import lazy_rgb
from ..io import table_metadata


def validate_product(store):
    import spatialdata as sd
    import zarr
    from anndata.io import read_elem
    s=sd.read_zarr(str(store),selection=('images','labels','shapes'))
    store=Path(store); results={'tables':[],'expression_matrices_loaded':False}
    if (store/'tables'/'bins_2um').exists():
        from ..storage import validate_store
        results.update(validate_store(store))
    if (store/'tables'/'cells').exists():
        obs,var=table_metadata(store,'cells')
        regions=obs.region.astype(str).unique()
        if len(regions)!=1 or regions[0] not in s.shapes: raise ValueError('Cell table shape relation is missing.')
        g=s.shapes[regions[0]]
        if not obs.index.is_unique or not g.index.is_unique or not np.array_equal(obs.instance_id.astype(str),g.index.astype(str)):
            raise ValueError('Cell instance IDs and shape order differ.')
        root=zarr.open_group(str(store/'tables'/'cells'),mode='r')
        xy=np.asarray(read_elem(root['obsm']['spatial']))
        if xy.shape!=(len(obs),2) or not np.isfinite(xy).all(): raise ValueError('Invalid cell coordinates.')
        if regions[0]=='cell_centroids':
            if not np.allclose(xy,np.column_stack([g.geometry.x,g.geometry.y])): raise ValueError('Centroid geometry drift.')
            if obs.cell_boundary_available.any(): raise ValueError('Centroids mislabeled as measured boundaries.')
        if 'cells' not in results['tables']: results['tables'].append('cells')
        results.update(n_cells=len(obs),n_cell_genes=len(var),cell_region=regions[0])
    if not results['tables']: raise ValueError('No bin or cell expression table.')
    return results


def assemble(cfg,spec):
    import anndata as ad
    import dask
    import geopandas as gpd
    import spatialdata as sd
    from spatialdata.models import Image2DModel,ShapesModel
    from spatialdata.transformations import Identity
    p=sample_paths(cfg,spec)
    have_bins=p['bins'].exists() and p['bin_obs'].exists()
    have_cells=cfg['cells']['mode']!='none' and p['cells'].exists()
    if not have_bins and not have_cells: raise FileNotFoundError('No prepared bins or cells.')
    cellpath=p['work']/'cell_geometry.json'
    ci=json.loads(cellpath.read_text()) if cellpath.exists() else {'region':'cell_boundaries','coordinate_system':'fullres'}
    geom=json.loads(p['geometry'].read_text()) if have_bins else ci.get('geometry',{})
    if (have_bins or ci.get('coordinate_system')=='fullres') and not spec.get('coordinates_reviewed'):
        raise RuntimeError('Review original-image alignment, then set coordinates_reviewed=true in the CSV.')
    inputs=([p['bins'],p['bin_obs'],p['geometry']] if have_bins else [])+([p['cells']] if have_cells else [])
    if have_cells and p['polygons'].exists(): inputs.append(p['polygons'])
    sig=digest({'version':VERSION,'files':[file_stamp(x) for x in inputs],'storage':cfg['storage'],'cell_info':ci})
    marker=p['report']/'storage_manifest.json'
    if p['store'].exists():
        if not marker.exists() or json.loads(marker.read_text()).get('signature')!=sig:
            raise FileExistsError('An existing product is not from this exact input/settings combination.')
        return validate_product(p['store'])
    partial=p['store'].with_name(p['store'].name+'.partial')
    if partial.exists(): raise FileExistsError(f'Inspect incomplete product: {partial}')
    p['store'].parent.mkdir(parents=True,exist_ok=True)
    with stage_lock(p['work'],'assemble'):
        images={}; labels={}; tables={}
        # With unregistered cells, images remain in fullres and cells in sample_native; no overlay is implied.
        if 'image' in geom:
            img=Image2DModel.parse(lazy_rgb(geom['image'],cfg['storage']['image_tile_size']),dims=('c','y','x'),
                                  c_coords=['r','g','b'],scale_factors=cfg['storage']['image_scale_factors'])
            images['he_original']=set_fullres_transform(img,np.linalg.inv(affine(geom['tenx_to_image'])))
        if have_bins:
            bins=ad.read_h5ad(p['bins']); obs=pd.read_parquet(p['bin_obs'])
            if not bins.obs_names.equals(obs.index): raise ValueError('Bin metadata order mismatch.')
            bins.obs=obs; lab,ids=make_bin_labels(obs,geom)
            labels['bins_2um_labels']=lab; tables['bins_2um']=parse_table(bins,'bins_2um_labels',ids)
            norm=json.loads((p['work']/'he_normalization.json').read_text())
            if cfg['storage']['include_normalized_he']:
                if not norm: raise ValueError('No normalization exists; reused QC does not invent normalized H&E.')
                img=Image2DModel.parse(lazy_rgb(geom['image'],cfg['storage']['image_tile_size'],norm),dims=('c','y','x'),
                                      c_coords=['r','g','b'],scale_factors=cfg['storage']['image_scale_factors'])
                images['he_reinhard']=set_fullres_transform(img,np.linalg.inv(affine(geom['tenx_to_image'])))
            if cfg['storage']['include_nuclei_prior']:
                prior=optional_nuclei(cfg,spec,p,geom)
                if prior is not None: labels['nuclei_prior']=prior
        shapes={}
        if have_cells and not have_bins:
            a=ad.read_h5ad(p['cells']); region=ci['region']; space=ci['coordinate_system']
            if region=='cell_boundaries':
                g=gpd.read_parquet(p['polygons'])
                if not g.index.astype(str).equals(a.obs_names): raise ValueError('Boundary ordering mismatch.')
            else:
                g=gpd.GeoDataFrame({'radius':np.ones(a.n_obs,dtype=float)},
                                   geometry=gpd.points_from_xy(a.obsm['spatial'][:,0],a.obsm['spatial'][:,1]),index=a.obs_names)
            shapes[region]=ShapesModel.parse(g,transformations={space:Identity()})
            tables['cells']=parse_table(a,region,a.obs_names.astype(str))
        s=sd.SpatialData(images=images,labels=labels,shapes=shapes,tables=tables,
                        attrs={'vhd_version':VERSION,'sample':spec['sample'],'dataset_id':spec['dataset_id'],
                               'geometry':geom,'source_signature':sig,'coordinate_system':ci.get('coordinate_system','fullres')})
        with dask.config.set(scheduler='threads',num_workers=cfg['storage']['io_workers']):
            s.write(partial,overwrite=False,consolidate_metadata=False)
        del s,images,labels,tables,shapes
        if have_cells and not have_bins: del a,g
        if have_bins: del bins,obs,lab
        gc.collect()
        if have_cells and have_bins:
            s=sd.read_zarr(str(partial),selection=('images','labels','shapes'))
            a=ad.read_h5ad(p['cells']); region=ci['region']; space=ci['coordinate_system']
            if region=='cell_boundaries':
                g=gpd.read_parquet(p['polygons'])
                if not g.index.astype(str).equals(a.obs_names): raise ValueError('Boundary ordering mismatch.')
            else:
                g=gpd.GeoDataFrame({'radius':np.ones(a.n_obs,dtype=float)},
                                   geometry=gpd.points_from_xy(a.obsm['spatial'][:,0],a.obsm['spatial'][:,1]),index=a.obs_names)
            s.shapes[region]=ShapesModel.parse(g,transformations={space:Identity()})
            s.tables['cells']=parse_table(a,region,a.obs_names.astype(str))
            s.write_element([region,'cells'],overwrite=False)
            del s,a,g; gc.collect()
        report=validate_product(partial)
        partial.rename(p['store'])
        report=validate_product(p['store'])
        dump_json({'signature':sig,'path':str(p['store']),'validation':report,
                   'source_stage':spec['_row']['stage'],'version':VERSION},marker)
        return report
