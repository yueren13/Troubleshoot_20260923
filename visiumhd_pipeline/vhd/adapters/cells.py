"""Standalone segmented AnnData import; polygons optional, biological boundaries never invented."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .anndata_input import read_selected
from ..control.manifest import scientific_row
from ..core import (VERSION,affine,completed,digest,dump_json,file_stamp,proseg_to_fullres,qc_metrics,
                    reusable,sample_paths,stage_lock,transform_xy,write_h5ad)
from ..images import image_info,thumbnail
from ..cells import read_polygons,transform_polygons,geometric_assign,aggregate_annotations,cycle_scores
from ..plots import standard_qc_plots,scatter_numeric,scatter_categorical,boundary_overlay


def geometry_context(cfg,spec):
    p=sample_paths(cfg,spec)
    if p['geometry'].exists(): return json.loads(p['geometry'].read_text())
    geom={'sample':spec['sample'],'source_mpp':spec.get('source_mpp_override'),
          'tenx_to_image':affine(spec['tenx_to_image_affine']).tolist()}
    if spec.get('image'):
        info=image_info(spec['image'],spec.get('image_mpp_override'))
        geom.update(image=info,image_mpp=info['mpp'])
    elif spec.get('image_mpp_override'):
        geom['image_mpp']=spec['image_mpp_override']
    if geom['source_mpp'] is None and spec.get('outs'):
        from ..io import resolve_spaceranger
        geom['source_mpp']=json.loads(resolve_spaceranger(spec)['scales'].read_text()).get('microns_per_pixel')
    return geom


def cell_matrix(mode,geom,explicit=''):
    if mode=='unregistered': return np.eye(3),'sample_native'
    if mode=='legacy_sr_row_col_um' and not geom.get('source_mpp'):
        raise ValueError('Legacy Proseg row/column microns require source_mpp; do not guess.')
    if mode=='image_xy_um' and 'image_mpp' not in geom:
        raise ValueError('image_xy_um requires image MPP on both axes.')
    from ..control.manifest import numbers
    return proseg_to_fullres(mode,geom,numbers(explicit) if explicit else None),'fullres'


def import_cells(cfg,spec):
    p=sample_paths(cfg,spec); row=spec['_row']; source=spec['_inputs']['cell_input']
    poly=spec['_inputs'].get('cell_boundaries')
    inputs=[file_stamp(source)]+([file_stamp(poly)] if poly else [])
    sig=digest({'version':VERSION,'inputs':inputs,'row':scientific_row(row),'settings':cfg['cells'],
                'transfer':cfg['metadata_transfer'],'cell_cycle':cfg['cell_cycle'],
                'metadata':spec.get('metadata',{}),'image':file_stamp(spec['image']) if spec.get('image') else None,
                'bins':file_stamp(p['bin_obs']) if p['bin_obs'].exists() else None})
    marker=p['work']/'05_import_cells_complete.json'; cellgeom=p['work']/'cell_geometry.json'
    required=[p['cells'],cellgeom]+([p['polygons']] if poly else [])
    if reusable(marker,sig,required): return json.loads(cellgeom.read_text())
    with stage_lock(p['work'],'import_cells'):
        a=read_selected(source,row['cell_input_kind'],row['cell_table'],row['counts_source'],
                        max_matrix_gib=cfg['import']['max_matrix_gib'])
        if not a.n_obs or not a.n_vars: raise ValueError('Cell input is empty.')
        if not a.obs_names.is_unique: raise ValueError('Duplicate source cell obs_names.')
        source_obs=a.obs_names.astype(str).to_numpy()
        idcol=row['cell_id_column']
        if idcol and idcol not in a.obs: raise KeyError(f'cell_id_column {idcol!r} is absent.')
        if idcol and a.obs[idcol].isna().any(): raise ValueError('Missing cell IDs.')
        ids=pd.Index(a.obs[idcol].astype(str) if idcol else a.obs_names.astype(str))
        if not ids.is_unique or ids.isna().any() or np.any(ids==''): raise ValueError('Invalid/duplicate cell IDs.')
        a.obs_names=ids
        fid=row['feature_id_column']
        if fid:
            if fid not in a.var or a.var[fid].isna().any(): raise ValueError(f'Invalid feature_id_column {fid}')
            a.var['input_var_name']=a.var_names.astype(str)
            a.var_names=pd.Index(a.var[fid].astype(str))
        if not a.var_names.is_unique: raise ValueError('Feature IDs must be unique; no silent suffixing.')
        sym=row['gene_symbol_column']
        if sym and sym not in a.var: raise KeyError(sym)
        a.var['gene_symbol']=a.var[sym].astype(str) if sym else a.var.get('gene_symbol',pd.Series(a.var_names,index=a.var_names)).astype(str)
        a.var['gene_id']=a.var_names.astype(str)
        coord_key=row['spatial_key']
        if coord_key not in a.obsm: raise KeyError(f'Spatial coordinates absent: obsm[{coord_key!r}]. Use export_only for nonspatial export.')
        geom=geometry_context(cfg,spec)
        matrix,space=cell_matrix(row['cell_coordinate_mode'],geom,row['cell_to_fullres_affine'])
        xy=transform_xy(a.obsm[coord_key],matrix)
        a.obsm['vhd_input_spatial']=np.asarray(a.obsm[coord_key],dtype=np.float64).copy()
        a.obsm['spatial']=xy
        for axis,key in enumerate(('centroid_x','centroid_y')):
            if key in a.obs: a.obs['input__'+key]=a.obs[key].to_numpy()
            a.obs[key]=xy[:,axis]
        g=None; region='cell_centroids'
        if poly:
            if row['boundary_id_column']=='__index__':
                import geopandas as gpd
                from ..cells import clean_local_geometry
                g=clean_local_geometry(gpd.read_parquet(poly)); g.index=g.index.astype(str)
            else: g=read_polygons(poly,row['boundary_id_column'])
            if g.geometry.isna().any() or g.geometry.is_empty.any() or not g.geometry.is_valid.all():
                raise ValueError('Missing, empty, or invalid cell polygons; no automatic geometry repair.')
            if not g.geometry.geom_type.isin(['Polygon','MultiPolygon']).all(): raise ValueError('Boundaries must be Polygon/MultiPolygon.')
            if not g.index.is_unique or set(g.index)!=set(ids):
                raise ValueError('Cell IDs and boundary IDs must have exactly the same one-to-one set.')
            g=transform_polygons(g.loc[ids].copy(),matrix); region='cell_boundaries'
        new_ids=pd.Index([f"{spec['sample']}::{v}" for v in ids],name=None)
        a.obs['vhd_source_obs_name']=source_obs; a.obs['vhd_local_cell_id']=ids.to_numpy()
        a.obs_names=new_ids
        if g is not None: g.index=new_ids
        for k in ('sample','dataset_id','sample_id','total_counts','n_genes_by_counts','pct_counts_mt','pct_counts_ribo','qc_pass'):
            if k in a.obs: a.obs['input__'+k]=a.obs[k].to_numpy()
        a.obs['sample']=spec['sample']; a.obs['sample_id']=row['sample_id']; a.obs['dataset_id']=row['dataset_id']
        for key,val in spec.get('metadata',{}).items():
            if key in a.obs: raise ValueError(f'Metadata override would replace existing obs column {key!r}.')
            a.obs[key]=val
        metrics=qc_metrics(a.X,a.var.gene_symbol); metrics.index=new_ids
        for key in metrics: a.obs[key]=metrics[key]
        a.obs['cell_boundary_available']=g is not None
        if g is not None and space=='fullres' and 'image_mpp' in geom:
            physical=np.diag([*geom['image_mpp'],1.])@affine(geom['tenx_to_image'])
            area=transform_polygons(g,physical).geometry.area.to_numpy()
            a.obs['cell_area_um2']=area
            a.obs['counts_per_um2']=np.divide(a.obs.total_counts,area,out=np.full(a.n_obs,np.nan),where=area>0)
        elif g is not None and row['cell_coordinate_mode']=='legacy_sr_row_col_um':
            area=g.geometry.area.to_numpy()*float(geom['source_mpp'])**2
            a.obs['cell_area_um2']=area
        # If no polygons exist, never derive area from the visualization glyph radius.
        q=cfg['cells']['qc']; keep=(a.obs.total_counts>=q['min_counts'])&(a.obs.n_genes_by_counts>=q['min_genes'])
        for setting,col,op in [('max_pct_mt','pct_counts_mt','le'),('min_area_um2','cell_area_um2','ge'),('max_area_um2','cell_area_um2','le')]:
            if q.get(setting) is not None:
                if col not in a.obs: raise ValueError(f'{setting} cannot be applied without measured cell area.')
                keep &= getattr(a.obs[col],op)(q[setting]) & a.obs[col].notna()
        if row['cell_qc_column']:
            key=row['cell_qc_column']
            if key not in a.obs or not pd.api.types.is_bool_dtype(a.obs[key]):
                raise ValueError('cell_qc_column must name a boolean existing QC-pass column.')
            keep &= a.obs[key]
        a.obs['qc_pass']=np.asarray(keep,bool)
        # Preserve accepted bin QC but do not imply every imported segmentation used that exact bin scope.
        transfer_report={'performed':False,'reason':'no registered bin/polygon pair'}
        if cfg['metadata_transfer']['enabled'] and p['bin_obs'].exists() and g is not None and space=='fullres':
            bins=pd.read_parquet(p['bin_obs'])
            if cfg['metadata_transfer']['bin_scope']=='qc_keep_only': bins=bins.loc[bins.qc_keep].copy()
            mapping=geometric_assign(bins[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(),g,cfg['metadata_transfer']['chunk_size'])
            transferred=aggregate_annotations(bins,mapping,a.n_obs,cfg['metadata_transfer']); transferred.index=a.obs_names
            for key in transferred:
                if key in a.obs: a.obs['input__'+key]=a.obs[key].to_numpy()
                a.obs[key]=transferred[key]
            transfer_report={'performed':True,'method':'unique bin-center geometric intersection; not Proseg transcript assignments',
                             'n_assigned':int((mapping>=0).sum()),'n_ambiguous':int((mapping==-2).sum())}
        if cfg['cell_cycle']['enabled']:
            for key in ('S_score','G2M_score','phase'):
                if key in a.obs: a.obs['input__'+key]=a.obs[key].to_numpy()
            cycle_scores(a,cfg['cell_cycle'])
        if space=='fullres' and 'image' in geom:
            a.obsm['spatial_image_px']=transform_xy(xy,geom['tenx_to_image'])
            image_xy=a.obsm['spatial_image_px']; image=geom['image']
            a.obs['image_in_bounds']=(image_xy[:,0]>=0)&(image_xy[:,1]>=0)&(image_xy[:,0]<image['width'])&(image_xy[:,1]<image['height'])
        a.uns.pop('spatialdata_attrs',None)
        a.uns['vhd_import']={'counts_source':row['counts_source'],'source':row['cell_input'],'coordinate_system':space,
                             'region':region,'boundary_available':g is not None,
                             'expression_layers_not_selected_are_not_copied':True,
                             'original_full_object_is_untouched':True,'version':VERSION}
        a.obs['region']=pd.Categorical([region]*a.n_obs); a.obs['instance_id']=a.obs_names.astype(str)
        write_h5ad(a,p['cells'])
        if g is not None: g.to_parquet(p['polygons'])
        cellinfo={'region':region,'coordinate_system':space,'geometry':geom,'n_cells':a.n_obs,
                  'counts_source':row['counts_source'],'has_boundaries':g is not None,
                  'centroid_radius_is_visualization_only':g is None,'metadata_transfer':transfer_report}
        dump_json(cellinfo,cellgeom)
        standard_qc_plots(a.obs,p['report'],'cells',cfg['plots']['dpi'])
        if 'spatial_image_px' in a.obsm and space=='fullres' and 'image' in geom:
            info=geom['image']; thumb=thumbnail(info,cfg['plots']['thumbnail_max_side'])
            scatter_categorical(a.obsm['spatial_image_px'],a.obs.qc_pass,'Imported cell QC',p['report']/'cells_qc_on_HE.png',
                                thumb=thumb,image_size=(info['width'],info['height']),dpi=cfg['plots']['dpi'])
            if g is not None:
                boundary_overlay(transform_polygons(g,geom['tenx_to_image']),info,thumb,
                                 p['report']/'cells_boundaries_whole.png',dpi=cfg['plots']['dpi'])
        else:
            scatter_categorical(xy,a.obs.qc_pass,'Imported cells: native coordinates (not registered to H&E)',
                                p['report']/'cells_native_coordinates.png',dpi=cfg['plots']['dpi'])
        a.obs.to_parquet(p['report']/'cell_obs.parquet'); a.var.to_csv(p['report']/'cell_var.csv')
        dump_json(transfer_report,p['report']/'import_transfer_report.json')
        completed(marker,sig,required,n_cells=a.n_obs,n_qc_pass=int(keep.sum()))
        del a,g; gc.collect()
    return cellinfo
