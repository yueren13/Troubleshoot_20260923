"""Import already-created bins using original coordinate columns, not transformed obsm."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .anndata_input import read_selected
from ..control.manifest import scientific_row
from ..core import (VERSION,affine,canonical_qc,completed,digest,dump_json,file_stamp,fit_grid_affine,
                    integer_csr,qc_metrics,reusable,sample_paths,stage_lock,transform_xy,write_h5ad,KEEP_CLASSES)
from ..images import image_info,thumbnail
from ..plots import standard_qc_plots,scatter_categorical,scatter_numeric


def import_bins(cfg,spec):
    row=spec['_row']; p=sample_paths(cfg,spec)
    source=spec['_inputs']['bin_input']
    sig=digest({'source':file_stamp(source),'row':scientific_row(row),'version':VERSION,'image':file_stamp(spec['image'])})
    marker=p['work']/'01_import_bins_complete.json'
    if reusable(marker,sig,[p['bins'],p['geometry']]): return
    with stage_lock(p['work'],'import_bins'):
        a=read_selected(source,row['bin_input_kind'],row['bin_table'],row['bin_counts_source'],
                        max_matrix_gib=cfg['import']['max_matrix_gib'])
        if not a.obs_names.is_unique or not a.var_names.is_unique: raise ValueError('Duplicate bin/feature IDs.')
        required=['array_row','array_col','pxl_col_in_fullres','pxl_row_in_fullres']
        if not set(required)<=set(a.obs):
            if not spec.get('outs'): raise ValueError('Original coordinate columns missing; supply spaceranger_outs.')
            from ..io import resolve_spaceranger
            sr=resolve_spaceranger(spec)
            pos=pd.read_parquet(sr['positions']).set_index('barcode')
            if not pos.index.is_unique: raise ValueError('Duplicate original position barcodes.')
            pos=pos.reindex(a.obs_names)
            for k in required: a.obs[k]=pos[k]
        if a.obs[required].isna().any().any(): raise ValueError('Missing original full-resolution bin coordinates.')
        for key in ('array_row','array_col'):
            val=a.obs[key].to_numpy(float)
            if np.any(val<0) or not np.equal(val,np.rint(val)).all(): raise ValueError('Invalid grid positions.')
            a.obs[key]=val.astype(np.int64)
        if a.obs[['array_row','array_col']].duplicated().any(): raise ValueError('Duplicate array grid positions.')
        xy=a.obs[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(float)
        grid,residual=fit_grid_affine(a.obs.array_row,a.obs.array_col,xy,cfg['coordinates']['grid_max_residual_px'])
        info=image_info(spec['image'],spec.get('image_mpp_override'))
        mpp=spec.get('source_mpp_override'); scales={}
        if mpp is None and spec.get('outs'):
            from ..io import resolve_spaceranger
            scales=json.loads(resolve_spaceranger(spec)['scales'].read_text()); mpp=scales.get('microns_per_pixel')
        if mpp is None or not np.isfinite(mpp) or float(mpp)<=0:
            raise ValueError('Supply source_mpp or spaceranger_outs; no hires-scale guessing.')
        if 'spatial' in a.obsm: a.obsm['legacy_spatial']=a.obsm['spatial'].copy()
        a.obsm['spatial']=xy; a.obsm['spatial_image_px']=transform_xy(xy,spec['tenx_to_image_affine'])
        a.obs['barcode']=a.obs_names.astype(str); a.obs['sample']=spec['sample']
        feature=row.get('bin_feature_id_column','')
        if feature:
            if feature not in a.var or a.var[feature].isna().any(): raise ValueError(f'Invalid bin feature ID field: {feature}')
            a.var['input_feature_name']=a.var_names.astype(str)
            a.var_names=pd.Index(a.var[feature].astype(str))
            if not a.var_names.is_unique: raise ValueError('Duplicate bin feature IDs.')
        symbol=row.get('bin_gene_symbol_column','') or ('gene_symbol' if 'gene_symbol' in a.var else '')
        a.var['gene_symbol']=a.var[symbol].astype(str) if symbol else a.var_names.astype(str)
        a.var['gene_id']=a.var_names.astype(str)
        if 'legacy_unique_symbol' not in a.var:
            from anndata.utils import make_index_unique
            a.var['legacy_unique_symbol']=make_index_unique(pd.Index(a.var.gene_symbol)).to_numpy()
        metrics=qc_metrics(a.X,a.var.gene_symbol); metrics.index=a.obs_names
        for k in metrics:
            if k in a.obs: a.obs['legacy__'+k]=a.obs[k].to_numpy()
            a.obs[k]=metrics[k]
        geom={'sample':spec['sample'],'image':info,'image_mpp':info['mpp'],'source_mpp':float(mpp),
              'tenx_to_image':affine(spec['tenx_to_image_affine']).tolist(),'grid_index_to_fullres':grid.tolist(),
              'grid_fit':residual,'grid_shape':[int(a.obs.array_row.max())+1,int(a.obs.array_col.max())+1],
              'n_bins':a.n_obs,'n_genes':a.n_vars,'input_total_umis':int(a.X.sum(dtype=np.int64)),
              'source_scalefactors':scales,'bin_scope':'provided bins only; missing raw bins are not reconstructed',
              'source_coordinate_policy':'original coordinate columns; old obsm never assumed full-resolution'}
        a.uns.pop('spatialdata_attrs',None)
        write_h5ad(a,p['bins']); dump_json(geom,p['geometry'])
        completed(marker,sig,[p['bins'],p['geometry']])
        del a; gc.collect()


def reuse_qc(cfg,spec):
    from ..io import h5_metadata
    row=spec['_row']; p=sample_paths(cfg,spec)
    sig=digest({'bins':file_stamp(p['bins']),'qc_column':row['bin_qc_column']})
    marker=p['work']/'02_reuse_qc_complete.json'; norm=p['work']/'he_normalization.json'
    if reusable(marker,sig,[p['bin_obs'],norm]): return
    obs,_=h5_metadata(p['bins']); key=row['bin_qc_column']
    if key not in obs: raise KeyError(f'Stage says QC complete but bin table lacks {key!r}.')
    obs['qc_class_raw']=obs[key].astype('string')
    obs['qc_class']=canonical_qc(obs['qc_class_raw']).to_numpy()
    obs['qc_keep']=obs.qc_class.isin(KEEP_CLASSES).to_numpy(bool)
    obs['qc_method']='reused_input_qc_classes_no_rethreshold'
    geom=json.loads(p['geometry'].read_text())
    xy=transform_xy(obs[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(),geom['tenx_to_image'])
    info=geom['image']; valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<info['width'])&(xy[:,1]<info['height'])
    obs['image_in_bounds']=valid
    if np.any(obs.qc_keep & ~valid): raise ValueError('Retained input QC bins fall outside image. Review registration.')
    obs.to_parquet(p['bin_obs']); dump_json({},norm)
    standard_qc_plots(obs,p['report'],'bins',cfg['plots']['dpi'])
    thumb=thumbnail(info,cfg['plots']['thumbnail_max_side'])
    scatter_categorical(xy,obs.qc_class,'Reused bin QC classes',p['report']/'bins_qc_class_on_HE.png',
                        thumb=thumb,image_size=(info['width'],info['height']),dpi=cfg['plots']['dpi'],
                        max_points=cfg['plots']['max_spatial_points'])
    dump_json({'n_bins':len(obs),'n_qc_keep':int(obs.qc_keep.sum()),'source':'reused; no fresh QC',
               'n_unassessed':int(obs.qc_class.eq('unassessed').sum())},p['report']/'bin_qc_report.json')
    completed(marker,sig,[p['bin_obs'],norm])
