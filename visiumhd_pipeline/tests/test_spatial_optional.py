"""Run in the TARGET SpatialData environment before any full-resolution real-sample work."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad=pytest.importorskip('anndata')
sd=pytest.importorskip('spatialdata')
pytest.importorskip('dask')
import geopandas as gpd
from shapely.geometry import box
from spatialdata.models import Image2DModel,ShapesModel
from spatialdata.transformations import Identity

from vhd.core import affine,transform_xy,fit_grid_affine
from vhd.storage import make_bin_labels,parse_table,set_fullres_transform,validate_store
from vhd.io import load_table,load_spatial_with_table,table_metadata


def test_two_table_spatial_roundtrip(tmp_path):
    row,col=np.mgrid[:4,:5];row=row.ravel();col=col.ravel()
    grid=affine([2,.1,5,.1,2,4]);xy=transform_xy(np.c_[col,row],grid)
    obs=pd.DataFrame({'array_row':row,'array_col':col,'pxl_col_in_fullres':xy[:,0],
                      'pxl_row_in_fullres':xy[:,1],'qc_keep':True},index=[f'b{i}' for i in range(len(row))])
    geom={'grid_shape':[4,5],'grid_index_to_fullres':grid.tolist(),
          'grid_fit':{'max_residual_px':0},'tenx_to_image':np.eye(3).tolist(),
          'image':{'width':32,'height':32}}
    labels,ids=make_bin_labels(obs,geom)
    a=ad.AnnData(X=sparse.csr_matrix(np.arange(60).reshape(20,3)),obs=obs)
    a.obsm['spatial']=xy;a=parse_table(a,'bins_2um_labels',ids)
    image=Image2DModel.parse(np.zeros((3,32,32),np.uint8),dims=('c','y','x'),c_coords=['r','g','b'],scale_factors=[2])
    image=set_fullres_transform(image,np.eye(3))
    c=ad.AnnData(X=sparse.csr_matrix([[2,3,4],[3,4,5]]),obs=pd.DataFrame({'qc_pass':[True,True]},index=['s::0','s::1']))
    c.obsm['spatial']=np.array([[10,10],[20,20]],float)
    g=gpd.GeoDataFrame(geometry=[box(8,8,12,12),box(18,18,22,22)],index=c.obs_names)
    s=sd.SpatialData(images={'he_original':image},labels={'bins_2um_labels':labels},
                     shapes={'cell_boundaries':ShapesModel.parse(g,transformations={'fullres':Identity()})},
                     tables={'bins_2um':a,'cells':parse_table(c,'cell_boundaries',c.obs_names.astype(str))},
                     attrs={'geometry':geom})
    path=tmp_path/'test.zarr';s.write(path)
    report=validate_store(path)
    assert report['n_bins']==20 and report['n_cells']==2
    only_cells=load_table(path,'cells')
    assert only_cells.shape==(2,3)
    assert (only_cells.X!=c.X).nnz==0
    small=load_spatial_with_table(path,'cells')
    assert set(small.tables)=={'cells'}
    meta,_=table_metadata(path,'bins_2um')
    assert meta.index.equals(obs.index)


def test_zero_gene_annotation_writeback_and_named_loader(tmp_path):
    # Separate tiny store to test the critical zero-variable annotation-table API.
    c=ad.AnnData(X=sparse.csr_matrix([[2,3],[4,5]]),obs=pd.DataFrame({'qc_pass':[True,False]},index=['s::0','s::1']))
    c.obsm['spatial']=np.array([[0,0],[2,0]],float)
    g=gpd.GeoDataFrame(geometry=[box(-1,-1,1,1),box(1,-1,3,1)],index=c.obs_names)
    s=sd.SpatialData(shapes={'cell_boundaries':ShapesModel.parse(g,transformations={'fullres':Identity()})},
                     tables={'cells':parse_table(c,'cell_boundaries',c.obs_names.astype(str))})
    path=tmp_path/'annotation.zarr';s.write(path)
    obs=pd.DataFrame({'included_in_scvi':[True,False],
                      'leiden_0p1':pd.Categorical(['0','not_integrated'])},index=c.obs_names)
    ann=ad.AnnData(X=sparse.csr_matrix((2,0)),obs=obs)
    ann.obsm['X_scVI']=np.array([[1,2,3],[np.nan,np.nan,np.nan]],np.float32)
    ann.obsm['X_umap']=np.array([[4,5],[np.nan,np.nan]],np.float32)
    s=sd.read_zarr(path,selection=('shapes',))
    s.tables['cells_scvi']=parse_table(ann,'cell_boundaries',c.obs_names.astype(str))
    s.write_element('cells_scvi')
    merged=load_table(path,'cells',merge_scvi=True)
    assert merged.shape==(2,2)
    assert (merged.X!=c.X).nnz==0
    assert merged.obs['leiden_0p1'].astype(str).tolist()==['0','not_integrated']
    assert np.isnan(merged.obsm['X_umap'][1]).all()
