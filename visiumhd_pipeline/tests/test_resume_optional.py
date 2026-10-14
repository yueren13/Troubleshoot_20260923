"""Real tiny AnnData/SpatialData tests; skipped when target packages are not installed."""
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
ad=pytest.importorskip('anndata')
sd=pytest.importorskip('spatialdata')
pytest.importorskip('pyarrow')
from vhd.adapters.anndata_input import read_selected
from vhd.control.manifest import COLUMNS,load_project,sample_layout
from vhd.control.runner import run_action
from vhd.io import load_table
BASE=Path(__file__).parents[1]


def project_for(tmp_path,source,stage='cell_segmented',goal='cell_annotated'):
    cfg=json.loads((BASE/'config/settings.g5_24xlarge.example.json').read_text())
    cfg['storage_policy'].update(enforce_allowlists=False,min_free_tmp_gib=0)
    cfg['metadata_transfer']['enabled']=False;cfg['plots']['dpi']=60
    for k in ('control_dir','tmp_dir','output_object_dir','output_plot_dir','output_data_dir'):
        cfg['integration'][k]=str(tmp_path/'integration'/k)
    cfgpath=tmp_path/'settings.json';cfgpath.write_text(json.dumps(cfg))
    r=dict.fromkeys(COLUMNS,'');r.update(enabled='true',dataset_id='tiny',sample_id='one',stage=stage,goal=goal,
        cell_input=str(source),cell_input_kind='anndata',counts_source='layers:counts',spatial_key='spatial',
        cell_coordinate_mode='unregistered',output_object_dir=str(tmp_path/'objects'),
        output_plot_dir=str(tmp_path/'plots'),output_data_dir=str(tmp_path/'data'),tmp_dir=str(tmp_path/'scratch'))
    csvpath=tmp_path/'samples.csv'
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerow(r)
    return load_project(csvpath,cfgpath)


def source_data(path):
    counts=sparse.csr_matrix([[1,0,3],[0,2,1],[2,3,0]],dtype=np.int32)
    a=ad.AnnData(X=np.full((3,3),.123,dtype=np.float32),
      obs=pd.DataFrame({'original_annotation':pd.Categorical(['A','B','A'])},index=['01','02','03']),
      var=pd.DataFrame({'gene_symbol':['Actb','mt-Co1','Rplp0']},index=['g1','g2','g3']))
    a.layers['counts']=counts;a.obsm['spatial']=np.array([[1,2],[2,3],[3,4]],float)
    a.obsm['X_umap']=np.array([[4,5],[5,6],[6,7]],float)
    if path.suffix=='.h5ad': a.write_h5ad(path)
    else: a.write_zarr(path)
    return a


@pytest.mark.parametrize('suffix',['.h5ad','.zarr'])
def test_only_selected_counts_read(suffix,tmp_path):
    p=tmp_path/('source'+suffix);a=source_data(p)
    b=read_selected(p,'anndata',source='layers:counts')
    assert (b.X!=a.layers['counts']).nnz==0
    assert len(b.layers)==0 and b.obs.original_annotation.equals(a.obs.original_annotation)


def test_cell_only_full_adapter_roundtrip(tmp_path):
    src=tmp_path/'source.h5ad';a=source_data(src);project=project_for(tmp_path,src)
    run_action(project,'tiny__one','import_cells')
    run_action(project,'tiny__one','assemble')
    p=sample_layout(project,project['rows'][0]);b=load_table(p['store'],'cells')
    assert b.shape==(3,3) and (b.X!=a.layers['counts']).nnz==0
    assert b.obs.region.astype(str).unique().tolist()==['cell_centroids']
    assert not b.obs.cell_boundary_available.any()
    assert 'bins_2um' not in sd.read_zarr(p['store']).tables
    assert list(b.obs.original_annotation)==list(a.obs.original_annotation)
    original=ad.read_h5ad(src);assert np.array_equal(original.X,a.X)


def test_export_is_byte_preserving(tmp_path):
    src=tmp_path/'source.h5ad';source_data(src);project=project_for(tmp_path,src,'cell_annotated','export_only')
    run_action(project,'tiny__one','export_existing')
    p=sample_layout(project,project['rows'][0]);target=p['objects']/'annotated_input.h5ad'
    assert target.read_bytes()==src.read_bytes()
