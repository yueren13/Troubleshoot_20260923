import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import geopandas as gpd
from shapely.geometry import box

from vhd.core import (affine, transform_xy, fit_grid_affine, proseg_to_fullres,
                      crop_mask_transform, integer_csr, qc_metrics, canonical_qc,
                      safe_uns, digest, completed, reusable, dataframe_digest)
from vhd.segmentation import gate_prior
from vhd.cells import geometric_assign, aggregate_annotations, transform_polygons
from vhd.palette import BREWER_PLUS, distinct_palette, natural_order
from vhd.integration import consensus_hvg


def test_affine_roundtrip_nontrivial():
    a=affine([2,.1,100,-.2,3,200])
    xy=np.array([[0,0],[4,8],[83.22,70.1]])
    assert np.allclose(transform_xy(transform_xy(xy,a),np.linalg.inv(a)),xy)


@pytest.mark.parametrize('values',[[1,0,0,0,0,0],np.ones((2,3)),[1,0,np.nan,0,1,0]])
def test_bad_affine(values):
    with pytest.raises(ValueError):
        affine(values)


def test_grid_fit_preserves_rotation_and_offset():
    r,c=np.mgrid[:20,:15]; a=affine([7.7,.11,200,.12,7.8,50])
    xy=transform_xy(np.c_[c.ravel(),r.ravel()],a)
    fit,report=fit_grid_affine(r.ravel(),c.ravel(),xy,tolerance_px=1e-8)
    assert np.allclose(fit,a)
    assert report['max_residual_px']<1e-8


def test_nonaffine_grid_rejected():
    r,c=np.mgrid[:10,:10];xy=np.c_[c.ravel(),r.ravel()].astype(float)
    xy[-1,0]+=3
    with pytest.raises(ValueError,match='not sufficiently affine'):
        fit_grid_affine(r.ravel(),c.ravel(),xy,tolerance_px=.1)


def test_legacy_swapped_coordinates_are_not_xy():
    geom={'source_mpp':.25,'image_mpp':[.249,.249],'tenx_to_image':np.eye(3).tolist()}
    matrix=proseg_to_fullres('legacy_sr_row_col_um',geom)
    assert np.array_equal(transform_xy([[20,50]],matrix),[[200,80]])
    new=proseg_to_fullres('image_xy_um',geom)
    assert np.allclose(transform_xy([[.249*200,.249*80]],new),[[200,80]])


def test_resize_transform_actual_dimensions_and_centers():
    a=crop_mask_transform(10,20,101,83,50,40)
    assert np.isclose(a[0,0],101/50)
    assert np.allclose(transform_xy([[(50-1)/2,(40-1)/2]],a),[[10+50,20+41]])
    b=crop_mask_transform(10,20,100,80,100,80)
    assert np.array_equal(transform_xy([[0,0],[99,79]],b),[[10,20],[109,99]])


def test_integer_counts_and_qc_correct_for_zero_rows():
    x=sparse.csr_matrix([[1,2,0],[0,0,0],[3,0,1]])
    out=qc_metrics(x,['mt-Co1','Rplp0','Actb'])
    assert out.total_counts.tolist()==[3,0,4]
    assert out.n_genes_by_counts.tolist()==[2,0,2]
    assert np.allclose(out.pct_counts_mt,[100/3,0,75])
    assert np.allclose(out.pct_counts_ribo,[200/3,0,0])


@pytest.mark.parametrize('x',[[[.5,1]],[[-1,0]],[[np.nan,1]],[[np.inf,0]]])
def test_bad_counts_rejected(x):
    with pytest.raises(ValueError):
        integer_csr(sparse.csr_matrix(x))


def test_duplicate_sparse_entries_count_as_one_gene():
    x=sparse.csr_matrix((np.array([2,3,0]),np.array([0,0,1]),np.array([0,3])),shape=(1,2))
    y=integer_csr(x)
    assert y.nnz==1 and y[0,0]==5


def test_legacy_typo_not_changed_biologically():
    s=canonical_qc(['tissue-high_trancript-low','tissue-low_transcript-high',None])
    assert s.tolist()==['tissue-high_transcript-low','tissue-low_transcript-high','unassessed']
    with pytest.raises(ValueError):
        canonical_qc(['tissue-med_transcript-high'])


def test_prior_gating_preserves_whole_supported_nucleus(tmp_path):
    mask=np.array([[0,1,1,0],[0,1,1,2],[3,3,0,2]],dtype=np.uint32)
    xy=np.array([[1,0],[3,1],[0,2],[3,2]],float)
    keep=np.array([True,False,True,False])
    dest=tmp_path/'prior.npy'
    audit=gate_prior(mask,xy,keep,dest)
    cleaned=np.load(dest)
    assert set(np.unique(cleaned))=={0,1,2}
    assert np.all(cleaned[mask==1]==1)
    assert np.all(cleaned[mask==2]==0)
    assert np.all(cleaned[mask==3]==2)
    assert audit.n_qc_keep_bins.tolist()==[1,0,1]
    assert audit.n_qc_drop_bins.tolist()==[0,2,0]


def test_geometric_assignment_ambiguous_shared_boundary_excluded():
    g=gpd.GeoDataFrame(geometry=[box(0,0,2,2),box(2,0,4,2)])
    xy=np.array([[1,1],[3,1],[2,1],[5,1],[0,0]])
    result=geometric_assign(xy,g,chunk_size=2)
    assert result.tolist()==[0,1,-2,-1,0]


def test_geometry_transform_area_and_coordinate_conversion():
    g=gpd.GeoDataFrame({'cell':['0']},geometry=[box(0,0,2,3)],index=['0'])
    a=affine([0,4,0,4,0,0]);out=transform_polygons(g,a)
    assert out.geometry.area.iloc[0]==96
    assert out.crs is None


def test_metadata_fractions_missing_denominators_and_ties():
    obs=pd.DataFrame({'qc_class':['a','b','b',None],'qc_keep':pd.Series([True,False,True,pd.NA],dtype='boolean'),
                      'score':[1.,3.,np.nan,np.nan]})
    assignment=np.array([0,0,1,1])
    out=aggregate_annotations(obs,assignment,3,{'categorical':['qc_class'],'boolean':['qc_keep'],
                                              'continuous':{'score':['mean','sum','median']}})
    assert out['geometric__qc_class__dominant'].astype(str).tolist()==['mixed','b','missing']
    assert np.allclose(out['geometric__qc_keep__fraction_true'][:2],[.5,1])
    assert np.isnan(out['geometric__qc_keep__fraction_true'][2])
    assert np.isnan(out['geometric__score__sum'][1])
    assert out['geometric__score__mean'][0]==2


def test_boolean_strings_are_not_interpreted_as_true():
    with pytest.raises(ValueError,match='real boolean'):
        aggregate_annotations(pd.DataFrame({'flag':['False','True']}),np.array([0,0]),1,{'boolean':['flag']})


def test_palette_no_silent_recycling():
    assert len(BREWER_PLUS)==41 and len(set(BREWER_PLUS))==41
    assert distinct_palette(3)==BREWER_PLUS[:3]
    with pytest.raises(ValueError):
        distinct_palette(42,allow_extension=False)
    with pytest.warns(UserWarning):
        extra=distinct_palette(45)
    assert extra[:41]==BREWER_PLUS and len(set(extra))==45
    assert natural_order(['10','1','2'])==['1','2','10']


def test_consensus_hvg_balances_samples():
    index=pd.Index(['g1','g2','g3','g4'],name='gene_id')
    result=consensus_hvg({'s1':pd.Series([0,1,np.nan,np.nan],index=index),
                          's2':pd.Series([np.nan,1,0,np.nan],index=index)},2)
    assert result.index[0]=='g2'
    assert result.selected_hvg.sum()==2
    assert result.loc['g4','selected_hvg']==False


def test_serialization_lists_of_dicts_are_safe_mapping():
    out=safe_uns({'reports':[{'a':1},{'b':None}]})
    assert out['reports']['item_0000']=={'a':1}
    assert out['reports']['item_0001']=={'b':''}


def test_stage_reuse_checks_modification_and_signatures(tmp_path):
    output=tmp_path/'one.txt';output.write_text('data');marker=tmp_path/'done.json'
    completed(marker,'abc',[output])
    assert reusable(marker,'abc',[output])
    with pytest.raises(RuntimeError):
        reusable(marker,'def',[output])
    output.write_text('changed!')
    with pytest.raises(RuntimeError):
        reusable(marker,'abc',[output])


def test_metadata_digest_sensitive_to_index_and_values():
    a=pd.DataFrame({'a':[1,2],'b':['x','y']})
    assert dataframe_digest(a)==dataframe_digest(a.copy())
    assert dataframe_digest(a)!=dataframe_digest(a.iloc[::-1])


def test_float_roundoff_does_not_truncate_count():
    x = sparse.csr_matrix([[0.99999999, 2.00000001]])
    assert integer_csr(x).toarray().tolist() == [[1, 2]]
