import copy
import csv
import json
from pathlib import Path
import pytest
from vhd.control.manifest import (COLUMNS,read_manifest,normalize_stage,row_plan,validate_row,
                                  as_bool,under,load_project,check_policy,row_config,scientific_row,save_plan)
from vhd.control.storage import publish_tree,localize,inventory,join_uri
from vhd.compute.launch import concurrency_budget,worker_env,scvi_command
from vhd.compute.clustering import seed_kw
from vhd.core import file_stamp

BASE=Path(__file__).parents[1]
def base_row(tmp_path,**kw):
    r=dict.fromkeys(COLUMNS,'')
    r.update(enabled='true',dataset_id='study',sample_id='s01',stage='cell_segmented',goal='cell_annotated',
             cell_input=str(tmp_path/'source'/'cells.h5ad'),counts_source='layers:counts',
             cell_coordinate_mode='unregistered',output_object_dir=str(tmp_path/'objects'),
             output_plot_dir=str(tmp_path/'plots'),output_data_dir=str(tmp_path/'data'),tmp_dir=str(tmp_path/'scratch'))
    r.update(kw); return r

def write_csv(tmp_path,rows):
    p=tmp_path/'samples.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
    return p

@pytest.mark.parametrize('s,expected',[('raw','raw'),('bin_inputed','bin_imported'),('BIN_QCED','bin_QCed'),('Cell_Annotated','cell_annotated')])
def test_stage_alias(s,expected): assert normalize_stage(s)==expected

def test_unknown_stage():
    with pytest.raises(ValueError): normalize_stage('almost_done')

def test_cell_only_plan_never_starts_segmentation(tmp_path):
    r=read_manifest(write_csv(tmp_path,[base_row(tmp_path)]))[0]
    assert r['_plan']==['import_cells','assemble','publish']

def test_annotated_export_only_no_matrix_source_needed(tmp_path):
    r=base_row(tmp_path,stage='cell_annotated',goal='',counts_source='',cell_coordinate_mode='')
    out=read_manifest(write_csv(tmp_path,[r]))[0]
    assert out['_plan']==['export_existing']

def test_raw_to_bins_plan(tmp_path):
    r=base_row(tmp_path,stage='raw',goal='bin_QCed',spaceranger_outs=str(tmp_path/'source'/'outs'),segmentation_image=str(tmp_path/'source'/'he.TIF'))
    out=read_manifest(write_csv(tmp_path,[r]))[0]
    assert out['_plan']==['ingest_raw','bin_qc','assemble','publish']

def test_qc_bins_skip_thresholds(tmp_path):
    r=base_row(tmp_path,stage='bin_QCed',bin_input=str(tmp_path/'source'/'bins.zarr'),segmentation_image=str(tmp_path/'source'/'he.svs'))
    out=read_manifest(write_csv(tmp_path,[r]))[0]
    assert 'reuse_bin_qc' in out['_plan'] and 'bin_qc' not in out['_plan']
    assert 'stardist' in out['_plan']

@pytest.mark.parametrize('val',['TIF','.tif','btf','TIFF'])
def test_image_case(tmp_path,val): assert validate_row(base_row(tmp_path,image_type=val))['image_type']=='tiff'

def test_ambiguous_counts_rejected(tmp_path):
    with pytest.raises(ValueError): read_manifest(write_csv(tmp_path,[base_row(tmp_path,counts_source='auto')]))

def test_remote_scratch_rejected(tmp_path):
    with pytest.raises(ValueError): validate_row(base_row(tmp_path,tmp_dir='s3://bucket/tmp'))

def test_bin_cell_unregistered_rejected(tmp_path):
    with pytest.raises(ValueError): validate_row(base_row(tmp_path,bin_input=str(tmp_path/'source'/'bins.zarr')))

def test_source_output_overlap_rejected(tmp_path):
    with pytest.raises(ValueError): validate_row(base_row(tmp_path,output_object_dir=str(tmp_path/'source')))

def test_no_unrequested_crossstudy_integration(tmp_path):
    r=read_manifest(write_csv(tmp_path,[base_row(tmp_path)]))[0]
    assert r['integration_group']==''

def test_duplicate_global_id_rejected(tmp_path):
    r=base_row(tmp_path)
    with pytest.raises(ValueError): read_manifest(write_csv(tmp_path,[r,r]))

def test_prefix_boundary():
    assert under('s3://b/safe/x','s3://b/safe')
    assert not under('s3://b/safeevil','s3://b/safe')
    assert not under('/safe_bad','/safe')

def test_review_flags_do_not_invalidate_counts(tmp_path):
    r=base_row(tmp_path); r2={**r,'coordinates_reviewed':'true','output_plot_dir':'/new'}
    assert scientific_row(r)==scientific_row(r2)

@pytest.mark.parametrize('name',['samples.four_studies.example.csv','samples.cells_only.example.csv','samples.legacy_migration.example.csv'])
def test_examples_are_valid(name):
    p=load_project(BASE/'config'/name,BASE/'config/settings.g5_24xlarge.example.json')
    assert check_policy(p)
    for r in p['rows']:
        c,s=row_config(p,r)
        assert s['sample']==r['sample_key'] and c['proseg']['threads_per_sample']==20

def test_publication_local_commit_reuse_and_immutability(tmp_path):
    src=tmp_path/'payload';src.mkdir();(src/'part').mkdir();(src/'part'/'data').write_bytes(b'counts')
    dst=tmp_path/'dest';policy={}
    preview=publish_tree(src,str(dst),policy,execute=False)
    assert preview['files']==1 and not dst.exists()
    out=publish_tree(src,str(dst),policy,execute=True)
    assert out['executed'] and (dst/'_SUCCESS.json').exists() and (dst/'_MANIFEST.json').exists()
    assert publish_tree(src,str(dst),policy,execute=True)['reused']
    (src/'part'/'data').write_bytes(b'changed')
    with pytest.raises(FileExistsError): publish_tree(src,str(dst),policy,execute=True)

def test_same_size_tampered_local_publication_rejected(tmp_path):
    src=tmp_path/'src';src.mkdir();(src/'x').write_bytes(b'abc');dst=tmp_path/'dst'
    publish_tree(src,str(dst),{},execute=True);(dst/'x').write_bytes(b'xyz')
    with pytest.raises(IOError): publish_tree(src,str(dst),{},execute=True)

def test_incomplete_prefix_never_overwritten(tmp_path):
    src=tmp_path/'src';src.mkdir();(src/'x').write_text('data')
    dst=tmp_path/'dst';dst.mkdir();(dst/'x').write_text('partial')
    with pytest.raises(FileExistsError): publish_tree(src,str(dst),{},execute=True)

def test_remote_copy_optin(tmp_path):
    with pytest.raises(PermissionError): localize('s3://bucket/private',tmp_path,{'allow_remote_staging':False})

def test_remote_publish_optin(tmp_path):
    src=tmp_path/'src';src.mkdir();(src/'x').write_text('x')
    with pytest.raises(PermissionError): publish_tree(src,'s3://bucket/dest',{},execute=True)

def test_no_symbolic_publication_escape(tmp_path):
    src=tmp_path/'src';src.mkdir();(tmp_path/'secret').write_text('x');(src/'link').symlink_to(tmp_path/'secret')
    with pytest.raises(ValueError): inventory(src)

def test_unknown_ram_forces_pilot(monkeypatch):
    monkeypatch.setattr('vhd.compute.launch.cpu_count',lambda:96)
    p=concurrency_budget(4,20,available_gib=384)
    assert p['workers']==1 and p['usable_cpu_threads']==88

def test_g5_measured_ram_budget(monkeypatch):
    monkeypatch.setattr('vhd.compute.launch.cpu_count',lambda:96)
    assert concurrency_budget(4,20,60,available_gib=384)['workers']==4
    assert concurrency_budget(4,20,120,available_gib=384)['workers']==2

def test_gpu_env_isolated_caches(tmp_path):
    env=worker_env(tmp_path,[2],8)
    assert env['CUDA_VISIBLE_DEVICES']=='2' and env['OMP_NUM_THREADS']=='8'
    assert all(under(env[k],str(tmp_path)) for k in ('TMPDIR','KERAS_HOME','CUPY_CACHE_DIR','TORCH_HOME'))

def test_torchrun_correct_command():
    cmd=scvi_command('/env/python','/scratch/config.json','ddp',4)
    assert '--nproc_per_node=4' in cmd and cmd[1:3]==['-m','torch.distributed.run']
    assert '--module' in cmd

def test_seed_api_adaptation():
    def old(*,random_state=0): pass
    def new(*,rng=None): pass
    assert seed_kw(old,42)=={'random_state':42}
    assert seed_kw(new,42)=={'rng':42}

def test_recursive_directory_fingerprint(tmp_path):
    d=tmp_path/'zarr';d.mkdir();(d/'data').write_bytes(b'x');one=file_stamp(d)
    (d/'data').write_bytes(b'xyz');two=file_stamp(d)
    assert one['directory_inventory_sha256']!=two['directory_inventory_sha256']


def test_synchronous_process_launch_and_resource_log(tmp_path):
    import sys
    from vhd.compute.launch import run_logged
    log=tmp_path/'logs'/'worker.log'
    out=run_logged([sys.executable,'-c','print("worker-finished")'],worker_env(tmp_path/'runtime',[],1),log)
    assert out['status']=='complete' and 'worker-finished' in log.read_text()
    report=json.loads(log.with_suffix('.resources.json').read_text())
    assert report['returncode']==0 and report['elapsed_seconds']>=0


def test_failed_worker_surfaces_saved_error(tmp_path):
    import sys
    from vhd.compute.launch import run_logged
    log=tmp_path/'fail.log'
    with pytest.raises(RuntimeError,match='real-test-failure'):
        run_logged([sys.executable,'-c','raise RuntimeError("real-test-failure")'],
                   worker_env(tmp_path/'runtime',[],1),log)
    assert json.loads(log.with_suffix('.resources.json').read_text())['returncode']!=0


def test_policy_mismatch_is_not_auto_relaxed():
    p=load_project(BASE/'config/samples.cells_only.example.csv',BASE/'config/settings.g5_24xlarge.example.json')
    p['rows'][0]['output_object_dir']='/not-approved/location'
    with pytest.raises(PermissionError): check_policy(p)


def test_metadata_cannot_be_remote(tmp_path):
    with pytest.raises(ValueError,match='local reviewed'):
        validate_row(base_row(tmp_path,metadata_json='s3://private/meta.json'))
