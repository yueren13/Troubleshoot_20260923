"""Stage dispatch, independent of notebook cell state and of biological tool environments."""
from __future__ import annotations
import copy
import json
import shutil
import time
from pathlib import Path
from ..core import dump_json,file_stamp,digest,sample_paths,environment_report
from .manifest import row_config,sample_layout,check_policy
from .storage import localize,free_space_gate,publish_tree,join_uri


def prepare_context(project,row,action):
    check_policy(project)
    cfg,spec=row_config(project,row); p=sample_layout(project,row)
    for key in ('root','work','report','objects','plots','data'): p[key].mkdir(parents=True,exist_ok=True)
    free_space_gate(p['root'],reserve_gib=cfg['storage_policy'].get('min_free_tmp_gib',20))
    # Never stage raw inputs for a standalone cell import/export merely because paths exist in the row.
    needed={'ingest_raw':['spaceranger_outs','segmentation_image'],
            'import_bins':['bin_input','segmentation_image','spaceranger_outs'],
            'import_cells':['cell_input','cell_boundaries','segmentation_image','spaceranger_outs'],
            'export_existing':['cell_input'],
            'bin_qc':[],'reuse_bin_qc':[],'stardist':[],'proseg':[],
            'postprocess_cells':[],'assemble':[]}.get(action,[])
    inputs={}
    for key in needed:
        if row.get(key): inputs[key]=localize(row[key],p['root'],cfg['storage_policy'])
    for source,target in [('spaceranger_outs','outs'),('segmentation_image','image')]:
        if source in inputs: spec[target]=inputs[source]
    spec['_inputs']=inputs
    # Downstream engines use resolved paths recorded in geometry.json; no raw restaging.
    return cfg,spec


def export_existing(cfg,spec):
    from ..adapters.anndata_input import metadata
    row=spec['_row']; p=sample_paths(cfg,spec); source=Path(spec['_inputs']['cell_input'])
    destination=p['objects']/('annotated_input'+('.h5ad' if source.is_file() else '.zarr'))
    marker=p['report']/'export_existing.json'
    sig=digest({'source':file_stamp(source),'kind':row['cell_input_kind'],'table':row['cell_table']})
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text())['signature']!=sig:
            raise FileExistsError('Existing export does not match source. Use a new run_id.')
        return json.loads(marker.read_text())
    obs,var=metadata(source,row['cell_input_kind'],row['cell_table'])
    # Byte-for-byte source copy, retaining corrected X, raw layers, embeddings, graphs and all uns.
    total=sum(f.stat().st_size for f in source.rglob('*') if f.is_file()) if source.is_dir() else source.stat().st_size
    free_space_gate(p['root'],total,cfg['storage_policy']['min_free_tmp_gib'])
    if source.is_dir(): shutil.copytree(source,destination)
    else: shutil.copy2(source,destination)
    obs.to_parquet(p['report']/'exported_cell_obs.parquet'); var.to_csv(p['report']/'exported_cell_var.csv')
    report={'signature':sig,'n_cells':len(obs),'n_genes':len(var),'source':row['cell_input'],
            'expression_normalized_or_reclustered':False,'full_source_copy':str(destination),
            'source_format_preserved':True}
    dump_json(report,marker); return report


def run_action(project,sample_key,action):
    matches=[r for r in project['rows'] if r['sample_key']==sample_key]
    if len(matches)!=1: raise KeyError(sample_key)
    row=matches[0]
    if action not in row['_plan']: return {'sample':sample_key,'action':action,'status':'not_applicable'}
    if action=='publish': return publish_sample(project,row,execute=True)
    cfg,spec=prepare_context(project,row,action)
    p=sample_layout(project,row); status=p['report']/f'status_{action}.json'
    start=time.time()
    dump_json({'sample':sample_key,'action':action,'status':'running','started_unix':start},status)
    try:
        if action=='ingest_raw':
            from ..io import ingest
            result=ingest(cfg,spec)
        elif action=='import_bins':
            from ..adapters.bins import import_bins
            result=import_bins(cfg,spec)
        elif action=='reuse_bin_qc':
            from ..adapters.bins import reuse_qc
            result=reuse_qc(cfg,spec)
        elif action=='bin_qc':
            from ..qc import build_bin_qc
            result=build_bin_qc(cfg,spec)
        elif action=='stardist':
            from ..segmentation import stardist_prior
            result=stardist_prior(cfg,spec)
        elif action=='proseg':
            from ..segmentation import run_proseg
            result=run_proseg(cfg,spec)
        elif action=='postprocess_cells':
            from ..cells import build_cells
            result=build_cells(cfg,spec)
        elif action=='import_cells':
            from ..adapters.cells import import_cells
            result=import_cells(cfg,spec)
        elif action=='assemble':
            from ..adapters.assemble import assemble
            result=assemble(cfg,spec)
        elif action=='export_existing': result=export_existing(cfg,spec)
        else: raise ValueError(action)
        # The CSV source-stage is not edited. Verified progress is a separate auditable product.
        dump_json({'sample':sample_key,'entry_stage':row['stage'],'action':action,'status':'complete',
                   'elapsed_seconds':time.time()-start,'environment':environment_report()},status)
        return {'sample':sample_key,'action':action,'status':'complete'}
    except Exception as exc:
        dump_json({'sample':sample_key,'action':action,'status':'failed','error':repr(exc),
                   'elapsed_seconds':time.time()-start},status)
        raise


def copy_reports(p):
    """Only plot files go to plot output; metadata/tables/logs go to data output."""
    images={'.png','.pdf','.svg','.jpg','.jpeg','.webp','.tif','.tiff'}
    for f in p['report'].rglob('*'):
        if not f.is_file(): continue
        root=p['plots'] if f.suffix.lower() in images else p['data']
        dest=root/f.relative_to(p['report']); dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest)


def publish_sample(project,row,execute=False):
    check_policy(project)
    p=sample_layout(project,row)
    if row['goal']!='export_only' and not (p['report']/'storage_manifest.json').exists():
        raise RuntimeError(f"{row['sample_key']}: assemble and validate the product before publishing.")
    if row['goal']=='export_only' and not (p['report']/'export_existing.json').exists():
        raise RuntimeError('Export-only artifact has not been prepared.')
    for k in ('plots','data','objects'): p[k].mkdir(parents=True,exist_ok=True)
    copy_reports(p)
    result=[]
    for category,col in [('objects','output_object_dir'),('plots','output_plot_dir'),('data','output_data_dir')]:
        target=join_uri(row[col],row['dataset_id'],row['sample_id'],project['settings']['run_id'],category)
        result.append(publish_tree(p[category],target,project['settings']['storage_policy'],execute=execute))
    if execute:
        dump_json({'sample':row['sample_key'],'publications':result},p['root']/'PUBLICATION_RECEIPT.json')
    return result


def integration_config(project,group):
    rows=[r for r in project['rows'] if r['integration_group']==group]
    if len(rows)<2: raise ValueError('An integration group requires at least two explicit sample rows.')
    check_policy(project)
    cfg=copy.deepcopy(project['settings'])
    settings=cfg['integration']
    work=Path(settings['tmp_dir'])/'vhd_integration'/group/cfg['run_id']
    cfg['_integration_work']=str(work); cfg['output_root']=str(work); cfg['work_root']=str(work)
    cfg['scvi']['run_name']=group+'__'+cfg['run_id']
    cfg['samples']=[row_config(project,r)[1] for r in rows]
    cfg['_integration_group']=group
    return cfg


def publish_integration(project,group,execute=False):
    from ..integration import integration_dir
    cfg=integration_config(project,group); work=integration_dir(cfg)
    if not (work/'10_cluster_complete.json').exists(): raise RuntimeError('Finish clustering before publishing integration.')
    roots={k:work/('publish_'+k) for k in ('objects','plots','data')}
    for p in roots.values(): p.mkdir(exist_ok=True)
    for f in work.rglob('*'):
        rel=f.relative_to(work)
        if not f.is_file() or any(part.startswith('publish_') for part in rel.parts) or rel.parts[0]=='staged_hvg': continue
        if f.name.startswith('.') or f.name.endswith('.lock'): continue
        if f.suffix in ('.png','.pdf','.svg'): cat='plots'
        elif f.suffix in ('.h5ad','.npy','.npz','.pt','.ckpt') or 'model' in rel.parts: cat='objects'
        else: cat='data'
        dest=roots[cat]/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dest)
    output=[]
    for cat in roots:
        root=cfg['integration'][{'objects':'output_object_dir','plots':'output_plot_dir','data':'output_data_dir'}[cat]]
        target=join_uri(root,'integration',group,cfg['run_id'],cat)
        output.append(publish_tree(roots[cat],target,cfg['storage_policy'],execute=execute))
    return output
