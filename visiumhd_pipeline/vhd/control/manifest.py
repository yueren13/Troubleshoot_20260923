"""Fixed-format CSV -> validated per-sample plans. No biological data read on import."""
from __future__ import annotations
import copy
import csv
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from ..core import affine, digest, dump_json

STAGES = ('raw', 'bin_imported', 'bin_QCed', 'cell_segmented', 'cell_annotated')
ALIASES = {s.lower(): s for s in STAGES} | {'bin_inputed': 'bin_imported', 'bin_imported': 'bin_imported'}
GOALS = ('bin_QCed', 'cell_annotated', 'export_only')
# Fixed headers; blank optional values are intentional. Never parse sample IDs as numbers.
COLUMNS = [
    'enabled','dataset_id','sample_id','stage','goal','integration_group',
    'spaceranger_outs','segmentation_image','image_type',
    'bin_input','bin_input_kind','bin_table','bin_counts_source','bin_qc_column',
    'bin_feature_id_column','bin_gene_symbol_column',
    'cell_input','cell_input_kind','cell_table','counts_source',
    'feature_id_column','gene_symbol_column','cell_id_column','spatial_key',
    'cell_coordinate_mode','cell_to_fullres_affine','cell_qc_column',
    'cell_boundaries','boundary_id_column','tenx_to_image_affine',
    'image_mpp_x','image_mpp_y','source_mpp','coordinates_reviewed','fresh_qc_reviewed',
    'output_object_dir','output_plot_dir','output_data_dir','tmp_dir',
    'metadata_json','overrides_json',
]
INPUT_PATHS = ('spaceranger_outs','segmentation_image','bin_input','cell_input','cell_boundaries',
               'metadata_json','overrides_json')
OUTPUT_PATHS = ('output_object_dir','output_plot_dir','output_data_dir','tmp_dir')
KINDS = ('auto','anndata','spatialdata')
COORDS = ('fullres_xy_pixels','image_xy_um','legacy_sr_row_col_um','affine','unregistered')

def as_bool(value, default=False):
    value = str(value).strip().lower()
    if value == '': return default
    if value in ('true','1','yes'): return True
    if value in ('false','0','no'): return False
    raise ValueError(f'Expected true/false, not {value!r}')

def normalize_stage(value):
    try: return ALIASES[str(value).strip().lower()]
    except KeyError: raise ValueError(f'Unknown stage {value!r}; use {STAGES}') from None

def is_remote(value):
    return str(value).lower().startswith('s3://')

def path_value(value, base):
    value = str(value).strip()
    if not value: return ''
    if is_remote(value):
        u = urlsplit(value)
        if not u.netloc or u.query or u.fragment or not u.path.strip('/'):
            raise ValueError(f'Use an explicit S3 bucket/prefix, without query/fragment: {value}')
        if '..' in u.path.split('/'): raise ValueError('Parent traversal is not allowed in S3 keys.')
        return value.rstrip('/')
    if '://' in value: raise ValueError(f'Only local paths and s3:// are supported: {value}')
    p = Path(value).expanduser()
    return str((base/p).resolve() if not p.is_absolute() else p.resolve())

def under(child, root):
    if is_remote(child) != is_remote(root): return False
    if is_remote(child): return child.rstrip('/') == root.rstrip('/') or child.startswith(root.rstrip('/')+'/')
    c,r = Path(child).resolve(),Path(root).resolve()
    return c == r or r in c.parents

def numbers(value, default=None):
    if not str(value).strip(): return default
    s = str(value).strip()
    x = json.loads(s) if s.startswith('[') else [float(v.strip()) for v in s.split(';')]
    return affine(x).tolist()

def row_plan(row):
    """Source stage is an entry point, NOT permission to skip validation."""
    if row['goal'] == 'export_only': return ['export_existing']
    out = []
    has_bins = bool(row.get('bin_input')) or row['stage'] == 'raw'
    if row['stage'] == 'raw': out += ['ingest_raw','bin_qc']
    elif has_bins:
        out += ['import_bins']
        out += ['reuse_bin_qc' if row['stage'] != 'bin_imported' else 'bin_qc']
    if row['goal'] == 'bin_QCed': return out + ['assemble','publish']
    if row['stage'] in ('raw','bin_imported','bin_QCed'):
        out += ['stardist','proseg','postprocess_cells']
    else:
        out += ['import_cells']
    return out + ['assemble','publish']

def validate_row(row):
    for key in ('dataset_id','sample_id'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row[key]):
            raise ValueError(f'{key} must be a nonempty filesystem-safe ID: {row[key]!r}')
    row['stage'] = normalize_stage(row['stage'])
    row['goal'] = row['goal'] or ('export_only' if row['stage']=='cell_annotated' else 'cell_annotated')
    if row['goal'].lower()=='bin_qced': row['goal']='bin_QCed'
    if row['goal'] not in GOALS: raise ValueError(f'Unknown goal: {row["goal"]}')
    for key in ('metadata_json','overrides_json'):
        if is_remote(row[key]): raise ValueError(f'{key} must be a local reviewed JSON configuration file.')
    for key in ('coordinates_reviewed','fresh_qc_reviewed'): as_bool(row[key])
    for key in ('tenx_to_image_affine','cell_to_fullres_affine'):
        if row[key]: numbers(row[key])
    for key in ('image_mpp_x','image_mpp_y','source_mpp'):
        if row[key] and (not __import__('math').isfinite(float(row[key])) or float(row[key])<=0):
            raise ValueError(f'{key} must be finite and positive.')
    for key in OUTPUT_PATHS:
        if not row[key]: raise ValueError(f'{key} is required even if this run produces no such output.')
    if is_remote(row['tmp_dir']): raise ValueError('tmp_dir must be an approved local/POSIX path, not s3://.')
    for key in ('bin_input_kind','cell_input_kind'):
        row[key] = (row[key] or 'auto').lower()
        if row[key] not in KINDS: raise ValueError(f'Invalid {key}: {row[key]}')
    row['image_type'] = (row['image_type'] or 'auto').lower().lstrip('.')
    if row['image_type'] in ('tif','btf'): row['image_type']='tiff'
    if row['image_type'] not in ('auto','tiff','svs'): raise ValueError('image_type: auto, tif/TIF/tiff or svs.')
    row['spatial_key'] = row['spatial_key'] or 'spatial'
    row['bin_qc_column'] = row['bin_qc_column'] or 'qc_class'
    row['bin_counts_source'] = row['bin_counts_source'] or 'X'
    for key in ('counts_source','bin_counts_source'):
        v=row[key]
        if v and v!='X' and not (v.startswith('layers:') and len(v)>7):
            raise ValueError(f'{key}: use X or layers:counts (not an inferred normalized matrix).')
    if row['goal']=='export_only':
        if row['stage']!='cell_annotated' or not row['cell_input']:
            raise ValueError('export_only requires stage=cell_annotated and a cell_input.')
    elif row['stage'] == 'raw':
        if not row['spaceranger_outs'] or not row['segmentation_image']:
            raise ValueError('raw requires spaceranger_outs and segmentation_image.')
    elif row['stage'] in ('bin_imported','bin_QCed'):
        if not row['bin_input'] or not row['segmentation_image']:
            raise ValueError('Bin entry stages require bin_input and segmentation_image.')
    else:
        if row['goal']=='bin_QCed': raise ValueError('Do not backtrack cell stages to bins; add a separate bin job.')
        if not row['cell_input'] or not row['counts_source']:
            raise ValueError('Cell import requires cell_input and explicit counts_source; export_only is exempt.')
        if row['cell_coordinate_mode'] not in COORDS:
            raise ValueError(f'Explicit cell_coordinate_mode required: {COORDS}')
        if row['cell_coordinate_mode']=='affine' and not row['cell_to_fullres_affine']:
            raise ValueError('affine mode requires cell_to_fullres_affine.')
        if row['bin_input'] and row['cell_coordinate_mode']=='unregistered':
            raise ValueError('Cannot co-register bins and unregistered cells. Supply a reviewed affine.')
        if row['cell_boundaries'] and not row['boundary_id_column']:
            raise ValueError('Provide boundary_id_column, or __index__ for a GeoParquet index.')
    if row['bin_input'] and row['goal']!='export_only' and not row['segmentation_image']:
        raise ValueError('Packaging bins requires the original image.')
    for inp in INPUT_PATHS[:5]:
        if not row[inp]: continue
        for out in OUTPUT_PATHS:
            if under(row[out],row[inp]) or under(row[inp],row[out]):
                raise ValueError(f'Source and output/scratch must not overlap: {inp} vs {out}')
    if row['integration_group']:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',row['integration_group']):
            raise ValueError('integration_group must be a filesystem-safe ID.')
        if row['goal']!='cell_annotated':
            raise ValueError('integration_group requires a cell_annotated goal, not export-only or bins-only.')
    return row

def read_manifest(path):
    path=Path(path).resolve()
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise ValueError('Missing or duplicate CSV headers.')
        if set(reader.fieldnames)!=set(COLUMNS):
            raise ValueError(f'Fixed CSV schema mismatch. Missing={set(COLUMNS)-set(reader.fieldnames)}; '
                             f'unknown={set(reader.fieldnames)-set(COLUMNS)}. Use the supplied template.')
        rows=[]
        for line,raw in enumerate(reader,2):
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f'CSV line {line}: wrong field count; quote commas in paths.')
            r={k:v.strip() for k,v in raw.items()}
            if not any(r.values()): continue
            if not as_bool(r['enabled'],True): continue
            try:
                for key in INPUT_PATHS+OUTPUT_PATHS: r[key]=path_value(r[key],path.parent)
                r=validate_row(r)
            except Exception as e: raise ValueError(f'CSV line {line}: {e}') from e
            r['_csv_line']=line; r['sample_key']=r['dataset_id']+'__'+r['sample_id']
            r['_plan']=row_plan(r); rows.append(r)
    keys=[r['sample_key'] for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate/colliding dataset + sample IDs.')
    if not rows: raise ValueError('No enabled rows.')
    return rows

def deep_update(base, extra):
    out=copy.deepcopy(base)
    for k,v in extra.items():
        out[k]=deep_update(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else copy.deepcopy(v)
    return out

def load_project(manifest_path,settings_path):
    rows=read_manifest(manifest_path)
    settings_path=Path(settings_path).resolve()
    settings=json.loads(settings_path.read_text())
    rid=settings['run_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',rid): raise ValueError('Invalid run_id.')
    for block in ('execution','storage_policy','integration','import'):
        if block not in settings: raise ValueError(f'Missing settings block {block}')
    # Resolve execution/configuration paths relative to settings, not notebook working directory.
    for k in ('python_spatial','python_stardist','python_scvi','python_rapids'):
        settings['execution'][k]=path_value(settings['execution'][k],settings_path.parent)
        if is_remote(settings['execution'][k]): raise ValueError('Python executables must be local.')
    settings['proseg']['executable']=path_value(settings['proseg']['executable'],settings_path.parent)
    for block,key in [('he_normalization','reference_image'),('cell_cycle','gene_lists_json')]:
        if settings[block].get(key): settings[block][key]=path_value(settings[block][key],settings_path.parent)
    ex=settings['execution']
    if len(ex['gpu_ids'])!=len(set(map(str,ex['gpu_ids']))): raise ValueError('Duplicate GPU IDs.')
    if ex['scvi']['mode'] not in ('single','ddp'): raise ValueError('scVI mode: single or ddp.')
    if settings['clustering']['backend'] not in ('scanpy','rapids','rapids_dask'): raise ValueError('Invalid clustering backend.')
    for k in ('control_dir','tmp_dir','output_object_dir','output_plot_dir','output_data_dir'):
        if k in settings['integration']:
            settings['integration'][k]=path_value(settings['integration'][k],settings_path.parent)
    for k in ('approved_tmp_roots','approved_object_roots','approved_plot_roots','approved_data_roots'):
        settings['storage_policy'][k]=[path_value(v,settings_path.parent) for v in settings['storage_policy'].get(k,[])]
    if not settings['integration']['control_dir'] or is_remote(settings['integration']['control_dir']):
        raise ValueError('integration.control_dir must be local for plans, locks, logs and launch configuration.')
    return {'manifest_path':str(Path(manifest_path).resolve()),'settings_path':str(settings_path),
            'rows':rows,'settings':settings}

def check_policy(project):
    policy=project['settings']['storage_policy']
    mapping={'tmp_dir':'approved_tmp_roots','output_object_dir':'approved_object_roots',
             'output_plot_dir':'approved_plot_roots','output_data_dir':'approved_data_roots'}
    checks=project['rows']+[project['settings']['integration']]
    for r in checks:
        for col,allow in mapping.items():
            val=r.get(col,'')
            if not val: raise ValueError(f'{col} is empty.')
            if policy.get('enforce_allowlists',True) and not any(under(val,x) for x in policy.get(allow,[])):
                raise PermissionError(f'{col}={val} is not under {allow}. Obtain approval, then configure it.')
        if is_remote(r['tmp_dir']): raise ValueError('Local scratch is required.')
    control=project['settings']['integration']['control_dir']
    if policy.get('enforce_allowlists',True) and not any(under(control,x) for x in policy['approved_tmp_roots']):
        raise PermissionError('control_dir must be within an approved_tmp_root (it contains paths and logs).')
    return True

def sample_layout(project,row):
    run=project['settings']['run_id']
    w=Path(row['tmp_dir'])/'vhd'/row['dataset_id']/row['sample_id']/run
    # All computation is POSIX. Publishing sends each category ONLY to its declared destination.
    return {'work':w/'work','report':w/'reports','store':w/'objects'/f"{row['sample_id']}.zarr",
            'objects':w/'objects','plots':w/'publish_plots','data':w/'publish_data',
            'root':w,'proseg':w/'work'/'proseg'}

def row_config(project,row):
    settings=copy.deepcopy(project['settings'])
    overrides=row.get('overrides_json')
    if overrides:
        extra=json.loads(Path(overrides).read_text())
        forbidden=set(extra)-{'bin_qc','he_normalization','stardist','proseg','cells','metadata_transfer','cell_cycle','plots','storage'}
        if forbidden: raise ValueError(f'Sample overrides cannot change routing/storage policy: {forbidden}')
        settings=deep_update(settings,extra)
    p=sample_layout(project,row)
    spec={'sample':row['sample_key'],'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],
          'outs':row['spaceranger_outs'],'image':row['segmentation_image'],
          'tenx_to_image_affine':numbers(row['tenx_to_image_affine'],[1,0,0,0,1,0]),
          'image_mpp_override':([float(row['image_mpp_x']),float(row['image_mpp_y'])]
                                if row['image_mpp_x'] and row['image_mpp_y'] else None),
          'source_mpp_override':float(row['source_mpp']) if row['source_mpp'] else None,
          'coordinates_reviewed':as_bool(row['coordinates_reviewed']),
          'fresh_qc_reviewed':as_bool(row['fresh_qc_reviewed']),
          'metadata':json.loads(Path(row['metadata_json']).read_text()) if row['metadata_json'] else {},
          '_paths':{k:str(v) for k,v in p.items()},'_row':row}
    if bool(row['image_mpp_x']) != bool(row['image_mpp_y']): raise ValueError('Supply both image MPP axes.')
    settings['work_root']=str(p['root']); settings['output_root']=str(p['root'])
    settings['samples']=[spec]
    settings['cells']['mode']='none' if row['goal']=='bin_QCed' else ('proseg' if row['stage'] in STAGES[:3] else 'legacy')
    settings['bin_qc']['mode']='fresh' if row['stage'] in ('raw','bin_imported') else 'legacy'
    settings['bin_qc']['legacy_class_column']=row['bin_qc_column']
    if 'execution' in settings:
        settings['proseg']['threads_per_sample']=settings['execution']['proseg']['threads_per_worker']
    settings['_project']=project; settings['_row']=row
    return settings,spec

def save_plan(project):
    import pandas as pd
    base=Path(project['settings']['integration']['control_dir'])/project['settings']['run_id']
    base.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame([{'dataset_id':r['dataset_id'],'sample_id':r['sample_id'],'entry_stage':r['stage'],
                         'goal':r['goal'],'integration_group':r['integration_group'],
                         'actions':' -> '.join(r['_plan']),**{k:r[k] for k in OUTPUT_PATHS}} for r in project['rows']])
    frame.to_csv(base/'execution_plan.csv',index=False)
    snapshot=base/('resolved_project_'+digest(project)[:16]+'.json')
    if not snapshot.exists(): dump_json(project,snapshot)
    dump_json({'snapshot':str(snapshot)},base/'latest_plan.json')
    return frame,snapshot


def scientific_row(row):
    """Review sign-offs and publication destinations do not invalidate imported expression."""
    omit={'coordinates_reviewed','fresh_qc_reviewed','_plan','_csv_line',
          'output_object_dir','output_plot_dir','output_data_dir','integration_group','enabled','goal'}
    return {k:v for k,v in row.items() if k not in omit}
