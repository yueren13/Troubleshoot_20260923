"""Explicit local scratch staging and immutable publication; never mutate input data."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from ..core import dump_json, digest
from .manifest import is_remote, under


def join_uri(root,*parts):
    return root.rstrip('/')+'/'+('/'.join(str(p).strip('/') for p in parts)) if is_remote(root) else str(Path(root).joinpath(*map(str,parts)))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def free_space_gate(directory,required_bytes=0,reserve_gib=20):
    p=Path(directory)
    while not p.exists(): p=p.parent
    free=shutil.disk_usage(p).free
    required=int(required_bytes)+float(reserve_gib)*2**30
    if free<required: raise OSError(f'Scratch space: {free/2**30:.1f} GiB free < {required/2**30:.1f} GiB required.')


def fs_for(uri):
    import fsspec
    # Standard environment/IAM/profile authentication only; credentials never enter the CSV.
    return fsspec.core.url_to_fs(uri)


def localize(uri,scratch,policy):
    if not uri: return ''
    if not is_remote(uri):
        p=Path(uri)
        if not p.exists(): raise FileNotFoundError(p)
        return str(p)
    if not policy.get('allow_remote_staging',False):
        raise PermissionError('S3-to-scratch staging is disabled. Review approved storage policy first.')
    scratch=Path(scratch)
    if policy.get('enforce_allowlists',True) and not any(under(str(scratch),r) for r in policy['approved_tmp_roots']):
        raise PermissionError('Remote input would be staged outside approved scratch.')
    fs,key=fs_for(uri)
    if not fs.exists(key): raise FileNotFoundError(uri)
    isdir=fs.isdir(key)
    listing=fs.find(key,detail=True) if isdir else {key:fs.info(key)}
    listing={k:v for k,v in listing.items() if v.get('type')!='directory'}
    if not listing: raise ValueError(f'Empty source prefix: {uri}')
    # ETags are change tokens, NOT interpreted as MD5 checksums.
    entries=[{'key':k,'size':int(v.get('size',0)),
              'etag':str(v.get('ETag',v.get('etag',''))),'version_id':str(v.get('VersionId','')),
              'modified':str(v.get('LastModified',v.get('mtime','')))} for k,v in sorted(listing.items())]
    sig=digest({'uri':uri,'entries':entries})
    folder=scratch/'staged_inputs'/sig[:20]
    dest=folder/Path(urlsplit(uri).path).name
    marker=folder/'stage_complete.json'
    if marker.exists():
        report=json.loads(marker.read_text())
        good=report['signature']==sig and dest.exists()
        if good:
            for entry in entries:
                target=dest/(entry['key'][len(key.rstrip('/'))+1:]) if isdir else dest
                good=good and target.is_file() and target.stat().st_size==entry['size']
        if good: return str(dest)
        raise RuntimeError(f'Modified/incomplete staged copy: {folder}')
    if folder.exists(): raise FileExistsError(f'Incomplete stage requires review: {folder}')
    total=sum(x['size'] for x in entries)
    limit=policy.get('max_remote_stage_gib')
    if limit is not None and total>float(limit)*2**30: raise MemoryError('Remote source exceeds staging-size gate.')
    free_space_gate(scratch,total,policy.get('min_free_tmp_gib',20))
    folder.mkdir(parents=True)
    for entry in entries:
        rel=entry['key'][len(key.rstrip('/'))+1:] if isdir else ''
        if '..' in Path(rel).parts: raise ValueError('Unsafe remote relative path.')
        target=dest/rel if isdir else dest
        target.parent.mkdir(parents=True,exist_ok=True)
        fs.get_file(entry['key'],str(target))
        if target.stat().st_size!=entry['size']: raise IOError(f'Staged size mismatch: {entry["key"]}')
    dump_json({'signature':sig,'source':uri,'n_files':len(entries),'bytes':total,
               'validation':'per-file byte sizes plus remote version/ETag/modified change tokens'},marker)
    return str(dest)


def inventory(source):
    source=Path(source)
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Publication rejects symlinks; supply real files in the approved workspace.')
    return [{'path':p.relative_to(source).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)}
            for p in sorted(source.rglob('*')) if p.is_file()]


def publish_tree(source,destination,policy,*,execute=False):
    """A category is committed only after all payload files are copied and verified.

    S3 has no atomic multi-key directory transaction. Readers must require _SUCCESS.json.
    Existing committed runs are immutable. Incomplete prefixes are never auto-deleted.
    """
    source=Path(source)
    entries=inventory(source)
    manifest={'schema':'vhd-publication-v2','files':entries}
    sig=digest(manifest)
    report={'source':str(source),'destination':destination,'signature':sig,'files':len(entries),
            'bytes':sum(e['bytes'] for e in entries),'executed':False}
    if not execute: return report
    if is_remote(destination) and not policy.get('allow_remote_publication',False):
        raise PermissionError('S3 publication is disabled in storage_policy.')
    fs,key=fs_for(destination)
    success=key.rstrip('/')+'/_SUCCESS.json'
    marker=key.rstrip('/')+'/_MANIFEST.json'
    verify_hash=not is_remote(destination) or policy.get('verify_remote_sha256',False)
    if fs.exists(success):
        with fs.open(success) as f: old=json.load(f)
        if old.get('signature')!=sig: raise FileExistsError('Immutable output differs; choose a new run_id.')
        for entry in entries:
            target=key.rstrip('/')+'/'+entry['path']
            if not fs.exists(target) or fs.info(target)['size']!=entry['bytes']:
                raise IOError('Committed publication has a missing/changed file.')
            if verify_hash:
                h=hashlib.sha256()
                with fs.open(target,'rb') as f:
                    for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
                if h.hexdigest()!=entry['sha256']: raise IOError('Committed checksum changed.')
        return {**report,'executed':True,'reused':True}
    if fs.exists(key) and fs.find(key):
        raise FileExistsError(f'Uncommitted output prefix requires inspection, not automatic overwrite: {destination}')
    fs.makedirs(key,exist_ok=True)
    for entry in entries:
        target=key.rstrip('/')+'/'+entry['path']
        fs.makedirs(target.rsplit('/',1)[0],exist_ok=True)
        fs.put_file(str(source/entry['path']),target)
        if fs.info(target)['size']!=entry['bytes']: raise IOError(f'Published size mismatch: {target}')
        if verify_hash:
            h=hashlib.sha256()
            with fs.open(target,'rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
            if h.hexdigest()!=entry['sha256']: raise IOError(f'Published SHA256 mismatch: {target}')
    with fs.open(marker,'wt') as f: json.dump(manifest,f,indent=2)
    # Commit marker is deliberately last, including for local publication.
    with fs.open(success,'wt') as f:
        json.dump({'signature':sig,'n_files':len(entries),
                   'verified':'SHA256 readback' if verify_hash else 'byte-size readback; source SHA256 in manifest'},f,indent=2)
    return {**report,'executed':True,'reused':False}
