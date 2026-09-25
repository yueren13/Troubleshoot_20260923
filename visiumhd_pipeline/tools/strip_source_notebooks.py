#!/usr/bin/env python3
"""Create a shareable code-only audit bundle. Never executes notebook source."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--helper-dir',type=Path)
    args=parser.parse_args()
    src=args.source_dir.resolve();out=args.output_dir.resolve()
    if src==out or src in out.parents:
        raise ValueError('Choose an output directory outside the source repository directory.')
    out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for path in sorted(src.glob('*.ipynb')):
        data=path.read_bytes();nb=json.loads(data)
        source_blocks=[]
        for i,cell in enumerate(nb.get('cells',[])):
            if cell.get('cell_type')=='code':
                source=''.join(cell.get('source',[])) if isinstance(cell.get('source'),list) else cell.get('source','')
                source_blocks.append(f'# %% Original notebook cell {i}\n'+source+'\n')
                cell['outputs']=[];cell['execution_count']=None
            # Drop widget/rendering payloads, but leave source and markdown untouched.
            cell.pop('attachments',None)
            cell['metadata']={}
        nb['metadata'].pop('widgets',None)
        nb['metadata'].pop('signature',None)
        target=out/(path.stem+'.stripped.ipynb')
        if target.exists():
            raise FileExistsError(f'No silent overwrite: {target}')
        target.write_text(json.dumps(nb,indent=1,ensure_ascii=False))
        (out/(path.stem+'.source.py')).write_text('\n\n'.join(source_blocks))
        manifest.append({'source':str(path),'source_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                         'stripped_notebook':target.name,'n_code_cells':len(source_blocks)})
    missing=[]
    for name in ['cfs_spatialdata_202412.py','cfs_normhe_202605.py']:
        candidates=[src/name]
        if args.helper_dir:
            candidates.append(args.helper_dir/name)
        source=next((p for p in candidates if p.is_file()),None)
        if source is None:
            missing.append(name);continue
        target=out/name
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source,target)
        manifest.append({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'copied':target.name})
    (out/'source_manifest.json').write_text(json.dumps({'files':manifest,'missing_helpers':missing},indent=2))
    print(f'Wrote {len(manifest)} source records to {out}')
    if missing:
        print('Still missing:',', '.join(missing))
    print('No notebook source was executed. Review for confidential paths/content before sharing.')


if __name__=='__main__':
    main()
