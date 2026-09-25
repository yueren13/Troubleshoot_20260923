#!/usr/bin/env python3
"""Optional synchronous CLI; default dry-run. Notebooks remain the documented review workflow."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vhd.control.manifest import load_project,check_policy,save_plan
from vhd.compute.launch import launch_samples,launch_integration

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--settings',required=True)
    p.add_argument('--action',choices=['plan','ingest_raw','import_bins','bin_qc','reuse_bin_qc','stardist','proseg',
                                     'postprocess_cells','import_cells','assemble','export_existing',
                                     'prepare','train','cluster','diagnostics','writeback'],default='plan')
    p.add_argument('--sample',action='append',help='dataset_id__sample_id; repeat to select multiple')
    p.add_argument('--group',help='Explicit integration group')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();project=load_project(args.manifest,args.settings);check_policy(project)
    if args.action=='plan':
        frame,path=save_plan(project);print(frame.to_string(index=False));print(path)
    elif args.action in ('prepare','train','cluster','diagnostics','writeback'):
        if not args.group: p.error('--group is required for integration actions')
        launch_integration(project,args.group,args.action,execute=args.execute)
    else: launch_samples(project,args.action,execute=args.execute,sample_keys=args.sample)

if __name__=='__main__': main()
