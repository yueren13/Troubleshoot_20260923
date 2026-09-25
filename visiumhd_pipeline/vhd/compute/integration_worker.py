"""Non-TensorFlow integration actions, run in a user-selected interpreter."""
import argparse
import json

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=['prepare','cluster','diagnostics','writeback'],required=True)
    args=parser.parse_args(); cfg=json.load(open(args.config))
    from vhd.integration import prepare_scvi,leiden_umaps,diagnostics,writeback_annotations
    if args.phase=='prepare': print(prepare_scvi(cfg))
    elif args.phase=='cluster': print(leiden_umaps(cfg))
    elif args.phase=='diagnostics': print(diagnostics(cfg))
    else:
        for spec in cfg['samples']: print(writeback_annotations(cfg,spec))

if __name__=='__main__': main()
