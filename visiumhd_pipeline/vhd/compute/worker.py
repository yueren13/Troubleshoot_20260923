"""Fresh per-sample process; GPU and cache paths are configured by the parent BEFORE import."""
import argparse
import json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    parser.add_argument('--sample',required=True)
    parser.add_argument('--action',required=True)
    args=parser.parse_args()
    from vhd.control.runner import run_action
    project=json.load(open(args.project))
    print(run_action(project,args.sample,args.action),flush=True)

if __name__=='__main__': main()
