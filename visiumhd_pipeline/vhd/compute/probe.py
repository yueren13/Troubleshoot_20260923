"""Environment inventory; does not import TensorFlow/PyTorch into the notebook process."""
from __future__ import annotations
import os
import shutil
import subprocess
import psutil
from ..core import environment_report


def hardware_inventory():
    out=environment_report()
    out['available_cpu_threads']=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    ram=psutil.virtual_memory();out['ram_gib']={'total':ram.total/2**30,'available':ram.available/2**30}
    executable=shutil.which('nvidia-smi')
    if executable:
        for name,query in [('gpu','--query-gpu=index,name,memory.total,memory.free'),
                           ('processes','--query-compute-apps=gpu_uuid,pid,process_name,used_memory')]:
            p=subprocess.run([executable,query,'--format=csv'],capture_output=True,text=True)
            out['nvidia_'+name]=p.stdout if p.returncode==0 else p.stderr
    else: out['nvidia_gpu']='nvidia-smi unavailable in this environment'
    return out

if __name__=='__main__':
    import json
    print(json.dumps(hardware_inventory(),indent=2))
