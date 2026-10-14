"""Process launchers. Notebook kernels never initialize CUDA on behalf of workers."""
from __future__ import annotations
import copy
import json
import os
import queue
import shlex
import subprocess
import sys
import time
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..core import dump_json, stage_lock
from ..control.manifest import sample_layout, save_plan, check_policy

PACKAGE_ROOT=Path(__file__).resolve().parents[2]

def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)

def concurrency_budget(requested,threads,ram_per_job_gib=None,available_gib=None,reserve_cpus=8,reserve_gib=40):
    """Unknown per-job RAM => one pilot process, never four blind whole-slide allocations."""
    import psutil
    if requested<1 or threads<1: raise ValueError('Positive worker and thread counts required.')
    cpus=max(1,cpu_count()-int(reserve_cpus))
    if threads>cpus: raise ValueError(f'{threads} threads/job exceeds available CPU budget {cpus}.')
    avail=psutil.virtual_memory().available/2**30 if available_gib is None else float(available_gib)
    usable=avail-float(reserve_gib)
    if usable<=0: raise MemoryError('No free RAM after the configured reserve.')
    reason='Measured RAM estimate supplied.'
    if ram_per_job_gib is None:
        mem_workers=1; reason='RAM/job unknown: run one representative pilot and measure peak RSS before increasing concurrency.'
    else:
        if float(ram_per_job_gib)<=0: raise ValueError('RAM estimate must be positive.')
        mem_workers=int(usable//float(ram_per_job_gib))
        if mem_workers<1: raise MemoryError(f'Estimated job needs {ram_per_job_gib} GiB; budget {usable:.1f} GiB.')
    return {'workers':min(int(requested),cpus//threads,mem_workers),'threads_per_job':threads,
            'usable_cpu_threads':cpus,'usable_ram_gib':usable,'reason':reason}

def worker_env(tmp,gpus,threads):
    """Explicit writable caches under approved scratch; no credentials copied to config."""
    tmp=Path(tmp); tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
    env['PYTHONPATH']=str(PACKAGE_ROOT)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for name in ('TMPDIR','TMP','TEMP'): env[name]=str(tmp)
    for name in ('NUMBA_CACHE_DIR','CUPY_CACHE_DIR','XDG_CACHE_HOME','MPLCONFIGDIR',
                 'TORCH_HOME','KERAS_HOME','CUDA_CACHE_PATH','HF_HOME'):
        d=tmp/name.lower(); d.mkdir(exist_ok=True); env[name]=str(d)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]=str(threads)
    env['TF_NUM_INTRAOP_THREADS']=str(threads); env['TF_NUM_INTEROP_THREADS']='2'
    env['PYTHONUNBUFFERED']='1'; env['MPLBACKEND']='Agg'
    return env

def run_logged(command,env,log):
    import psutil
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); peak=0
    with log.open('w',buffering=1) as handle:
        handle.write(shlex.join(command)+'\n')
        process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,
                                 cwd=PACKAGE_ROOT,start_new_session=True)
        try:
            parent=psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    procs=[parent]+parent.children(recursive=True)
                    rss=0
                    for item in procs:
                        try: rss+=item.memory_info().rss
                        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                    peak=max(peak,rss)
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                time.sleep(0.5)
            returncode=process.wait()
        except BaseException:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    resource={'elapsed_seconds':time.monotonic()-started,'peak_sum_process_tree_rss_gib':peak/2**30,
              'rss_note':'Sampled every 0.5 s; summed RSS can count shared pages more than once, and can miss short peaks.',
              'returncode':returncode,'command':command}
    resource_path=log.with_suffix('.resources.json');dump_json(resource,resource_path)
    if returncode:
        tail='\n'.join(log.read_text(errors='replace').splitlines()[-35:])
        raise RuntimeError(f'Worker failed (exit={returncode}). Full log: {log}\n{tail}')
    return {'status':'complete','log':str(log),'resource_report':str(resource_path)}

def launch_samples(project,actions,execute=False,sample_keys=None):
    """Actions are barriers; GPUs are exclusive tokens, not an unsafe round-robin assignment."""
    check_policy(project)
    if isinstance(actions,str): actions=[actions]
    chosen=set(sample_keys) if sample_keys is not None else {r['sample_key'] for r in project['rows']}
    if chosen-{r['sample_key'] for r in project['rows']}: raise KeyError('Unknown selected sample key.')
    ex=project['settings']['execution']; outputs=[]
    for action in actions:
        rows=[r for r in project['rows'] if r['sample_key'] in chosen and action in r['_plan']]
        if not rows: continue
        if action=='publish': raise ValueError('Use publication notebook: publication is not a compute action.')
        gpu=action=='stardist'
        section=ex['stardist'] if gpu else ex['proseg'] if action=='proseg' else ex['cpu']
        gpus=list(ex['gpu_ids']) if gpu else []
        if gpu and not gpus: raise ValueError('StarDist GPU IDs are empty.')
        budget=concurrency_budget(min(section['workers'],len(gpus)) if gpu else section['workers'],
                                  section['threads_per_worker'],section.get('estimated_ram_gib_per_worker'),
                                  reserve_cpus=ex['reserve_cpus'],reserve_gib=ex['reserve_ram_gib'])
        executable=ex['python_stardist'] if gpu else ex['python_spatial']
        plan={'action':action,'samples':[r['sample_key'] for r in rows],
              'python':executable,'gpu_ids':gpus,**budget}
        print(json.dumps(plan,indent=2)); outputs.append(plan)
        if not execute: continue
        launch_project=copy.deepcopy(project)
        if action=='proseg':
            launch_project['settings']['proseg']['execute']=True
            launch_project['settings']['proseg']['threads_per_sample']=section['threads_per_worker']
        _,project_file=save_plan(launch_project)
        # Each action waits for all submitted processes before the next phase. No TF/scVI/RAPIDS overlap.
        tokens=queue.Queue()
        for i in range(budget['workers']): tokens.put(gpus[i] if gpu else None)
        def job(row):
            token=tokens.get()
            try:
                p=sample_layout(launch_project,row)
                env=worker_env(p['root']/'runtime'/action,[] if token is None else [token],section['threads_per_worker'])
                command=[executable,'-m','vhd.compute.worker','--project',str(project_file),
                         '--sample',row['sample_key'],'--action',action]
                with stage_lock(p['work'],'worker_'+action):
                    return run_logged(command,env,p['report']/('worker_'+action+'.log'))
            finally: tokens.put(token)
        failures=[]
        with ThreadPoolExecutor(max_workers=budget['workers']) as pool:
            jobs={pool.submit(job,r):r['sample_key'] for r in rows}
            for future in as_completed(jobs):
                try: print(jobs[future],future.result())
                except Exception as e: failures.append((jobs[future],str(e)))
        if failures: raise RuntimeError('Some samples failed; no later actions were launched.\n'+str(failures))
    return outputs

def scvi_command(executable,config_path,mode,devices):
    args=['--config',str(config_path),'--phase','train']
    if mode=='single': return [executable,'-m','vhd.compute.scvi_worker',*args]
    if mode!='ddp' or devices<2: raise ValueError('scVI mode is single or ddp (at least two GPUs).')
    return [executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={devices}',
            '--module','vhd.compute.scvi_worker',*args]

def launch_integration(project,group,phase,execute=False):
    from ..control.runner import integration_config
    cfg=integration_config(project,group); ex=cfg['execution']
    work=Path(cfg['_integration_work']); work.mkdir(parents=True,exist_ok=True)
    gpus=list(ex['gpu_ids']); threads=ex['integration_threads']
    if phase=='train':
        devices=ex['scvi']['devices'] if ex['scvi']['mode']=='ddp' else 1
        if devices>len(gpus): raise ValueError('scVI requests more GPU IDs than configured.')
        gpus=gpus[:devices]; exe=ex['python_scvi']
    elif phase=='cluster':
        backend=cfg['clustering']['backend']
        exe=ex['python_spatial'] if backend=='scanpy' else ex['python_rapids']
        if backend=='scanpy': gpus=[]
        elif backend=='rapids': gpus=gpus[:1]
    else:
        exe=ex['python_spatial']; gpus=[]
    if threads>max(1,cpu_count()-ex['reserve_cpus']): raise ValueError('Integration thread budget exceeds CPU quota.')
    path=work/'resolved_integration.json'
    command=scvi_command(exe,path,ex['scvi']['mode'],len(gpus)) if phase=='train' else [
        exe,'-m','vhd.compute.integration_worker','--config',str(path),'--phase',phase]
    plan={'phase':phase,'group':group,'command':shlex.join(command),'gpu_ids':gpus,'scratch':str(work)}
    print(json.dumps(plan,indent=2))
    if not execute: return plan
    dump_json(cfg,path)
    env=worker_env(work/'runtime'/phase,gpus,threads)
    with stage_lock(work,'phase_'+phase):
        result=run_logged(command,env,work/('worker_'+phase+'.log'))
        if phase=='train':
            # Single-writer, single-GPU inference after every DDP rank has exited.
            command=[exe,'-m','vhd.compute.scvi_worker','--config',str(path),'--phase','infer']
            env=worker_env(work/'runtime'/'infer',gpus[:1],threads)
            run_logged(command,env,work/'worker_infer.log')
    return result
