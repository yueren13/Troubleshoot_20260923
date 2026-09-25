"""CPU, single-GPU RAPIDS, or optional multi-GPU Leiden with Dask.

UMAP fitting remains single GPU. Dask spill is not a promise that all CUDA algorithms spill.
"""
from __future__ import annotations
import inspect
import os
import warnings
from pathlib import Path


def seed_kw(function,seed):
    params=inspect.signature(function).parameters
    if 'rng' in params: return {'rng':seed}
    if 'random_state' in params: return {'random_state':seed}
    raise RuntimeError(f'No supported random-seed argument on {function.__name__}; inspect installed API.')


def graph_and_cluster(adata,cfg,resolutions):
    settings=cfg['clustering']; backend=settings.get('backend','scanpy')
    if backend=='scanpy':
        import scanpy as sc
        sc.pp.neighbors(adata,use_rep='X_scVI',n_neighbors=settings['n_neighbors'],random_state=settings['seed'])
        sc.tl.umap(adata,min_dist=settings['umap_min_dist'],random_state=settings['seed'])
        for res in resolutions:
            key=f'leiden_{res:.1f}'.replace('.','p')
            sc.tl.leiden(adata,resolution=res,key_added=key,flavor='igraph',directed=False,
                         n_iterations=settings['leiden_iterations'],random_state=settings['seed'])
        return {'backend':backend,'multi_gpu_umap_fit':False}
    if backend not in ('rapids','rapids_dask'): raise ValueError(f'Unknown clustering backend {backend}.')
    import rapids_singlecell as rsc
    import cupy as cp
    import numpy as np
    use_dask=backend=='rapids_dask'
    if not cp.cuda.runtime.getDeviceCount(): raise RuntimeError('RAPIDS environment does not see CUDA.')
    if use_dask and 'use_dask' not in inspect.signature(rsc.tl.leiden).parameters:
        raise RuntimeError('Installed RAPIDS Leiden lacks use_dask. Install a supported version; no silent fallback.')
    if use_dask and adata.n_obs<10_000_000:
        warnings.warn('RAPIDS recommends single-GPU Leiden below 10 million cells. Dask is explicitly requested; benchmark both.')
    algorithm=settings.get('rapids_neighbor_algorithm','cagra')
    if algorithm.startswith('mg_') and not use_dask:
        raise ValueError('Use rapids_dask with multiple visible GPUs for an mg_ neighbor algorithm.')
    # Only the scVI latent coordinates and sparse neighbor graph go through RAPIDS.
    # The all-gene expression matrix is never copied to CUDA for clustering.
    neighbor_kw=dict(use_rep='X_scVI',n_neighbors=settings['n_neighbors'],algorithm=algorithm,
                     **seed_kw(rsc.pp.neighbors,settings['seed']))
    options=settings.get('rapids_neighbor_options',{})
    if options:
        if 'algorithm_kwds' not in inspect.signature(rsc.pp.neighbors).parameters:
            raise RuntimeError('Installed neighbors does not accept algorithm_kwds.')
        neighbor_kw['algorithm_kwds']=options
    rsc.pp.neighbors(adata,**neighbor_kw)
    cluster=client=None
    try:
        if use_dask:
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            d=cfg['execution']['dask']
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
            if len(visible)<2 or not all(visible): raise RuntimeError('Dask backend requires at least two assigned GPUs.')
            scratch=Path(cfg['_integration_work'])/'runtime'/'dask_spill'; scratch.mkdir(parents=True,exist_ok=True)
            cluster=LocalCUDACluster(CUDA_VISIBLE_DEVICES=','.join(visible),n_workers=len(visible),
                threads_per_worker=d['threads_per_worker'],memory_limit=d['host_memory_limit_per_worker'],
                device_memory_limit=d['device_memory_limit'],local_directory=str(scratch),
                dashboard_address=None,protocol='tcp')
            client=Client(cluster); client.wait_for_workers(len(visible))
        for res in resolutions:
            kw=dict(resolution=res,key_added=f'leiden_{res:.1f}'.replace('.','p'),
                    n_iterations=settings['rapids_leiden_iterations'],**seed_kw(rsc.tl.leiden,settings['seed']))
            if use_dask: kw['use_dask']=True
            rsc.tl.leiden(adata,**kw)
    finally:
        if client is not None: client.close()
        if cluster is not None: cluster.close()
    # Free Dask worker reservations before single-device UMAP fitting.
    kw=dict(min_dist=settings['umap_min_dist'],**seed_kw(rsc.tl.umap,settings['seed']))
    if 'key_added' in inspect.signature(rsc.tl.umap).parameters: kw['key_added']='X_umap'
    rsc.tl.umap(adata,**kw)
    if 'X_umap' not in adata.obsm: raise RuntimeError('RAPIDS UMAP did not populate X_umap.')
    for key in list(adata.obsm):
        if hasattr(adata.obsm[key],'get'): adata.obsm[key]=adata.obsm[key].get()
    for key in list(adata.obsp):
        if hasattr(adata.obsp[key],'get'): adata.obsp[key]=adata.obsp[key].get()
    cp.get_default_memory_pool().free_all_blocks()
    return {'backend':backend,'rapids_version':getattr(rsc,'__version__','unknown'),
            'neighbor_algorithm':algorithm,'dask_leiden':use_dask,'multi_gpu_umap_fit':False}
