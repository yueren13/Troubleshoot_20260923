"""scVI script/DDP training; all expression must be explicitly selected raw counts.

DDP workers train only. Inference runs in a new single-GPU process after torchrun exits.
This avoids notebook spawning and multiple ranks competing to write latent arrays.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def execute(cfg,phase):
    import numpy as np
    import pandas as pd
    import psutil
    import anndata as ad
    import scvi
    import torch
    from ..core import VERSION,digest,dump_json,file_stamp,integer_csr,reusable,completed,environment_report
    from ..integration import integration_dir,sparse_h5_ram_estimate
    p=integration_dir(cfg); settings=cfg['scvi']; execution=cfg['execution']['scvi']
    ddp=execution['mode']=='ddp'; world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0'))
    sig=digest({'version':VERSION,'input':file_stamp(p/'scvi_counts_hvg.h5ad'),
                'scvi_version':scvi.__version__,'scvi':settings,'execution':execution})
    required=[p/'model'/'model.pt',p/'latent.npy',p/'training_obs.parquet',p/'training_var.csv']
    marker=p/'09_train_complete.json'; receipt=p/'09_model_saved.json'
    if reusable(marker,sig,required): return {'status':'reused'}
    estimate=sparse_h5_ram_estimate(p/'scvi_counts_hvg.h5ad')
    ranks=execution['devices'] if ddp and phase=='train' else 1
    needed=estimate['conservative_working_gib']*ranks
    # Rank-zero preflight must occur before ranks allocate. Other ranks follow the same estimate;
    # do not multiply *currently shrinking* free memory gates after ranks begin to allocate.
    available=psutil.virtual_memory().available/2**30
    budget=min(float(settings['max_training_ram_gib']),float(cfg['execution'].get('host_ram_budget_gib',300)))
    if needed>budget: raise MemoryError(f'DDP-aware RAM estimate {needed:.1f} GiB ({ranks} replicas) > configured {budget:.1f} GiB.')
    if phase=='train' and rank==0 and needed>available-cfg['execution']['reserve_ram_gib']:
        raise MemoryError(f'Training estimate {needed:.1f} GiB exceeds currently free host RAM minus reserve.')
    if settings['require_gpu'] and not torch.cuda.is_available(): raise RuntimeError('PyTorch CUDA is not available.')
    if ddp and phase=='train' and world!=execution['devices']:
        raise RuntimeError('Launch DDP with the supplied torchrun launcher, not a notebook cell.')
    if phase=='train' and receipt.exists():
        if json.loads(receipt.read_text())['signature']!=sig: raise FileExistsError('Existing model has a different configuration. Use new run_id.')
        return {'status':'model_reused_waiting_for_inference'}
    if phase=='train' and (p/'model').exists(): raise FileExistsError('Incomplete model; inspect it and select new run_id. No automatic overwrite.')
    if phase=='infer' and (not receipt.exists() or json.loads(receipt.read_text())['signature']!=sig):
        raise RuntimeError('No matching trained model receipt.')
    scvi.settings.seed=settings['seed']; torch.set_float32_matmul_precision('high')
    a=ad.read_h5ad(p/'scvi_counts_hvg.h5ad'); a.X=integer_csr(a.X).astype(np.float32)
    a.obs['sample']=a.obs['sample'].astype('category')
    scvi.model.SCVI.setup_anndata(a,batch_key='sample',layer=None)
    if phase=='infer':
        model=scvi.model.SCVI.load(str(p/'model'),adata=a,accelerator='gpu' if torch.cuda.is_available() else 'cpu',device=0)
        z=np.asarray(model.get_latent_representation(batch_size=settings['batch_size']),dtype=np.float32)
        if z.shape!=(a.n_obs,settings['n_latent']) or not np.isfinite(z).all(): raise ValueError('Invalid latent output.')
        np.save(p/'latent.npy',z); a.obs.to_parquet(p/'training_obs.parquet'); a.var.to_csv(p/'training_var.csv')
        completed(marker,sig,required,n_cells=a.n_obs,n_genes=a.n_vars,
                  training_mode=execution['mode'],inference_mode='single_gpu_single_writer',
                  batch_key='sample',ddp_does_not_pool_vram=True)
        return {'status':'inference_complete','n_cells':a.n_obs}
    model=scvi.model.SCVI(a,n_latent=settings['n_latent'],n_hidden=settings['n_hidden'],
                          n_layers=settings['n_layers'],gene_likelihood=settings['gene_likelihood'])
    kw=dict(max_epochs=settings['max_epochs'],accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=execution['devices'] if ddp else 1,batch_size=settings['batch_size'],
            plan_kwargs={'lr':settings['learning_rate']},enable_checkpointing=False,
            enable_progress_bar=False,logger=False,default_root_dir=str(p/'runtime'/'lightning'))
    if ddp:
        # Current official scVI multi-GPU guide: early stopping is unsupported.
        kw.update(strategy='ddp_find_unused_parameters_true',train_size=1.0,validation_size=0.0,
                  early_stopping=False,check_val_every_n_epoch=None)
    else:
        kw.update(train_size=settings['train_size'],early_stopping=True,
                  early_stopping_patience=settings['early_stopping_patience'])
    model.train(**kw)
    if rank==0:
        model.save(str(p/'model'),overwrite=False,save_anndata=False)
        history=p/'history'; history.mkdir(exist_ok=True)
        for key,value in model.history.items():
            frame=pd.DataFrame(value); frame.to_csv(history/f'{key}.csv')
            numeric=frame.select_dtypes(include='number')
            if not numeric.empty:
                import matplotlib.pyplot as plt
                from ..plots import finish
                fig,ax=plt.subplots(figsize=(8,5))
                for col in numeric: ax.plot(numeric[col].to_numpy(),label=str(col))
                ax.set(title=f'scVI: {key}',xlabel='Recorded epoch/step',ylabel=str(key)); ax.legend()
                finish(fig,history/f'{key}.png',cfg['plots']['dpi'])
        dump_json({'signature':sig,'world_size':world,'per_device_batch_size':settings['batch_size'],
                   'nominal_global_batch_size':settings['batch_size']*world,
                   'no_heldout_validation_in_ddp':ddp,'early_stopping':not ddp,
                   'estimated_host_ram_gib':needed,'environment':environment_report()},receipt)
    return {'status':'trained','rank':rank}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=['train','infer'],required=True)
    args=parser.parse_args(); print(execute(json.load(open(args.config)),args.phase),flush=True)

if __name__=='__main__': main()
