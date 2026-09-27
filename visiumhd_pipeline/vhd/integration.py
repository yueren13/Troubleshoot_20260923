from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (VERSION, completed, digest, dump_json, file_stamp, integer_csr,
                   reusable, safe_frame, sample_paths, stage_lock, write_h5ad)
from .io import load_table, table_metadata
from .palette import distinct_palette, natural_order
from .plots import scatter_categorical, scatter_numeric, finish


def integration_dir(cfg):
    p=Path(cfg['_integration_work']) if cfg.get('_integration_work') else Path(cfg['output_root'])/'integration'/cfg['scvi']['run_name']
    p.mkdir(parents=True,exist_ok=True)
    return p


def eligible(obs,cfg):
    if 'qc_pass' not in obs or not pd.api.types.is_bool_dtype(obs.qc_pass):
        raise ValueError('Cell table requires a real boolean qc_pass column.')
    return (obs.qc_pass.to_numpy(bool)
            & (obs.total_counts.to_numpy()>=cfg['scvi']['min_counts'])
            & (obs.n_genes_by_counts.to_numpy()>=cfg['scvi']['min_genes']))


def consensus_hvg(rankings,number):
    """Batch-balanced consensus: number of top-HVG batches, then median within-batch rank."""
    ranks=pd.DataFrame(rankings).rename_axis(None)
    out=pd.DataFrame(index=ranks.index)
    out['n_batches_hvg']=ranks.notna().sum(axis=1)
    out['median_rank']=ranks.median(axis=1,skipna=True)
    out['gene_id']=out.index.astype(str)
    out=out.sort_values(['n_batches_hvg','median_rank','gene_id'],ascending=[False,True,True],na_position='last')
    out['selected_hvg']=False
    available=out.index[out.n_batches_hvg>0]
    if len(available)<number:
        raise ValueError(f'Only {len(available)} genes have usable within-sample HVG ranks; requested {number}.')
    out.loc[available[:number],'selected_hvg']=True
    return out


def prepare_scvi(cfg):
    """Two pass, one sample in RAM at a time; concat_on_disk assembles the shared-HVG input."""
    import anndata as ad
    import scanpy as sc
    from anndata.experimental import concat_on_disk
    p=integration_dir(cfg); stage=p/'staged_hvg'; stage.mkdir(exist_ok=True)
    specs=cfg['samples']
    if len(specs)<2:
        raise ValueError('Batch integration requires at least two samples.')
    store_paths={s['sample']:sample_paths(cfg,s)['store'] for s in specs}
    source_manifests=[sample_paths(cfg,s)['report']/'storage_manifest.json' for s in specs]
    settings=cfg['scvi']
    sig=digest({'version':VERSION,'sources':[file_stamp(x) for x in source_manifests],
                'settings':{k:settings[k] for k in ('n_hvg','min_counts','min_genes','hvg_spans','exclude_prefixes')}})
    required=[p/'scvi_counts_hvg.h5ad',p/'features.csv',p/'sample_qc_counts.csv']
    marker=p/'08_prepare_complete.json'
    if reusable(marker,sig,required):
        return pd.read_csv(p/'sample_qc_counts.csv')
    with stage_lock(p,'prepare'):
        shared=None; summary=[]
        for name,store in store_paths.items():
            obs,var=table_metadata(store,'cells')
            if not var.index.is_unique:
                raise ValueError(f'{name}: duplicate stable gene IDs.')
            gene_set=set(var.index.astype(str))
            shared=gene_set if shared is None else shared & gene_set
            summary.append({'sample':name,'n_cells':len(obs),'n_qc_eligible':int(eligible(obs,cfg).sum()),
                            'n_assayed_genes':len(var)})
        common=pd.Index(sorted(shared))
        if len(common)<settings['n_hvg']:
            raise ValueError(f'{len(common)} common assayed genes < requested {settings["n_hvg"]} HVGs. '
                             'Review gene identifiers/panels or reduce n_hvg explicitly; no missing-gene zero-fill.')
        if any(x['n_qc_eligible']<20 for x in summary):
            raise ValueError('Each sample must have >=20 eligible cells for this HVG/scVI workflow.')
        rankings={}; hvg_audit=[]
        for name,store in store_paths.items():
            print(f'HVG pass: {name}')
            a=load_table(store,'cells')
            a=a[eligible(a.obs,cfg),common].copy(); a.X=integer_csr(a.X)
            if settings['exclude_prefixes']:
                symbols=a.var['gene_symbol'].astype(str).str.upper()
                allow=~symbols.str.startswith(tuple(x.upper() for x in settings['exclude_prefixes']))
            else:
                allow=np.ones(a.n_vars,bool)
            work=a[:,np.asarray(allow)].copy()
            n=min(settings['n_hvg'],work.n_vars)
            for span in settings['hvg_spans']:
                try:
                    sc.pp.highly_variable_genes(work,flavor='seurat_v3',n_top_genes=n,span=span,inplace=True)
                    break
                except ValueError as exc:
                    hvg_audit.append({'sample':name,'span':span,'error':str(exc)})
            else:
                raise RuntimeError(f'All seurat_v3 LOESS fits failed for {name}; no alternate method was substituted.')
            ranks=work.var['highly_variable_rank'].astype(float).copy()
            ranks.loc[~work.var['highly_variable'].to_numpy(bool)]=np.nan
            rankings[name]=ranks.reindex(common)
            del a,work; gc.collect()
        consensus=consensus_hvg(rankings,settings['n_hvg'])
        chosen=consensus.index[consensus.selected_hvg]
        dump_json({'method':'Per-sample seurat_v3; consensus sorts n_batches_hvg desc, median_rank asc, gene_id asc. '
                            'This is an explicit batch-balanced consensus, not a claim of identical scanpy batch_key implementation.',
                   'loess_retries':hvg_audit,'common_assayed_genes':len(common)},p/'hvg_method.json')
        consensus.to_csv(p/'features.csv',index=False)
        manifest={}; total_cells=0
        for name,store in store_paths.items():
            print(f'Staging selected HVGs: {name}')
            a=load_table(store,'cells')
            a=a[eligible(a.obs,cfg),chosen].copy()
            a.X=integer_csr(a.X)
            # A common all-gene QC threshold does not guarantee nonzero library size on HVGs.
            nonzero=np.asarray(a.X.sum(axis=1)).ravel()>0
            if not nonzero.all():
                pd.DataFrame({'cell_id':a.obs_names[~nonzero], 'reason':'zero_selected_HVG_counts'}).to_csv(
                    p/f'{name}_excluded_zero_HVG.csv',index=False)
                a=a[nonzero].copy()
            if a.n_obs<20:
                raise ValueError(f'{name}: fewer than 20 cells have nonzero selected-HVG counts.')
            a.obs['sample']=pd.Categorical([name]*a.n_obs)
            a.uns={}
            # Keep original-image coordinates in per-sample stores, not in a misleading pooled spatial graph.
            a.obsm.clear(); a.obsp.clear(); a.varm.clear(); a.varp.clear(); a.raw=None
            for key in list(a.layers):
                del a.layers[key]
            path=stage/f'{name}.h5ad'
            write_h5ad(a,path); manifest[name]=str(path)
            for row in summary:
                if row['sample']==name:
                    row.update(n_training_cells=a.n_obs,n_zero_hvg_excluded=int((~nonzero).sum()),
                               selected_hvg_umis=int(a.X.sum(dtype=np.int64)))
            total_cells+=a.n_obs
            del a; gc.collect()
        temporary=p/'scvi_counts_hvg.partial.h5ad'
        if temporary.exists():
            raise FileExistsError(temporary)
        concat_on_disk(manifest,str(temporary),axis=0,join='inner',merge='same',
                       label='sample',index_unique=None)
        check=ad.read_h5ad(temporary,backed='r')
        try:
            if check.n_obs!=total_cells or check.n_vars!=len(chosen) or not check.obs_names.is_unique:
                raise ValueError('On-disk concatenation did not preserve the expected cell/features set.')
            if not check.var_names.equals(pd.Index(chosen)):
                raise ValueError('HVG order changed during concat_on_disk.')
            if set(check.obs['sample'].astype(str))!=set(store_paths):
                raise ValueError('At least one sample is absent from the pooled training input.')
        finally:
            check.file.close()
        temporary.rename(p/'scvi_counts_hvg.h5ad')
        pd.DataFrame(summary).to_csv(p/'sample_qc_counts.csv',index=False)
        completed(marker,sig,required,n_cells=total_cells,n_hvg=len(chosen),n_samples=len(specs))
    return pd.DataFrame(summary)


def sparse_h5_ram_estimate(path):
    import h5py
    with h5py.File(path,'r') as f:
        x=f['X']
        if not isinstance(x,h5py.Group) or 'data' not in x:
            raise ValueError('Expected sparse CSR training X, not a dense matrix.')
        raw=sum(x[k].size*x[k].dtype.itemsize for k in ('data','indices','indptr'))
        return {'sparse_X_gib':raw/2**30,'conservative_working_gib':raw/2**30*4+4}


def train_scvi(cfg):
    import anndata as ad
    import scvi
    import torch
    import psutil
    p=integration_dir(cfg); settings=cfg['scvi']
    input_path=p/'scvi_counts_hvg.h5ad'
    sig=digest({'version':VERSION,'input':file_stamp(input_path),
                'scvi_version':scvi.__version__,'settings':settings})
    marker=p/'09_train_complete.json'
    required=[p/'latent.npy',p/'training_obs.parquet',p/'model'/'model.pt']
    if reusable(marker,sig,required):
        return {'status':'reused','model_directory':str(p/'model')}
    estimate=sparse_h5_ram_estimate(input_path)
    budget=min(float(settings['max_training_ram_gib']),psutil.virtual_memory().available/2**30*0.8)
    if estimate['conservative_working_gib']>budget:
        raise MemoryError(f'Training memory gate: {estimate}; budget {budget:.1f} GiB. '
                          'Use more RAM or explicitly design streaming training; this version does not pretend to stream scVI.')
    if settings['require_gpu'] and not torch.cuda.is_available():
        raise RuntimeError('PyTorch GPU unavailable. Select the scVI kernel; TensorFlow GPU visibility is independent.')
    with stage_lock(p,'train'):
        if (p/'model').exists():
            raise FileExistsError('Partial/existing model without completion marker; choose a new scvi.run_name.')
        scvi.settings.seed=settings['seed']
        torch.set_float32_matmul_precision('high')
        a=ad.read_h5ad(input_path); a.X=integer_csr(a.X).astype(np.float32)
        a.obs['sample']=a.obs['sample'].astype('category')
        scvi.model.SCVI.setup_anndata(a,layer=None,batch_key='sample')
        model=scvi.model.SCVI(a,n_latent=settings['n_latent'],n_hidden=settings['n_hidden'],
                             n_layers=settings['n_layers'],gene_likelihood=settings['gene_likelihood'])
        model.train(max_epochs=settings['max_epochs'],accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=1,batch_size=settings['batch_size'],train_size=settings['train_size'],
                    early_stopping=True,early_stopping_patience=settings['early_stopping_patience'],
                    plan_kwargs={'lr':settings['learning_rate']})
        z=np.asarray(model.get_latent_representation(),dtype=np.float32)
        if z.shape!=(a.n_obs,settings['n_latent']) or not np.isfinite(z).all():
            raise ValueError('Invalid scVI latent representation.')
        model.save(str(p/'model'),overwrite=False,save_anndata=False)
        np.save(p/'latent.npy',z)
        a.obs.to_parquet(p/'training_obs.parquet')
        a.var.to_csv(p/'training_var.csv')
        # scVI model.save does not retain full trainer history; persist each available metric separately.
        history_dir=p/'history'; history_dir.mkdir(exist_ok=True)
        import matplotlib.pyplot as plt
        for key,values in model.history.items():
            frame=pd.DataFrame(values)
            frame.to_csv(history_dir/f'{key}.csv')
            numeric=frame.select_dtypes(include='number')
            if numeric.empty:
                continue
            fig,ax=plt.subplots(figsize=(8,5))
            for column in numeric:
                ax.plot(np.arange(len(numeric)),numeric[column].to_numpy(),label=str(column))
            ax.set(xlabel='Recorded step / epoch index',ylabel=str(key),title=f'scVI training: {key}')
            ax.legend(); finish(fig,history_dir/f'{key}.png',cfg['plots']['dpi'])
        from .core import environment_report
        dump_json(environment_report(),p/'environment_scvi.json')
        report={'n_cells':a.n_obs,'n_hvg':a.n_vars,'n_samples':a.obs['sample'].nunique(),
                'scvi_version':scvi.__version__,'memory_estimate':estimate,
                'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'seed':settings['seed'],'training_X':'raw integer counts cast to float32, not normalized',
                'sample_batch_does_not_make_treatment_confounding_identifiable':True}
        completed(marker,sig,required,**report)
        del a,model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def leiden_umaps(cfg):
    import anndata as ad
    import scanpy as sc
    sc.settings.n_jobs=cfg.get('execution',{}).get('integration_threads',8)
    from scipy import sparse
    p=integration_dir(cfg); settings=cfg['clustering']
    resolutions=[round(i/10,1) for i in range(1,11)]
    sig=digest({'version':VERSION,'latent':file_stamp(p/'latent.npy'),
                'obs':file_stamp(p/'training_obs.parquet'),'settings':settings,'resolutions':resolutions})
    marker=p/'10_cluster_complete.json'; required=[p/'embedding_clusters.h5ad',p/'cluster_summary.csv',p/'colors.csv']
    if reusable(marker,sig,required):
        return pd.read_csv(p/'cluster_summary.csv')
    with stage_lock(p,'cluster'):
        obs=pd.read_parquet(p/'training_obs.parquet'); z=np.load(p/'latent.npy')
        if len(obs)!=len(z) or not obs.index.is_unique:
            raise ValueError('Latent rows do not match unique training observation rows.')
        a=ad.AnnData(X=sparse.csr_matrix((len(obs),0),dtype=np.float32),obs=safe_frame(obs))
        a.obsm['X_scVI']=z
        from .compute.clustering import graph_and_cluster
        backend_report=graph_and_cluster(a,cfg,resolutions)
        dump_json(backend_report,p/'clustering_backend.json')
        # Exactly one embedding: changing Leiden resolution changes labels, not UMAP coordinates.
        out=p/'plots'; out.mkdir(exist_ok=True)
        summaries=[]; color_rows=[]
        for res in resolutions:
            key=f'leiden_{res:.1f}'.replace('.','p')
            cats=natural_order(a.obs[key].astype(str).unique())
            a.obs[key]=pd.Categorical(a.obs[key].astype(str),categories=cats,ordered=True)
            colors=distinct_palette(len(cats),settings['allow_palette_extension'])
            palette=dict(zip(cats,colors)); a.uns[key+'_colors']=np.asarray(colors,dtype=str)
            scatter_categorical(a.obsm['X_umap'],a.obs[key],f'scVI integration | Leiden resolution {res:.1f}',
                                out/f'UMAP_{key}.png',colors=palette,max_points=settings['max_umap_points'],
                                labels=('UMAP 1','UMAP 2'),dpi=cfg['plots']['dpi'],
                                point_size=settings['point_size'])
            counts=pd.crosstab(a.obs[key],a.obs['sample'],dropna=False)
            counts.to_csv(p/f'{key}_cluster_by_sample_counts.csv')
            counts.div(counts.sum(axis=1),axis=0).to_csv(p/f'{key}_within_cluster_sample_fractions.csv')
            counts.div(counts.sum(axis=0),axis=1).to_csv(p/f'{key}_within_sample_cluster_fractions.csv')
            sizes=a.obs[key].value_counts(sort=False)
            sizes.rename('n_cells').to_csv(p/f'{key}_cluster_sizes.csv')
            summaries.append({'resolution':res,'key':key,'n_clusters':len(cats),'n_cells':a.n_obs,
                              'smallest_cluster':int(sizes.min()),'largest_cluster':int(sizes.max()),
                              'n_non_microviz_colors':max(0,len(cats)-41)})
            color_rows.extend({'key':key,'cluster':c,'hex':palette[c],
                               'source':'microViz brewerPlus' if i<41 else 'LAB-distance extension'}
                              for i,c in enumerate(cats))
            print(summaries[-1])
        a.uns['vhd_scvi_embedding']={'one_umap_for_all_resolutions':True,'batch_key':'sample',
                                     'latent_source':str(p/'latent.npy'),'run_name':cfg['scvi']['run_name']}
        write_h5ad(a,p/'embedding_clusters.h5ad')
        pd.DataFrame(summaries).to_csv(p/'cluster_summary.csv',index=False)
        pd.DataFrame(color_rows).to_csv(p/'colors.csv',index=False)
        completed(marker,sig,required,n_resolutions=10,n_cells=a.n_obs)
    return pd.DataFrame(summaries)


def diagnostics(cfg):
    import anndata as ad
    p=integration_dir(cfg); a=ad.read_h5ad(p/'embedding_clusters.h5ad')
    out=p/'diagnostics'; out.mkdir(exist_ok=True)
    xy=a.obsm['X_umap']
    scatter_categorical(xy,a.obs['sample'],'scVI UMAP colored by sample',out/'UMAP_by_sample.png',
                        max_points=cfg['clustering']['max_umap_points'],labels=('UMAP 1','UMAP 2'),dpi=cfg['plots']['dpi'])
    for column in ['total_counts','n_genes_by_counts','pct_counts_mt','cell_area_um2','S_score','G2M_score']:
        if column in a.obs:
            scatter_numeric(xy,a.obs[column],f'scVI UMAP: {column}',out/f'UMAP_by_{column}.png',
                            coordinate_labels=('UMAP 1','UMAP 2'),dpi=cfg['plots']['dpi'],
                            max_points=cfg['clustering']['max_umap_points'],
                            log1p=column in ('total_counts','n_genes_by_counts'))
    for column in cfg['scvi']['diagnostic_metadata_columns']:
        if column not in a.obs:
            raise KeyError(f'Requested diagnostic metadata absent: {column}')
        scatter_categorical(xy,a.obs[column],f'scVI UMAP: {column}',out/f'UMAP_by_{column}.png',
                            labels=('UMAP 1','UMAP 2'),dpi=cfg['plots']['dpi'],
                            max_points=cfg['clustering']['max_umap_points'])
        pd.crosstab(a.obs['sample'],a.obs[column]).to_csv(out/f'sample_by_{column}_confounding_audit.csv')
    rows=[]
    for key in [x for x in a.obs if x.startswith('leiden_')]:
        for column in ['total_counts','n_genes_by_counts','pct_counts_mt','cell_area_um2']:
            if column in a.obs:
                stats=a.obs.groupby(key,observed=True)[column].agg(['count','median','mean'])
                stats.to_csv(out/f'{key}_{column}_summary.csv')
    return out


def writeback_annotations(cfg,spec):
    """Append a SMALL zero-gene annotation table. Never rewrite huge bins or duplicate cell counts."""
    import anndata as ad
    import spatialdata as sd
    from scipy import sparse
    from .storage import parse_table
    p=integration_dir(cfg); sp=sample_paths(cfg,spec)
    if not (sp['report']/'storage_manifest.json').exists():
        raise RuntimeError('Writeback is restricted to stores built by this new workflow, not legacy originals.')
    a=ad.read_h5ad(p/'embedding_clusters.h5ad')
    selected=a.obs['sample'].astype(str).eq(spec['sample']).to_numpy()
    a=a[selected].copy()
    obs,_=table_metadata(sp['store'],'cells')
    if not a.obs_names.isin(obs.index).all():
        raise ValueError('Integrated cell IDs are not a subset of the target sample cell table.')
    indexer=obs.index.get_indexer(a.obs_names)
    annotation=pd.DataFrame(index=obs.index.copy())
    annotation['sample']=spec['sample']
    annotation['included_in_scvi']=False
    annotation.loc[a.obs_names,'included_in_scvi']=True
    for key in [x for x in a.obs if x.startswith('leiden_')]:
        values=pd.Series('not_integrated',index=obs.index)
        values.loc[a.obs_names]=a.obs[key].astype(str).to_numpy()
        annotation[key]=pd.Categorical(values,categories=[*a.obs[key].cat.categories,'not_integrated'])
    b=ad.AnnData(X=sparse.csr_matrix((len(obs),0),dtype=np.float32),obs=annotation)
    for key in ('X_scVI','X_umap'):
        values=np.full((len(obs),a.obsm[key].shape[1]),np.nan,dtype=np.float32)
        values[indexer]=a.obsm[key]; b.obsm[key]=values
    for key,value in a.uns.items():
        if key.endswith('_colors'):
            # Explicit neutral category, only for excluded cells in writeback; not part of cluster palette.
            b.uns[key]=np.asarray([*list(value),'#BDBDBD'],dtype=str)
    b.uns['vhd_scvi_run']=cfg['scvi']['run_name']
    sig=digest({'embedding':file_stamp(p/'embedding_clusters.h5ad'),
                'store_signature':json.loads((sp['report']/'storage_manifest.json').read_text())['signature'],
                'sample':spec['sample']})
    b.uns['vhd_scvi_signature']=sig
    marker=sp['report']/'scvi_writeback.json'
    if (sp['store']/'tables'/'cells_scvi').exists():
        previous=load_table(sp['store'],'cells_scvi')
        if previous.uns.get('vhd_scvi_signature')!=sig:
            raise FileExistsError('cells_scvi exists from a different run. No automatic replacement. '
                                 'Use another output store or deliberately archive this annotation element.')
        return {'sample':spec['sample'],'status':'reused'}
    with stage_lock(sp['work'],'scvi_writeback'):
        s=sd.read_zarr(sp['store'],selection=('shapes',))
        s.tables['cells_scvi']=parse_table(b,str(obs.region.astype(str).iloc[0]),obs['instance_id'].astype(str))
        s.write_element('cells_scvi',overwrite=False)
        check=load_table(sp['store'],'cells_scvi')
        if not check.obs_names.equals(obs.index) or check.uns['vhd_scvi_signature']!=sig:
            raise ValueError('Annotation table read-back failed.')
        annotation.to_parquet(sp['report']/'scvi_cell_annotations.parquet')
        report={'sample':spec['sample'],'n_integrated':a.n_obs,'n_all_cells':len(obs),
                'signature':sig,'counts_duplicated':False,'table':'cells_scvi'}
        dump_json(report,marker)
    return report
