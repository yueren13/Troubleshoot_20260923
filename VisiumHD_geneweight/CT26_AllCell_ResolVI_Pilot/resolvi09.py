"""Single-sample ResolVI comparison on matched cells and marker-masked features."""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

HOLD = ("Cd4", "Cd8a", "Cd8b1")
PAN = ("Cd3d", "Cd3e", "Cd3g", "Trac", "Trbc1", "Trbc2")
PANELS = {"TCR": PAN, "myeloid": ("C1qa", "C1qb", "C1qc", "Lyz2", "Adgre1", "Csf1r"),
          "epithelial": ("Epcam", "Krt8", "Krt18", "Krt19"),
          "stromal": ("Col1a1", "Col1a2", "Dcn", "Lum")}
DEFAULT_TYPES = {"T": r"^T[ _-]*cells?(?:\b|_)", "macrophage": r"macroph|monocyt",
                 "tumor": r"tumou?r|epithel"}
GIB = 1024**3


def _sample_geometry(coords, sections, mode, target, halo, section, center):
    if mode == "full":
        return np.arange(len(coords)), np.ones(len(coords), bool), {}
    if mode != "tile":
        raise ValueError("mode must be tile or full.")
    section = str(section) if section is not None else pd.Series(sections).value_counts().index[0]
    region = np.flatnonzero(sections == section)
    if not len(region):
        raise ValueError(f"Unknown section {section!r}. Available: {np.unique(sections)}")
    if center is None:
        # Center on an actual segmented cell so a gap between tissue pieces cannot be the center.
        med = np.median(coords[region], axis=0)
        center = coords[region[np.argmin(np.sum((coords[region]-med)**2, axis=1))]]
    center = np.asarray(center, float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_um needs finite x,y coordinates.")
    d = np.linalg.norm(coords[region] - center, axis=1)
    n = min(target, len(region))
    inner = float(np.partition(d, n-1)[n-1])
    outer = inner + halo
    fit = region[d <= outer]
    return fit, (d[d <= outer] <= inner), dict(section=section, center_um=center.tolist(),
             inner_radius_um=inner, outer_radius_um=outer, entire_section=len(fit)==len(region))


def _read_rows(path, rows, columns=None, size=2048):
    """Bound raw HDF5 reads by source-row span, including random feature-selection rows."""
    import h5py
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0 or np.any(np.diff(rows) <= 0):
        raise ValueError("Need a nonempty set of sorted unique source rows.")
    parts = []
    with h5py.File(path, "r") as h:
        group = h["X"]
        if not isinstance(group, h5py.Group) or group.attrs.get("encoding-type") != "csr_matrix":
            raise TypeError("Expected standardized sparse CSR observed counts in X.")
        nrow, ncol = map(int, group.attrs["shape"])
        if rows[0] < 0 or rows[-1] >= nrow:
            raise IndexError("Cell positions outside the source matrix.")
        by_window = rows // size
        starts = np.r_[0, np.flatnonzero(np.diff(by_window)) + 1, len(rows)]
        for lo, hi in zip(starts[:-1], starts[1:]):
            requested = rows[lo:hi]
            first, last = int(requested[0]), int(requested[-1])
            ptr = group["indptr"][first:last+2]
            a, b = int(ptr[0]), int(ptr[-1])
            block = sparse.csr_matrix((group["data"][a:b], group["indices"][a:b], ptr-a),
                                      shape=(last-first+1, ncol))
            block = block[requested-first]
            if np.any(block.data < 0) or not np.isfinite(block.data).all() or np.any(block.data != np.rint(block.data)):
                raise ValueError("Input must be nonnegative integer-valued observed counts.")
            parts.append(block if columns is None else block[:, columns])
    return sparse.vstack(parts, format="csr")


def _select_genes(X, genes, n):
    # Select features on a bounded sample. Mask direct lineage genes BEFORE normalization and selection.
    low = np.char.lower(np.asarray(genes, dtype=str))
    allow = ~np.isin(low, [g.lower() for g in HOLD])
    allow &= ~np.char.startswith(low, "mt-")
    x = X[:, allow].astype(np.float32)
    total = np.asarray(x.sum(1)).ravel()
    x = x.multiply((10000 / np.maximum(total,1))[:,None]).tocsr()
    x.data = np.log1p(x.data)
    mu = np.asarray(x.mean(0)).ravel()
    variance = np.maximum(np.asarray(x.power(2).mean(0)).ravel() - mu*mu, 0)
    # Equalize abundance bias with within-mean-bin dispersion ranks.
    detected = np.asarray(x.getnnz(axis=0)).ravel()
    good = (detected >= 3) & (variance>0)
    scores = np.full(len(mu), -np.inf)
    if np.any(good):
        groups = pd.qcut(mu[good], 20, labels=False, duplicates="drop")
        if np.all(pd.isna(groups)):
            groups = np.zeros(good.sum(), int)
        disp = pd.Series(variance[good] / np.maximum(mu[good], 1e-8))
        scores[good] = disp.groupby(np.asarray(groups)).rank(pct=True).fillna(0).to_numpy()
    pool = np.flatnonzero(allow)
    chosen = pool[np.argsort(-scores,kind="stable")[:min(n,int(good.sum()))]]
    control = np.array([g.lower() for members in PANELS.values() for g in members])
    chosen = np.union1d(chosen,pool[np.isin(low[pool],control) & (detected>0)])
    if len(chosen)<100:
        raise ValueError(f"Only {len(chosen)} usable features. Check the assay and feature selection.")
    return chosen


def _spatial_graph(xy, sections, k):
    # Same keys, units and squared-distance convention as the scvi-tools RESOLVI preparation.
    nn=np.empty((len(xy),k),np.int64)
    dd=np.empty((len(xy),k),np.float32)
    for section in np.unique(sections):
        ids=np.flatnonzero(sections==section)
        if len(ids)<=k:
            raise ValueError(f"Section {section} has {len(ids)} model-eligible cells for k={k}.")
        d,loc=cKDTree(xy[ids]).query(xy[ids],k=k+1)
        # Remove self by identity, including coincident-coordinate ties.
        order=np.argsort(np.where(loc==np.arange(len(ids))[:,None],np.inf,d),axis=1,kind="stable")[:,:k]
        nn[ids]=ids[np.take_along_axis(loc,order,axis=1)]
        dd[ids]=np.take_along_axis(d,order,axis=1)**2
    if np.median(np.partition(dd,5,axis=1)[:,5]) <= 0:
        raise ValueError("Zero median sixth-neighbor distance. Verify centroids, units and duplicates.")
    return nn,dd


def _physical_dist(xy, groups, donor):
    out=np.full(len(xy),np.nan)
    for g in np.unique(groups):
        rows=np.flatnonzero(groups==g)
        source=rows[donor[rows]]
        if len(source):
            out[rows]=cKDTree(xy[source]).query(xy[rows])[0]
    return out


def _anchor_metric(z, raw, tcell, xy, sections, block_um, seed):
    mask=tcell & (raw["CD4_only"] | raw["CD8_only"]) & raw["panT"]
    pos=np.flatnonzero(mask)
    y=raw["CD8_only"][pos]
    blocks=np.floor(xy[pos]/block_um).astype(int)
    groups=np.array([f"{s}|{x}|{yy}" for s,(x,yy) in zip(sections[pos],blocks)])
    empty=dict(n_anchors=len(pos), n_scored=0, n_blocks=len(np.unique(groups)), AUC=np.nan)
    if len(np.unique(groups))<2 or len(np.unique(y))<2: return empty
    estimates=np.full(len(pos),np.nan)
    cv=GroupKFold(min(5,len(np.unique(groups))))
    for train,test in cv.split(z[pos],y,groups):
        if len(np.unique(y[train]))<2: continue
        knn=KNeighborsClassifier(n_neighbors=min(25,len(train)),weights="distance")
        estimates[test]=knn.fit(z[pos[train]],y[train]).predict_proba(z[pos[test]])[:,1]
    measured=np.isfinite(estimates)
    if len(np.unique(y[measured]))<2:return empty
    return dict(n_anchors=len(pos), n_scored=int(measured.sum()),n_blocks=len(np.unique(groups)),
                AUC=float(roc_auc_score(y[measured],estimates[measured])))


def _graph(z, resolution, seed):
    import anndata as ad
    import scanpy as sc
    tmp=ad.AnnData(X=sparse.csr_matrix((len(z),1),dtype=np.float32))
    tmp.obsm["latent"]=np.ascontiguousarray(z,dtype=np.float32)
    sc.pp.neighbors(tmp,n_neighbors=min(20,len(z)-1),use_rep="latent",random_state=seed)
    sc.tl.leiden(tmp,resolution=resolution,random_state=seed,key_added="labels",flavor="igraph",
                 n_iterations=2,directed=False)
    return tmp.obs["labels"].astype(str).to_numpy()


def _raw_pca(adata, latent, seed):
    import scanpy as sc
    # Restore original sparse counts on the same AnnData after the transformation.
    counts=adata.X
    try:
        adata.X=counts.astype(np.float32,copy=True)
        sc.pp.normalize_total(adata,target_sum=10000)
        sc.pp.log1p(adata)
        sc.pp.pca(adata,n_comps=latent,zero_center=True,svd_solver="arpack",random_state=seed)
        return adata.obsm.pop("X_pca")
    finally:
        adata.X=counts


def _aggregate(expr, genes, groups, idx, distance, where):
    rows=[]
    names=pd.Index(genes)
    for panel,members in PANELS.items():
        kept = names.isin(members)
        if not kept.any(): continue
        scores=expr[:,kept].sum(axis=1)
        for typ,mask in groups.items():
            for zone,zone_mask in [("all",np.ones(len(idx),bool)),("near_macrophage_25um",distance[idx]<=25),
                                   ("far_from_macrophage_50um",distance[idx]>=50)]:
                select=mask[idx] & zone_mask
                if select.any():
                    rows.append(dict(method=where,panel=panel,compartment=typ,zone=zone,
                                     n=int(select.sum()),mean_per_10k=float(np.mean(scores[select])),
                                     median_per_10k=float(np.median(scores[select])),
                                     genes=",".join(names[kept])))
    return rows


def _write_report(dest,metrics,signals,labels,xy,core,seed):
    import matplotlib.pyplot as plt
    metrics.to_csv(dest/"embedding_comparison.csv",index=False)
    signals.to_csv(dest/"marker_controls.csv",index=False)
    lookup = signals.set_index(["method", "panel", "compartment", "zone"])["mean_per_10k"]
    contrasts = []
    for method in metrics.method:
        for panel,typ,reference in [("myeloid","T_panT_observed","macrophage"),
                                    ("TCR","T_panT_observed","macrophage"),
                                    ("epithelial","T_panT_observed","tumor")]:
            numerator = lookup.get((method,panel,typ,"all"), np.nan)
            denominator = lookup.get((method,panel,reference,"all"),np.nan)
            near = lookup.get((method,panel,typ,"near_macrophage_25um"),np.nan)
            far = lookup.get((method,panel,typ,"far_from_macrophage_50um"),np.nan)
            contrasts.append(dict(method=method,panel=panel,compartment=typ,reference=reference,
                                  focal_mean_per_10k=numerator,reference_mean_per_10k=denominator,
                                  focal_to_reference=numerator/denominator if denominator>0 else np.nan,
                                  near_minus_far_per_10k=near-far))
    pd.DataFrame(contrasts).to_csv(dest/"specificity_and_distance.csv",index=False)
    methods=list(metrics.method)
    rng=np.random.default_rng(seed)
    ii=rng.choice(np.flatnonzero(core),size=min(35000,int(core.sum())),replace=False)
    fig,axes=plt.subplots(1,len(methods),figsize=(5*len(methods),4.4),squeeze=False,constrained_layout=True)
    for ax,method in zip(axes[0],methods):
        code=pd.Categorical(labels[method]).codes
        ax.scatter(*xy[ii].T,c=code[ii],s=1.5,cmap="tab20",linewidths=0,rasterized=True)
        ax.set_title(f"{method}  ·  {len(np.unique(code))} clusters");ax.set_aspect("equal")
        ax.set_xlabel("x (µm)");ax.set_ylabel("y (µm)")
    fig.savefig(dest/"spatial_leiden.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,panel in zip(axes,["myeloid","TCR"]):
        data=signals.loc[signals.panel.eq(panel)&signals.zone.eq("all")]
        if len(data):
            data.pivot(index="compartment",columns="method",values="mean_per_10k").plot.bar(ax=ax)
        ax.set_title(panel+" signal by compartment");ax.set_ylabel("Mean panel sum / 10,000 model-gene UMIs")
    fig.savefig(dest/"marker_controls.png",dpi=175);plt.close(fig)


def _freeze_zip(dest,files,target):
    tmp=target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if (dest/f).is_file(): z.write(dest/f,f)
    os.replace(tmp,target)


def _prepare_gpu(torch, runtime):
    # Lightning and ResolVI's auto-selected prediction paths use the first visible GPU.
    # This index is local to the kernel, including when CUDA_VISIBLE_DEVICES is set.
    device = torch.device("cuda:0")
    runtime.update(logical_device=str(device), cuda_probe_succeeded=False,
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        runtime["cuda_available"] = torch.cuda.is_available()
        runtime["visible_device_count"] = torch.cuda.device_count()
        if not runtime["cuda_available"] or runtime["visible_device_count"] < 1:
            raise RuntimeError("PyTorch cannot see a usable CUDA device.")
        # A tensor allocation initializes the allocator. Reading the result back to the CPU
        # also verifies that this kernel can run a CUDA operation before fitting either model.
        result = torch.ones(1, device=device).add_(1).item()
        if result != 2:
            raise RuntimeError(f"The CUDA test returned {result!r}, expected 2.")
        runtime["cuda_probe_succeeded"] = True
    except (RuntimeError, AssertionError, ValueError) as exc:
        raise RuntimeError(
            f"09B could not run a one-element tensor on {device}. Check the GPU assignment "
            "and CUDA-enabled PyTorch in this notebook kernel. See runtime.json in the "
            f"diagnostic ZIP. PyTorch reported: {exc}"
        ) from exc

    try:
        runtime["gpu"] = torch.cuda.get_device_name(device)
    except (RuntimeError, ValueError, AttributeError) as exc:
        runtime["gpu_name_error"] = str(exc)
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except (RuntimeError, ValueError, AttributeError) as exc:
        # Peak allocation is optional telemetry. A failed reset invalidates the run-specific peak.
        runtime["peak_reset_error"] = str(exc)
        print(f"GPU compute check passed. Peak memory reporting is unavailable: {exc}", flush=True)
    return device


def _record_gpu_peak(torch, device, runtime):
    runtime["peak_gpu_allocated_GiB"] = None
    if "peak_reset_error" not in runtime:
        try:
            runtime["peak_gpu_allocated_GiB"] = round(torch.cuda.max_memory_allocated(device) / GIB, 3)
        except (RuntimeError, ValueError, AttributeError) as exc:
            runtime["peak_read_error"] = str(exc)


def _training_batch_size(n_cells, requested):
    """Use the largest allowed fixed batch size that keeps at least two cells per step."""
    if not isinstance(requested, (int, np.integer)) or requested < 2:
        raise ValueError("batch_size must be an integer of at least 2 for batch normalization.")
    if n_cells < 2:
        raise ValueError("At least two model-eligible cells are needed for batch normalization.")
    for size in range(min(int(requested), n_cells), 1, -1):
        if n_cells % size != 1:
            return size
    raise ValueError(
        f"No fixed batch size from 2 through {requested} can partition {n_cells} cells "
        "without a one-cell batch. Increase batch_size to allow a larger minibatch."
    )


def run(h5ad_path,output_root,*,mode="tile",target_core_cells=12000,halo_um=100,section=None,
        center_um=None,microns_per_unit=1.0,n_features=1200,n_neighbors=8,batch_size=64,
        epochs=40,latent_dim=16,resolution=0.6,min_model_counts=5,run_scvi=True,
        accelerator="gpu",seed=17,block_um=200,marker_cell_cap=30000,type_patterns=None):
    """Matched raw PCA/scVI/ResolVI. Output a review ZIP and local cell assignments."""
    import anndata as ad
    import scvi
    import torch
    from scvi.external import RESOLVI
    if not all(k in inspect.signature(RESOLVI.setup_anndata).parameters for k in ["prepare_data","batch_key"]):
        raise RuntimeError("This package expects the documented scvi.external.RESOLVI setup API.")
    if accelerator not in {"cpu","gpu"}:
        raise ValueError("accelerator must be 'gpu' or 'cpu'.")
    if min(target_core_cells,epochs,n_features,batch_size,marker_cell_cap)<1 or n_neighbors<6 or not 2<=latent_dim<100:
        raise ValueError("Check positive settings, n_neighbors >=6 and latent_dim between 2 and 99.")
    if min_model_counts<1 or microns_per_unit<=0 or halo_um<0 or block_um<=0:raise ValueError("Invalid spatial or count settings.")
    path=Path(h5ad_path).expanduser().resolve()
    root=Path(output_root).expanduser().resolve()
    settings={k:v for k,v in locals().copy().items() if k in inspect.signature(run).parameters and k not in {"h5ad_path","output_root"}}
    config=dict(source=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                settings=settings,scvi=version("scvi-tools"),source_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    rid=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()[:12]
    root.mkdir(parents=True,exist_ok=True)
    dest=root/("09B_"+rid)
    review=root/f"09B_return_to_chat_{rid}.zip"
    files=["complete.json","settings.json","runtime.json","embedding_comparison.csv","marker_controls.csv",
           "mixture_by_cell_type.csv","specificity_and_distance.csv","spatial_leiden.png","marker_controls.png","model_genes.csv","training_history.csv"]
    if (dest/"complete.json").is_file():
        if not review.is_file(): _freeze_zip(dest,files,review)
        return dest,review
    # A failed run can be retried; keep diagnostic artifacts for each attempt.
    import tempfile
    stage=Path(tempfile.mkdtemp(prefix="09B_work_",dir=root))
    started=time.monotonic()
    runtime = dict(accelerator=accelerator, python=sys.version.split()[0],
                   torch=str(torch.__version__), cuda_build=torch.version.cuda,
                   scvi_tools=config["scvi"])
    try:
        (stage/"settings.json").write_text(json.dumps(config,indent=2,default=str))
        device = None
        model_devices = 1  # Lightning uses a device count for CPU.
        if accelerator == "gpu":
            device = _prepare_gpu(torch, runtime)
            model_devices = [device.index]
        (stage/"runtime.json").write_text(json.dumps(runtime,indent=2))
        print("09B runtime",json.dumps(runtime),flush=True)
        scvi.settings.seed=seed
        rng=np.random.default_rng(seed)
        b=ad.read_h5ad(path,backed="r")
        try:
            names=np.asarray(b.var_names,dtype=str)
            if not b.var_names.is_unique or not b.obs_names.is_unique: raise ValueError("Gene symbols and cell identifiers must be unique.")
            if not {"cell_type","section_id"}<=set(b.obs): raise KeyError("09B expects cell_type and section_id from the pilot input adapter.")
            if b.obs.section_id.isna().any():raise ValueError("Section identifiers missing.")
            if "sample" in b.obs and b.obs["sample"].nunique(dropna=False)!=1:raise ValueError("Give 09B one mouse at a time.")
            if not set(HOLD)<=set(names):raise ValueError("Check mouse gene symbols. Need Cd4, Cd8a, Cd8b1 for blind validation.")
            xy=np.asarray(b.obsm["spatial"],dtype=float)[:,:2]*microns_per_unit
            if xy.shape!=(b.n_obs,2) or not np.isfinite(xy).all():raise ValueError("Need finite 2D cell centroids.")
            secs=b.obs.section_id.astype(str).to_numpy()
            fit_rows,core,geometry=_sample_geometry(xy,secs,mode,target_core_cells,halo_um,section,center_um)
            lookup_rows=np.sort(rng.choice(fit_rows,min(6000,len(fit_rows)),replace=False))
            chosen=_select_genes(_read_rows(path,lookup_rows),names,n_features)
            held_idx=np.flatnonzero(np.isin(names,HOLD))
            used=np.union1d(chosen,held_idx)
            X=_read_rows(path,fit_rows,used)
            hold=X[:,np.searchsorted(used,held_idx)].toarray()>0
            held={names[i]:hold[:,j] for j,i in enumerate(held_idx)}
            counts=X[:,np.searchsorted(used,chosen)].astype(np.float32);del X
            counts.eliminate_zeros()
            keep=np.asarray(counts.sum(1)).ravel()>=min_model_counts
            raw={"CD4_only":(held["Cd4"] & ~(held["Cd8a"]|held["Cd8b1"]))[keep],
                 "CD8_only":(~held["Cd4"] & (held["Cd8a"]|held["Cd8b1"]))[keep]}
            fit_rows=fit_rows[keep];counts=counts[keep];core=core[keep]
            obs=b.obs.iloc[fit_rows][["cell_type","section_id"]].copy()
            ids=np.asarray(b.obs_names[fit_rows],dtype=str)
            measured=names[chosen];coords=xy[fit_rows];secs=secs[fit_rows]
        finally:b.file.close()
        if len(obs)<max(35,latent_dim+2):raise ValueError("Too few model-eligible cells.")
        # Spatial halo and count filtering both affect the 100% training split used by each model.
        n_fit = len(obs)
        batch_plan = {"fitted_cells": n_fit}
        runtime["training_batches"] = batch_plan
        train_batch_size = _training_batch_size(n_fit, batch_size)
        full_batches, remainder = divmod(n_fit, train_batch_size)
        batch_plan.update(requested_batch_size=int(batch_size),
                          effective_batch_size=train_batch_size,
                          batches_per_epoch=full_batches + bool(remainder),
                          final_batch_cells=remainder or train_batch_size)
        (stage/"runtime.json").write_text(json.dumps(runtime,indent=2))
        log=dict(source_cells=int(len(xy)),fitted_cells=len(obs),below_minimum_model_counts=int((~keep).sum()),
                 gene_selection_cells=len(lookup_rows),training_genes=len(measured),geometry=geometry,
                 original_core=int(core.sum()),withheld_direct_genes=list(HOLD),training_batches=batch_plan)
        nbr,dist=_spatial_graph(coords,secs,n_neighbors)
        if mode=="tile" and not geometry["entire_section"]:
            radial=np.linalg.norm(coords-np.asarray(geometry["center_um"]),axis=1)
            # Guarantee every evaluated core cell's fitted kth neighbor is closer than the edge of the training disk.
            core &= radial+np.sqrt(dist.max(1)) < geometry["outer_radius_um"]
        if core.sum()<20:raise ValueError("Insufficient core cells with complete neighborhoods. Increase halo or choose another center.")
        log["evaluation_core_cells"]=int(core.sum())
        log["spatial_neighbor_distance_p50_p95_um"]=np.quantile(np.sqrt(dist),[.5,.95]).tolist()
        log["hypothetical_dense_fit_GiB"]=round(len(obs)*len(measured)*4/GIB,3)
        log["count_CSR_GiB"]=round((counts.data.nbytes+counts.indptr.nbytes+counts.indices.nbytes)/GIB,3)
        pd.DataFrame({"model_gene":measured}).to_csv(stage/"model_genes.csv",index=False)
        print("Fit and memory profile",json.dumps(log,default=str),flush=True)
        core_ids=np.flatnonzero(core)
        patterns=dict(DEFAULT_TYPES,**(type_patterns or {}))
        groups={k:obs.cell_type.astype("string").str.contains(regex,case=False,regex=True,na=False).to_numpy(dtype=bool)
                for k,regex in patterns.items()}
        raw["panT"]=np.asarray(counts[:,np.isin(measured,PAN)].sum(1)).ravel()>0
        groups["T_panT_observed"]=groups["T"] & raw["panT"]
        distance=_physical_dist(coords,secs,groups["macrophage"])
        select=np.sort(rng.choice(core_ids,min(marker_cell_cap,len(core_ids)),replace=False))
        marker_names=measured[np.isin(measured,[g for p in PANELS.values() for g in p])]
        if not len(marker_names): raise ValueError("No panel genes in training data. Check symbol mapping.")
        model_total=np.asarray(counts[select].sum(1)).ravel()
        profile=counts[select][:,np.isin(measured,marker_names)].toarray()*10000/model_total[:,None]
        signals=_aggregate(profile,marker_names,groups,select,distance,"raw_PCA")
        # Make positional row codes explicit for upstream index-neighbor parameter tables.
        obs.index=pd.Index([f"p{i:09d}" for i in range(len(obs))])
        a=ad.AnnData(X=counts,obs=obs,var=pd.DataFrame(index=measured))
        a.obsm["X_spatial"]=coords.astype(np.float32)
        embed={"raw_PCA":_raw_pca(a,latent_dim,seed)}
        if run_scvi:
            scvi.settings.seed=seed
            scvi.model.SCVI.setup_anndata(a)
            simple=scvi.model.SCVI(a,n_latent=latent_dim,n_hidden=64)
            fit_start=time.monotonic()
            simple.train(max_epochs=epochs,batch_size=train_batch_size,train_size=1.0,
                         accelerator=accelerator,devices=model_devices,enable_progress_bar=False)
            log["scvi_fit_minutes"]=round((time.monotonic()-fit_start)/60,3)
            embed["scVI"]=simple.get_latent_representation(batch_size=batch_size)
            pred=simple.get_normalized_expression(indices=select,gene_list=marker_names.tolist(),library_size=10000,
                      n_samples=3,return_mean=True,return_numpy=False,batch_size=batch_size)
            signals+=_aggregate(pred.loc[:,marker_names].to_numpy(),marker_names,groups,select,distance,"scVI")
            del simple,pred;gc.collect()
            if accelerator=="gpu":torch.cuda.empty_cache()
        scvi.settings.seed=seed
        a.obsm["index_neighbor"]=nbr;a.obsm["distance_neighbor"]=dist
        RESOLVI.setup_anndata(a,prepare_data=False)
        rv=RESOLVI(a,n_latent=latent_dim,n_hidden=64,semisupervised=False)
        fit_start=time.monotonic()
        rv.train(max_epochs=epochs,batch_size=train_batch_size,accelerator=accelerator,device=model_devices,
                 n_epochs_kl_warmup=max(1,min(20,epochs//2)),enable_progress_bar=False)
        log["resolvi_fit_minutes"]=round((time.monotonic()-fit_start)/60,3)
        embed["ResolVI"]=rv.get_latent_representation(batch_size=batch_size)
        pred=rv.get_normalized_expression(indices=select,gene_list=marker_names.tolist(),library_size=10000,
                  n_samples=3,return_mean=True,return_numpy=False,batch_size=batch_size)
        signals+=_aggregate(pred.loc[:,marker_names].to_numpy(),marker_names,groups,select,distance,"ResolVI")
        history=getattr(rv,"history",None)
        if isinstance(history,dict):
            curves=[pd.Series(np.asarray(s).flatten(),name=k) for k,s in history.items() if np.asarray(s).ndim<=2]
            if curves:pd.concat(curves,axis=1).to_csv(stage/"training_history.csv",index_label="step")
        # Upstream Pyro inference can be expensive. Limit mixture sampling to a fixed diagnostic cohort.
        midx=np.sort(rng.choice(select,min(1000,len(select)),replace=False))
        mix=rv.sample_posterior(indices=midx,return_sites=["mixture_proportions"],num_samples=3,
             batch_size=batch_size,accelerator=accelerator,device=model_devices,return_samples=False,summary_frequency=1)
        fractions=np.asarray(mix["post_sample_means"]["mixture_proportions"]).reshape(len(midx),3)
        mass=fractions.sum(axis=1)
        if not np.isfinite(fractions).all() or (fractions<0).any() or (mass<=0).any():
            raise ValueError("Unexpected nonfinite or negative ResolVI mixture weights.")
        # The 1.3.3 and 1.5.0.post1 guides add numerical epsilons to softmax outputs
        # (including diffusion_eps). Keep the actual weights as well as their relative shares.
        log["raw_mixture_weight_sum_range"]=[float(mass.min()),float(mass.max())]
        mixdf=pd.DataFrame(fractions/mass[:,None],columns=["intrinsic_share","diffusion_share","background_share"])
        for j,k in enumerate(["intrinsic","diffusion","background"]):
            mixdf[k+"_weight"]=fractions[:,j]
        mixdf["cell_type"]=obs.cell_type.astype(str).to_numpy()[midx]
        mixdf.groupby("cell_type").agg(["size","median","mean"]).to_csv(stage/"mixture_by_cell_type.csv")
        assignments=pd.DataFrame({"cell_id":ids,"cell_type":obs.cell_type.astype(str).to_numpy(),
                                  "section_id":secs,"evaluation_core":core})
        metrics=[];labels={}
        for name,z in embed.items():
            if not np.isfinite(z).all(): raise ValueError(f"Nonfinite embedding for {name}.")
            np.save(stage/(name+"_latent.npy"),np.asarray(z,np.float32),allow_pickle=False)
            labels[name]=_graph(z,resolution,seed)
            assignments[name+"_leiden"]=labels[name]
            m=_anchor_metric(z,raw,groups["T"] & core,coords,secs,block_um,seed)
            m.update(method=name,n_core=int(core.sum()),n_clusters=len(np.unique(labels[name][core])),
                     annotation_AMI=adjusted_mutual_info_score(obs.cell_type.astype(str).to_numpy()[core],labels[name][core]))
            metrics.append(m)
        assignments.to_csv(stage/"local_cell_assignments.csv.gz",index=False)
        _write_report(stage,pd.DataFrame(metrics),pd.DataFrame(signals),labels,coords,core,seed)
        log["minutes"]=round((time.monotonic()-started)/60,2)
        if device is not None:
            _record_gpu_peak(torch, device, runtime)
            log["gpu"] = runtime.get("gpu")
            log["peak_gpu_allocated_GiB"] = runtime["peak_gpu_allocated_GiB"]
            (stage/"runtime.json").write_text(json.dumps(runtime,indent=2))
        log["type_patterns"]=patterns;log["diagnostic_marker_cells"]=len(select)
        log["diagnostic_mixture_cells"]=len(midx)
        (stage/"complete.json").write_text(json.dumps(log,indent=2))
        if dest.exists():raise FileExistsError(f"Another run already created {dest}.")
        stage.rename(dest)
        _freeze_zip(dest,files,review)
        print("09B aggregate review ZIP:",review,flush=True)
        return dest,review
    except Exception as exc:
        import traceback
        if stage.exists():
            (stage/"error.txt").write_text(traceback.format_exc())
            (stage/"runtime.json").write_text(json.dumps(runtime,indent=2))
            failure=root/(stage.name+"_diagnostics.zip")
            _freeze_zip(stage,["settings.json","runtime.json","error.txt","model_genes.csv","training_history.csv"],failure)
            exc.diagnostics_zip = failure
            print("09B diagnostic ZIP:",failure,flush=True)
        raise
