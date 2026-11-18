"""Streaming single-mouse gene maps and matched positive-control comparisons for 09A."""
from __future__ import annotations

from pathlib import Path
from importlib import metadata
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

PANELS = {
    "T cell": ["Cd3d", "Cd3e", "Cd3g", "Trac", "Trbc1", "Trbc2"],
    "B cell": ["Cd79a", "Cd79b", "Ms4a1", "Cd19"],
    "Myeloid": ["C1qa", "C1qb", "C1qc", "Csf1r", "Lyz2", "Adgre1"],
    "Epithelial": ["Epcam", "Krt8", "Krt18", "Krt19"],
    "Endothelial": ["Cdh5", "Pecam1", "Kdr", "Emcn"],
    "Fibroblast": ["Col1a1", "Col1a2", "Dcn", "Lum"],
    "Cycling": ["Mki67", "Top2a", "Cenpf", "Cdk1"],
    "CD8": ["Cd8a", "Cd8b1"], "CD4 companion": ["Cd4", "Zbtb7b", "Cd40lg"],
    "Regulatory": ["Foxp3", "Il2ra", "Ctla4", "Ikzf2"],
}
# Prespecified pairs prevent choosing markers based on favorable embeddings.
PAIRS = {
    "T cell": [("Cd3d", "Cd3e"), ("Trac", "Cd3g")],
    "B cell": [("Cd79a", "Cd79b"), ("Cd79a", "Ms4a1")],
    "Myeloid": [("C1qa", "C1qb"), ("C1qa", "Csf1r")],
    "Epithelial": [("Epcam", "Krt18"), ("Krt8", "Krt18")],
    "Endothelial": [("Cdh5", "Pecam1"), ("Cdh5", "Kdr")],
    "Fibroblast": [("Col1a1", "Col1a2"), ("Dcn", "Lum")],
    "Cycling": [("Mki67", "Top2a"), ("Cenpf", "Cdk1")],
    "CD8": [("Cd8a", "Cd8b1")],
    "CD4 companion": [("Cd4", "Zbtb7b"), ("Cd4", "Cd40lg")],
    "Regulatory": [("Foxp3", "Il2ra"), ("Foxp3", "Ctla4")],
}
MARKERS = {g for group in PANELS.values() for g in group}
VIEWS = ("all_cells", "matched_all", "matched_T")
TITLE = {"all_cells": "Whole tissue · all cells", "matched_all": "Whole tissue · matched budget",
         "matched_T": "T cells · matched budget"}


def _json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n", encoding="utf8")


def _identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _rows(h, group, rows, columns=None):
    """Read selected rows with h5py; gather sparse intervals in bounded contiguous windows."""
    rows = np.asarray(rows, dtype=np.int64)
    dim = int(group.attrs["shape"][1])
    index = h["ptr"]
    # Splitting at large gaps avoids reading sparse arrays for the whole sample to collect a few cells.
    cuts = np.r_[0, np.where(np.diff(rows) > 48)[0] + 1, len(rows)]
    pieces = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        seg = rows[lo:hi]
        first, last = int(seg[0]), int(seg[-1]) + 1
        start, stop = int(index[first]), int(index[last])
        local = sparse.csr_matrix((group["data"][start:stop], group["indices"][start:stop],
                                   index[first:last+1] - start), shape=(last-first, dim))
        if len(seg) != last-first:
            local = local[seg-first]
        pieces.append(local)
    one = pieces[0] if len(pieces) == 1 else sparse.vstack(pieces, format="csr")
    if columns is not None:
        one = one[:, columns].tocsr()
    one.sum_duplicates()
    one.eliminate_zeros()
    if one.data.size and (not np.isfinite(one.data).all() or (one.data < 0).any()):
        raise ValueError("Counts must be finite and nonnegative.")
    one.data = np.ones(one.nnz, np.float32)
    return one


def _matched(obs, tmask, max_n, bins, rng):
    depth = obs["total_counts"].to_numpy(float)
    strata = np.empty(len(obs), dtype=int)
    offset = 0
    for section in pd.unique(obs["section_id"]):
        ids = np.flatnonzero(obs["section_id"].to_numpy() == section)
        breaks = np.unique(np.quantile(depth[ids], np.arange(1, bins) / bins))
        s = np.searchsorted(breaks, depth[ids], side="left")
        strata[ids] = s + offset
        offset += len(breaks)+1
    _, strata = np.unique(strata, return_inverse=True)
    ts = np.flatnonzero(tmask)
    if not ts.size:
        raise ValueError("No T cells matched t_labels. Inspect cell_type labels in the standardized sample.")
    select_t = np.sort(rng.choice(ts, min(max_n, len(ts)), replace=False))
    blocks = []
    for s, n in zip(*np.unique(strata[select_t], return_counts=True)):
        pool = np.flatnonzero(strata == s)
        blocks.append(rng.choice(pool, n, replace=False))
    select_all = np.sort(np.concatenate(blocks))
    if not np.array_equal(np.bincount(strata[select_t]), np.bincount(strata[select_all])):
        raise AssertionError("Matched depth/section composition differs.")
    return strata, {"all_cells": np.arange(len(obs)), "matched_all": select_all, "matched_T": select_t}


def _summarize(h, group, rows, strata, genes, pos, chunk, sketch_size, rng):
    """First pass over the requested rows. Exact strata moments and anchor co-detection."""
    n_strata = int(strata.max()) + 1
    anchors = [g for g in sorted(MARKERS) if g in pos]
    ai = [pos[g] for g in anchors]
    counts = np.zeros((len(genes), n_strata), dtype=np.float64)
    co = np.zeros((len(genes), len(ai)), dtype=np.float64)
    sketch = np.zeros((len(genes), sketch_size), dtype=np.float32)
    stratum_sketch = np.zeros((n_strata, sketch_size), dtype=np.float32)
    ns = np.bincount(strata[rows], minlength=n_strata)
    for lo in range(0, len(rows), chunk):
        ids = rows[lo:lo+chunk]
        x = _rows(h, group, ids)
        rr = np.arange(len(ids))
        mem = sparse.csr_matrix((np.ones(len(ids), np.float32), (rr, strata[ids])),
                                shape=(len(ids), n_strata))
        counts += (x.T @ mem).toarray()
        if len(ai):
            co += (x.T @ x[:, ai]).toarray()
        buckets = rng.integers(sketch_size, size=len(ids))
        signs = rng.choice(np.array([-1, 1], np.float32), size=len(ids))
        proj = sparse.csr_matrix((signs, (rr, buckets)), shape=(len(ids), sketch_size))
        sketch += (x.T @ proj).toarray()
        stratum_sketch += (mem.T @ proj).toarray()
    means = (counts / np.maximum(ns, 1)).astype(np.float32)
    n = counts.sum(axis=1)
    ss = np.maximum(n - (counts * means.astype(float)).sum(axis=1), 0)
    co -= (counts / np.maximum(ns, 1)) @ counts[ai].T
    denom = np.sqrt(ss[:, None] * ss[ai])
    corr = np.divide(co, denom, out=np.full_like(co, np.nan), where=denom > 1e-8)
    sketch -= means @ stratum_sketch
    weight = np.zeros(len(ss), np.float32)
    np.divide(1, np.sqrt(ss), out=weight, where=ss > 1e-8)
    sketch *= weight[:, None]
    return dict(n=n, ss=ss, means=means, sketch=sketch, weights=weight,
                corr=corr, anchors=anchors, ai=ai, rows=rows)


def _coords(h, group, summ, axis, strata, chunk, n_pcs):
    """Refine a sketched basis by one streamed exact residual covariance application (Nyström)."""
    s = summ["sketch"][axis]
    s = s - s.mean(axis=0)
    q, _ = np.linalg.qr(s)
    q = (q - q.mean(axis=0)).astype(np.float32)
    w = summ["weights"][axis]
    wq = w[:, None] * q
    means = summ["means"][axis].T
    offset = means @ wq
    csum = np.zeros_like(q, dtype=np.float64)
    nstr = means.shape[0]
    for lo in range(0, len(summ["rows"]), chunk):
        ids = summ["rows"][lo:lo+chunk]
        x = _rows(h, group, ids, columns=axis)
        group_id = strata[ids]
        u = x @ wq - offset[group_id]
        strsum = np.zeros((nstr, u.shape[1]), dtype=np.float64)
        np.add.at(strsum, group_id, u)
        csum += (x.T @ u - means.T @ strsum) * w[:, None]
    csum -= csum.mean(axis=0)
    small = q.T @ csum
    val, vec = np.linalg.eigh(.5 * (small + small.T))
    good = val > max(1e-10, val.max()*1e-9)
    if good.sum() < 2:
        raise ValueError("The selected genes have fewer than two informative dimensions.")
    f = csum @ (vec[:, good] / np.sqrt(val[good]))
    u, s, _ = np.linalg.svd(f, full_matrices=False)
    k = min(n_pcs, len(s))
    return (u[:, :k] * s[:k]).astype(np.float32)


def _graph(pc, k, resolution, seed):
    k = min(k, len(pc)-1)
    neighbor = NearestNeighbors(n_neighbors=k+1).fit(pc)
    dist, inds = neighbor.kneighbors(pc)
    keep = np.zeros((len(pc), k), dtype=int)
    for i, row in enumerate(inds):
        keep[i] = np.flatnonzero(row != i)[:k]
    inds = np.take_along_axis(inds, keep, axis=1)
    dist = np.take_along_axis(dist, keep, axis=1)
    wt = np.exp(-np.square(dist/np.maximum(dist[:, -1:], 1e-10)))
    graph = sparse.csr_matrix((wt.ravel(), (np.repeat(np.arange(len(pc)), k), inds.ravel())),
                              shape=(len(pc), len(pc)))
    graph = graph.maximum(graph.T)
    try:
        import umap
    except ImportError:
        coords, projection = pc[:, :2], "PCA"
    else:
        coords = umap.UMAP(n_components=2, n_neighbors=k, random_state=seed, n_jobs=1,
                           min_dist=.2, init="random").fit_transform(pc)
        projection = "UMAP"
    try:
        import igraph
        import leidenalg
    except ImportError:
        labels = AgglomerativeClustering(n_clusters=min(18, max(2, len(pc)//30)),
                                        connectivity=graph).fit_predict(pc)
        clustering = "Ward"
    else:
        tri = sparse.triu(graph, k=1).tocoo()
        g = igraph.Graph(n=len(pc), edges=list(zip(tri.row.tolist(), tri.col.tolist())))
        labels = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                          weights=tri.data.tolist(), seed=seed,
                                          resolution_parameter=resolution).membership
        clustering = "Leiden"
    return coords, np.asarray(labels), projection, clustering


def _controls(summary, axis, pc, modules, genes, min_cells):
    position = {g:i for i,g in enumerate(genes)}
    ap = {g:i for i,g in enumerate(summary["anchors"])}
    positions = {int(i): j for j,i in enumerate(axis)}
    out = []
    good = ((summary["n"] >= min_cells) & (len(summary["rows"]) - summary["n"] >= min_cells)
            & (summary["ss"] > 1e-8))
    for family, pairs in PAIRS.items():
        for a,b in pairs:
            row = dict(panel=family, anchor=a, partner=b, cells=len(summary["rows"]), direct_r=np.nan,
                       direct_rank=np.nan, PC_rank=np.nan, same_module=np.nan, decoys=0,
                       matched_null_percentile=np.nan, anchor_detections=np.nan, partner_detections=np.nan)
            if a in ap and b in position:
                i,j = position[a],position[b]
                row.update(anchor_detections=int(summary["n"][i]),partner_detections=int(summary["n"][j]))
                if good[i] and good[j]:
                    corr = summary["corr"][:, ap[a]]
                    row.update(direct_r=float(corr[j]), direct_rank=int(1+((corr>corr[j]) & good & (np.arange(len(genes))!=i)).sum()))
                    allowed = good & np.array([g not in MARKERS for g in genes]) & np.isfinite(corr)
                    null = np.flatnonzero(allowed)
                    nearest = np.argsort(np.abs(summary["n"][null] - summary["n"][j]), kind="stable")[:100]
                    v = corr[null[nearest]]
                    if len(v):
                        row.update(decoys=len(v),matched_null_percentile=float(np.mean((v<corr[j]) + .5*(v==corr[j]))))
                    if i in positions and j in positions:
                        x,y = positions[i],positions[j]
                        d = np.linalg.norm(pc-pc[x],axis=1)
                        row.update(PC_rank=int(1+((d<d[y]) & (np.arange(len(axis))!=x)).sum()),
                                   same_module=bool(modules[x]==modules[y]))
            out.append(row)
    return pd.DataFrame(out)


def _show_plot(maps, output):
    fig, axes = plt.subplots(1,len(VIEWS),figsize=(16,6),constrained_layout=True)
    palette = plt.get_cmap("tab10").colors
    for ax,name in zip(axes,VIEWS):
        frame = maps[name]; proj=frame.projection.iloc[0]
        ax.scatter(frame.x,frame.y,s=4,color="#BFC3C9",alpha=.55,rasterized=True)
        for i,(label,members) in enumerate(PANELS.items()):
            sub=frame[frame.gene.isin(members)]
            ax.scatter(sub.x,sub.y,s=27,color=palette[i],label=label,alpha=.95,linewidths=.2,edgecolors="black")
        ax.set(title=TITLE[name],xlabel=f"{proj} 1",ylabel=f"{proj} 2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc="outside lower center",frameon=False)
    fig.savefig(output/"gene_maps.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1,len(VIEWS),figsize=(16,5),constrained_layout=True)
    for ax,name in zip(axes,VIEWS):
        frame=maps[name];proj=frame.projection.iloc[0]
        ax.scatter(frame.x,frame.y,s=6,c=frame.module,cmap="tab20",alpha=.72,rasterized=True)
        ax.set(title=f"{TITLE[name]} · {frame.module.nunique()} {frame.clustering.iloc[0]} groups",
               xlabel=f"{proj} 1",ylabel=f"{proj} 2")
    fig.savefig(output/"gene_modules.png",dpi=190,bbox_inches="tight")
    plt.close(fig)


def run_gene_pilot(sample_h5ad, output_root, *, seed=17, max_genes=6000,
                   match_cells=30000, t_labels=("T cell 1","T cell 2"),
                   min_cells=12, chunksize=1536, n_depth_bins=10, sketch_size=72,
                   n_pcs=35, neighbors=20, resolution=1.0):
    """Return a completed immutable run folder for one standardized all-cell sample H5AD."""
    from anndata import read_h5ad
    if (min(max_genes,match_cells,sketch_size,n_pcs,neighbors)<2 or
        min(min_cells,chunksize,n_depth_bins)<1 or resolution<=0):
        raise ValueError("Check gene-map and QC parameters.")
    source=Path(sample_h5ad).expanduser().resolve()
    root=Path(output_root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    data=read_h5ad(source,backed="r")
    try:
        obs=data.obs[["sample","cell_type","section_id","total_counts"]].copy()
        genes=np.asarray(data.var_names.astype(str))
    finally:
        data.file.close()
    samples=obs["sample"].dropna().unique()
    if len(samples)!=1 or not obs.index.is_unique or not pd.Index(genes).is_unique:
        raise ValueError("Provide one standardized sample with unique genes and cell IDs.")
    if obs[["sample","section_id","total_counts"]].isna().any().any():
        raise ValueError("Missing sample, section or depth metadata.")
    if not np.isfinite(obs.total_counts).all() or (obs.total_counts<0).any():
        raise ValueError("Invalid observed library size.")
    sample=str(samples[0]); tmask=obs.cell_type.astype(str).isin(t_labels).to_numpy()
    obs["section_id"]=obs.section_id.astype(str)
    stats=source.stat()
    versions={}
    for lib in ("numpy","scipy","pandas","scikit-learn","matplotlib","anndata","h5py","umap-learn","igraph","leidenalg"):
        try: versions[lib]=metadata.version(lib)
        except metadata.PackageNotFoundError: versions[lib]=None
    settings=dict(sample=sample,input_path=str(source),size=stats.st_size,mtime=stats.st_mtime_ns,
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions=versions,
              seed=seed,max_genes=max_genes,match_cells=match_cells,t_labels=list(t_labels),min_cells=min_cells,
              chunksize=chunksize,n_depth_bins=n_depth_bins,sketch_size=sketch_size,n_pcs=n_pcs,
              neighbors=neighbors,resolution=resolution)
    run=root/("09A_"+_identity(settings)); receipt=run/"complete.json"
    if receipt.exists():
        payload=json.loads(receipt.read_text())
        if payload["settings"]!=settings or any(not (run/n).is_file() or (run/n).stat().st_size!=size for n,size in payload["files"].items()):
            raise RuntimeError(f"Incomplete or changed results in {run}. Choose a new output root or restore the run.")
        print("Reusing",run);return run
    tmp=Path(tempfile.mkdtemp(prefix=".09A-",dir=root))
    try:
        rng=np.random.default_rng(seed)
        strata,views=_matched(obs,tmask,match_cells,n_depth_bins,rng)
        pos={g:i for i,g in enumerate(genes)}
        maps={}; pair_frames=[]; setting_rows=[]; summaries={}
        with h5py.File(source,"r") as file:
            group=file["X"]
            if group.attrs.get("encoding-type")!="csr_matrix":
                raise ValueError("Expected raw CSR counts in X of the standardized one-mouse H5AD.")
            h={"ptr":group["indptr"][:]}
            for name in VIEWS:
                print(f"{sample} · {name}: detection profiles from {len(views[name]):,} cells",flush=True)
                summaries[name]=_summarize(h,group,views[name],strata,genes,pos,chunksize,
                                           max(sketch_size,n_pcs+4),np.random.default_rng(seed))
            # The two matched maps use exactly the same genes. Unobserved genes still appear
            # with explicit unmapped/NA status in the per-pair and per-gene tables.
            informative={name:(summaries[name]["n"]>=min_cells) &
                         (len(views[name])-summaries[name]["n"]>=min_cells) &
                         (summaries[name]["ss"]>1e-8) for name in VIEWS}
            required=np.isin(genes,list(MARKERS))
            def choose(mask):
                must=np.flatnonzero(mask&required); other=np.flatnonzero(mask&~required)
                order=np.argsort(-summaries["all_cells"]["n"][other],kind="stable")
                axis=np.sort(np.r_[must,other[order[:max(0,max_genes-len(must))]]])
                if len(axis)<max(neighbors+1,4):
                    raise ValueError(f"Only {len(axis)} informative genes for this view. Check depth, T-cell labels and gene coverage.")
                return axis
            all_axis=choose(informative["all_cells"])
            common=choose(informative["matched_all"]&informative["matched_T"])
            for name in VIEWS:
                print(f"{sample} · {name}: refining gene PCs and building graph",flush=True)
                summary=summaries[name]
                axis=all_axis if name=="all_cells" else common
                pc=_coords(h,group,summary,axis,strata,chunksize,n_pcs)
                coords,cl,proj,algorithm=_graph(pc,neighbors,resolution,seed)
                frame=pd.DataFrame(dict(gene=genes[axis],x=coords[:,0],y=coords[:,1],module=cl,
                                        projection=proj,clustering=algorithm,
                                        detected_in_cells=summary["n"][axis].astype(int)))
                frame=pd.concat([frame,pd.DataFrame(pc,columns=[f"PC{i+1}" for i in range(pc.shape[1])])],axis=1)
                frame.to_csv(tmp/(name+"_gene_map.csv.gz"),index=False,float_format="%.7g")
                maps[name]=frame
                corrs=pd.DataFrame(summary["corr"],index=pd.Index(genes,name="gene"),columns=summary["anchors"])
                good=informative[name]
                corrs.loc[~good,:]=np.nan
                for g in summary["anchors"]:
                    if not good[pos[g]]: corrs[g]=np.nan
                corrs.insert(0,"detected_cells",summary["n"].astype(int))
                corrs.to_csv(tmp/(name+"_anchor_correlations.csv.gz"),float_format="%.6g")
                pairs=_controls(summary,axis,pc,cl,genes,min_cells)
                pairs.insert(0,"view",name);pairs.insert(0,"sample",sample);pair_frames.append(pairs)
                setting_rows.append(dict(sample=sample,view=name,n_cells=len(views[name]),n_T=int(tmask[views[name]].sum()),
                                   n_zero_depth=int(obs.total_counts.iloc[views[name]].eq(0).sum()),n_genes_mapped=len(axis),
                                   median_depth=float(obs.total_counts.iloc[views[name]].median()),projection=proj,
                                   clustering=algorithm,modules=int(len(set(cl)))))
        _show_plot(maps,tmp)
        pd.concat(pair_frames).to_csv(tmp/"marker_pair_diagnostics.csv",index=False)
        pd.DataFrame(setting_rows).to_csv(tmp/"view_sizes.csv",index=False)
        hist=[]
        for name,ids in views.items():
            t=pd.crosstab(obs.section_id.iloc[ids].to_numpy(),strata[ids]).stack().rename("n_cells").reset_index()
            t.columns=["section","depth_section_stratum","n_cells"];t["view"]=name;hist.append(t)
        pd.concat(hist).to_csv(tmp/"matching_audit.csv",index=False)
        if source.stat().st_mtime_ns!=stats.st_mtime_ns or source.stat().st_size!=stats.st_size:
            raise RuntimeError("The source H5AD changed during the run.")
        files={p.name:p.stat().st_size for p in tmp.iterdir() if p.is_file()}
        _json(tmp/"complete.json",dict(settings=settings,files=files,matched_T_and_all_cell_overlap=int(
              len(np.intersect1d(views["matched_T"],views["matched_all"],assume_unique=True)))))
        try: tmp.rename(run)
        except OSError:
            if not receipt.is_file(): raise
        return run
    finally:
        if tmp.is_dir(): shutil.rmtree(tmp)


def review_gene_runs(runs, output_root):
    """Create a per-notebook aggregate ZIP that pairs supported comparisons on identical gene pairs."""
    root=Path(output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    runs=[Path(p).resolve() for p in runs]
    if not runs or len(set(runs))!=len(runs):
        raise ValueError("Choose one or more distinct completed runs.")
    reports=[json.loads((p/"complete.json").read_text()) for p in runs]
    samples=[x["settings"]["sample"] for x in reports]
    if len(samples)!=len(set(samples)):
        raise ValueError("Use one result per unique mouse for the replication view.")
    frames=[pd.read_csv(p/"marker_pair_diagnostics.csv") for p in runs]
    combined=pd.concat(frames,ignore_index=True)
    keys=["panel","anchor","partner"]
    compare=combined.pivot(index=["sample",*keys],columns="view",
                           values=["direct_r","PC_rank","matched_null_percentile"]).copy()
    compare.columns=["_".join(c) for c in compare.columns]
    compare=compare.reset_index()
    compare["matched_all_minus_T_r"]=compare.direct_r_matched_all-compare.direct_r_matched_T
    replicate=compare.groupby(keys).agg(evaluable_mice=("direct_r_matched_all","count"),
                  positive_all_cells_r_mice=("direct_r_all_cells",lambda x:(x>0).sum()),
                  evaluable_delta_mice=("matched_all_minus_T_r","count"),
                  matched_all_exceeds_T_mice=("matched_all_minus_T_r",lambda x:(x>0).sum())).reset_index()
    archive=root/("09A_return_to_chat_"+_identity([p.name for p in runs])+".zip")
    fd,name=tempfile.mkstemp(dir=root,prefix=".09A-review-");os.close(fd)
    try:
        with zipfile.ZipFile(name,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            z.writestr("matched_view_comparison.csv",compare.to_csv(index=False))
            z.writestr("per_pair_recurrence.csv",replicate.to_csv(index=False))
            for i,(p,report) in enumerate(zip(runs,reports)):
                for n in ["view_sizes.csv","marker_pair_diagnostics.csv","matching_audit.csv","gene_maps.png","gene_modules.png"]:
                    z.write(p/n,f"mouse_{i+1}_{p.name}/{n}")
                public={k:v for k,v in report["settings"].items() if k not in ("input_path",)}
                z.writestr(f"mouse_{i+1}_{p.name}/settings.json",json.dumps(public,indent=2))
        os.replace(name,archive)
    finally:
        Path(name).unlink(missing_ok=True)
    return archive,compare,replicate
