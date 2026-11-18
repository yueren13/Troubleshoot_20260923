"""Discovery-mouse gene co-detection, gene map, and candidate lineage signatures."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# Each multi-gene group is an indicator of any detected member. These pooled
# references stabilize sparse probes. Individual direct markers also get rows.
REFERENCE = {
    "Cd4": ["Cd4"], "Cd8a": ["Cd8a"], "Cd8b1": ["Cd8b1"],
    "Cd8_any": ["Cd8a", "Cd8b1"], "Foxp3": ["Foxp3"],
    "Zbtb7b": ["Zbtb7b"], "Cd40lg": ["Cd40lg"],
    "Il2ra": ["Il2ra"], "Ctla4": ["Ctla4"], "Ikzf2": ["Ikzf2"],
    "Tnfrsf18": ["Tnfrsf18"],
    "pan_T": ["Cd3d", "Cd3e", "Cd3g", "Trac", "Trbc1", "Trbc2"],
    "cytotoxic": ["Nkg7", "Prf1", "Gzma", "Gzmb", "Gzmk", "Ccl5", "Ctsw"],
    "memory": ["Tcf7", "Lef1", "Ccr7", "Sell", "Il7r", "Klf2"],
    "checkpoint": ["Pdcd1", "Lag3", "Havcr2", "Tox", "Tigit", "Entpd1"],
    "cycling": ["Mki67", "Top2a", "Ube2c", "Cenpf", "Birc5", "Ccnb1", "Pclaf"],
    "interferon": ["Isg15", "Ifit1", "Ifit3", "Oas1a", "Mx1", "Ifitm3", "Stat1", "Irf7"],
    "regulatory": ["Foxp3", "Il2ra", "Ctla4", "Ikzf2", "Tnfrsf18"],
    "myeloid": ["C1qa", "C1qb", "C1qc", "Csf1r", "Lyz2", "Adgre1", "Lst1"],
    "epithelial": ["Epcam", "Krt8", "Krt18", "Krt19"],
    "stroma": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Lum"],
    "endothelium": ["Cdh5", "Pecam1", "Emcn", "Kdr"],
    "B_cell": ["Cd79a", "Ms4a1", "Cd79b"],
    "NK": ["Ncr1", "Klrb1c", "Klrk1", "Klrd1", "Tyrobp"],
    "gamma_delta": ["Trdc", "Trgc1", "Trgc2"],
    "TCR_activation": ["Nr4a1", "Nr4a2", "Egr1", "Egr2", "Cd69", "Icos"],
    "acute_stress": ["Fos", "Jun", "Junb", "Egr1", "Dusp1", "Atf3"],
}
STATE_CONTROLS = ["cytotoxic", "memory", "checkpoint", "cycling", "interferon",
                  "regulatory", "acute_stress", "TCR_activation", "NK", "gamma_delta", "pan_T"]
ENV_CONTROLS = ["myeloid", "epithelial", "stroma", "endothelium", "B_cell"]


def _references(binary, positions, panels):
    columns = []
    for members in panels.values():
        indices = [positions[g.casefold()] for g in members if g.casefold() in positions]
        if indices:
            detected = np.asarray(binary[:, indices].sum(axis=1)).ravel() > 0
        else:
            detected = np.zeros(binary.shape[0], dtype=bool)
        columns.append(detected)
    return np.column_stack(columns).astype(np.float32)


def _mouse_profiles(counts, depth, anchors, n_bins, sketch_dim, rng, min_detection):
    """Exact within-depth-stratum binary correlations and a compact gene map input."""
    binary = counts.tocsr(copy=True).astype(np.float32, copy=False)
    binary.sum_duplicates()
    binary.eliminate_zeros()
    binary.data.fill(1)
    n_cells = binary.shape[0]
    cuts = np.unique(np.quantile(depth, np.arange(1, n_bins) / n_bins))
    _, group = np.unique(np.searchsorted(cuts, depth), return_inverse=True)
    n_group = np.bincount(group)
    row = np.arange(n_cells)
    membership = sparse.csr_matrix((np.ones(n_cells, np.float32), (row, group)),
                                   shape=(n_cells, len(n_group)))
    # Float64 for subtracting the depth-expected co-detections, especially for common genes.
    g_count = (binary.T @ membership).toarray().astype(np.float64)
    detected = g_count.sum(axis=1)
    g_mean = g_count / n_group
    g_ss = np.maximum(detected - (g_count * g_mean).sum(axis=1), 0)
    anchor_matrix = _references(binary, *anchors)
    a_count = np.asarray(membership.T @ anchor_matrix, dtype=np.float64)
    a_ss = np.maximum(a_count.sum(axis=0) - (a_count ** 2 / n_group[:, None]).sum(axis=0), 0)
    numerator = binary.T @ anchor_matrix - g_mean @ a_count
    denominator = np.sqrt(g_ss[:, None] * a_ss[None, :])
    corr = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 1e-10)
    n_anchor = a_count.sum(axis=0)
    informative_g = (detected >= min_detection) & ((n_cells - detected) >= min_detection)
    informative_a = (n_anchor >= min_detection) & ((n_cells - n_anchor) >= min_detection)
    corr[~informative_g, :] = np.nan
    corr[:, ~informative_a] = np.nan

    # Signed feature hashing compresses standardized per-gene profiles across
    # cells before PCA. In expectation it preserves cell-space dot products.
    # Every mouse contributes a unit-norm profile for each informative gene.
    buckets = rng.integers(0, sketch_dim, n_cells)
    signs = rng.choice(np.array([-1, 1], dtype=np.float32), size=n_cells)
    projection = sparse.csr_matrix((signs, (row, buckets)), shape=(n_cells, sketch_dim))
    projected = (binary.T @ projection).toarray()
    group_sketch = (membership.T @ projection).toarray()
    projected -= g_mean.astype(np.float32) @ group_sketch
    projected /= np.sqrt(np.maximum(g_ss, 1e-8)).astype(np.float32)[:, None]
    projected[~informative_g] = 0
    return corr.astype(np.float32), detected.astype(int), n_anchor.astype(int), projected, informative_g


def _median(array, axis=0):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(array, axis=axis)


def _strongest(frame, columns):
    values = frame[columns]
    # Replacing NaN with -inf would make an entirely unavailable control appear measured.
    maximum = values.max(axis=1, skipna=True)
    winner = values.fillna(-np.inf).idxmax(axis=1).where(maximum.notna(), "")
    return maximum, winner


def _nominate(correlations, conditions, genes, panels, audit, settings):
    names = list(panels)
    median = pd.DataFrame(_median(correlations), index=genes, columns=names)
    # Coverage filter per anchor prevents tiny or absent controls from vetoing candidates.
    median = median.where(np.isfinite(correlations).sum(axis=0) >= settings["min_mice"])
    state_r, state_name = _strongest(median, STATE_CONTROLS)
    env_r, env_name = _strongest(median, ENV_CONTROLS)
    excluded = {s.casefold() for members in panels.values() for s in members}
    genes_lower = pd.Index([g.casefold() for g in genes])
    technical = (genes_lower.str.startswith(("mt-", "rpl", "rps", "trav", "traj", "trbv",
                                              "trbj", "trdv", "trgv", "igh", "igk", "igl"))
                 | genes_lower.isin(["malat1", "xist", "tsix", "ddx3y", "eif2s3y",
                                     "kdm5d", "uty", "actb", "b2m", "gapdh"]))
    tables, signatures = [], {}
    for lineage, target, other in [("CD4", "Cd4", "Cd8_any"), ("CD8", "Cd8_any", "Cd4")]:
        target_r = correlations[:, :, names.index(target)]
        difference = target_r - correlations[:, :, names.index(other)]
        eligible = np.isfinite(difference).sum(axis=0)
        positive = ((difference > 0) & (target_r > 0)).sum(axis=0)
        arm_support = np.zeros(len(genes), dtype=int)
        arms_measured = np.zeros(len(genes), dtype=int)
        for arm in pd.unique(np.asarray(conditions)):
            arm_difference = _median(difference[np.array(conditions) == arm])
            arms_measured += np.isfinite(arm_difference)
            arm_support += arm_difference > 0
        frame = audit.copy()
        frame["lineage"] = lineage
        frame["median_target_r"] = median[target]
        frame["median_opposite_r"] = median[other]
        frame["median_paired_r_difference"] = _median(difference)
        frame["mice_evaluable"] = eligible
        frame["mice_supporting"] = positive
        frame["support_fraction"] = positive / np.maximum(eligible, 1)
        frame["conditions_evaluable"] = arms_measured
        frame["conditions_supporting"] = arm_support
        frame["closest_state_control"] = state_name
        frame["strongest_state_control_r"] = state_r
        frame["closest_nonT_control"] = env_name
        frame["strongest_nonT_control_r"] = env_r
        frame["n_replicated_nonT_controls"] = median[ENV_CONTROLS].notna().sum(axis=1)
        frame["n_replicated_state_controls"] = median[STATE_CONTROLS].notna().sum(axis=1)
        # Non-T sentinels in the same segments are only contamination/niche proxies.
        # A whole-cohort ratio is descriptive and is never used to freeze a signature.
        keep_ratio = settings["ratio_scope"] == "discovery"
        ratio = pd.to_numeric(frame["env_ratio"], errors="coerce")
        reject = {
            "reference_or_known_state": genes_lower.isin(excluded),
            "technical_or_receptor_family": technical,
            "insufficient_mice": eligible < settings["min_mice"],
            "insufficient_agreement": frame["support_fraction"] < settings["min_agreement"],
            "insufficient_conditions": arm_support < min(2, len(set(conditions))),
            "weak_target": ~(frame["median_target_r"] >= settings["min_r"]),
            "weak_specificity": ~(frame["median_paired_r_difference"] >= settings["min_margin"]),
            "unknown_nonT_control": frame["n_replicated_nonT_controls"] == 0,
            "unknown_state_control": frame["n_replicated_state_controls"] == 0,
            "stronger_nonT_control": env_r > frame["median_target_r"],
            "stronger_state_control": state_r > frame["median_target_r"],
            "discovery_env_ratio_high": (ratio > settings["max_env_ratio"]) & keep_ratio,
            "discovery_env_ratio_missing": ratio.isna() & keep_ratio,
        }
        if keep_ratio and "env_ratio_discovery_mice" in frame:
            reject["discovery_env_ratio_few_mice"] = frame["env_ratio_discovery_mice"].fillna(0) < min(3, settings["min_mice"])
        reason = np.array([""] * len(frame), dtype=object)
        for label, condition in reject.items():
            mask = np.asarray(condition, bool)
            reason[mask] = [f"{x}|{label}" if x else label for x in reason[mask]]
        frame["exclusion_reasons"] = reason
        frame["eligible"] = frame["exclusion_reasons"] == ""
        ordered = frame.sort_values(["eligible", "median_paired_r_difference", "median_target_r"],
                                    ascending=False, kind="stable")
        selected = ordered.index[ordered["eligible"]][:settings["max_signature"]].tolist()
        signatures[lineage + "_support"] = selected
        frame["selected"] = frame.index.isin(selected)
        tables.append(frame)
    return signatures, pd.concat(tables), median


def _gene_map(sketch, genes, mandatory, informative, detect, out, n_pcs, k, resolution,
              cap, seed, use_umap, use_leiden, sketch_dim):
    import matplotlib.pyplot as plt
    from scipy import sparse as sp

    repeated = informative.sum(axis=0)
    valid = repeated >= min(2, len(informative))
    measured = repeated > 0
    required = measured & np.array([g in mandatory for g in genes])
    order = np.lexsort((-detect.sum(axis=0), -repeated))
    remaining = order[valid[order] & ~required[order]]
    index = np.sort(np.r_[np.flatnonzero(required), remaining[:max(0, cap-required.sum())]])
    if len(index) < 4:
        raise ValueError("Fewer than four measured gene profiles for a gene map. Inspect gene coverage or lower min_detection.")
    # Missing mice contribute zero; equalize estimated row energy for genes with
    # different mouse coverage. Every retained gene has at least one measured mouse.
    profiles = np.concatenate([matrix[index] for matrix in sketch], axis=1)
    profiles /= np.sqrt(repeated[index]).astype(np.float32)[:, None]
    pca = PCA(n_components=min(n_pcs, len(index)-1, profiles.shape[1]-1),
              svd_solver="randomized", random_state=seed)
    pcs = pca.fit_transform(profiles)
    del profiles
    ks = min(k, len(index)-1)
    distances, neighbors = NearestNeighbors(n_neighbors=ks+1, metric="euclidean").fit(pcs).kneighbors(pcs)
    # Ties can place the query gene outside column zero; remove by identity.
    chosen = [np.flatnonzero(nn != i)[:ks] for i, nn in enumerate(neighbors)]
    nn = np.stack([neighbors[i, cols] for i, cols in enumerate(chosen)])
    dd = np.stack([distances[i, cols] for i, cols in enumerate(chosen)])
    scale = np.maximum(dd[:, -1], 1e-8)
    weights = np.exp(-np.square(dd / scale[:, None]))
    graph = sp.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(index)), ks), nn.ravel())),
                          shape=(len(index), len(index)))
    graph = graph.maximum(graph.T)
    # UMAP uses the same input PCs and metric. Modules use the weighted symmetric kNN graph.
    if use_umap:
        try:
            from umap import UMAP
        except ImportError:
            use_umap = False
    embedding = "UMAP" if use_umap else "PCA"
    if use_umap:
        coords = UMAP(n_components=2, n_neighbors=ks, metric="euclidean", min_dist=0.2,
                      random_state=seed, n_jobs=1, init="random").fit_transform(pcs)
    else:
        coords = pcs[:, :2]
    if use_leiden:
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            use_leiden = False
    if use_leiden:
        edges = sp.triu(graph, k=1).tocoo()
        g = ig.Graph(n=len(index), edges=list(zip(edges.row.tolist(), edges.col.tolist())))
        modules = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                            weights=edges.data.tolist(), resolution_parameter=resolution,
                                            seed=seed, n_iterations=-1).membership
        clustering = "Leiden"
    else:
        # Explicitly labelled fallback for an environment lacking optional graph packages.
        groups = min(len(index)-1, max(2, min(35, round(np.sqrt(len(index)/ks)))))
        modules = AgglomerativeClustering(n_clusters=groups, connectivity=graph).fit_predict(pcs)
        clustering = "graph-constrained Ward"
    names = np.asarray(genes)[index]
    frame = pd.DataFrame({"gene": names, "module": modules, "embedding": embedding,
                          "clustering": clustering, "map_x": coords[:, 0], "map_y": coords[:, 1],
                          "mice_adequate_detection": repeated[index]})
    for j in range(pcs.shape[1]):
        frame[f"PC{j+1}"] = pcs[:, j]
    frame.to_csv(out / "gene_map.csv.gz", index=False)
    pd.DataFrame({"gene": np.repeat(names, ks), "neighbor": names[nn.ravel()],
                  "rank": np.tile(np.arange(1, ks+1), len(names)), "PC_distance": dd.ravel(),
                  "directed_weight": weights.ravel()}).to_csv(out/"gene_map_neighbors.csv.gz", index=False)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.scatter(*coords.T, c=modules, s=5, cmap="tab20", alpha=.65, rasterized=True)
    text_genes = {"cd4", "cd8a", "cd8b1", "foxp3", "zbtb7b", "cd40lg", "cd3d", "nkg7", "c1qa", "epcam"}
    for name, xy in zip(names, coords):
        if name.casefold() in text_genes:
            ax.scatter(*xy, s=32, facecolors="none", edgecolors="black")
            ax.annotate(name, xy, xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(title=f"Discovery-mouse gene profiles · {clustering} modules",
           xlabel=embedding+" 1", ylabel=embedding+" 2")
    fig.savefig(out/"gene_map.png", dpi=210)
    plt.close(fig)
    return {"embedding": embedding, "clustering": clustering, "n_genes_mapped": len(names),
            "n_pcs": pcs.shape[1], "n_neighbors": ks, "n_modules": int(len(set(modules))),
            "sketch_dimensions_per_mouse": sketch_dim,
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist()}


def _map_evidence(out, median, audit, signatures):
    """Put candidate, compartment and anchor evidence onto the same fixed map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    frame = pd.read_csv(out/"gene_map.csv.gz").set_index("gene")
    frame = frame.join(audit[[c for c in audit if c.startswith("env_ratio")]])
    for anchor in ["Cd4", "Cd8_any", "Foxp3", *ENV_CONTROLS]:
        frame["r_" + anchor] = median[anchor].reindex(frame.index)
    for lineage, selected in signatures.items():
        frame[lineage] = frame.index.isin(selected)
    frame.to_csv(out/"gene_map.csv.gz", float_format="%.7g")
    embedding = frame["embedding"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [("r_Cd4", "CD4"), ("r_Cd8_any", "CD8A or CD8B1"),
              ("r_Foxp3", "Foxp3"), ("env_ratio", "Local non-T / focal-T expression")]
    x, y = frame["map_x"], frame["map_y"]
    for ax, (column, label) in zip(axes.flat, panels):
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(x, y, s=4, color="0.85", rasterized=True)
        if not values.empty:
            if column.startswith("r_"):
                bound = max(.005, float(values.abs().quantile(.98)))
                norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
                colors = ax.scatter(x.loc[values.index], y.loc[values.index], c=values.clip(-bound,bound),
                                    cmap="RdBu_r", norm=norm, s=5, rasterized=True)
                legend = "Median depth-adjusted detection r"
            else:
                high = max(1e-7, float(values.quantile(.98)))
                colors = ax.scatter(x.loc[values.index], y.loc[values.index], c=values.clip(0,high),
                                    cmap="viridis", vmin=0, vmax=high, s=5, rasterized=True)
                legend = "Non-T / T expression ratio"
            fig.colorbar(colors, ax=ax, shrink=.78, label=legend, extend="both" if column.startswith("r_") else "max")
        else:
            ax.text(.03, .96, "No eligible measurements", va="top", transform=ax.transAxes)
        ax.set(title=label, xlabel=embedding+" 1", ylabel=embedding+" 2")
    fig.savefig(out/"gene_map_evidence.png", dpi=210)
    plt.close(fig)

    references = {s.casefold() for members in REFERENCE.values() for s in members}
    pc = [c for c in frame if c.startswith("PC")]
    module_rows = []
    for module, block in frame.groupby("module", sort=True):
        p = block[pc].to_numpy()
        closest = np.argsort(np.square(p - p.mean(axis=0)).sum(axis=1))[:8]
        module_refs = block.index[block.index.str.lower().isin(references)].tolist()
        row = {"module":module, "n_genes":len(block),
               "central_genes":"|".join(block.index[closest]),
               "reference_genes":"|".join(module_refs),
               "CD4_support_genes":"|".join(block.index[block["CD4_support"]]),
               "CD8_support_genes":"|".join(block.index[block["CD8_support"]]),
               "median_env_ratio":block["env_ratio"].median() if block["env_ratio"].notna().any() else np.nan}
        for anchor, val in median.reindex(block.index).median(axis=0).items():
            row["median_r_" + anchor] = val
        module_rows.append(row)
    pd.DataFrame(module_rows).to_csv(out/"gene_module_summary.csv", index=False, float_format="%.6g")


def discover(X, obs, genes, gmeta, out, *, depth_col=None, max_cells_per_mouse=30000,
             min_detection=12, min_mice=None, min_agreement=.75, min_r=.015,
             min_margin=.01, max_signature=16, ratio_scope="descriptive",
             max_env_ratio=1.0, n_depth_bins=10, sketch_dim=192, n_pcs=35,
             map_neighbors=20, max_map_genes=8000, resolution=1.0, seed=17,
             use_umap=True, use_leiden=True):
    """Fit on discovery mice only. Writes ranked tables, a gene map, and frozen signatures.

    X: original cell-by-gene CSR counts. obs: aligned sample/condition/split and
    optional whole-library depth column. gmeta: optional uniquely gene-indexed
    frame with env_ratio (larger = more local non-T enrichment). Keep ratio_scope
    descriptive for ratios estimated on pooled mice. Choosing discovery asserts
    the supplied ratio was calculated exclusively from the discovery mice.
    """
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    genes = [str(g) for g in genes]
    positions = {g.casefold(): i for i, g in enumerate(genes)}
    if len(positions) != len(genes):
        raise ValueError("Gene symbols must be unique ignoring case. Collapse duplicate symbols upstream.")
    if X.shape != (len(obs), len(genes)):
        raise ValueError("obs, genes and count matrix dimensions disagree")
    if not sparse.isspmatrix_csr(X) and getattr(X, "format", "") != "csr":
        raise ValueError("Provide CSR observed counts for efficient row selection")
    if ratio_scope not in {"descriptive", "discovery"}:
        raise ValueError("ratio_scope must be descriptive or discovery")
    if min_detection < 1 or n_depth_bins < 1 or sketch_dim < 4 or n_pcs < 2 or map_neighbors < 2 or max_map_genes < 4:
        raise ValueError("Invalid detection, depth-bin or gene-map parameters")
    if max_signature < 1 or not 0 < min_agreement <= 1 or (max_cells_per_mouse is not None and max_cells_per_mouse < 4):
        raise ValueError("Invalid signature, replication or mouse-cap parameters")
    for key in ["sample", "condition", "split"] + ([depth_col] if depth_col else []):
        if key not in obs or obs[key].isna().any():
            raise ValueError(f"Missing values or missing column: {key}")
    if not set(obs["split"].astype(str)) <= {"discovery", "evaluation"}:
        raise ValueError("Use discovery/evaluation split labels")
    design = obs[["sample","condition","split"]].drop_duplicates()
    if design["sample"].duplicated().any():
        raise ValueError("Conflicting condition or split for one sample")
    design = design.loc[design["split"].eq("discovery")].sort_values("sample")
    if len(design) < 2:
        raise ValueError("At least two discovery mice are required")
    if min_mice is None:
        min_mice = max(2, int(np.ceil(.625 * len(design))))
    if not 2 <= min_mice <= len(design):
        raise ValueError("min_mice must be between two and the number of discovery mice")
    audit = pd.DataFrame(index=pd.Index(genes, name="gene"))
    audit["env_ratio"] = np.nan
    if gmeta is not None and "env_ratio" in gmeta:
        ratio = pd.to_numeric(gmeta["env_ratio"], errors="coerce").copy()
        ratio.index = pd.Index([str(x).casefold() for x in ratio.index])
        if not ratio.index.is_unique:
            raise ValueError("gmeta index must be unique ignoring case")
        audit["env_ratio"] = ratio.reindex([g.casefold() for g in genes]).to_numpy()
    for key in ["env_ratio_pooled", "env_ratio_discovery_mice"]:
        if gmeta is not None and key in gmeta:
            series = pd.to_numeric(gmeta[key], errors="coerce").copy()
            series.index = pd.Index([str(x).casefold() for x in series.index])
            if not series.index.is_unique:
                raise ValueError("gmeta index must be unique ignoring case")
            audit[key] = series.reindex([g.casefold() for g in genes]).to_numpy()
    if ratio_scope == "discovery" and not audit["env_ratio"].notna().any():
        raise ValueError("A discovery-only ratio was requested but no gene has env_ratio.")
    rng = np.random.default_rng(seed)
    corr, det, sketches, info, rows, anchors_measured, per_mouse = [], [], [], [], [], [], []
    sample_obs = obs["sample"].to_numpy()
    for record in design.itertuples(index=False):
        idx = np.flatnonzero(sample_obs == record.sample)
        source_n = len(idx)
        if max_cells_per_mouse is not None and len(idx) > max_cells_per_mouse:
            idx = np.sort(rng.choice(idx, max_cells_per_mouse, replace=False))
        counts = X[idx]
        if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
            raise ValueError(f"Invalid observed counts in {record.sample}")
        depth = (obs.iloc[idx][depth_col].to_numpy(dtype=float) if depth_col
                 else np.asarray(counts.sum(axis=1)).ravel())
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f"Invalid library depth for {record.sample}")
        c, d, n_anchor, projection, informative = _mouse_profiles(
            counts, depth, (positions, REFERENCE), n_depth_bins, sketch_dim, rng, min_detection)
        corr.append(c)
        det.append(d)
        sketches.append(projection)
        info.append(informative)
        rows.append({"sample":record.sample, "condition":record.condition, "n_available":source_n,
                     "n_profiled":len(idx), "median_depth":float(np.median(depth))})
        for j, (anchor, members) in enumerate(REFERENCE.items()):
            anchors_measured.append({"sample":record.sample, "condition":record.condition, "anchor":anchor,
                                     "assayed_members":"|".join(genes[positions[g.casefold()]] for g in members
                                                               if g.casefold() in positions),
                                     "n_detected_cells":int(n_anchor[j]), "n_profiled":len(idx)})
        cf = pd.DataFrame(c, index=audit.index, columns=list(REFERENCE))
        cf.insert(0,"detected_cells",d)
        cf.insert(0,"sample",record.sample)
        per_mouse.append(cf)
    corr, det, info = np.stack(corr), np.stack(det), np.stack(info)
    pd.DataFrame(rows).to_csv(out/"discovery_mouse_counts.csv", index=False)
    pd.DataFrame(anchors_measured).to_csv(out/"reference_detection_by_mouse.csv", index=False)
    pd.concat(per_mouse).to_csv(out/"per_mouse_gene_associations.csv.gz", float_format="%.6g")
    del per_mouse
    audit["detected_discovery_cells"] = det.sum(axis=0)
    audit["mice_adequate_detection"] = info.sum(axis=0)
    cfg = {"min_mice":int(min_mice), "min_agreement":min_agreement, "min_r":min_r,
           "min_margin":min_margin, "max_signature":max_signature, "ratio_scope":ratio_scope,
           "max_env_ratio":max_env_ratio}
    signatures, candidates, median = _nominate(corr, design["condition"].tolist(), genes, REFERENCE,
                                                audit, cfg)
    candidates.to_csv(out/"lineage_candidate_audit.csv.gz", float_format="%.6g")
    rank_tables = []
    for j, anchor in enumerate(REFERENCE):
        tab = audit.copy()
        tab["anchor"] = anchor
        tab["median_r"] = _median(corr[:, :, j])
        tab["mice_with_r"] = np.isfinite(corr[:, :, j]).sum(axis=0)
        tab["mice_positive"] = (corr[:, :, j] > 0).sum(axis=0)
        tab["mouse_replicated_r"] = median[anchor]
        tab["anchor_member"] = tab.index.str.lower().isin([g.lower() for g in REFERENCE[anchor]])
        tab["neighbor_rank"] = median[anchor].where(~tab["anchor_member"]).rank(ascending=False, method="min")
        rank_tables.append(tab.sort_values("neighbor_rank"))
    pd.concat(rank_tables).to_csv(out/"anchor_rankings.csv.gz", float_format="%.6g")
    all_required = set(g.casefold() for members in REFERENCE.values() for g in members)
    mandatory = {g for g in genes if g.casefold() in all_required} | set(sum(signatures.values(), []))
    map_meta = _gene_map(sketches, genes, mandatory, info, det, out, n_pcs, map_neighbors,
                         resolution, max_map_genes, seed, use_umap, use_leiden, sketch_dim)
    _map_evidence(out, median, audit, signatures)
    cfg.update({"reference_panels":REFERENCE, "discovery_samples":[str(v) for v in design["sample"]],
                "n_depth_bins":n_depth_bins, "min_detection":min_detection,
                "max_cells_per_mouse":max_cells_per_mouse, "depth_column":depth_col or "row sum of counts",
                "seed":seed, "max_map_genes":max_map_genes, "resolution":resolution, **map_meta})
    (out/"discovery_settings.json").write_text(json.dumps(cfg, indent=2)+"\n")
    (out/"signatures.json").write_text(json.dumps(signatures, indent=2)+"\n")
    return signatures
