"""Native-square coverage, matched hard-mask controls and a targeted SPARKLE pilot."""
from pathlib import Path
import json
import importlib.metadata
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree

PANELS = {
    "panT": ["Cd3d", "Cd3e", "Cd3g", "Trac", "Trbc1", "Trbc2"],
    "myeloid": ["C1qa", "C1qb", "C1qc", "Lyz2", "Csf1r", "Adgre1"],
    "B": ["Cd79a", "Cd79b", "Ms4a1", "Cd19"],
    "stroma": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Lum"],
    "epithelial": ["Epcam", "Krt8", "Krt18", "Krt19"],
    "endothelial": ["Pecam1", "Cdh5", "Kdr", "Emcn"],
    "Cd4": ["Cd4"], "CD8": ["Cd8a", "Cd8b1"],
}
TARGETS = set().union(*PANELS.values())


def _groups(codes, n):
    return sparse.csr_matrix((np.ones(len(codes), dtype=np.int32),
                              (np.arange(len(codes)), codes)), shape=(len(codes), n))


def _coverage(name, matrix):
    x = sparse.csc_matrix(matrix)
    totals = np.asarray(x.sum(axis=0)).ravel()
    ngenes = x.getnnz(axis=0)
    return dict(unit=name, n=len(totals), total_UMIs=int(totals.sum()),
                fraction_zero=np.mean(totals == 0) if len(totals) else np.nan,
                median_UMIs=np.median(totals) if len(totals) else np.nan,
                p90_UMIs=np.quantile(totals, .9) if len(totals) else np.nan,
                median_features=np.median(ngenes) if len(ngenes) else np.nan)


def _panel_counts(matrix, symbols):
    result = {}
    for name, members in PANELS.items():
        genes = np.flatnonzero(symbols.isin(members))
        if len(genes):
            result[name] = np.asarray(matrix[genes].sum(axis=0)).ravel()
        else:
            result[name] = np.full(matrix.shape[1], np.nan)
    return pd.DataFrame(result)


def _original(run):
    meta = json.loads((run / "settings.json").read_text())
    receipt = json.loads((run / "complete.json").read_text())
    core_info = pd.read_csv(run / "local_cell_assignments.csv.gz", dtype=str, keep_default_na=False)
    core = core_info.evaluation_core.astype(str).str.lower().map({"true":True, "false":False})
    if (core.isna().any() or not core_info.cell_id.is_unique or not core.any()
            or len(core_info) != receipt["fitted_cells"] or core.sum() != receipt["evaluation_core_cells"]):
        raise ValueError("09B assignment IDs or evaluation-core counts differ from the completed receipt.")
    src = Path(meta["source"])
    if not src.is_file():
        src = run.parent / "inputs" / src.parent.name / src.name
    if not src.is_file() or src.stat().st_size != meta["size"]:
        raise ValueError(f"Cannot match the prepared 09B H5AD to its saved receipt: {src}")
    if src.stat().st_mtime_ns != meta["mtime_ns"]:
        warnings.warn("Prepared H5AD timestamp differs from the 09B receipt. Check that this is the same input copy.")
    a = ad.read_h5ad(src, backed="r")
    try:
        idx = a.obs_names.get_indexer(core_info.cell_id)
        if (np.any(idx < 0) or np.any(np.diff(idx) <= 0) or a.n_obs != receipt["source_cells"]
                or not a.var_names.is_unique):
            raise ValueError("Prepared H5AD dimensions, symbols, or cell IDs do not match the 09B run.")
        for column in ("cell_type", "section_id"):
            if not np.array_equal(a.obs[column].iloc[idx].astype(str).to_numpy(), core_info[column].to_numpy()):
                raise ValueError(f"09B {column} differs between the assignment file and prepared H5AD.")
        xy = np.asarray(a.obsm["spatial"])[idx, :2] * meta["settings"]["microns_per_unit"]
        sym = pd.Index(a.var_names)
        chosen = np.flatnonzero(sym.isin(TARGETS))
        cells = core_info.loc[core, ["cell_id","cell_type"]].reset_index(drop=True)
        selected = idx[core.to_numpy()]
        blocks = []
        for rows in np.split(selected, np.flatnonzero(np.diff(selected // 2048)) + 1):
            block = a.X[rows[0]:rows[-1]+1][rows-rows[0]].tocsr()
            blocks.append(block[:,chosen])
        original = sparse.vstack(blocks, format="csr")
        original.sum_duplicates()
        original.eliminate_zeros()
        panel = _panel_counts(original.T.tocsr(), sym[chosen])
        pan = np.flatnonzero(sym[chosen].isin(PANELS["panT"]))
        cells["panT_09B_detected_symbols"] = original[:,pan].getnnz(axis=1)
        # Match 09C: fewer than two measured symbols makes support unassessable.
        supported = cells.panT_09B_detected_symbols.ge(2).astype("boolean")
        cells["panT_09B_observed"] = supported if len(pan) >= 2 else pd.array([pd.NA] * len(cells), dtype="boolean")
        depth = a.obs["total_counts"].iloc[selected].to_numpy()
        patterns = receipt["type_patterns"]
        is_mac = core_info.cell_type.str.contains(patterns["macrophage"], case=False, na=False).to_numpy()
        cells["T_label_09B"] = cells.cell_type.str.contains(patterns["T"], case=False, na=False)
        cells["macrophage_label_09B"] = is_mac[core]
        distance = np.full(len(xy), np.nan)
        for section in core_info.section_id.unique():
            here = core_info.section_id.eq(section).to_numpy()
            donors = here & is_mac
            if donors.any():
                distance[here] = cKDTree(xy[donors]).query(xy[here])[0]
        cells["distance_to_mac_um"] = distance[core]
        cells["09B_UMIs"] = depth
        cells["09B_features"] = a.obs["n_genes_by_counts"].iloc[selected].to_numpy()
        cells["09B_x_um"], cells["09B_y_um"] = xy[core].T
        return cells, panel
    finally:
        a.file.close()


def _hard_cells(tile):
    spots = tile["spots"]
    owner = spots.cell_index.to_numpy()
    inside = owner >= 0
    ids, codes, area = np.unique(owner[inside], return_inverse=True, return_counts=True)
    hard = (tile["counts"][:, inside] @ _groups(codes, len(ids))).tocsr()
    hard.eliminate_zeros()
    return ids, area, hard


def summarize(tile):
    spots, x = tile["spots"], tile["counts"]
    sym = pd.Index(tile["genes"]["gene_symbol"])
    ids, area, hard = _hard_cells(tile)
    saved, original = _original(Path(tile["run_dir"]))
    all_ids = pd.Index(tile["cell_ids"], dtype=str)
    if not all_ids.is_unique or (all_ids.get_indexer(saved.cell_id) < 0).any():
        raise ValueError("09B core IDs do not match Step 02 Proseg IDs. Verify the sample and segmentation.")
    lookup = pd.Index(all_ids[ids]).get_indexer(saved.cell_id)
    keep = lookup >= 0
    cells = saved.loc[keep].reset_index(drop=True)
    col = lookup[keep]
    cells["square_count"] = area[col]
    cells["represented_area_um2"] = area[col] * 4
    cells["hard_UMIs"] = np.asarray(hard[:,col].sum(axis=0)).ravel()
    if tile.get("cell_areas") is not None:
        cells["polygon_area_um2"] = np.asarray(tile["cell_areas"])[ids[col]]
    owner = spots.cell_index.to_numpy()
    assigned = owner >= 0
    all_xy = spots[["x_um", "y_um"]].to_numpy()
    codes = np.searchsorted(ids, owner[assigned])
    hard_xy = np.column_stack([np.bincount(codes, weights=v, minlength=len(ids)) / area
                               for v in all_xy[assigned].T])
    cells["centroid_offset_um"] = np.linalg.norm(
        hard_xy[col] - cells[["09B_x_um", "09B_y_um"]].to_numpy(), axis=1)
    core = spots.in_core.to_numpy(dtype=bool)
    # Parent blocks align to the native array lattice, regardless of slide rotation.
    ij = spots.loc[core,["array_row","array_col"]].to_numpy(dtype=np.int64)
    if len(ij) == 0 or len(np.unique(ij,axis=0)) != len(ij):
        raise ValueError("Empty or duplicated 2-µm array locations in the evaluation core.")
    _, blocks, sizes = np.unique(ij // 4, axis=0, return_inverse=True, return_counts=True)
    whole = sizes == 16
    # Distinct owners reflect square-center assignment. Count mixed blocks over complete footprints.
    center_owners = spots.loc[core, "cell_index"].to_numpy(dtype=np.int64)
    owner_pairs = np.unique(np.column_stack([blocks, center_owners]), axis=0)
    n_owners = np.bincount(owner_pairs[owner_pairs[:, 1] >= 0, 0], minlength=len(sizes))
    touches_outside = np.bincount(owner_pairs[owner_pairs[:, 1] < 0, 0], minlength=len(sizes)) > 0
    n_complete = int(whole.sum())
    matrix = x[:,core]
    native8 = (matrix @ _groups(blocks, len(sizes)))[:, whole]
    coverage = [_coverage("2um core squares",matrix),
                _coverage("2um constituents of complete 8um bins",matrix[:,whole[blocks]]),
                _coverage("8um complete 4x4 blocks",native8),
                _coverage("hard-mask cells represented in 09B core",hard[:,col])]
    for name, mask in [("assigned",spots.cell_index.to_numpy()>=0),
                       ("out_of_mask",spots.cell_index.to_numpy()<0)]:
        coverage.append(_coverage(f"2um core {name}", x[:,core & mask]))
    coverage.append(_coverage("2um core, historical QC keep", x[:,core & spots.qc_keep.to_numpy(bool)]))
    for name, ref in [("observed 09B Proseg cells, entire core", saved),
                      ("observed 09B Proseg cells, matched core", cells)]:
        v = ref["09B_UMIs"]
        coverage.append(dict(unit=name, n=len(v), total_UMIs=int(v.sum()),
                             fraction_zero=float(v.eq(0).mean()), median_UMIs=v.median(),
                             p90_UMIs=v.quantile(.9), median_features=ref["09B_features"].median()))
    # Fractions describe the actual retained measurement domain; preserve zero-UMI squares.
    xy = spots[["x_um","y_um"]].to_numpy()
    width = 50.
    ij50 = np.floor((xy-xy.min(axis=0))/width).astype(int)
    occupied = pd.DataFrame(ij50, columns=["tile_x","tile_y"])
    occupied["unassigned_squares"] = (spots.cell_index.to_numpy()<0).astype(int)
    occupied["total_squares"] = 1
    occupied["unassigned_UMIs"] = (np.asarray(x.sum(axis=0)).ravel() *
                                   occupied.unassigned_squares.to_numpy())
    # The historical segmentation used the QC-retained transcripts. Quantify the possible effect on background.
    for suffix, mask in [("QC_keep", spots.qc_keep.to_numpy(bool)), ("outside_QC_keep", ~spots.qc_keep.to_numpy(bool))]:
        occupied["unassigned_" + suffix] = occupied.unassigned_squares.to_numpy() * mask
        occupied["unassigned_UMIs_" + suffix] = occupied.unassigned_UMIs.to_numpy() * mask
    background = occupied.groupby(["tile_x","tile_y"],as_index=False).sum()
    background["out_of_mask_fraction"] = background.unassigned_squares/background.total_squares
    hard_panel = _panel_counts(hard[:,col],sym)
    # Freeze comparisons to the labels and the observed pan-T status from the earlier run.
    base = original.loc[keep].reset_index(drop=True)
    info = {"n_original_core":len(saved), "n_matched_core":len(cells),
            "core_cells_without_retained_squares":int((~keep).sum()),
            "matched_centroid_offset_median_um":float(cells.centroid_offset_um.median()) if len(cells) else None,
            "matched_centroid_offset_p95_um":float(cells.centroid_offset_um.quantile(.95)) if len(cells) else None,
            "native8_complete":int(whole.sum()),"native8_partial":int((~whole).sum()),
            "native8_rule":"all 16 retained native array positions inside the core",
            "complete_8um_with_multiple_cells":int(np.sum(whole & (n_owners >= 2))),
            "complete_8um_with_cell_and_outside":int(np.sum(whole & (n_owners > 0) & touches_outside)),
            "fraction_complete_8um_with_multiple_cells":float(np.sum(whole & (n_owners >= 2)) / n_complete) if n_complete else None,
            "fraction_complete_8um_with_cell_and_outside":float(np.sum(whole & (n_owners > 0) & touches_outside) / n_complete) if n_complete else None,
            "diagnostic_grid_microns":width,"diagnostic_grid_origin_um":xy.min(axis=0).tolist()}
    for group,mask in [("T",saved.cell_type.str.contains(r"^T[ _-]*cell",case=False)),
                       ("B",saved.cell_type.str.contains(r"^B[ _-]*cell",case=False))]:
        info[f"{group}_original_core"] = int(mask.sum())
        info[f"{group}_without_squares"] = int((mask & ~keep).sum())
        group_owner = ids[col[np.asarray(mask[keep], dtype=bool)]]
        touches = np.unique(owner_pairs[np.isin(owner_pairs[:,1], group_owner),0])
        full_group = touches[whole[touches]]
        info[f"complete_8um_touching_{group}"] = len(full_group)
        info[f"complete_8um_touching_{group}_and_multiple_cells"] = int((n_owners[full_group] >= 2).sum())
        coverage.append(_coverage(f"hard-mask cells, 09B {group} label", hard[:,col[np.asarray(mask[keep], dtype=bool)]]))
    return dict(cells=cells, raw_panel=hard_panel, proseg_panel=base, info=info, ids=ids, col=col,
                coverage=pd.DataFrame(coverage), background=background)


def _save_targeted(output_dir, values):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sparkle_targeted_counts.npz"
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **values)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def fit_panel(tile, summary, *, background_genes=128, radius=200., display_progress=True, output_dir=None):
    from stambient import SPARKLE
    from stambient.binning import bin_empty_dnbs

    x = tile["counts"]
    owner = tile["spots"].cell_index.to_numpy()
    xy = tile["spots"][["x_um","y_um"]].to_numpy()
    sym = pd.Index(tile["genes"]["gene_symbol"])
    inside = owner >= 0
    ids, codes = np.unique(owner[inside], return_inverse=True)
    all_ids = pd.Index(tile["cell_ids"], dtype=str)
    if (not np.array_equal(ids, summary["ids"]) or not all_ids.is_unique
            or (len(ids) and ids[-1] >= len(all_ids))
            or not np.array_equal(all_ids[ids[summary["col"]]], summary["cells"].cell_id)):
        raise ValueError("SPARKLE cell order and matched analysis cells disagree with the hard-mask aggregation.")
    if not isinstance(background_genes, (int, np.integer)) or background_genes < 0:
        raise ValueError("background_genes must be a nonnegative integer.")
    bg = np.asarray(x[:,~inside].sum(axis=1)).ravel()
    positive = np.flatnonzero(bg > 0)
    if len(positive) == 0 or len(ids) < 2:
        raise ValueError("Fitting needs measured out-of-mask RNA and neighboring segmented cells.")
    top = positive[np.argsort(-bg[positive].astype(float),kind="stable")[:background_genes]]
    axis = np.union1d(top, np.flatnonzero(sym.isin(TARGETS)))
    if bg[axis].sum() == 0:
        raise ValueError("The selected genes have zero measured out-of-mask UMIs.")
    bin_size = 50.
    _, bin_areas, outside, bin_index = bin_empty_dnbs(xy, owner, bin_size)
    if len(bin_areas) < 2:
        raise ValueError(f"Only {len(bin_areas)} occupied out-of-mask 50-µm tile(s). At least two are needed to assess spatial variation.")
    selected = x[axis]
    background_counts = (selected[:,outside] @ _groups(bin_index,len(bin_areas))).toarray()
    bg_density = background_counts / bin_areas
    varies = ~np.isclose(bg_density.min(axis=1), bg_density.max(axis=1), atol=1e-12, rtol=1e-10)
    if not varies.any():
        raise ValueError("No selected gene varies in observed out-of-mask UMIs per square across SPARKLE’s 50-µm background tiles.")
    n_bg_positive = int(np.count_nonzero(background_counts.sum(axis=0)))
    model = SPARKLE(bin_size=bin_size, max_radius=radius, n_high_genes=None, use_gpu=False,
                    verbose=display_progress)
    corrected, diag = model.fit_transform(selected,xy,owner)
    if corrected.shape != (len(axis),len(ids)) or not np.isfinite(corrected).all() or (corrected<0).any():
        raise ValueError("Unexpected SPARKLE output for gene or cell axis.")
    if diag["empty_to_cell_graph_nnz"] == 0 or diag["cell_to_cell_graph_nnz"] == 0:
        raise ValueError(f"SPARKLE found no out-of-mask-to-cell or no cell-to-cell neighbors within {radius} µm.")
    raw = (selected[:,inside] @ _groups(codes,len(ids))).tocsr()
    for start in range(0, len(axis), 32):
        rows = slice(start, start + 32)
        if np.any(corrected[rows] > raw[rows].toarray() + 1e-8):
            raise ValueError("Corrected counts exceed the matching observed hard-mask counts.")
    fitted = np.asarray(diag["gene_indices"],dtype=int)
    if not np.array_equal(np.sort(fitted), np.arange(len(axis))):
        raise ValueError("SPARKLE diagnostics do not cover each selected gene exactly once.")
    genes = tile["genes"].iloc[axis[fitted]].reset_index(drop=True).copy()
    genes["matrix_row"] = fitted
    genes["out_of_mask_UMIs"] = bg[axis[fitted]]
    genes["model_background_bins_with_gene_UMIs"] = np.count_nonzero(background_counts, axis=1)[fitted]
    genes["model_background_density_varies"] = varies[fitted]
    genes["raw_hard_mask_UMIs"] = np.asarray(raw.sum(axis=1)).ravel()[fitted]
    genes["corrected_UMIs"] = corrected.sum(axis=1)[fitted]
    genes["UMIs_removed"] = genes.raw_hard_mask_UMIs - genes.corrected_UMIs
    genes["alpha"] = diag["alphas"]
    genes["weighted_r2"] = diag["r2_scores"]
    genes["passes_r2"] = genes.weighted_r2.ge(diag["r2_threshold"])
    panel = _panel_counts(corrected[:,summary["col"]],sym[axis])
    compact = {k:v for k,v in diag.items() if k not in ("gene_indices","alphas","r2_scores")}
    compact.update(version=importlib.metadata.version("stambient"), n_input_genes=len(sym),
                   n_fit_features=len(axis), selection="highest out-of-mask counts plus prespecified panels",
                   n_model_background_bins_with_selected_UMIs=n_bg_positive,
                   n_genes_with_count_reduction=int(genes.UMIs_removed.gt(1e-8).sum()))
    if output_dir is not None:
        features = tile["genes"].iloc[axis]
        path = _save_targeted(output_dir, dict(corrected_counts=corrected, raw_hard_mask_counts=raw.toarray(),
                              gene_id=features.gene_id.to_numpy(dtype=str), gene_symbol=features.gene_symbol.to_numpy(dtype=str),
                              cell_id=np.asarray(all_ids[ids], dtype=str), cell_index=ids,
                              evaluation_core=np.isin(np.arange(len(ids)), summary["col"])))
        compact["local_matrix_file"] = path.name
    return panel, genes, compact


def comparisons(summary, corrected=None):
    c = summary["cells"]
    t = c.T_label_09B
    mac = c.macrophage_label_09B
    b = c.cell_type.str.contains(r"^B[ _-]*cell",case=False)
    supported = c.panT_09B_observed.fillna(False).astype(bool)
    one_symbol = c.panT_09B_detected_symbols.gt(0)
    near, far = c.distance_to_mac_um.le(25), c.distance_to_mac_um.ge(50)
    masks = {"all_matched":np.ones(len(c),bool), "T_label":t, "B_label":b, "macrophage_label":mac,
             "panT09C":supported, "panT09C_near25":supported & near, "panT09C_far50":supported & far,
             "panT09C_nonmac_near25":supported & ~mac & near, "panT09C_nonmac_far50":supported & ~mac & far,
             "T_panT09C":t & supported, "T_panT09C_near25":t & supported & near,
             "T_panT09C_far50":t & supported & far,
             "T_panT09B":t & one_symbol, "T_panT09B_near25":t & one_symbol & near, "T_panT09B_far50":t & one_symbol & far}
    for var in ["hard_UMIs","represented_area_um2"]:
        if t.sum() and c.loc[t,var].nunique() > 1:
            bins = pd.qcut(c.loc[t,var],3,duplicates="drop")
            for k in bins.cat.categories:
                masks[f"T_{var}_{k}"] = t & c.index.isin(bins.index[bins.eq(k)])
    methods = {"observed_09B_Proseg":summary["proseg_panel"],"observed_same_hard_mask":summary["raw_panel"]}
    if corrected is not None: methods["SPARKLE_same_hard_mask"] = corrected
    rows=[]
    for name,mask in masks.items():
        for method,panel in methods.items():
            for target in PANELS:
                v=panel.loc[mask,target].dropna()
                rows.append(dict(group=name,method=method,panel=target,n_cells=int(np.sum(mask)),
                                 n_measured=len(v),mean_panel_counts=v.mean(),
                                 total_panel_counts=v.sum() if len(v) else np.nan,
                                 cells_positive=int(v.gt(0).sum()) if len(v) else np.nan))
    return pd.DataFrame(rows)


def plot_audit(tile,summary):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    c=summary["coverage"].iloc[[0,2,3]]
    axes[0].bar(["2 µm","8 µm\ncomplete","cell\nhard mask"],c.fraction_zero,color=["#27766f","#668f93","#284a64"])
    axes[0].set(ylabel="Fraction with zero observed UMIs",ylim=(0,1),title="Sparsity in the original 09B core")
    for i,(_,r) in enumerate(c.iterrows()):
        if r.n:axes[0].text(i, min(.97,r.fraction_zero+.025), f"{r.fraction_zero:.1%}",ha="center",fontsize=9)
    grid=summary["background"]
    s=axes[1].scatter(grid.tile_x,grid.tile_y,c=grid.out_of_mask_fraction,cmap="viridis",vmin=0,vmax=1,
                      marker="s",s=23,linewidths=0)
    axes[1].set(xlabel="50-µm tile along Proseg x (image row)",ylabel="50-µm tile along Proseg y (image column)",aspect="equal",
                title="Out-of-mask coverage in the buffered region")
    fig.colorbar(s,ax=axes[1],label="Fraction of retained squares")
    return fig
