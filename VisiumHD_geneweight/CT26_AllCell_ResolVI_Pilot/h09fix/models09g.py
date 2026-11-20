"""Fit both regional models on one barcode-aligned native input and verify the common cell counts."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from compare09d import TARGETS, _groups
from readout09f import _counts, _flag, _input, _table
from run09d import _write_json
from run09f import _run_r


def input_digest(input_dir):
    """Fingerprint raw counts, axes, roles and annotations without loading them into memory."""
    parts = {}
    for name in ("cells.tsv", "counts.mtx", "genes.tsv", "positions.tsv"):
        with (Path(input_dir) / name).open("rb") as stream:
            parts[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    overall = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return overall, parts


def _native(input_dir):
    key, checksums = input_digest(input_dir)
    genes, spots, cells, counts = _input(Path(input_dir))
    tissue = spots.in_tissue.to_numpy(bool)
    assigned = spots.cell_id.ne("").to_numpy()
    if not assigned.any() or not (~assigned).any() or np.any(assigned & ~tissue):
        raise ValueError("Need assigned Proseg cells, unassigned positions, and all assigned squares modeled as tissue.")
    rc = spots[["array_row", "array_col"]].to_numpy(float)
    widths = np.ptp(rc, axis=0) + 1
    if (not np.isfinite(rc).all() or (rc != np.floor(rc)).any() or widths[0] != widths[1] or
            widths.prod() != len(rc) or len(np.unique(rc, axis=0)) != len(rc) or len(rc) > 4096):
        raise ValueError("Expected a complete native square window of at most 64 × 64 positions.")
    ids, code = np.unique(spots.loc[assigned, "cell_id"].to_numpy(dtype=str), return_inverse=True)
    aggregation = _groups(code, len(ids))
    labels = np.full(len(spots), -1, dtype=np.int64)
    labels[assigned] = code
    members = genes.isin(TARGETS)
    listed = _table(Path(input_dir) / "genes.tsv", "gene")
    if not np.array_equal(_flag(listed.target).to_numpy(bool), members):
        raise ValueError("Marker flags in the native input differ from the registered lineage panels.")
    markers = np.flatnonzero(members)
    if len(markers) == 0:
        raise ValueError("Native input has no requested marker genes.")
    raw_markers = (counts[markers][:, assigned] @ aggregation).toarray()
    totals = np.asarray(counts[:, tissue].sum(axis=1)).ravel()
    positive = np.flatnonzero(totals > 0)
    ranked = sorted(positive, key=lambda g: (-int(totals[g]), genes[g]))
    spatial = np.union1d(ranked[:200], np.flatnonzero(members & (totals > 0))).astype(int)
    if len(spatial) < 2:
        raise ValueError("Both methods need at least two expressed genes for the common spatial fit.")
    xy = 2. * (rc - rc.min(axis=0))
    # Both estimators use the same native lattice. The reader maps complete polygon
    # footprints to these barcodes using separate, registered anatomical coordinates.
    complete = _flag(cells.complete_core).reindex(ids).to_numpy(bool)
    centroid = np.asarray((aggregation.T @ xy[assigned])) / np.bincount(code)[:, None]
    edge_distance = np.minimum(centroid + 1., 2. * widths - 1. - centroid).min(axis=1)
    return dict(genes=genes, spots=spots, counts=counts, area=aggregation, ids=ids, labels=labels, xy=xy,
                markers=markers, raw=raw_markers, plan=spatial, assigned=assigned, tissue=tissue,
                input_sha256=key, input_files_sha256=checksums, edge_dist=edge_distance[complete])


def _finish(out, input_dir, data, corrected, gene_fits, info):
    if input_digest(input_dir)[0] != data["input_sha256"]:
        raise ValueError("Native input changed during the fit.")
    if (corrected.shape != data["raw"].shape or not np.isfinite(corrected).all() or (corrected < 0).any()):
        raise ValueError("Expected finite, nonnegative marker estimates on the common gene and cell axes.")
    target = data["genes"].take(data["markers"]).to_numpy(dtype=str)
    info.update(state="complete", input_sha256=data["input_sha256"], input_files_sha256=data["input_files_sha256"],
                n_positions=len(data["labels"]), n_cells=len(data["ids"]), n_assayed_targets=len(target),
                n_spatial_genes=len(data["plan"]), n_assigned_squares=int(data["assigned"].sum()),
                n_spotclean_background_squares=int((~data["tissue"]).sum()),
                n_sparkle_background_squares=int((~data["assigned"]).sum()),
                native_pitch_um=2, coordinate_system="relative native array row and column in micrometres")
    gene_fits.to_csv(out / "gene_fits.csv", index=False)
    np.savez_compressed(out / "cell_counts.npz", genes=target, cell_ids=data["ids"],
                        raw=data["raw"], corrected=corrected)
    _write_json(out / "fit.json", info)
    return info


def _plan(out, data):
    frame = pd.DataFrame({"gene": data["genes"].take(data["plan"])})
    frame.to_csv(out / "shared_plan.tsv", sep="\t", index=False)
    return frame.gene.to_numpy(str)


def fit_sparkle(input_dir, out_dir, *, bin_um=12., radius_um=50.):
    """Estimate one local SPARKLE model on all measured positions in the submitted window."""
    from stambient import SPARKLE
    from stambient.binning import bin_empty_dnbs

    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    if (out / "fit.json").exists():
        raise FileExistsError(f"Completed fit exists at {out}. Choose a fresh fit directory.")
    started = time.monotonic()
    data = _native(input_dir)
    if not np.isfinite(bin_um) or not np.isfinite(radius_um) or min(bin_um, radius_um) <= 0:
        raise ValueError("SPARKLE aggregation width and neighbor radius must be positive finite micrometres.")
    plan_genes = _plan(out, data)
    counts = data["counts"][data["plan"]]
    xy, labels = data["xy"], data["labels"]
    _, areas, outside, group = bin_empty_dnbs(xy, labels, bin_um)
    if len(data["ids"]) < 2 or len(areas) < 2:
        raise ValueError("SPARKLE needs at least two represented cells and two occupied out-of-mask background bins.")
    bg = (counts[:, outside] @ _groups(group, len(areas))).toarray()
    ng = np.count_nonzero(bg, axis=1)
    spread = np.ptp(bg / areas, axis=1)
    if not (spread > 1e-12).any():
        raise ValueError("Spatially fitted genes lack variation in out-of-mask bin density.")
    grid = [4., 8., 12., 20., 32.]
    grid = [v for v in grid if v <= radius_um]
    if not grid:
        raise ValueError("SPARKLE radius must be at least 4 µm for this regional decay grid.")
    model = SPARKLE(bin_size=bin_um, max_radius=radius_um, lambda_grid=grid,
                    n_high_genes=None, r2_threshold=.01, use_gpu=False, verbose=False)
    fit_started = time.monotonic()
    result, diag = model.fit_transform(counts, xy, labels)
    fit_time = time.monotonic() - fit_started
    if not diag["empty_to_cell_graph_nnz"] or not diag["cell_to_cell_graph_nnz"]:
        raise ValueError("The fit has no out-of-mask-to-cell or no cell-to-cell neighbors.")
    plan_raw = (counts[:, data["assigned"]] @ data["area"]).toarray()
    if (result.shape != plan_raw.shape or not np.isfinite(result).all() or (result < 0).any() or
            not np.all(result <= plan_raw + 1e-8)):
        raise ValueError("SPARKLE result fails the raw hard-mask reconciliation.")
    ix = np.asarray(diag["gene_indices"], dtype=int)
    if not np.array_equal(np.sort(ix), np.arange(len(plan_genes))):
        raise ValueError("SPARKLE diagnostics differ from the common spatial gene set.")
    frame = pd.DataFrame(dict(gene=plan_genes[ix], raw_cell_UMIs=plan_raw.sum(axis=1)[ix],
                              corrected_cell_sum=result.sum(axis=1)[ix], alpha=diag["alphas"],
                              weighted_r2=diag["r2_scores"], out_of_mask_UMIs=bg.sum(axis=1)[ix],
                              positive_background_bins=ng[ix]))
    frame["passes_r2"] = frame.weighted_r2.ge(diag["r2_threshold"])
    frame["amount_removed"] = frame.raw_cell_UMIs - frame.corrected_cell_sum
    frame["status"] = np.select([frame.amount_removed.gt(1e-8), frame.passes_r2],
                                ["subtracted", "gate_passed_no_subtraction"], default="below_r2_gate")
    target_axis = data["genes"].take(data["markers"])
    target_pos = pd.Index(plan_genes).get_indexer(target_axis)
    measured = target_pos >= 0
    corrected = data["raw"].astype(float)
    corrected[measured] = result[target_pos[measured]]
    if np.any(~measured):
        # A gene observed only in provisional background has zero raw counts in every cell.
        if corrected[~measured].any():
            raise ValueError("An observed target was omitted from the shared spatial panel.")
        missing = pd.DataFrame(dict(gene=target_axis[~measured], status="no_positive_tissue_signal",
                                    raw_cell_UMIs=0, corrected_cell_sum=0., amount_removed=0.))
        frame = pd.concat([frame, missing], ignore_index=True)
    lam = float(diag["lambda_estimated"])
    info = dict(method="SPARKLE", version=metadata.version("stambient"), background_tile_um=bin_um,
                lambda_um=lam, lambda_grid_um=grid, lambda_search_residuals=diag["lambda_search_rss"],
                lambda_at_search_boundary=lam in (grid[0], grid[-1]),
                lambda_grid_flat=bool(np.allclose(diag["lambda_search_rss"], diag["lambda_search_rss"][0])),
                out_of_mask_to_cell_links=int(diag["empty_to_cell_graph_nnz"]),
                cell_to_cell_links=int(diag["cell_to_cell_graph_nnz"]),
                radius_um=radius_um, background_bins=len(areas),
                background_bins_with_counts=int((bg.sum(axis=0) > 0).sum()),
                r2_threshold=diag["r2_threshold"], genes_passing_gate=int(frame.passes_r2.eq(True).sum()),
                targets_passing_gate=int((frame.gene.isin(target_axis) & frame.passes_r2.eq(True)).sum()),
                targets_subtracted=int((frame.gene.isin(target_axis) & frame.amount_removed.gt(1e-8)).sum()),
                complete_cells_within_radius_of_crop_edge=int((data["edge_dist"] < radius_um).sum()),
                n_complete_cells=int(len(data["edge_dist"])), time_in_model_seconds=round(fit_time, 3),
                wall_seconds=round(time.monotonic() - started, 3), backend=diag["compute_backend"])
    return _finish(out, input_dir, data, corrected, frame, info)


def _collect_spotclean(input_dir, out, data, metrics):
    """Validate SpotClean gene/position axes and aggregate tissue bins on the same Proseg cells."""
    fit_axis = _table(out / "decont_genes.tsv", "gene").index
    tissue_bcs = _table(out / "decont_positions.tsv", "barcode").index
    gene_receipt = _table(out / "gene_status.tsv", "gene")
    spots = data["spots"]
    mask = data["assigned"]
    if (set(tissue_bcs) != set(spots.index[data["tissue"]]) or set(gene_receipt.index) != set(data["genes"])
            or set(gene_receipt.index[_flag(gene_receipt.in_output)]) != set(fit_axis)):
        raise ValueError("The SpotClean output axes disagree with the common regional counts.")
    expr = _counts(out / "decont.mtx", (len(fit_axis), len(tissue_bcs)))
    cols = tissue_bcs.get_indexer(spots.index[mask])
    target = data["genes"].take(data["markers"])
    rows = fit_axis.get_indexer(target)
    if (cols < 0).any():
        raise ValueError("SpotClean omitted an assigned square from its corrected tissue positions.")
    corrected = np.zeros_like(data["raw"], dtype=float)
    present = rows >= 0
    corrected[present] = (expr[rows[present]][:, cols] @ data["area"]).toarray()
    omitted = gene_receipt.loc[target[~present]]
    if not omitted.status.isin(["background_only", "undetected", "zero_after_fit"]).all():
        raise ValueError("A marker disappeared without a model receipt.")
    plan = data["genes"].take(data["plan"])
    plan_status = gene_receipt.loc[plan, "status"]
    if not plan_status.isin(["spatial_em", "zero_after_fit"]).all():
        raise ValueError("SpotClean did not spatially fit the common gene plan.")
    result_genes = data["genes"][data["genes"].isin(set(plan) | set(target))]
    frame = gene_receipt.loc[result_genes].rename_axis("gene").reset_index()
    frame["raw_cell_UMIs"] = (data["counts"][data["genes"].get_indexer(result_genes)][:, mask]
                               .sum(axis=1).A1)
    frame["corrected_cell_sum"] = 0.
    in_corrected = result_genes.isin(fit_axis)
    if in_corrected.any():
        expressed_rows = fit_axis.get_indexer(result_genes[in_corrected])
        frame.loc[in_corrected, "corrected_cell_sum"] = expr[expressed_rows][:, cols].sum(axis=1).A1
    r_fit = pd.read_csv(out / "fit.tsv", sep="\t")
    if len(r_fit) != 1:
        raise ValueError("Expected one SpotClean region fit receipt.")
    fit = r_fit.iloc[0].to_dict()
    for key in ("distal_rate", "bleeding_rate"):
        if not np.isfinite(fit[key]) or not 0 <= fit[key] <= 1:
            raise ValueError(f"Invalid fitted {key}.")
    plan_tissue = gene_receipt.loc[plan, "raw_tissue_umi"].sum()
    plan_bg = gene_receipt.loc[plan, "raw_background_umi"].sum()
    bleed_lower = float(plan_bg / (plan_bg + plan_tissue))
    fit.update(method="SpotClean", **metrics, version=fit["spotclean_version"],
               bleeding_rate_lower_bound=bleed_lower,
               bleeding_rate_at_constraint=np.isclose(fit["bleeding_rate"], [bleed_lower, 1.], atol=1e-6, rtol=0).any().item(),
               distal_rate_at_constraint=np.isclose(fit["distal_rate"], [.1, 1.], atol=1e-6, rtol=0).any().item(),
               radius_at_search_boundary=fit["radius_um"] in (4., 20.),
               iterations_reached_max=bool(fit["em_iterations"] >= fit["maxit"]))
    return _finish(out, input_dir, data, corrected, frame, fit)


def fit_spotclean(input_dir, out_dir, *, rscript, fit_minutes=60):
    """Fit R SpotClean, record per-process wall time and peak RSS and aggregate the corrected native positions."""
    out = Path(out_dir).resolve()
    out.mkdir(exist_ok=True, parents=True)
    if (out / "fit.tsv").exists() or (out / "fit.json").exists():
        raise FileExistsError(f"Existing fit at {out}. Choose a fresh fit directory.")
    if not np.isfinite(fit_minutes) or fit_minutes <= 0:
        raise ValueError("SpotClean wall-time limit must be positive and finite.")
    inputs = Path(input_dir).resolve()
    data = _native(inputs)
    _plan(out, data)
    runner = Path(__file__).with_name("spotclean09f.R")
    metrics = {}
    try:
        _run_r([str(rscript), "--vanilla", str(runner), str(inputs), str(out), "--09g-shared-plan"],
               out / "spotclean.log", metrics, timeout=60 * fit_minutes)
    finally:
        _write_json(out / "runtime.json", metrics)
    return _collect_spotclean(inputs, out, data, metrics)
