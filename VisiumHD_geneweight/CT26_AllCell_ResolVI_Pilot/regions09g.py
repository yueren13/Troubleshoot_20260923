"""Select shared native windows with measurable source and neighboring-cell readouts."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
import shapely

from bins09d import STUDY_ROOT, TEMP_PROJECT_ROOT, locate_paths, _rows, _raw_cell_order, _local_assignment
from compare09d import PANELS, TARGETS, _original
from raw09d import read_square_counts
from roi09f import _positions, _annotations
from run09d import _write_json


CELL_FIELDS = ["cell_type", "T_label_09B", "macrophage_label_09B", "panT_09B_observed",
               "distance_to_mac_um"]
STRICT = "score_low_outside_proseg_and_sr_tissue"
POOLED = "score_low_outside_proseg_including_sr_tissue"


def _geometry_lookup(path, seen_ids, all_ids):
    """Read only the boundaries belonging to cells represented in the search area."""
    compact = pd.Index(all_ids[seen_ids], dtype=str)
    polys = np.empty(len(compact), dtype=object)
    polys.fill(None)
    for part in _rows(path, ["cell", "geometry"], size=10000):
        ix = compact.get_indexer(pd.to_numeric(part.cell, errors="raise").astype(np.int64).astype(str))
        keep = ix >= 0
        if keep.any():
            polys[ix[keep]] = shapely.from_wkb(part.loc[keep, "geometry"].to_numpy())
    if shapely.is_missing(polys).any():
        raise ValueError("A represented Proseg cell lacks its original polygon.")
    return polys


def _lattice(pool):
    rc = pool[["array_row","array_col"]].to_numpy(int)
    lower = rc.min(axis=0)
    rr, cc = (rc - lower).T
    shape = (rr.max()+1, cc.max()+1)
    if np.prod(shape, dtype=np.int64) > 25_000_000:
        raise ValueError("Native coordinates are too dispersed for a bounded search around this 09B core.")
    grid = np.full(shape, -1, dtype=np.int32)
    grid[rr, cc] = np.arange(len(pool))
    return grid, lower


def _intersects(first, second, side):
    return (abs(first["row"] - second["row"]) < side and
            abs(first["col"] - second["col"]) < side)


def _pick(ranked, side, n):
    if not ranked or n == 1:
        return ranked[:n]
    # For the two-region comparison, consider each possible first choice. A
    # highest-ranked central window can otherwise block both members of a pair.
    for i, first in enumerate(ranked):
        for second in ranked[i+1:]:
            if not _intersects(first, second, side):
                return [first, second]
    return ranked[:1]


def _candidates(grid, lower, pool, meta, polys, core_disk, side, halo, stride, score_max,
                min_bg, strict, min_mac, min_t):
    """Inspect a fixed spatial lattice. Score and previous labels are available at this step."""
    owner = pool.cell_index.to_numpy()
    sr = pool.sr_in_tissue.to_numpy(bool)
    score = pool.he_tissue_score.to_numpy(float)
    xy = pool[["x_um","y_um"]].to_numpy(float)
    low = (owner < 0) & (score <= score_max)
    low_off = low & ~sr
    higher_sr = (score > score_max) & sr
    mac_flag = meta.macrophage_label_09B.astype("boolean")
    mac = mac_flag.fillna(False).to_numpy(bool)
    near = (meta.panT_09B_observed.astype("boolean") & ~mac_flag &
            meta.distance_to_mac_um.le(25)).fillna(False).to_numpy(bool)
    t_label = meta.T_label_09B.astype("boolean").fillna(False).to_numpy(bool)
    audit, candidates = [], []
    halo_edge = np.ones((side,side), dtype=bool)
    halo_edge[halo:side-halo, halo:side-halo] = False
    for r in range(0, grid.shape[0]-side+1, stride):
        for c in range(0, grid.shape[1]-side+1, stride):
            win = grid[r:r+side, c:c+side]
            rec = dict(row=int(r+lower[0]), col=int(c+lower[1]),
                       decision="incomplete_grid", selected=False)
            audit.append(rec)
            if (win < 0).any():
                continue
            mid = win[halo:side-halo, halo:side-halo].ravel()
            if not core_disk[mid].all():
                rec["decision"] = "core_outside_previous_evaluation_disk"
                continue
            ix = win.ravel()
            off, all_low = low_off[ix].sum(), low[ix].sum()
            n_min = int(np.ceil(min_bg * len(ix)))
            rec.update(n_bg_outside_sr=int(off), n_bg_either_sr=int(all_low),
                       n_measured_scores=int(np.isfinite(score[ix]).sum()),
                       n_higher_score_sr_tissue=int(higher_sr[ix].sum()))
            mode = STRICT if off >= n_min else POOLED
            if (strict and mode != STRICT) or all_low < n_min:
                rec["decision"] = "insufficient_provisional_background"
                continue
            if rec["n_higher_score_sr_tissue"] < np.ceil(.25*len(ix)):
                rec["decision"] = "insufficient_higher_score_tissue"
                continue
            present = np.unique(owner[mid])
            present = present[present >= 0]
            hull = shapely.convex_hull(shapely.multipoints(xy[mid]))
            complete = present[shapely.covers(hull, polys[present])]
            complete = complete[~np.isin(complete, owner[win[halo_edge]])]
            n_mac, n_near = int(mac[complete].sum()), int(near[complete].sum())
            rec.update(n_t_labels=int(t_label[complete].sum()), background_mode=mode, n_complete=int(len(complete)), n_mac=n_mac, n_near_t=n_near,
                       n_complete_prior_labeled=int(meta.cell_type.iloc[complete].notna().sum()))
            if n_mac < min_mac or n_near < min_t or rec["n_t_labels"] < min_t:
                rec["decision"] = "insufficient_complete_cells"
                continue
            rec["decision"] = "passed_geometry_and_label_screen"
            candidates.append((rec, ix, complete))
    return audit, candidates, mac, near, t_label


def _read_union(square_dir, barcodes):
    raw, features, path = read_square_counts(square_dir, barcodes)
    codes, symbols = pd.factorize(features.gene_symbol, sort=False)
    groups = sparse.csr_matrix((np.ones(len(codes),dtype=np.int64), (codes,np.arange(len(codes)))),
                               shape=(len(symbols),len(codes)))
    matrix = (groups @ raw).tocsc()
    genes = pd.DataFrame({"gene":symbols, "target":pd.Index(symbols).isin(TARGETS).astype(int)})
    return matrix, genes, path, len(features)


def _write_input(folder, spots, counts, genes, meta, complete, receipt, halo, side, score_max):
    folder.mkdir()
    origin = receipt["origin_rc"]
    rc = spots[["array_row","array_col"]].to_numpy() - origin
    spots["in_core"] = ((rc >= halo) & (rc < side-halo)).all(axis=1).astype(int)
    owner = spots.cell_index.to_numpy()
    assigned = owner >= 0
    ids = np.unique(owner[assigned])
    incl = meta.iloc[ids].copy()
    incl["complete_core"] = np.isin(ids, complete).astype(int)
    spots["cell_id"] = ""
    spots.loc[assigned, "cell_id"] = meta.index.to_numpy()[owner[assigned]]
    low = (~assigned) & (spots.he_tissue_score.to_numpy() <= score_max)
    off = spots.sr_in_tissue.eq(0).to_numpy()
    background = low & off if receipt["background_mode"] == STRICT else low
    spots["in_tissue"] = (~background).astype(int)
    spots["background_class"] = np.select(
        [background & off, background & ~off],
        ["low_score_outside_Proseg_and_Space_Ranger_tissue", "low_score_outside_Proseg_inside_Space_Ranger_tissue"],
        default="potential_tissue")
    spots["raw_total_umis"] = np.asarray(counts.sum(axis=0)).ravel()
    receipt.update(n_positions=len(spots), n_background=int(background.sum()), n_complete_cells=len(complete),
                   n_zero_positions=int(spots.raw_total_umis.eq(0).sum()), total_umis=int(spots.raw_total_umis.sum()),
                   score_missing=int(spots.he_tissue_score.isna().sum()),
                   roi_score_quantiles=spots.he_tissue_score.quantile([0,.05,.25,.5,.75,.95,1]).to_dict(),
                   background_classes=spots.background_class.value_counts().to_dict())
    spots.drop(columns="cell_index").to_csv(folder/"positions.tsv", sep="\t", index=False)
    incl.to_csv(folder/"cells.tsv", sep="\t", index_label="cell_id")
    genes.to_csv(folder/"genes.tsv", sep="\t",index=False)
    mmwrite(folder/"counts.mtx", counts, field="integer", symmetry="general")
    _write_json(folder/"roi_receipt.json",receipt)


def prepare_regions(out_dir, *, run_dir, study_root=STUDY_ROOT, temp_project_root=TEMP_PROJECT_ROOT,
                    overrides=None, side_bins=64, halo_bins=6, n_regions=2, score_max=.1,
                    min_background_fraction=.25, min_mac_cells=3, min_near_t_cells=3,
                    min_panel_umis=5, zarr_table="square_002um", require_sr_background=False,
                    stride_bins=None):
    """Return selected 09F-format input folders. Shortfalls are recorded before any fits."""
    if not isinstance(side_bins, int) or not 8 <= side_bins <= 64:
        raise ValueError("side_bins must be an integer between 8 and 64.")
    if not isinstance(halo_bins, int) or halo_bins < 1 or 2*halo_bins >= side_bins-1:
        raise ValueError("Leave at least two native square centers across the evaluation core.")
    if not isinstance(n_regions, int) or n_regions not in (1, 2):
        raise ValueError("Request one or two nonoverlapping biological pilots per section.")
    if not 0 <= score_max < 1 or not 0 < min_background_fraction < .75:
        raise ValueError("Choose score_max in [0,1) and min_background_fraction in (0,0.75).")
    if any(not isinstance(v, int) or v < 1 for v in (min_mac_cells, min_near_t_cells, min_panel_umis)):
        raise ValueError("The two group sizes and raw UMI minima must be positive integers.")
    stride = max(1, side_bins // 8) if stride_bins is None else stride_bins
    if not isinstance(stride, int) or not 1 <= stride <= side_bins:
        raise ValueError("stride_bins must be an integer within the window size.")
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    run = Path(run_dir).resolve()
    geom = json.loads((run/"complete.json").read_text())["geometry"]
    if float(json.loads((run/"settings.json").read_text())["settings"]["microns_per_unit"]) != 1.:
        raise ValueError("09G requires registered micron units in the completed 09B run.")
    sample = str(geom["section"])
    paths = locate_paths(sample, study_root, temp_project_root, overrides)
    pool, center = _positions(paths, geom, side_bins, None)
    sources = _annotations(paths, pool, zarr_table)
    all_ids, _ = _raw_cell_order(paths["raw_cells"])
    owner = _local_assignment(paths["boundaries"], pool, all_ids)
    # Compact index into the metadata and polygon arrays. Negative denotes unassigned.
    seen, compact = np.unique(owner[owner>=0], return_inverse=True)
    owner[owner>=0] = compact
    pool["cell_index"] = owner
    prior, _ = _original(run)
    meta = prior.set_index("cell_id", verify_integrity=True)[CELL_FIELDS].reindex(all_ids[seen])
    polys = _geometry_lookup(paths["boundaries"], seen, all_ids)
    rc_grid, lower = _lattice(pool)
    core_disk = np.linalg.norm(pool[["x_um","y_um"]].to_numpy()-center, axis=1) <= float(geom["inner_radius_um"])
    score_groups = pool.loc[core_disk].assign(out_of_mask=owner[core_disk] < 0).groupby(
        ["sr_in_tissue", "out_of_mask"])
    score_groups.he_tissue_score.describe(percentiles=[.05,.10,.25,.5,.75,.90,.95]).join(
        score_groups.size().rename("n_positions")).to_csv(out/"score_search.csv")
    audit, candidates, mac, near, t_label = _candidates(rc_grid, lower, pool, meta, polys, core_disk, side_bins,
                        halo_bins, stride, score_max, min_background_fraction, require_sr_background,
                        min_mac_cells, min_near_t_cells)
    # Rank using pre-existing group membership. Raw counts below are minimum
    # evaluability checks, and corrected counts never enter selection.
    candidates.sort(key=lambda x: (x[0]["background_mode"] != STRICT,
                    -min(x[0]["n_mac"]/min_mac_cells, x[0]["n_near_t"]/min_near_t_cells),
                    -x[0]["n_complete_prior_labeled"], x[0]["row"], x[0]["col"]))
    settings = dict(side_bins=side_bins, halo_bins=halo_bins, stride_bins=stride, score_max=score_max,
                    min_background_fraction=min_background_fraction, require_sr_background=require_sr_background,
                    min_mac_cells=min_mac_cells, min_near_t_cells=min_near_t_cells, min_panel_umis=min_panel_umis)
    region_dirs = []
    common = dict(sample=sample, run_dir=str(run), pitch_um=2, score_sources=sources, **settings)
    search = dict(requested=n_regions, **common, n_search_positions=len(pool), n_candidate_windows=len(audit),
                  n_geometry_label_pass=len(candidates), n_raw_positions_read=0,
                  selection_scope="One previous evaluation core. Candidates are enriched for both analysis groups. "
                  "This targeted comparison is descriptive and does not estimate section-wide performance.")
    if candidates:
        for rank, (rec, _, _) in enumerate(candidates,1):
            rec["rank_before_native_counts"] = rank
        pd.DataFrame(audit).to_csv(out/"candidates.csv", index=False)
        union = np.zeros(len(pool), dtype=bool)
        for _, ix, _ in candidates:
            union[ix] = True
        union_indices = np.flatnonzero(union)
        search.update(n_raw_positions_read=len(union_indices))
        _write_json(out/"selection.json",search)
        print(f"Reading the union of {len(union_indices):,} native barcodes once for "
              f"{len(candidates):,} geometry/label-qualified windows.",flush=True)
        mapping = np.full(len(pool),-1,dtype=int)
        mapping[union] = np.arange(union.sum())
        mat, genes, raw_path, n_features = _read_union(paths["square_dir"],pool.barcode.to_numpy()[union])
        total = np.asarray(mat.sum(axis=0)).ravel()
        gene_sums = {}
        for name in ("myeloid","panT","Cd4","CD8"):
            rows = genes.gene.isin(PANELS[name]).to_numpy()
            gene_sums[name] = np.asarray(mat[rows].sum(axis=0)).ravel() if rows.any() else None
        ready = []
        low_unassigned = (pool.he_tissue_score.to_numpy() <= score_max) & (owner < 0)
        off_sr = pool.sr_in_tissue.to_numpy() == 0
        for rec, ix, complete in candidates:
            ux = mapping[ix]
            bg = low_unassigned[ix]
            if rec["background_mode"] == STRICT:
                bg &= off_sr[ix]
            rec["raw_background_umis"] = int(total[ux[bg]].sum())
            mac_ix = np.isin(owner[ix],complete[mac[complete]])
            near_ix = np.isin(owner[ix],complete[near[complete]])
            t_ix = np.isin(owner[ix],complete[t_label[complete]])
            for name, panel, mask in (("Cd4_in_T_labels","Cd4",t_ix), ("CD8_in_T_labels","CD8",t_ix),
                                     ("myeloid_in_mac","myeloid",mac_ix),
                                     ("myeloid_in_near_t","myeloid",near_ix),
                                     ("panT_in_near_t","panT",near_ix)):
                vec = gene_sums[panel]
                rec[name] = int(vec[ux[mask]].sum()) if vec is not None else None
            if rec["raw_background_umis"] == 0:
                rec["decision"] = "no_observed_background_umi"
            elif any((rec[name] or 0) < min_panel_umis for name in
                     ("myeloid_in_mac","myeloid_in_near_t","panT_in_near_t")):
                rec["decision"] = "insufficient_raw_marker_denominator"
            else:
                rec["decision"] = "eligible"
                ready.append(rec)
        pick = _pick(ready,side_bins,n_regions)
        search.update(n_native_gate_pass=len(ready), raw_count_source=raw_path)
        by_origin = {(rec["row"],rec["col"]):(ix,complete) for rec,ix,complete in candidates}
        for i, rec in enumerate(pick,1):
            key = (rec["row"],rec["col"])
            ix, complete = by_origin[key]
            folder = out/f"region_{i:02d}"
            rec.update(selected=True, decision="selected", region=folder.name)
            receipt = dict(**common, region=folder.name, origin_rc=list(key),
                           background_mode=rec["background_mode"], raw_source=raw_path,
                           n_gene_ids=n_features, n_symbols=len(genes),
                           primary_readout_preflight=rec.copy())
            _write_input(folder,pool.iloc[ix].copy(),mat[:,mapping[ix]],genes,meta,complete,
                         receipt,halo_bins,side_bins,score_max)
            region_dirs.append(folder)
    search["n_selected"] = len(region_dirs)
    search["selected_regions"] = [p.name for p in region_dirs]
    search["decision_counts"] = pd.Series([r["decision"] for r in audit],dtype=str).value_counts().to_dict()
    search.setdefault("n_native_gate_pass",0)
    if len(region_dirs) < n_regions:
        search["shortfall"] = "Too few disjoint qualifying native windows on the searched lattice. Inspect candidates.csv " \
            "for the limiting geometry, group sizes, background fraction or observed UMI denominators. " \
            "Record any revised settings in a separate run."
    _write_json(out/"selection.json",search)
    pd.DataFrame(audit, columns=None if audit else ["row","col","decision","selected"]).to_csv(
        out/"candidates.csv", index=False)
    return region_dirs
