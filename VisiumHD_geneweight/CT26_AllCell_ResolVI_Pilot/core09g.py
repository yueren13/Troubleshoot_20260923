"""Partition the saved 09B evaluation disk with disjoint readouts and overlapping fit halos."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.spatial import cKDTree
import shapely

from bins09d import STUDY_ROOT, TEMP_PROJECT_ROOT, locate_paths, _local_assignment, _raw_cell_order
from compare09d import _original
from regions09g import CELL_FIELDS, POOLED, STRICT, _geometry_lookup, _lattice, _read_union
from roi09f import _positions, _annotations
from run09d import _write_json


def _centroid_positions(xy, centroids):
    """Nearest observed native position to original 09B centroids. Ties use lattice order."""
    tree = cKDTree(xy)
    dist, ix = tree.query(centroids, k=2)
    best = ix[:, 0].copy()
    for j in np.flatnonzero(np.isclose(dist[:, 0], dist[:, 1], rtol=0., atol=1e-10)):
        nearby = np.asarray(tree.query_ball_point(centroids[j], float(dist[j, 0]) + 1e-9))
        norms = np.linalg.norm(xy[nearby] - centroids[j], axis=1)
        best[j] = np.min(nearby[norms <= norms.min() + 1e-10])
    return best, dist[:, 0]


def _tile_plan(pool, center, radius, prior, side, halo):
    rc = pool[["array_row", "array_col"]].to_numpy(dtype=int)
    xy = pool[["x_um", "y_um"]].to_numpy(dtype=float)
    disk = np.linalg.norm(xy - center, axis=1) <= radius
    if not disk.any():
        raise ValueError("No measured native square centers fall in the saved 09B evaluation disk.")
    stride = side - 2 * halo
    anchor = rc[disk].min(axis=0)
    bin_keys = (rc - anchor) // stride
    centroids = prior[["09B_x_um", "09B_y_um"]].to_numpy(float)
    nearest, offsets = _centroid_positions(xy, centroids)
    cell_keys = bin_keys[nearest]
    keys = sorted(set(map(tuple, bin_keys[disk])) | set(map(tuple, cell_keys)))
    lookup = {key: i for i, key in enumerate(keys)}
    # Index all bins against a small array of integer tile ordinals.
    low = np.minimum(bin_keys.min(axis=0), np.min(keys, axis=0))
    lut = np.full(tuple(bin_keys.max(axis=0) - low + 1), -1, dtype=int)
    for key, i in lookup.items():
        lut[tuple(np.asarray(key)-low)] = i
    bin_tile = lut[tuple((bin_keys-low).T)]
    cell_tile = bin_tile[nearest]
    grid, rc_min = _lattice(pool)
    records, windows = [], {}
    target_counts = np.bincount(bin_tile[disk], minlength=len(keys))
    cells_per = np.bincount(cell_tile, minlength=len(keys))
    for i, key in enumerate(keys):
        core_start = anchor + stride * np.asarray(key)
        # Slide-edge windows are shifted into the measured grid whenever possible.
        # The readout ownership remains anchored to the unshifted disjoint partition.
        top = np.maximum(0, np.minimum(core_start - halo - rc_min, np.asarray(grid.shape) - side))
        r, c = top
        win = grid[r:r+side, c:c+side]
        ok = win.shape == (side,side) and np.all(win >= 0)
        name = f"region_{i+1:04d}"
        rec = dict(region=name, row=int((top+rc_min)[0]), col=int((top+rc_min)[1]),
                   ownership_origin_rc=core_start.tolist(), n_owned_disk_bins=int(target_counts[i]),
                   n_owned_target_cells=int(cells_per[i]),
                   decision="prepared" if ok else "incomplete_native_fit_grid")
        records.append(rec)
        if ok:
            windows[i] = win.ravel()
    return disk, bin_tile, cell_tile, offsets, records, windows


def _readiness(counts, genes, background, owner):
    """Test the SpotClean runner's numerical inputs separately from region membership."""
    tissue = ~background
    sums = np.asarray(counts[:, tissue].sum(axis=1)).ravel()
    positive = np.flatnonzero(sums > 0)
    ranked = sorted(positive, key=lambda j: (-int(sums[j]), genes.gene.iat[j]))
    plan = np.union1d(ranked[:200], np.flatnonzero((sums > 0) & genes.target.to_numpy(bool))).astype(int)
    source_positions = np.count_nonzero(np.asarray(counts[plan][:, tissue].sum(axis=0)))
    bg_umi = int(counts[plan][:, background].sum())
    n_cells = len(np.unique(owner[owner >= 0]))
    n_outside = int((owner < 0).sum())
    shared_requirements = [(genes.target.to_numpy(bool).any(), "no_assayed_target_gene"),
                           (len(plan) >= 2, "fewer_than_two_expressed_genes")]
    spotclean_requirements = [*shared_requirements,
                              (n_cells >= 1, "no_assigned_cells"),
                              (tissue.sum() >= 2, "fewer_than_two_tissue_positions"),
                              (background.sum() >= 2, "fewer_than_two_background_positions"),
                              (source_positions >= 2, "fewer_than_two_expressed_tissue_positions"),
                              (bg_umi > 0, "no_background_UMIs_on_spatial_gene_plan")]
    sparkle_requirements = [*shared_requirements,
                            (n_cells >= 2, "fewer_than_two_assigned_cells"),
                            (n_outside >= 2, "fewer_than_two_out_of_mask_positions"),
                            (counts[plan][:, owner < 0].sum() > 0, "no_out_of_mask_UMIs_on_spatial_gene_plan")]
    sp_reasons = [reason for passed, reason in sparkle_requirements if not passed]
    sc_reasons = [reason for passed, reason in spotclean_requirements if not passed]
    sparkle = dict(ready=not sp_reasons, reasons=sp_reasons, n_represented_cells=n_cells,
                   n_out_of_mask_positions=n_outside)
    spotclean = dict(ready=not sc_reasons, reasons=sc_reasons, n_spatial_genes=len(plan),
                     n_background=int(background.sum()), spatial_background_umis=bg_umi,
                     n_positive_tissue_positions=int(source_positions),
                     below_25pct_background=bool(background.mean() < .25))
    return sparkle, spotclean


def prepare_09b_core(out_dir, *, run_dir, study_root=STUDY_ROOT, temp_project_root=TEMP_PROJECT_ROOT,
                     overrides=None, side_bins=64, halo_bins=6, score_max=.1,
                     require_sr_background=False, zarr_table="square_002um"):
    """Write one native input per measurable tile and reconcile coverage to every saved core cell."""
    if not isinstance(side_bins, int) or not 8 <= side_bins <= 64:
        raise ValueError("side_bins must be an integer between 8 and 64.")
    if not isinstance(halo_bins, int) or halo_bins < 1 or side_bins - 2*halo_bins < 2:
        raise ValueError("Choose a positive halo and an ownership width of at least two bins.")
    if not 0 <= score_max < 1:
        raise ValueError("score_max must be in [0,1).")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Choose a new input directory or resume the prepared run: {out}")
    run = Path(run_dir).resolve()
    saved = json.loads((run/"complete.json").read_text())
    geom = saved["geometry"]
    if float(json.loads((run/"settings.json").read_text())["settings"]["microns_per_unit"]) != 1.:
        raise ValueError("The 09B run must use registered physical micrometres.")
    center = np.asarray(geom["center_um"], dtype=float)
    radius = float(geom["inner_radius_um"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("A finite saved 09B evaluation radius is required.")
    sample = str(geom["section"])
    paths = locate_paths(sample, study_root, temp_project_root, overrides)
    # Include at least one diagonal tile beyond the circular boundary so measured
    # corner bins are available for fits whose ownership intersects the disk.
    pool, _ = _positions(paths, geom, 2*side_bins, None)
    pool = pool.sort_values(["array_row", "array_col"]).reset_index(drop=True)
    sources = _annotations(paths, pool, zarr_table)
    all_ids, _ = _raw_cell_order(paths["raw_cells"])
    all_owner = _local_assignment(paths["boundaries"], pool, all_ids)
    represented, compact = np.unique(all_owner[all_owner >= 0], return_inverse=True)
    owner = np.full(len(pool), -1, dtype=int)
    owner[all_owner >= 0] = compact
    represented_ids = pd.Index(all_ids[represented].astype(str), name="cell_id")
    ref_area = np.bincount(compact, minlength=len(represented))
    polys = _geometry_lookup(paths["boundaries"], represented, all_ids) if len(represented) else []
    prior, _ = _original(run)
    prior = prior.set_index("cell_id", verify_integrity=True)
    disk, bin_tile, cell_tile, offsets, records, windows = _tile_plan(pool, center, radius, prior,
                                                                      side_bins, halo_bins)
    names = np.array([r["region"] for r in records], dtype=str)
    meta = prior[CELL_FIELDS].reindex(represented_ids)
    meta["prior_09b_core"] = meta.index.isin(prior.index).astype(int)
    meta["owner_tile"] = pd.Series(names[cell_tile], index=prior.index).reindex(meta.index).fillna("")
    prior_pos = prior.index.get_indexer(represented_ids)
    ledger = prior[CELL_FIELDS].copy()
    ledger["owner_tile"] = names[cell_tile]
    ledger["centroid_to_native_center_um"] = offsets
    ledger["reason"] = "no_assigned_native_square"
    ledger.loc[~ledger.index.isin(all_ids), "reason"] = "absent_from_segmentation_reference"
    for i in set(range(len(records))) - set(windows):
        ledger.loc[cell_tile == i, "reason"] = "incomplete_native_fit_grid"
    ledger["n_footprint_bins_reference"] = pd.Series(ref_area,index=represented_ids).reindex(ledger.index,fill_value=0)
    ledger["n_footprint_bins_tile"] = 0
    covered = np.zeros(len(pool), dtype=np.int16)
    for i, win in windows.items():
        owned = disk[win] & (bin_tile[win] == i)
        if int(owned.sum()) != records[i]["n_owned_disk_bins"]:
            raise ValueError("Shifted fitting window lost owned target disk positions.")
        covered[win[owned]] += 1
    if np.any(covered > 1) or np.any(covered[~disk]):
        raise ValueError("Native disk ownership overlaps or includes a square beyond the saved disk.")
    shared = dict(analysis_mode="09B_fixed_core",mode="fixed_09b_core",sample=sample,run_dir=str(run),
                  geometry=geom, n_disk_bins_reference=int(disk.sum()), n_prior_cells_reference=len(prior),
                  side_bins=side_bins, halo_bins=halo_bins, owned_stride_bins=side_bins - 2*halo_bins,
                  pitch_um=2, score_max=score_max, score_sources=sources,
                  require_sr_background=bool(require_sr_background),
                  background_mode=STRICT if require_sr_background else POOLED)
    bins_mask = np.zeros(len(pool), dtype=bool)
    for win in windows.values():
        bins_mask[win] = True
    ordered = np.flatnonzero(bins_mask)
    if len(ordered):
        print(f"Preparing {len(windows)} fit grids for the saved 09B core. Reading {len(ordered):,} distinct "
              "native count columns once.",flush=True)
        counts, genes, raw_path, n_features = _read_union(paths["square_dir"],pool.barcode.to_numpy()[ordered])
        shared.update(raw_source=raw_path,n_gene_ids=n_features,n_symbols=len(genes))
    column = np.full(len(pool),-1,dtype=int)
    column[ordered] = np.arange(len(ordered))
    ready_dirs = []
    for i, win in windows.items():
        rec = records[i]
        tile = out/rec["region"]
        tile.mkdir()
        spots = pool.iloc[win].copy()
        spots["in_core"] = (disk[win] & (bin_tile[win] == i)).astype(int)
        ow = owner[win]
        assigned = ow >= 0
        present, area = np.unique(ow[assigned],return_counts=True)
        local_meta = meta.iloc[present].copy()
        local_meta["n_footprint_bins_tile"] = area
        local_meta["n_footprint_bins_reference"] = ref_area[present]
        local_meta["complete_core"] = 0
        # The readout for a whole cell includes its full footprint in this fit,
        # including native squares outside the inner ownership partition.
        in_this = np.flatnonzero(local_meta.owner_tile.eq(rec["region"]).to_numpy())
        if len(in_this):
            idx = present[in_this]
            boundary = shapely.convex_hull(shapely.multipoints(spots[["x_um","y_um"]].to_numpy()))
            complete = (area[in_this] == ref_area[idx]) & shapely.covers(boundary, polys[idx])
            keys = represented_ids[idx]
            ledger.loc[keys, "n_footprint_bins_tile"] = area[in_this]
            ledger.loc[keys, "reason"] = np.where(complete, "complete_owned_footprint", "footprint_exceeds_fit_halo")
            local_meta.loc[keys, "complete_core"] = complete.astype(int)
        # A large displacement or a fragmented polygon can leave all its
        # observed footprint squares beyond its centroid's owner tile.
        represented_target = np.flatnonzero(prior_pos >= 0)
        lost = represented_target[(cell_tile[prior_pos[represented_target]] == i) & ~np.isin(represented_target, present)]
        ledger.loc[represented_ids[lost], "reason"] = "no_footprint_in_owner_tile"
        spots["cell_id"] = ""
        spots.loc[assigned,"cell_id"] = represented_ids.to_numpy()[ow[assigned]]
        score = spots.he_tissue_score.to_numpy(float)
        background = ~assigned & np.isfinite(score) & (score <= score_max)
        off = spots.sr_in_tissue.eq(0).to_numpy()
        if require_sr_background:
            background &= off
        spots["in_tissue"] = (~background).astype(int)
        spots["background_class"] = np.select([background & off,background & ~off],
            ["low_score_outside_Proseg_and_Space_Ranger_tissue","low_score_outside_Proseg_inside_Space_Ranger_tissue"],
            default="potential_tissue")
        sub = counts[:,column[win]]
        spots["raw_total_umis"] = np.asarray(sub.sum(axis=0)).ravel()
        sparkle, tech = _readiness(sub, genes, background, ow)
        tile_center = spots[["x_um", "y_um"]].to_numpy(float).mean(axis=0)
        rec.update(n_complete_cells=int(local_meta.complete_core.sum()), n_background=int(background.sum()),
                   center_distance_um=float(np.linalg.norm(tile_center - center)),
                   sparkle_ready=sparkle["ready"], sparkle_reasons=",".join(sparkle["reasons"]),
                   spotclean_ready=tech["ready"],spotclean_reasons=",".join(tech["reasons"]),
                   below_25pct_background=tech["below_25pct_background"])
        receipt = dict(**shared, **rec, origin_rc=[rec["row"],rec["col"]], n_positions=len(win),
                       n_zero_positions=int(spots.raw_total_umis.eq(0).sum()),
                       sparkle_readiness=sparkle, spotclean_readiness=tech, total_umis=int(spots.raw_total_umis.sum()),
                       n_finite_scores=int(np.isfinite(score).sum()))
        spots.to_csv(tile/"positions.tsv",sep="\t",index=False)
        local_meta.to_csv(tile/"cells.tsv",sep="\t",index_label="cell_id")
        genes.to_csv(tile/"genes.tsv",sep="\t",index=False)
        mmwrite(tile/"counts.mtx",sub,field="integer",symmetry="general")
        _write_json(tile/"roi_receipt.json",receipt)
        ready_dirs.append(tile)
    (out/"local").mkdir()
    ledger.to_csv(out/"local"/"cell_coverage.csv.gz")
    audit = pd.DataFrame(records)
    audit.to_csv(out/"candidates.csv",index=False)
    eligible = ledger.reason.eq("complete_owned_footprint")
    if int(eligible.sum()) != sum(r.get("n_complete_cells",0) for r in records):
        raise ValueError("Complete owned footprints do not reconcile to original 09B cell IDs.")
    coverage = dict(**shared, requested=len(records), n_selected=len(ready_dirs), n_tiles_planned=len(records),
                    n_tiles_prepared=len(ready_dirs), n_distinct_fit_bins=len(ordered),
                    n_disk_bins_prepared=int(covered.sum()), n_disk_bins_unavailable=int((disk & (covered == 0)).sum()),
                    n_complete_cells=int(eligible.sum()), prior_cell_counts_by_reason=ledger.reason.value_counts().to_dict(),
                    decision_counts=audit.decision.value_counts().to_dict(), selected_regions=[p.name for p in ready_dirs],
                    n_sparkle_ready=sum(r.get("sparkle_ready",False) for r in records),
                    n_spotclean_ready=sum(r.get("spotclean_ready",False) for r in records))
    _write_json(out/"coverage.json",coverage)
    _write_json(out/"selection.json",coverage)
    return ready_dirs
