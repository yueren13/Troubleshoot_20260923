"""Build a bounded SpotClean pilot from native barcodes, stored H&E scores and Proseg polygons."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.io import mmwrite

from bins09d import STUDY_ROOT, TEMP_PROJECT_ROOT, _local_assignment, _raw_cell_order, _rows, locate_paths
from raw09d import read_square_counts
from compare09d import TARGETS, _original


def _positions(paths, geometry, side, origin):
    mpp = float(json.loads(paths["scalefactors"].read_text())["microns_per_pixel"])
    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError("Invalid Space Ranger microns_per_pixel.")
    center = np.asarray(geometry["center_um"], dtype=float)
    radius = float(geometry["inner_radius_um"]) + 2 * side
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid 09B physical search geometry.")
    out = []
    cols = ["barcode","array_row","array_col","in_tissue","pxl_row_in_fullres","pxl_col_in_fullres"]
    for part in _rows(paths["positions"], cols):
        xy = part[["pxl_row_in_fullres","pxl_col_in_fullres"]].to_numpy(float) * mpp
        if origin is None:
            keep = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1]) <= radius
        else:
            keep = (part.array_row.between(origin[0], origin[0] + side - 1) &
                    part.array_col.between(origin[1], origin[1] + side - 1)).to_numpy()
        if np.any(keep):
            found = part.loc[keep, cols[:4]].copy()
            found["x_um"], found["y_um"] = xy[keep].T
            out.append(found)
    if not out:
        raise ValueError("No Space Ranger positions in the requested search area.")
    spots = pd.concat(out, ignore_index=True).rename(columns={"in_tissue":"sr_in_tissue"})
    ij = spots[["array_row", "array_col"]].to_numpy(float)
    if (not np.isfinite(ij).all() or (ij < 0).any() or (ij != np.floor(ij)).any()
            or spots.duplicated(["array_row","array_col"]).any()
            or not spots.barcode.is_unique or spots.barcode.isna().any()
            or not spots.sr_in_tissue.isin([0,1]).all()
            or not np.isfinite(spots[["x_um","y_um"]]).all().all()):
        raise ValueError("Invalid native barcodes, coordinates or Space Ranger tissue flags.")
    spots[["array_row","array_col"]] = ij.astype(int)
    return spots, center


def _slice_obs(node, start, stop):
    # AnnData numeric, string, categorical, and nullable encodings. Reading by
    # chunks avoids materializing the full 2-µm table or its expression matrix.
    if hasattr(node, "shape"):
        v = np.asarray(node[start:stop])
        return np.char.decode(v) if v.dtype.kind == "S" else v
    if "codes" in node and "categories" in node:
        return np.asarray(pd.Categorical.from_codes(node["codes"][start:stop],
                                                     node["categories"][:]).astype(object))
    if "values" in node and "mask" in node:
        v = np.asarray(node["values"][start:stop], dtype=object)
        v[np.asarray(node["mask"][start:stop], dtype=bool)] = None
        return v
    raise ValueError("Unsupported stored format for an AnnData observation column.")


def _zarr_chunks(path, table, fields):
    import zarr
    group = zarr.open_group(str(path), mode="r")
    table_path = f"tables/{table}/obs"
    if table_path not in group:
        raise ValueError(f"{path} lacks {table_path}. Check zarr_table or the zarr path override.")
    obs = group[table_path]
    index = obs.attrs.get("_index", "_index")
    if index not in obs:
        raise ValueError(f"{table_path} lacks its stored barcode index {index!r}.")
    idx = obs[index]
    count = idx.shape[0] if hasattr(idx, "shape") else (
        idx["codes"].shape[0] if "codes" in idx else idx["values"].shape[0])
    names = [key for key in fields if key in obs]
    if not names:
        return
    for lo in range(0, count, 131072):
        hi = min(lo + 131072, count)
        yield pd.DataFrame({"barcode":_slice_obs(idx, lo, hi),
                            **{key:_slice_obs(obs[key], lo, hi) for key in names}})


def _annotations(paths, spots, table):
    wanted = pd.Index(spots.barcode)
    annotated = pd.DataFrame(index=wanted)
    source = {}
    fields = ["he_tissue_score", "qc_class"]
    parquet = paths["aligned_qc"]
    if parquet.is_file():
        schema = pq.ParquetFile(parquet).schema_arrow.names
        selected = [key for key in fields if key in schema]
        if "qc_class" not in selected and "qc_class_raw" in schema:
            selected.append("qc_class_raw")
        xy = ["proseg_x_um","proseg_y_um"]
        selected += xy if set(xy).issubset(schema) else []
        prior = []
        for part in _rows(parquet, ["barcode", *selected]):
            hit = wanted.get_indexer(part.barcode) >= 0
            if hit.any():
                prior.append(part.loc[hit])
        if prior:
            hit = pd.concat(prior).set_index("barcode", verify_integrity=True).reindex(wanted)
            if set(xy).issubset(hit):
                both = hit[xy].notna().all(axis=1).to_numpy()
                if both.any() and not np.allclose(hit.loc[both, xy], spots.loc[both, ["x_um","y_um"]],
                                                   atol=0.05, rtol=0):
                    raise ValueError("Space Ranger coordinates disagree with the Step 00 Proseg registration.")
            for key in selected:
                if key not in xy:
                    dest = "qc_class" if key == "qc_class_raw" else key
                    annotated[dest] = hit[key]
                    source[dest] = str(parquet)
    missing = [key for key in fields if key not in annotated or annotated[key].isna().any()]
    if missing and paths["zarr"].is_dir():
        parts = []
        for part in _zarr_chunks(paths["zarr"], table, missing):
            pos = wanted.get_indexer(part.barcode)
            if (pos >= 0).any():
                parts.append(part.loc[pos >= 0])
        if parts:
            extra = pd.concat(parts).set_index("barcode", verify_integrity=True).reindex(wanted)
            for key in missing:
                if key in extra:
                    if key in annotated:
                        fill = annotated[key].isna() & extra[key].notna()
                        if fill.any():
                            annotated.loc[fill,key] = extra.loc[fill,key]
                            source[key] += f"; {paths['zarr']}/tables/{table}/obs"
                    else:
                        annotated[key] = extra[key]
                        source[key] = f"{paths['zarr']}/tables/{table}/obs"
    if "he_tissue_score" not in annotated:
        raise KeyError(f"Continuous he_tissue_score missing for this search. Checked {parquet} and "
                       f"{paths['zarr']}/tables/{table}/obs. Point overrides at the scored sample data.")
    score = pd.to_numeric(annotated.he_tissue_score, errors="raise").to_numpy(float, na_value=np.nan)
    measured = ~np.isnan(score)
    if not measured.any() or not np.isfinite(score[measured]).all() or ((score[measured] < 0) | (score[measured] > 1)).any():
        raise ValueError("Expected measured he_tissue_score values on the original [0,1] scale.")
    spots["he_tissue_score"] = score
    qc = annotated.get("qc_class", pd.Series("unassessed",index=wanted))
    spots["qc_class"] = (qc.astype("string").str.strip().str.lower().str.replace("trancript", "transcript",regex=False)
                         .fillna("unassessed").to_numpy())
    return source


def _window(spots, side, score_max, min_bg, center, require_sr_background):
    # Integral images permit testing all fully observed square windows, including
    # genuine zero-count positions, without reading a single count for selection.
    rc = spots[["array_row","array_col"]].to_numpy(int)
    lower = rc.min(axis=0)
    rr, cc = (rc - lower).T
    shape = (rr.max() + 1, cc.max() + 1)
    if min(shape) < side or np.prod(shape, dtype=np.int64) > 25_000_000:
        raise ValueError("Native lattice is too small or dispersed. Specify a local origin_rc and side_bins.")
    def totals(member):
        grid = np.zeros(shape, dtype=np.int32)
        grid[rr, cc] = np.asarray(member, dtype=np.int32)
        summed = np.pad(grid.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1,0),(1,0)))
        return summed[side:,side:] - summed[:-side,side:] - summed[side:,:-side] + summed[:-side,:-side]
    n = side**2
    full = totals(np.ones(len(spots),dtype=bool)) == n
    score = spots.he_tissue_score.to_numpy()
    sr = spots.sr_in_tissue.to_numpy(bool)
    low = (score <= score_max) & (spots.cell_index.to_numpy() < 0)
    strong = totals((score > score_max) & sr)
    # Use observed tissue and assigned cells to support the fit. A minimum of
    # one quarter measured higher-score tissue avoids a mostly empty window.
    eligible = full & (strong >= np.ceil(.25*n)) & (totals(spots.cell_index >= 0) > 0)
    off = totals(low & ~sr)
    pooled = totals(low)
    modes = [("score_low_outside_proseg_and_sr_tissue", off)]
    if not require_sr_background:
        modes.append(("score_low_outside_proseg_including_sr_tissue", pooled))
    # Array lattice is a rigid physical grid. Use a nearest observed position
    # as the deterministic anchor for choosing among qualifying windows.
    anchor = rc[np.argmin(np.sum((spots[["x_um","y_um"]].to_numpy() - center) ** 2, axis=1))]
    for mode, background in modes:
        ok = eligible & (background >= np.ceil(min_bg*n))
        ij = np.argwhere(ok) + lower
        if len(ij):
            dist2 = np.sum((ij + (side-1)/2 - anchor) ** 2, axis=1)
            winner = ij[np.argmin(dist2)]
            chosen = np.all((rc >= winner) & (rc < winner + side), axis=1)
            role = (low & ~sr) if mode == modes[0][0] else low
            return spots.loc[chosen].copy(), role[chosen], winner.tolist(), mode
    maximum = float(max((background[eligible].max(initial=0) for _,background in modes)) / n)
    raise ValueError(f"No complete {side}×{side} window had the requested {min_bg:.1%} provisional "
                     f"background and ≥25% higher-score Space Ranger tissue. Maximum eligible background "
                     f"was {maximum:.1%} at score_max={score_max:g}. Review score_search.csv before "
                     "changing score_max, setting a known origin_rc, or reducing side_bins.")


def _cell_annotations(paths, run, spots):
    import shapely
    assigned = spots.loc[spots.cell_index >= 0, "cell_id"].unique()
    wanted = pd.Index(assigned, name="cell_id")
    core = spots.loc[spots.in_core.eq(1)]
    # Conservatively require the entire polygon to fit inside the convex hull
    # of the inner bin centers. This respects physical slide rotation.
    boundary = shapely.convex_hull(shapely.multipoints(core[["x_um","y_um"]].to_numpy()))
    complete = pd.Series(False,index=wanted)
    for part in _rows(paths["boundaries"], ["cell","geometry"], size=10000):
        keys = part.cell.astype(str)
        take = keys.isin(wanted)
        if take.any():
            geometry = shapely.from_wkb(part.loc[take, "geometry"].to_numpy())
            complete.loc[keys.loc[take]] = shapely.covers(boundary, geometry)
    prior, _ = _original(run)
    fields = ["cell_type","T_label_09B","macrophage_label_09B","panT_09B_observed","distance_to_mac_um"]
    info = prior.set_index("cell_id",verify_integrity=True)[fields].reindex(wanted)
    info["complete_core"] = complete.astype(int)
    return info.reset_index()


def prepare_roi(input_dir, *, run_dir, study_root=STUDY_ROOT, temp_project_root=TEMP_PROJECT_ROOT,
                overrides=None, side_bins=40, halo_bins=4, score_max=.1, min_background_fraction=.25,
                origin_rc=None, zarr_table="square_002um", require_sr_background=False):
    """Write SpotClean matrix + axes and return a provenance receipt.

    The regular native grid has a halo included in side_bins. Spatial resolution
    is 2 µm. Thresholds use original stored H&E scores. Missing scores never
    qualify a position for background. The RNA half of qc_class is descriptive.
    """
    if not isinstance(side_bins, int) or not 8 <= side_bins <= 50:
        raise ValueError("side_bins must be an integer from 8 to 50 (at most 2,500 positions).")
    if not isinstance(halo_bins, int) or halo_bins < 1 or 2*halo_bins >= side_bins - 1:
        raise ValueError("halo_bins must leave at least two native rows and columns in the core.")
    if not 0 <= score_max < 1 or not 0 < min_background_fraction < .75:
        raise ValueError("Require 0 ≤ score_max < 1 and 0 < min_background_fraction < 0.75.")
    if origin_rc is not None:
        if len(origin_rc) != 2 or any(not isinstance(v, (int,np.integer)) or v < 0 for v in origin_rc):
            raise ValueError("origin_rc must contain two nonnegative native grid indices.")
    out = Path(input_dir)
    out.mkdir(parents=True,exist_ok=True)
    run = Path(run_dir).resolve()
    geometry = json.loads((run/"complete.json").read_text())["geometry"]
    if float(json.loads((run/"settings.json").read_text())["settings"]["microns_per_unit"]) != 1.:
        raise ValueError("Cannot compare this 09B geometry without verified physical micron units.")
    sample = str(geometry["section"])
    paths = locate_paths(sample, study_root, temp_project_root, overrides)
    pool, center = _positions(paths, geometry, side_bins, origin_rc)
    sources = _annotations(paths, pool, zarr_table)
    cell_ids, _ = _raw_cell_order(paths["raw_cells"])
    pool["cell_index"] = _local_assignment(paths["boundaries"], pool, cell_ids)
    diagnostic = pool.assign(out_of_mask=pool.cell_index.lt(0)).groupby(["sr_in_tissue","out_of_mask"])
    diagnostic.he_tissue_score.describe(percentiles=[.01,.05,.10,.25,.5,.75,.90,.95,.99]).join(
        diagnostic.size().rename("n_positions")).to_csv(out/"score_search.csv")
    spots, background, origin, mode = _window(pool, side_bins, score_max, min_background_fraction,
                                                center, require_sr_background)
    spots["in_tissue"] = (~background).astype(int)
    spots["background_class"] = np.select([background & spots.sr_in_tissue.eq(0),
                                            background & spots.sr_in_tissue.eq(1)],
              ["low_score_outside_Proseg_and_Space_Ranger_tissue",
               "low_score_outside_Proseg_inside_Space_Ranger_tissue"], default="potential_tissue")
    rc = spots[["array_row","array_col"]].to_numpy() - origin
    spots["in_core"] = ((rc >= halo_bins) & (rc < side_bins-halo_bins)).all(axis=1).astype(int)
    spots["cell_id"] = ""
    assigned = spots.cell_index >= 0
    spots.loc[assigned,"cell_id"] = cell_ids.to_numpy()[spots.loc[assigned,"cell_index"].to_numpy()]
    spots = spots.sort_values(["array_row","array_col"]).reset_index(drop=True)
    counts, raw_features, raw_path = read_square_counts(paths["square_dir"],spots.barcode.to_numpy())
    codes, symbols = pd.factorize(raw_features.gene_symbol, sort=False)
    grouping = sparse.csr_matrix((np.ones(len(codes),dtype=np.int64),(codes,np.arange(len(codes)))),
                                 shape=(len(symbols),len(codes)))
    counts = (grouping @ counts).tocsr()
    genes = pd.DataFrame({"gene":symbols, "target":pd.Index(symbols).isin(TARGETS).astype(int)})
    spots["raw_total_umis"] = np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)
    genes.to_csv(out/"genes.tsv",sep="\t",index=False)
    spots.drop(columns=["cell_index"]).to_csv(out/"positions.tsv",sep="\t",index=False)
    cells = _cell_annotations(paths, run, spots)
    cells.to_csv(out/"cells.tsv",sep="\t",index=False)
    mmwrite(out/"counts.mtx", counts, field="integer", symmetry="general")
    receipt = dict(sample=sample,run_dir=str(run),origin_rc=origin,side_bins=side_bins,halo_bins=halo_bins,
                   pitch_um=2,score_max=score_max,min_background_fraction=min_background_fraction,
                   require_sr_background=bool(require_sr_background),background_mode=mode,
                   background_classes={str(k):int(v) for k,v in spots.background_class.value_counts().items()},
                   score_sources=sources,score_missing=int(spots.he_tissue_score.isna().sum()),
                   score_missing_fraction=float(spots.he_tissue_score.isna().mean()),
                   off_sr_missing_score_as_potential_sources=int((spots.sr_in_tissue.eq(0) &
                       spots.he_tissue_score.isna() & spots.in_tissue.eq(1)).sum()),
                   roi_score_quantiles=spots.he_tissue_score.quantile([0,.05,.25,.5,.75,.95,1]).to_dict(),
                   n_search_positions=len(pool),n_positions=len(spots),n_background=int(spots.in_tissue.eq(0).sum()),
                   n_complete_cells=int(cells.complete_core.sum()),n_labeled_cells=int(cells.cell_type.notna().sum()),
                   raw_source=raw_path,n_gene_ids=len(codes),n_symbols=len(symbols),n_zero_positions=int(spots.raw_total_umis.eq(0).sum()),
                   total_umis=int(spots.raw_total_umis.sum()),annotation_scope="completed 09B core",
                   candidate_selection="nearest qualifying complete native window to 09B center; geometry and scores only",
                   fit_support="The potential-tissue role allows native expression; it is not a tissue-presence call. This includes unscored and ambiguous positions, even outside Space Ranger tissue.")
    (out/"roi_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
