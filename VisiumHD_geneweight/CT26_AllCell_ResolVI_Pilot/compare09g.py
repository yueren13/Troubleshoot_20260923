"""Evaluate two regional fits against the same native counts and complete Proseg footprints."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from compare09d import PANELS, TARGETS, _groups
from readout09f import REVIEW, _flag, _group_masks, _input, _unassigned


METHODS = ("sparkle", "spotclean")
PANEL_NAMES = ("myeloid", "panT", "Cd4", "CD8")
RETAIN_COLS = ["region", "group", "panel", "genes", "cells", "raw_UMIs", "raw_positive",
               "sparkle_counts", "sparkle_retention", "sparkle_positive",
               "spotclean_counts", "spotclean_retention", "spotclean_positive", "both_available"]
POOL_COLS = ["group", "panel", "genes", "paired_regions", "region_names", "cells", "raw_UMIs",
             "sparkle_counts", "sparkle_retention", "spotclean_counts", "spotclean_retention"]
COVER_COLS = ["region", "gene", "assayed", "raw_assigned_all", "raw_unassigned_all", "raw_complete",
              "sparkle_status", "sparkle_complete_counts", "spotclean_status", "spotclean_complete_counts", "bin_scope"]
STATUS_COLS = ["region","sample","native_positions","provisional_background","complete_cells",
               "macrophage_cells","near_panT_cells","input_sha256","sparkle","spotclean","pool_eligible",
               "analysis_mode","evaluation_positions","owned_prior_cells","disk_positions_reference",
               "prior_cells_reference"]


def _load_model(folder, method, digest, hashes, genes, raw_cells, raw_counts):
    """Reorder both axes by ID, checking the complete raw exchange, including halo cells."""
    receipt = json.loads((folder / "fit.json").read_text())
    if receipt["state"] != "complete" or str(receipt["method"]).lower() != method:
        raise ValueError("The model receipt has the wrong method or completion state.")
    if receipt["input_sha256"] != digest or receipt["input_files_sha256"] != hashes:
        raise ValueError("The fit input fingerprint differs from this native window.")
    with np.load(folder / "cell_counts.npz", allow_pickle=False) as result:
        fit_genes = pd.Index(result["genes"].astype(str))
        fit_cells = pd.Index(result["cell_ids"].astype(str))
        if not fit_genes.is_unique or not fit_cells.is_unique:
            raise ValueError("The model has duplicate gene or cell IDs.")
        if set(fit_genes) != set(genes) or set(fit_cells) != set(raw_cells):
            raise ValueError("The model gene or cell IDs differ from the native input.")
        corrected = result["corrected"]
        original = result["raw"]
        expected_shape = (len(genes), len(raw_cells))
        if corrected.shape != expected_shape or original.shape != expected_shape:
            raise ValueError("The model count arrays have invalid dimensions.")
        order = np.ix_(fit_genes.get_indexer(genes), fit_cells.get_indexer(raw_cells))
        if not np.array_equal(original[order], raw_counts):
            raise ValueError("Raw gene-by-cell counts do not reproduce native square aggregation.")
        corrected = corrected[order].astype(float)
    if np.isinf(corrected).any() or (corrected < 0).any():
        raise ValueError("The corrected matrix contains an invalid count.")
    states = pd.read_csv(folder / "gene_fits.csv", dtype={"gene":str}, keep_default_na=False)
    states = states.set_index("gene", verify_integrity=True)
    if not genes.isin(states.index).all() or states.loc[genes,"status"].eq("").any():
        raise ValueError("The fit is missing a target gene's processing status.")
    return pd.DataFrame(corrected.T, index=raw_cells, columns=genes), states.status, receipt


def _panel(frame, genes):
    # A missing fit for a measured member keeps the panel explicitly unassessed.
    return frame[genes].sum(axis=1, min_count=len(genes)) if genes else pd.Series(np.nan,index=frame.index)


def _summarize_cells(name, cells, members, available):
    masks = _group_masks(cells)
    rows = []
    for group, panel, _ in REVIEW:
        part = cells.loc[masks[group]]
        raw = part["raw_" + panel]
        raw_total = raw.sum(min_count=1)
        row = dict(region=name, group=group, panel=panel, genes=",".join(members[panel]), cells=len(part),
                   raw_UMIs=raw_total, raw_positive=int(raw.gt(0).sum()) if raw.notna().any() else np.nan)
        both = bool(len(part) and raw.notna().all() and all(available.values()))
        for m in METHODS:
            v = part[m + "_" + panel]
            known = available[m] and len(part) > 0 and v.notna().all() and raw.notna().all()
            amt = float(v.sum()) if known else np.nan
            row[m + "_counts"] = amt
            row[m + "_retention"] = amt/raw_total if raw_total > 0 else np.nan
            row[m + "_positive"] = int(v.gt(0).sum()) if known else np.nan
            both = both and known
        row["both_available"] = bool(both)
        rows.append(row)
    return rows


def _pool(regional, eligible):
    rows = []
    source = regional.loc[regional.region.isin(eligible) & regional.both_available.eq(True)]
    # Separate strata if input feature panels differ across native datasets.
    for (group, panel, genes), part in source.groupby(["group","panel","genes"],sort=False):
        raw = part.raw_UMIs.sum()
        item = dict(group=group, panel=panel, genes=genes, paired_regions=len(part),
                    region_names=",".join(part.region), cells=int(part.cells.sum()), raw_UMIs=raw)
        for m in METHODS:
            count = float(part[m + "_counts"].sum())
            item[m + "_counts"] = count
            item[m + "_retention"] = count/raw if raw > 0 else np.nan
        rows.append(item)
    return pd.DataFrame(rows,columns=POOL_COLS)


def _plot(regional, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = list(regional.region.unique())
    fig, axes = plt.subplots(1, max(1,len(names)), figsize=(4.6*max(1,len(names))+1.4,4.2),
                             squeeze=False, layout="constrained")
    if not names:
        axes[0,0].text(.5,.5,"No scored regions met the selection criteria",ha="center",va="center",
                       transform=axes[0,0].transAxes)
        axes[0,0].set_axis_off()
    colors = {"sparkle":"#197066","spotclean":"#b45b36"}
    labels = ["Myeloid in nearby T-supported cells","Myeloid in labeled macrophages",
              "Pan-T in T-supported cells","Cd4 in T labels","CD8 in T labels"]
    values = regional[["sparkle_retention", "spotclean_retention"]].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    xmax = max(115., float(finite.max()) * 105.) if finite.size else 115.
    for i, name in enumerate(names):
        ax = axes[0,i]
        sub = regional.loc[regional.region.eq(name)].set_index(["group","panel"])
        for j,(group, panel, _) in enumerate(REVIEW):
            row = sub.loc[group,panel]
            for m, dy in zip(METHODS,(-.10,.10)):
                ratio = row[m + "_retention"]
                if pd.notna(ratio):
                    ax.scatter(100*ratio,j+dy,label=m,s=42,color=colors[m])
        if not sub[["sparkle_retention","spotclean_retention"]].notna().any().any():
            ax.text(.5,.08,"No measured retention ratios", transform=ax.transAxes, ha="center", fontsize=9)
        ax.axvline(100,ls="--",color="#727272",lw=1)
        ax.set_yticks(range(len(labels)), labels if i==0 else [""]*len(labels), fontsize=9)
        ax.set_ylim(len(labels)-.5, -.5)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x",alpha=.18)
        ax.set(title=name,xlabel="Corrected / observed marker counts (%)")
    if names:
        fig.legend(handles=[Line2D([],[],marker="o",ls="",color=v,label=m) for m,v in
                            (("SPARKLE",colors["sparkle"]),("SpotClean",colors["spotclean"]))],
                   loc="outside upper center",ncols=2,frameon=False)
    fig.savefig(out/"paired_retention.png",dpi=155,bbox_inches="tight")
    plt.close(fig)


def _owned_cells(meta, pos, receipt, name):
    """Verify complete 09B footprints independently of central-bin ownership."""
    required = {"owner_tile", "prior_09b_core", "complete_core",
                "n_footprint_bins_reference", "n_footprint_bins_tile"}
    if not required.issubset(meta):
        raise ValueError(f"The fixed-core input lacks ownership metadata: {sorted(required - set(meta))}")
    assigned = pos.loc[pos.cell_id.ne("")]
    sizes = assigned.groupby("cell_id").size().reindex(meta.index, fill_value=0)
    tile_sizes = pd.to_numeric(meta.n_footprint_bins_tile, errors="raise")
    original_sizes = pd.to_numeric(meta.n_footprint_bins_reference, errors="raise")
    if not sizes.eq(tile_sizes).all():
        raise ValueError("A tile's footprint counts disagree with its native cell assignments.")
    core = meta.index[_flag(meta.complete_core)]
    represented = (meta.owner_tile.eq(name) & _flag(meta.prior_09b_core) &
                   sizes.gt(0) & sizes.eq(original_sizes) &
                   assigned.groupby("cell_id").in_tissue.all().reindex(meta.index, fill_value=False))
    if not represented.loc[core].all() or len(core) != receipt["n_complete_cells"]:
        raise ValueError("An original 09B cell lacks unique ownership or a complete measured tissue footprint.")
    if int(pos.in_core.sum()) != receipt["n_owned_disk_bins"]:
        raise ValueError("Native disk-bin ownership disagrees with the tile receipt.")
    return core


def _disk_tables(out, status, regional, scores):
    """Report paired and individual fit coverage against the fixed biological target."""
    if (status["disk_positions_reference"].nunique() != 1 or status["prior_cells_reference"].nunique() != 1
            or status["sample"].nunique() != 1):
        raise ValueError("Fixed tiles have inconsistent sample or reference counts.")
    scope = {"reviewed": pd.Series(True, index=status.index),
             "SPARKLE": status.sparkle.eq("verified"),
             "SpotClean": status.spotclean.eq("verified"),
             "paired": status.sparkle.eq("verified") & status.spotclean.eq("verified")}
    disk = int(status.disk_positions_reference.iloc[0])
    cells = int(status.prior_cells_reference.iloc[0])
    coverage = []
    marker = []
    for label, active in scope.items():
        contributing = status.loc[active]
        n_bins = int(contributing.evaluation_positions.sum())
        n_prior = int(contributing.owned_prior_cells.sum())
        n_complete = int(contributing.complete_cells.sum())
        if n_bins > disk or n_prior > cells or n_complete > n_prior:
            raise ValueError("Summed tile coverage exceeds the fixed disk or its original cell count.")
        coverage.append(dict(scope=label, tiles=len(contributing),
                             reference_disk_bins=disk, reviewed_owned_disk_bins=n_bins,
                             reference_prior_cells=cells, prior_cells_with_owner_in_tiles=n_prior,
                             whole_footprints=n_complete,
                             whole_footprints_fraction_of_prior=n_complete/cells if cells else np.nan))
        part = regional.loc[regional.region.isin(contributing.region)]
        for group, panel, genes in regional.groupby(["group","panel","genes"], sort=False).groups:
            use = part.loc[part.group.eq(group) & part.panel.eq(panel) & part.genes.eq(genes)]
            if label == "paired":
                use = use.loc[use.both_available]
            elif label in ("SPARKLE", "SpotClean"):
                use = use.loc[use[label.lower()+"_counts"].notna()]
            use = use.loc[use.raw_UMIs.notna()]
            raw = use.raw_UMIs.sum(min_count=1)
            item = dict(scope=label, group=group, panel=panel, genes=genes, tiles_with_measured_cells=len(use),
                        cells=int(use.cells.sum()), raw_UMIs=raw)
            for m in METHODS:
                report = label == "paired" or label.lower() == m
                count = use[m+"_counts"].sum(min_count=1) if report else np.nan
                item[m+"_counts"] = count
                item[m+"_retention"] = count/raw if raw > 0 else np.nan
            marker.append(item)
    fraction = status.provisional_background / status.native_positions
    sensitivity = []
    for label, flag in (("below_25_percent", fraction.lt(.25)),
                        ("at_least_25_percent", fraction.ge(.25))):
        selected = status.loc[flag]
        results = _pool(regional, selected.region)
        for group, panel, _ in REVIEW:
            template = regional.loc[regional.group.eq(group) & regional.panel.eq(panel)].iloc[0]
            match = results.loc[results.group.eq(group) & results.panel.eq(panel)]
            row = dict(background_fraction_band=label, prepared_tiles=len(selected), group=group, panel=panel,
                       genes=template.genes, paired_regions=0, cells=0, raw_UMIs=np.nan,
                       sparkle_counts=np.nan, spotclean_counts=np.nan,
                       sparkle_retention=np.nan, spotclean_retention=np.nan)
            if len(match):
                row.update(match.iloc[0].drop(labels="region_names").to_dict())
            sensitivity.append(row)
    pd.DataFrame(sensitivity).to_csv(out/"background_sensitivity.csv", index=False)
    frame = pd.DataFrame(coverage)
    frame.to_csv(out/"domain_coverage.csv", index=False)
    pd.DataFrame(marker).to_csv(out/"method_coverage.csv", index=False)
    by_score = scores.groupby(["gene", "assayed", "assignment", "he_score"], sort=False)
    totals = by_score[["n_squares","raw_total_UMIs","raw_marker_UMIs"]].sum(min_count=1).reset_index()
    totals["raw_marker_per_1000_UMIs"] = (1000 * totals.raw_marker_UMIs /
                                         totals.raw_total_UMIs.where(totals.raw_total_UMIs.gt(0)))
    totals.to_csv(out/"coreceptors_disk_by_score.csv", index=False)
    return frame


def compare_regions(regions, output_dir):
    """Produce region-specific and pooled estimates, admitting each method only upon verification.

    regions is a sequence of dicts with name, input, sparkle and spotclean paths.
    Either model path may be None. A failed fit never re-enters a paired denominator.
    """
    from models09g import input_digest
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = out / "local"
    local.mkdir(exist_ok=True)
    summaries, coverage, outside, statuses, edge_rows = [], [], [], [], []
    seen_spots, seen_cells, names = set(), set(), set()
    fixed = None
    reference = None
    for region in regions:
        name = str(region["name"])
        if name in names or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Region names must be unique directory basenames.")
        names.add(name)
        src = Path(region["input"])
        receipt = json.loads((src/"roi_receipt.json").read_text())
        current_fixed = receipt.get("analysis_mode") == "09B_fixed_core"
        if fixed is not None and current_fixed != fixed:
            raise ValueError("A pooled readout must use one evaluation design.")
        fixed = current_fixed
        native_genes, pos, meta, native = _input(src)
        digest, file_hashes = input_digest(src)
        targets = native_genes[native_genes.isin(TARGETS)]
        keep = pos.cell_id.ne("")
        cell_ids = pd.Index(pos.loc[keep,"cell_id"].unique(),name="cell_id")
        weights = _groups(cell_ids.get_indexer(pos.loc[keep,"cell_id"]),len(cell_ids))
        target_counts = native[native_genes.get_indexer(targets)]
        raw_counts = (target_counts[:,keep.to_numpy()] @ weights).toarray()
        raw = pd.DataFrame(raw_counts.T,index=cell_ids,columns=targets)
        if fixed:
            signature = (receipt["sample"], receipt.get("run_dir"), tuple(targets))
            if reference is not None and signature != reference:
                raise ValueError("The original 09B run, sample or target gene axis differs across fixed tiles.")
            reference = signature
            core = _owned_cells(meta, pos, receipt, name)
        else:
            support = pos.loc[keep].groupby("cell_id")[["in_tissue","in_core"]].all().all(axis=1)
            core = meta.index[_flag(meta.complete_core) & support.reindex(meta.index,fill_value=False)]
        cells = meta.loc[core].copy()
        observed = raw.loc[core]
        fitted, states, available, receipts = {}, {}, {}, {}
        region_stat = dict(region=name, sample=receipt["sample"], native_positions=len(pos),
                           provisional_background=int((~pos.in_tissue).sum()), complete_cells=len(cells),
                           input_sha256=digest, evaluation_positions=int(pos.in_core.sum()),
                           analysis_mode="09B_fixed_core" if fixed else "qualified_regions",
                           owned_prior_cells=receipt.get("n_owned_target_cells"),
                           disk_positions_reference=receipt.get("n_disk_bins_reference"),
                           prior_cells_reference=receipt.get("n_prior_cells_reference"))
        for m in METHODS:
            available[m] = False
            fitted[m] = pd.DataFrame(np.nan, index=core, columns=targets)
            states[m] = pd.Series("fit_unavailable",index=targets)
            if region.get(m) is None:
                region_stat[m] = "unavailable"
                continue
            try:
                corrected, gene_state, fit_meta = _load_model(Path(region[m]),m,digest,file_hashes,targets,cell_ids,raw_counts)
            except (KeyError, IndexError, OSError, ValueError) as exc:
                region_stat[m] = "failed validation: " + str(exc)
                continue
            fitted[m] = corrected.loc[core]
            states[m] = gene_state
            receipts[m] = fit_meta
            available[m] = True
            region_stat[m] = "verified"

        scope = pos.in_core.to_numpy(bool) if fixed else np.ones(len(pos), dtype=bool)
        all_totals = np.asarray(target_counts[:, scope].sum(axis=1)).ravel()
        assigned_totals = np.asarray(target_counts[:, scope & keep.to_numpy()].sum(axis=1)).ravel()
        for gene in sorted(TARGETS):
            j = targets.get_indexer([gene])[0]
            item = dict(region=name,gene=gene,assayed=j>=0,bin_scope="owned_disk" if fixed else "fit_window",
                        raw_assigned_all=assigned_totals[j] if j>=0 else np.nan,
                        raw_unassigned_all=all_totals[j]-assigned_totals[j] if j>=0 else np.nan,
                        raw_complete=observed[gene].sum(min_count=1) if j>=0 else np.nan)
            for m in METHODS:
                item[m+"_status"] = states[m].get(gene,"unassayed")
                item[m+"_complete_counts"] = fitted[m][gene].sum(min_count=len(core) or 1) if j>=0 else np.nan
            coverage.append(item)
        members = {k:[g for g in PANELS[k] if g in targets] for k in PANEL_NAMES}
        for key, frame in {"raw":observed, **fitted}.items():
            for panel, measured in members.items():
                cells[key+"_"+panel] = _panel(frame,measured)
            for gene in ("Cd8a","Cd8b1"):
                cells[key+"_"+gene] = frame[gene] if gene in frame.columns else np.nan
        groups = _group_masks(cells)
        region_stat.update(macrophage_cells=int(groups["macrophage"].sum()),
                           near_panT_cells=int(groups["near_panT"].sum()))
        summaries.extend(_summarize_cells(name,cells,members,available))
        # Native arrays may be rotated relative to H&E. On their own grid each
        # squared footprint lies half a capture pitch beyond its outermost center.
        radius = receipts.get("sparkle",{}).get("radius_um",np.nan)
        if np.isfinite(radius) and radius > 0 and {"array_row","array_col"}.issubset(pos):
            rc = 2.0 * pos[["array_row","array_col"]].astype(float)
            centroids = rc.loc[keep].groupby(pos.loc[keep, "cell_id"]).mean().reindex(cells.index)
            distances = np.minimum(centroids - (rc.min()-1.0), (rc.max()+1.0) - centroids).min(axis=1)
            cells["centroid_edge_distance_um"] = distances
            subsets = {"within_radius_of_border":distances.lt(radius), "full_radius_inside_window":distances.ge(radius)}
        else:
            cells["centroid_edge_distance_um"] = np.nan
            subsets = {"unknown_radius_or_geometry":pd.Series(True,index=cells.index)}
        for label, mask in subsets.items():
            for row in _summarize_cells(name,cells.loc[mask],members,available):
                edge_rows.append(dict(edge_context=label, donor_radius_um=radius, **row))
        cells.to_csv(local/(name+"_cells.csv.gz"))
        # Overlapping physical support would double count regions in a pooled estimate.
        sample = str(receipt["sample"])
        bin_ids = pos.index[pos.in_core] if fixed else pos.index
        spot_keys = {(sample, b) for b in bin_ids}
        cell_keys = {(sample, c) for c in core}
        eligible = not (spot_keys & seen_spots or cell_keys & seen_cells)
        region_stat["pool_eligible"] = eligible
        if fixed and not eligible:
            raise ValueError("Repeated disk-bin or complete-cell owner across fixed 09B tiles.")
        if eligible and (fixed or all(available.values())):
            seen_spots.update(spot_keys)
            seen_cells.update(cell_keys)
        statuses.append(region_stat)
        panel_out = _unassigned(pos,native,native_genes)
        panel_out.insert(0,"region",name)
        outside.append(panel_out)

    regional = pd.DataFrame(summaries,columns=RETAIN_COLS)
    status = pd.DataFrame(statuses,columns=STATUS_COLS)
    pooled = _pool(regional,status.loc[status.pool_eligible.eq(True),"region"])
    regional.to_csv(out/"regional_retention.csv",index=False)
    pooled.to_csv(out/"pooled_retention.csv",index=False)
    pd.DataFrame(edge_rows,columns=["edge_context","donor_radius_um",*RETAIN_COLS]).to_csv(
        out/"edge_sensitivity.csv",index=False)
    status.to_csv(out/"region_status.csv",index=False)
    pd.DataFrame(coverage,columns=COVER_COLS).to_csv(out/"marker_gene_coverage.csv",index=False)
    if outside:
        scores = pd.concat(outside,ignore_index=True)
        scores.to_csv(out/"coreceptors_by_score.csv",index=False)
    if fixed:
        area = _disk_tables(out, status, regional, scores).set_index("scope")
        # One fixed biological region may have hundreds of local fits. Show the
        # count-weighted paired estimates in one panel, with all tile rows in CSV.
        endpoints = pd.DataFrame([dict(group=g,panel=p) for g,p,_ in REVIEW])
        plot = endpoints.merge(pooled, on=["group","panel"], how="left", validate="one_to_one")
        plot["region"] = "09B core · paired tiles"
        _plot(plot, out)
    else:
        _plot(regional,out)
    lines = ["# Side-by-side correction on complete Proseg footprints","","Percentages divide each method’s "
             "summed corrected marker counts by observed counts on the same cells, genes and native squares. "
             "SpotClean can return values above 100%. A dash indicates unavailable data or zero observed counts.","",
             "| Region | Group and panel | Cells | Observed UMIs | SPARKLE | SpotClean |",
             "|---|---|---:|---:|---:|---:|"]
    lookup = {(g,p):label for g,p,label in REVIEW}
    def fmt(v, fraction=False):
        if pd.isna(v):
            return "—"
        return f"{100*v:.1f}%" if fraction else f"{v:,.0f}"
    if fixed:
        reviewed, paired = area.loc["reviewed"], area.loc["paired"]
        lines = ["# Original 09B evaluation core", "",
                 f"Readout covers {int(reviewed.reviewed_owned_disk_bins):,} of "
                 f"{int(reviewed.reference_disk_bins):,} native disk positions and "
                 f"{int(reviewed.whole_footprints):,} complete footprints from "
                 f"{int(reviewed.reference_prior_cells):,} original evaluation cells. "
                 f"Both methods returned verified outputs on {int(paired.tiles)} tiles containing "
                 f"{int(paired.whole_footprints):,} complete footprints. "
                 "An individual marker row can have a smaller denominator.", "",
                 "Native RNA totals use uniquely owned squares inside the disk. Cell totals use each "
                 "owned complete footprint, including bins extending across tile cores or the disk boundary. "
                 "The method coverage table gives raw and corrected counts for each method's available "
                 "tiles and for the shared subset. The background sensitivity table compares results "
                 "on tiles below and at or above 25% provisional SpotClean background, using identical "
                 "paired footprints for both methods within each stratum.", ""]
    for row in (() if fixed else regional.itertuples()):
        lbl = lookup[row.group,row.panel]
        lines.append(f"| {row.region} | {lbl} | {row.cells} | {fmt(row.raw_UMIs)} | "
                     f"{fmt(row.sparkle_retention,True)} | {fmt(row.spotclean_retention,True)} |")
    if not regional.empty:
        lines += ["","## Across regions","",
                  "The pooled table sums observed and corrected counts across nonoverlapping regions with both "
                  "fits available for the same gene panel. It reports its contributing regions and cell count. "
                  "A region with more observed UMIs contributes more to the pooled fraction. "
                  "These are selected regions within a specimen, so variation across them is descriptive.",""]
        if fixed:
            lines[-2] = ("Each pooled estimate uses the same measured cell footprints, marker genes and raw "
                         "denominators for both methods. Halos supply modeling context and may overlap. "
                         "Disjoint owners prevent repeated cell or disk-bin observations. Per-tile "
                         "results and fit availability are tabulated separately.")
        for r in pooled.itertuples():
            lines.append(f"- {lookup[r.group,r.panel]} across {r.paired_regions} region(s), {r.cells} cells "
                         f"and {fmt(r.raw_UMIs)} observed UMIs. SPARKLE retained {fmt(r.sparkle_retention,True)} "
                         f"and SpotClean retained {fmt(r.spotclean_retention,True)}.")
    else:
        lines += ["","No regions reached the readout stage. Inspect the selection and stage receipts."]
    lines += ["","## Interpretation","",
              "Read the two myeloid endpoints together. Loss of myeloid counts in nearby pan-T-supported "
              "cells is more persuasive when macrophage markers are retained in the same fit. These observations "
              "alone cannot assign the true source of an individual transcript.",
              "", "The edge sensitivity table repeats the group counts and retention estimates for centroids within one "
              "SPARKLE donor radius of the crop border and for centroids whose full radius lies inside the window. "
              "Inspect the group sizes when comparing these two subsets (edge_sensitivity.csv).",
              "", "Marker gene coverage records each gene’s fit status and observed counts in assigned and unassigned "
              "squares. An unchanged SPARKLE panel can reflect genes that failed the fit threshold. The coreceptor "
              "score table measures observed Cd4, Cd8a and Cd8b1 by mask assignment and H&E score in the inner core."]
    issues = [f"- {r.region} · {m} {getattr(r,m)}" for r in status.itertuples()
              for m in METHODS if getattr(r,m) != "verified"]
    if issues and fixed:
        lines += ["", f"{len(issues)} tile/method combinations lack a verified fit. "
                  "See region_status.csv for each tile and domain_coverage.csv for coverage."]
    elif issues:
        lines.extend(["","## Model availability","",*issues])
    (out/"paired_readout.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return dict(n_regions=len(status), n_verified_pairs=int((status.sparkle.eq("verified") &
                status.spotclean.eq("verified")).sum()), pooled_regions=sorted(set(",".join(pooled.region_names).split(","))-{""}))
