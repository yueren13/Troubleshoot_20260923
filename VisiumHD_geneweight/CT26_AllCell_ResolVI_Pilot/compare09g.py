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
              "sparkle_status", "sparkle_complete_counts", "spotclean_status", "spotclean_complete_counts"]
STATUS_COLS = ["region","sample","native_positions","provisional_background","complete_cells",
               "macrophage_cells","near_panT_cells","input_sha256","sparkle","spotclean","pool_eligible"]


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
    for region in regions:
        name = str(region["name"])
        if name in names or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Region names must be unique directory basenames.")
        names.add(name)
        src = Path(region["input"])
        receipt = json.loads((src/"roi_receipt.json").read_text())
        native_genes, pos, meta, native = _input(src)
        digest, file_hashes = input_digest(src)
        targets = native_genes[native_genes.isin(TARGETS)]
        keep = pos.cell_id.ne("")
        cell_ids = pd.Index(pos.loc[keep,"cell_id"].unique(),name="cell_id")
        weights = _groups(cell_ids.get_indexer(pos.loc[keep,"cell_id"]),len(cell_ids))
        target_counts = native[native_genes.get_indexer(targets)]
        raw_counts = (target_counts[:,keep.to_numpy()] @ weights).toarray()
        raw = pd.DataFrame(raw_counts.T,index=cell_ids,columns=targets)
        support = pos.loc[keep].groupby("cell_id")[["in_tissue","in_core"]].all().all(axis=1)
        core = meta.index[_flag(meta.complete_core) & support.reindex(meta.index,fill_value=False)]
        cells = meta.loc[core].copy()
        observed = raw.loc[core]
        fitted, states, available, receipts = {}, {}, {}, {}
        region_stat = dict(region=name, sample=receipt["sample"], native_positions=len(pos),
                           provisional_background=int((~pos.in_tissue).sum()), complete_cells=len(cells),
                           input_sha256=digest)
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

        all_totals = np.asarray(target_counts.sum(axis=1)).ravel()
        assigned_totals = raw_counts.sum(axis=1)
        for gene in sorted(TARGETS):
            j = targets.get_indexer([gene])[0]
            item = dict(region=name,gene=gene,assayed=j>=0,
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
        spot_keys = {(sample, b) for b in pos.index}
        cell_keys = {(sample, c) for c in core}
        eligible = not (spot_keys & seen_spots or cell_keys & seen_cells)
        region_stat["pool_eligible"] = eligible
        if eligible and all(available.values()):
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
        pd.concat(outside,ignore_index=True).to_csv(out/"coreceptors_by_score.csv",index=False)
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
    for row in regional.itertuples():
        lbl = lookup[row.group,row.panel]
        lines.append(f"| {row.region} | {lbl} | {row.cells} | {fmt(row.raw_UMIs)} | "
                     f"{fmt(row.sparkle_retention,True)} | {fmt(row.spotclean_retention,True)} |")
    if not regional.empty:
        lines += ["","## Across regions","",
                  "The pooled table sums observed and corrected counts across nonoverlapping regions with both "
                  "fits available for the same gene panel. It reports its contributing regions and cell count. "
                  "A region with more observed UMIs contributes more to the pooled fraction. "
                  "These are selected regions within a specimen, so variation across them is descriptive.",""]
        for r in pooled.itertuples():
            lines.append(f"- {lookup[r.group,r.panel]} across {r.paired_regions} region(s), {r.cells} cells "
                         f"and {fmt(r.raw_UMIs)} observed UMIs. SPARKLE retained {fmt(r.sparkle_retention,True)} "
                         f"and SpotClean retained {fmt(r.spotclean_retention,True)}.")
    else:
        lines += ["","No regions reached the readout stage. Inspect the selection and stage receipts."]
    lines += ["","## Interpretation","",
              "Read the first two rows of each region together. Loss of myeloid counts in nearby pan-T-supported "
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
    if issues:
        lines.extend(["","## Model availability","",*issues])
    (out/"paired_readout.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return dict(n_regions=len(status), n_verified_pairs=int((status.sparkle.eq("verified") &
                status.spotclean.eq("verified")).sum()), pooled_regions=sorted(set(",".join(pooled.region_names).split(","))-{""}))
