"""Evaluate the native-square SpotClean pilot on observed Proseg footprints."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from compare09d import PANELS


REVIEW = [
    ("near_panT", "myeloid", "Pan-T supported, near macrophages · myeloid"),
    ("macrophage", "myeloid", "Macrophage / monocyte labels · myeloid"),
    ("panT_supported", "panT", "Pan-T supported · pan-T markers"),
    ("T_label", "Cd4", "Prior T labels · Cd4"),
    ("T_label", "CD8", "Prior T labels · Cd8a / Cd8b1"),
]
CD_GENES = ("Cd4", "Cd8a", "Cd8b1")
MEASURES = {name: PANELS[name] for name in ("myeloid", "panT", "Cd4", "CD8")}
MEASURES.update({gene: [gene] for gene in CD_GENES})


def _table(path, key):
    df = pd.read_csv(path, sep="\t", dtype={key: str, "cell_id": str}, keep_default_na=False)
    if not df[key].is_unique or df[key].eq("").any():
        raise ValueError(f"Missing or duplicate {key} in {path.name}")
    return df.set_index(key)


def _counts(path, shape, integer=False):
    matrix = sparse.csr_matrix(mmread(path))
    matrix.sum_duplicates()
    if (matrix.shape != shape or not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
            or (integer and (matrix.data != np.floor(matrix.data)).any())):
        raise ValueError(f"Unexpected shape or invalid counts in {path.name}")
    return matrix


def _flag(values, allow_missing=False):
    mapped = values.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False,
                                                  "1.0": True, "0.0": False})
    if (mapped.isna() & ~values.astype(str).isin(["", "nan", "<NA>"])).any() or (
            not allow_missing and mapped.isna().any()):
        raise ValueError(f"Invalid indicator: {values.name}")
    return mapped.astype("boolean")


def _group_masks(cells):
    pan = _flag(cells.panT_09B_observed, True)
    mac = _flag(cells.macrophage_label_09B, True)
    distance = pd.to_numeric(cells.distance_to_mac_um, errors="coerce")
    return {
        "all_complete": pd.Series(True, index=cells.index),
        "macrophage": mac.fillna(False),
        "panT_supported": pan.fillna(False),
        "near_panT": (pan & ~mac & distance.le(25)).fillna(False),
        "T_label": _flag(cells.T_label_09B, True).fillna(False),
    }


def _input(input_dir):
    pos = _table(input_dir / "positions.tsv", "barcode")
    genes = _table(input_dir / "genes.tsv", "gene").index
    cells = _table(input_dir / "cells.tsv", "cell_id")
    raw = _counts(input_dir / "counts.mtx", (len(genes), len(pos)), integer=True)
    pos["in_core"] = _flag(pos.in_core).astype(bool)
    pos["in_tissue"] = _flag(pos.in_tissue).astype(bool)
    pos["sr_in_tissue"] = _flag(pos.sr_in_tissue).astype(bool)
    pos["he_tissue_score"] = pd.to_numeric(pos.he_tissue_score.replace("", np.nan), errors="raise")
    scores = pos.he_tissue_score.dropna()
    if not scores.between(0, 1).all():
        raise ValueError("The input contains he_tissue_score outside [0,1].")
    depth = np.asarray(raw.sum(axis=0)).ravel()
    if not np.array_equal(depth, pd.to_numeric(pos.raw_total_umis).to_numpy()):
        raise ValueError("Raw UMI totals disagree with the position table.")
    assigned = pos.cell_id.ne("")
    if not pos.loc[assigned, "cell_id"].isin(cells.index).all():
        raise ValueError("A measured Proseg footprint is missing its cell metadata row.")
    return genes, pos, cells, raw


def _unassigned(pos, raw, genes):
    # Fixed boundaries are comparable across ROIs. Include score 1 and the unknown stratum.
    labels = ["0–<0.1", "0.1–<0.25", "0.25–<0.5", "0.5–<0.75", "0.75–1", "missing"]
    edges = [0, .1, .25, .5, .75, np.nextafter(1., 2.)]
    strata = pd.cut(pos.he_tissue_score, edges, labels=labels[:-1], right=False)
    strata = strata.cat.add_categories("missing").fillna("missing")
    core = pos.in_core.to_numpy(bool)
    outside = pos.cell_id.eq("").to_numpy()
    rows = []
    for gene in CD_GENES:
        measured = gene in genes
        counts = raw.getrow(genes.get_loc(gene)).toarray().ravel() if measured else None
        for region in labels:
            eligible = core & strata.eq(region).to_numpy()
            for state, status in [(True, "out_of_mask"), (False, "assigned")]:
                ix = eligible & (outside == state)
                total = int(pos.loc[ix, "raw_total_umis"].sum())
                k = int(counts[ix].sum()) if measured else None
                rows.append(dict(he_score=region, assignment=status, gene=gene, assayed=measured,
                                 n_squares=int(ix.sum()), raw_total_UMIs=total, raw_marker_UMIs=k,
                                 raw_marker_per_1000_UMIs=1000*k/total if measured and total else np.nan))
    return pd.DataFrame(rows)


def _summarize_groups(cells):
    rows = []
    for group, mask in _group_masks(cells).items():
        subset = cells.loc[mask]
        for panel in ("panT", "myeloid", "Cd4", "CD8"):
            a, b = subset[f"raw_{panel}"], subset[f"spotclean_{panel}"]
            paired = a.notna() & b.notna()
            n = int(paired.sum())
            raw_total, corrected = a[paired].sum(min_count=1), b[paired].sum(min_count=1)
            rows.append(dict(group=group, panel=panel, n_cells=len(subset), n_with_paired_counts=n,
                             raw_UMIs_all_assayed=a.sum(min_count=1), raw_UMIs_paired=raw_total,
                             spotclean_counts_paired=corrected,
                             retained=corrected/raw_total if raw_total > 0 else np.nan,
                             positive_raw_paired=int(a[paired].gt(0).sum()) if n else np.nan,
                             positive_spotclean_paired=int(b[paired].gt(0).sum()) if n else np.nan))
    return pd.DataFrame(rows)


def _historical(paired, roi, pos, cells):
    """Compare only complete cells having identical native-square support and raw totals."""
    rows, notes = [], []
    if paired is None:
        return rows, ["No 09E paired-fit receipt supplied."]
    receipt = json.loads(Path(paired).read_text())
    run = Path(roi["run_dir"]).resolve()
    if Path(receipt["parameters"]["run_dir"]).resolve() != run:
        return rows, ["The supplied 09E receipt belongs to another 09B run."]
    groups = _group_masks(cells)
    myspots = pos.loc[pos.cell_id.isin(cells.index)]
    for domain in ("all_in_tissue", "qc_retained_in_tissue"):
        folder = Path(receipt["fits"][domain]["folder"])
        try:
            state = json.loads((folder / "status.json").read_text())
            source = json.loads((folder / "source.json").read_text())
            if (state["state"] != "complete" or Path(state["run_09B"]).resolve() != run
                    or source["sample"] != roi["sample"] or source["domain"] != domain):
                raise ValueError("fit provenance differs")
            prev = pd.read_csv(folder / "local_matched_cells.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
            if not prev.index.is_unique:
                raise ValueError("duplicate previous cell IDs")
            take = myspots.sr_in_tissue.copy()
            if domain == "qc_retained_in_tissue":
                take &= myspots.qc_class.str.startswith("tissue-high_")
            agrees = myspots.in_tissue.eq(take).groupby(myspots.cell_id).all()
            prev = prev.reindex(cells.index)
            ok = (agrees.reindex(cells.index, fill_value=False) & prev.square_count.eq(cells.n_squares)
                  & prev.hard_UMIs.eq(cells.raw_total_umis))
            for panel in ("panT", "myeloid", "Cd4", "CD8"):
                ok &= prev[f"hard_mask_{panel}"].eq(cells[f"raw_{panel}"]) | (
                    prev[f"hard_mask_{panel}"].isna() & cells[f"raw_{panel}"].isna())
            notes.append(f"{domain}: {int(ok.sum())} complete cells have matched square support and raw counts.")
            for group, panel, _ in REVIEW:
                shared = ok & groups[group] & cells[f"spotclean_{panel}"].notna() & prev[f"sparkle_{panel}"].notna()
                raw = cells.loc[shared, f"raw_{panel}"].sum(min_count=1)
                rows.append(dict(domain=domain, group=group, panel=panel, shared_cells=int(shared.sum()),
                                 raw_UMIs=raw, spotclean_retained=(cells.loc[shared,f"spotclean_{panel}"].sum()/raw
                                                                  if raw > 0 else np.nan),
                                 sparkle_retained=(prev.loc[shared,f"sparkle_{panel}"].sum()/raw
                                                   if raw > 0 else np.nan)))
        except (KeyError, ValueError, OSError) as exc:
            notes.append(f"{domain}: historical fit unavailable ({exc}).")
    return rows, notes


def _readout_plot(summary, outside, target, fit_available):
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios":[1.2,1]},
                                layout="constrained")
    lookup = summary.set_index(["group", "panel"])
    for i, (group, panel, _) in enumerate(REVIEW):
        row = lookup.loc[(group, panel)]
        value = row.retained*100 if fit_available else row.raw_UMIs_all_assayed
        if pd.notna(value):
            a.scatter(value, i, c="#275d64", s=45)
        else:
            a.annotate("unavailable", (0, i), va="center", color="#757575", fontsize=8)
    a.set_yticks(range(len(REVIEW)), [r[2] for r in REVIEW], fontsize=9)
    a.set_ylim(len(REVIEW)-.5,-.5)
    if fit_available:
        a.set_xlim(left=-3, right=max(108, a.get_xlim()[1]))
        a.axvline(100, ls="--", lw=1, color="#888888")
    else:
        a.set_xlim(left=-.25, right=max(1., a.get_xlim()[1]))
    a.set(xlabel="SpotClean counts / observed counts (%)" if fit_available else "Observed marker UMIs",
          title="Complete Proseg cells")
    a.grid(axis="x", alpha=.2)
    dat = outside[outside.assignment.eq("out_of_mask")]
    for gene, color in zip(CD_GENES, ("#a44c35", "#275d64", "#766199")):
        part = dat[dat.gene.eq(gene)]
        b.plot(np.arange(len(part)), part.raw_marker_per_1000_UMIs, ".-", lw=1.4, color=color, label=gene)
    b.set_xticks(np.arange(dat.he_score.nunique()), dat.he_score.unique(), rotation=35, ha="right")
    b.set(xlabel="Stored H&E tissue score", ylabel="Marker UMIs per 1,000 observed UMIs",
          title="Core squares outside Proseg masks")
    b.legend(frameon=False, ncols=3, loc="upper left", fontsize=8)
    b.grid(axis="y", alpha=.2)
    fig.savefig(target, dpi=155, bbox_inches="tight")
    plt.close(fig)


def make_readout(input_dir, fit_dir, output_dir, *, paired_fits=None):
    """Write aggregate review files and a separate local complete-cell table."""
    input_dir, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    genes, pos, meta, raw = _input(input_dir)
    fit_available = fit_dir is not None
    if fit_available:
        fit_dir = Path(fit_dir)
        fit_genes = _table(fit_dir / "decont_genes.tsv", "gene").index
        fit_spots = _table(fit_dir / "decont_positions.tsv", "barcode").index
        correct = _counts(fit_dir / "decont.mtx", (len(fit_genes), len(fit_spots)))
        if (not fit_genes.isin(genes).all() or set(fit_spots) != set(pos.index[pos.in_tissue])):
            raise ValueError("SpotClean axes differ from the submitted tissue positions or input genes.")
        status = _table(fit_dir / "gene_status.tsv", "gene")
        if set(status.index) != set(genes) or set(status.index[_flag(status.in_output)]) != set(fit_genes):
            raise ValueError("SpotClean gene receipt disagrees with its input or corrected gene axis.")
    else:
        fit_spots = pos.index[pos.in_tissue]
        fit_genes = pd.Index([], dtype=str)
        status = pd.DataFrame({"status":"fit_unavailable"}, index=genes)

    outside = _unassigned(pos, raw, genes)
    outside.to_csv(out / "coreceptors_by_score.csv", index=False)
    assigned = pos.cell_id.ne("")
    support = pos.loc[assigned].groupby("cell_id")[["in_core","in_tissue"]].all().all(axis=1)
    complete = meta.index[_flag(meta.complete_core) & support.reindex(meta.index, fill_value=False)]
    cells = meta.loc[complete].copy()
    incl = pos.cell_id.isin(complete)
    owner = complete.get_indexer(pos.loc[incl,"cell_id"])
    weights = sparse.csr_matrix((np.ones(len(owner), dtype=int), (np.arange(len(owner)), owner)),
                                shape=(len(owner), len(complete)))
    tissue_positions = fit_spots.get_indexer(pos.index[incl])
    if (tissue_positions < 0).any():
        raise ValueError("An evaluation cell is missing corrected tissue squares.")
    cells["n_squares"] = np.asarray(weights.sum(axis=0)).ravel()
    cells["raw_total_umis"] = np.asarray(pos.loc[incl, "raw_total_umis"].to_numpy() @ weights).ravel()
    panel_notes = []
    for name, members in MEASURES.items():
        measured = [g for g in members if g in genes]
        fitted = [g for g in measured if g in fit_genes]
        failed = [g for g in measured if g not in fit_genes and status.at[g,"status"] not in
                  ("background_only", "undetected")]
        if measured:
            arr = raw[genes.get_indexer(measured)][:,incl.to_numpy()] @ weights
            cells[f"raw_{name}"] = np.asarray(arr.sum(axis=0)).ravel()
        else:
            cells[f"raw_{name}"] = np.nan
        if fitted and not failed:
            arr = correct[fit_genes.get_indexer(fitted)][:,tissue_positions] @ weights
            cells[f"spotclean_{name}"] = np.asarray(arr.sum(axis=0)).ravel()
        else:
            cells[f"spotclean_{name}"] = np.nan
        panel_notes.append(dict(panel=name, requested=",".join(members), assayed=",".join(measured),
                                in_fit_output=",".join(fitted), status="; ".join(
                                    f"{g}={status.at[g,'status'] if g in status.index else 'unassayed'}" for g in members)))
    pd.DataFrame(panel_notes).to_csv(out / "marker_fit_coverage.csv", index=False)
    for gene in CD_GENES:
        for method in ("raw", "spotclean"):
            v = cells[f"{method}_{gene}"]
            cells[f"{method}_{gene}_detected"] = v.gt(0).astype("boolean").where(v.notna())
    (out / "local").mkdir(exist_ok=True)
    cells.to_csv(out / "local" / "complete_cells.csv.gz", index=True)
    summary = _summarize_groups(cells)
    summary.to_csv(out / "marker_retention.csv", index=False)
    roi = json.loads((input_dir / "roi_receipt.json").read_text())
    try:
        if fit_available:
            history, notes = _historical(paired_fits, roi, pos, cells)
        else:
            history, notes = [], ["SpotClean fit unavailable. Only observed counts are reported."]
    except (KeyError, ValueError, OSError) as exc:
        history, notes = [], [f"Historical comparison unavailable ({exc})."]
    cols = ["domain","group","panel","shared_cells","raw_UMIs","spotclean_retained","sparkle_retained"]
    pd.DataFrame(history, columns=cols).to_csv(out / "paired_sparkle.csv", index=False)
    _readout_plot(summary, outside, out / "marker_readout.png", fit_available)
    assigned_by_gene = []
    for gene in CD_GENES:
        z = outside[outside.gene.eq(gene)]
        full = z.raw_marker_UMIs.sum(min_count=1)
        unassigned = z.loc[z.assignment.eq("out_of_mask"),"raw_marker_UMIs"].sum(min_count=1)
        phrase = f"{int(unassigned)} of {int(full)} raw core UMIs outside Proseg masks" if pd.notna(full) else "unassayed"
        assigned_by_gene.append(f"- {gene}: {phrase}")
    labels_known = int(cells.cell_type.fillna("").ne("").sum())
    counts = dict(spotclean_fit_available=fit_available, complete_cells=len(cells), complete_cells_with_prior_label=labels_known,
                  core_squares=int(pos.in_core.sum()), core_unassigned_squares=int((pos.in_core & ~assigned).sum()))
    mode = "SpotClean fit completed." if fit_available else "Raw-only review. No SpotClean fit was available."
    lines = ["# Marker readout", "", mode, "", f"{len(cells)} complete Proseg cell footprints enter the comparison. "
             f"{labels_known} have a previous cell-type label. Evaluation uses the same native squares for "
             "raw and SpotClean counts. Any cell touching the crop boundary or containing a modeled background "
             "square is excluded.", "", "| Group and panel | Cells | Observed UMIs | SpotClean / raw |",
             "|---|---:|---:|---:|"]
    lookup = summary.set_index(["group","panel"])
    for group, panel, label in REVIEW:
        row = lookup.loc[(group,panel)]
        raw_text = f"{row.raw_UMIs_all_assayed:.0f}" if pd.notna(row.raw_UMIs_all_assayed) else "—"
        ret = f"{100*row.retained:.1f}%" if pd.notna(row.retained) else "—"
        lines.append(f"| {label} | {int(row.n_cells)} | {raw_text} | {ret} |")
    lines.extend(["", "Ratios use summed counts and may exceed 100%. A dash indicates insufficient paired "
                  "data or a zero denominator. Pan-T support and macrophage distances come from the previous "
                  "09B analysis. The nearby group requires pan-T support, a non-macrophage label and distance "
                  "to a labeled macrophage of at most 25 µm. See marker_fit_coverage.csv for assayed and modeled genes.",
                  "", "## Observed coreceptor RNA in the core", "", *assigned_by_gene, "",
                  "The score table includes measured zero-count squares. An out-of-mask count identifies a "
                  "candidate for segmentation review. Its cell of origin remains unknown.", "",
                  "## Historical SPARKLE comparison", "", *notes, "",
                  "paired_sparkle.csv uses shared complete cells with matching native-square domains and raw counts. "
                  "The original SPARKLE fit used a wider neighborhood. A tiny or empty group has little power "
                  "to distinguish retention at equivalent spillover reduction.", ""])
    (out / "marker_readout.md").write_text("\n".join(lines))
    return counts
