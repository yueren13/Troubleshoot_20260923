"""Describe native RNA by historical QC class and distance from annotated myeloid cells."""
from pathlib import Path
import json

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from bins09d import _rows
from compare09d import PANELS


QC_CLASSES = tuple(f"tissue-{he}_transcript-{rna}"
                   for he in ("high", "low") for rna in ("high", "low"))
DISTANCE_UPPER = np.array([10., 25., 50., 100.])
DISTANCE_LABELS = ("0–10", ">10–25", ">25–50", ">50–100", ">100")


def _read_classes(tile):
    spots = tile["spots"]
    source = Path(tile["meta"]["inputs"]["aligned_qc"])
    names = pq.ParquetFile(source).schema_arrow.names
    column = next((s for s in ("qc_class", "qc_class_raw") if s in names), None)
    if column is None:
        raise ValueError(f"{source} lacks qc_class and qc_class_raw. Use the Step 00 export with four-way labels.")
    wanted = pd.Index(spots.barcode)
    if wanted.hasnans or not wanted.is_unique:
        raise ValueError("Native tile barcodes must be unique and nonmissing.")
    labels = np.full(len(wanted), "unassessed", dtype=object)
    found = np.zeros(len(wanted), dtype=bool)
    for chunk in _rows(source, ["barcode", column]):
        where = wanted.get_indexer(chunk.barcode)
        hit = where >= 0
        destination = where[hit]
        if len(np.unique(destination)) != len(destination) or found[destination].any():
            raise ValueError("A requested barcode has duplicate Step 00 QC records.")
        values = (chunk.loc[hit, column].astype("string").str.strip().str.lower()
                  .str.replace("trancript", "transcript", regex=False))
        values = values.fillna("unassessed")
        values = values.replace({"": "unassessed", "<na>": "unassessed", "nan": "unassessed"})
        unknown = values.loc[~values.isin((*QC_CLASSES, "unassessed"))].unique().tolist()
        if unknown:
            raise ValueError(f"Unrecognized Step 00 QC class from {column}: {unknown[:8]}")
        labels[destination] = values.to_numpy()
        found[destination] = True
    if not np.array_equal(found, spots.qc_record_present.to_numpy(bool)):
        raise ValueError("Barcode matches for the four-way labels differ from the 09D Step 00 alignment.")
    if not np.array_equal(np.isin(labels, QC_CLASSES[:2]), spots.qc_keep.to_numpy(bool)):
        raise ValueError("The two tissue-high classes disagree with the Step 00 qc_keep flag.")
    return labels, column


def _source_centroids(tile):
    """Use the full prepared single-section annotation so artificial tile edges do not censor distance."""
    run = Path(tile["run_dir"])
    settings = json.loads((run / "settings.json").read_text())
    receipt = json.loads((run / "complete.json").read_text())
    src = Path(settings["source"])
    if not src.is_file():
        src = run.parent / "inputs" / src.parent.name / src.name
    if not src.is_file() or src.stat().st_size != settings["size"]:
        raise ValueError(f"Cannot match the 09B prepared input to its saved receipt: {src}")
    fit = pd.read_csv(run / "local_cell_assignments.csv.gz", dtype=str, keep_default_na=False)
    section = str(tile["meta"]["sample"])
    if len(fit) != receipt["fitted_cells"] or not fit.cell_id.is_unique:
        raise ValueError("Saved 09B cell assignment IDs differ from the completed fit receipt.")
    a = ad.read_h5ad(src, backed="r")
    try:
        pos = a.obs_names.get_indexer(fit.cell_id)
        if (pos < 0).any() or a.n_obs != receipt["source_cells"]:
            raise ValueError("The 09B assignment IDs or dimensions differ from its prepared H5AD.")
        for field in ("cell_type", "section_id"):
            if not np.array_equal(fit[field].to_numpy(), a.obs[field].iloc[pos].astype(str).to_numpy()):
                raise ValueError(f"{field} disagrees between the 09B fit and prepared H5AD.")
        section_mask = a.obs.section_id.astype(str).eq(section).to_numpy()
        if not section_mask.any() or not fit.section_id.eq(section).all():
            raise ValueError(f"09B fit or prepared cells do not identify a single section {section!r}.")
        mac = a.obs.cell_type.astype("string").str.contains(receipt["type_patterns"]["macrophage"],
                                                            case=False, na=False).to_numpy(bool)
        xy = np.asarray(a.obsm["spatial"])[section_mask & mac, :2].astype(float)
        xy *= float(settings["settings"]["microns_per_unit"])
        if not np.isfinite(xy).all():
            raise ValueError("Nonfinite prior macrophage/monocyte centroids in the prepared H5AD.")
        return xy, src
    finally:
        a.file.close()


def _marker_counts(tile):
    symbols = pd.Index(tile["genes"]["gene_symbol"].astype(str))
    definitions = {s: [s] for s in PANELS["myeloid"]}
    definitions.update({key: PANELS[key] for key in ("myeloid", "panT", "epithelial", "endothelial")})
    counts, coverage = {}, {}
    for name, requested in definitions.items():
        rows = np.flatnonzero(symbols.isin(requested))
        coverage[name] = {"requested": requested, "measured": list(symbols[rows].unique())}
        if len(rows):
            counts[name] = np.asarray(tile["counts"][rows].sum(axis=0)).ravel().astype(np.int64)
    return counts, coverage


def _coverage(labels, core, outside, totals, markers):
    classes = list(QC_CLASSES) + (["unassessed"] if np.any(labels == "unassessed") else [])
    rows, marker_rows = [], []
    for scope, area in [("buffered", np.ones(len(labels), bool)), ("core", core), ("halo_only", ~core)]:
        for assignment, region in [("out_of_mask", area & outside), ("assigned", area & ~outside)]:
            region_count = int(region.sum())
            region_umis = int(totals[region].sum())
            for qc in classes:
                take = region & (labels == qc)
                depth = totals[take]
                n = len(depth)
                umi = int(depth.sum())
                n0 = int(np.count_nonzero(depth == 0))
                n1 = int(np.count_nonzero(depth == 1))
                n2 = n - n0 - n1
                mismatch = (n0 + n1) if qc.endswith("transcript-high") else n2
                rows.append(dict(scope=scope, assignment=assignment, qc_class=qc,
                                 n_squares=n, UMIs=umi, n_zero=n0, n_one=n1, n_two_or_more=n2,
                                 UMIs_per_square=umi/n if n else np.nan,
                                 square_fraction_of_assignment=n/region_count if region_count else np.nan,
                                 UMI_fraction_of_assignment=umi/region_umis if region_umis else np.nan,
                                 n_disagree_raw_1p6=(mismatch if qc != "unassessed" else np.nan)))
                for marker, values in markers.items():
                    k = int(values[take].sum())
                    marker_rows.append(dict(scope=scope, assignment=assignment, qc_class=qc,
                                            marker=marker, n_squares=n, total_UMIs=umi, marker_UMIs=k,
                                            marker_UMIs_per_square=k/n if n else np.nan,
                                            marker_per_1000_total_UMIs=1000*k/umi if umi else np.nan))
    marker_table = pd.DataFrame(marker_rows, columns=["scope", "assignment", "qc_class", "marker", "n_squares",
                                                    "total_UMIs", "marker_UMIs", "marker_UMIs_per_square",
                                                    "marker_per_1000_total_UMIs"])
    return pd.DataFrame(rows), marker_table


def _gradient(labels, core_outside, distances, totals, markers):
    """Six strata permit tests pooled across RNA status while preserving the four original classes."""
    classes = [(s, labels == s) for s in QC_CLASSES]
    classes.extend([("tissue-low_all", np.isin(labels, QC_CLASSES[2:])),
                    ("tissue-high_all", np.isin(labels, QC_CLASSES[:2]))])
    rank = np.searchsorted(DISTANCE_UPPER, distances, side="left")
    valid = core_outside & np.isfinite(distances)
    rows = []
    for group, member in classes:
        for i, interval in enumerate(DISTANCE_LABELS):
            take = valid & member & (rank == i)
            n = int(take.sum())
            depth = int(totals[take].sum())
            for key, counts in markers.items():
                observed = counts[take]
                k = int(observed.sum())
                rows.append(dict(qc_group=group, distance_band_um=interval, band_order=i,
                                 marker=key, n_squares=n, total_UMIs=depth, marker_UMIs=k,
                                 marker_positive_squares=np.count_nonzero(observed),
                                 marker_UMIs_per_square=k/n if n else np.nan,
                                 marker_per_1000_total_UMIs=1000*k/depth if depth else np.nan))
    return pd.DataFrame(rows, columns=["qc_group","distance_band_um","band_order","marker","n_squares",
                                       "total_UMIs","marker_UMIs","marker_positive_squares",
                                       "marker_UMIs_per_square","marker_per_1000_total_UMIs"])


def _block_comparison(xy, keep, distances, totals, markers):
    """Compare nearby and more distant squares within the same fixed 200-µm block."""
    near = distances <= 25.
    far = (distances > 50.) & (distances <= 100.)
    ix = np.flatnonzero(keep & (near | far))
    coords = np.floor(xy[ix] / 200.).astype(np.int64)
    cols = ["block_x","block_y","marker","n_near","n_far","total_near","total_far","marker_near",
            "marker_far","near_minus_far_UMIs_per_square","near_minus_far_per_1000_total_UMIs"]
    if not len(ix):
        return pd.DataFrame(columns=cols)
    blocks, groups = np.unique(coords, axis=0, return_inverse=True)
    # Each half has its own denominator, including true zero squares. Unpaired blocks remain visible.
    def sums(mask, v=None):
        return np.bincount(groups[mask], weights=v[ix[mask]] if v is not None else None,
                           minlength=len(blocks)).astype(np.int64)
    n_near, n_far = sums(near[ix]), sums(far[ix])
    total_near, total_far = sums(near[ix],totals), sums(far[ix],totals)
    rows = []
    for marker, counts in markers.items():
        k_near, k_far = sums(near[ix],counts), sums(far[ix],counts)
        for j, (bx, by) in enumerate(blocks):
            n1, n2 = n_near[j], n_far[j]
            t1, t2 = total_near[j], total_far[j]
            k1, k2 = k_near[j], k_far[j]
            rows.append((bx,by,marker,n1,n2,t1,t2,k1,k2,
                         k1/n1 - k2/n2 if n1 and n2 else np.nan,
                         1000*(k1/t1 - k2/t2) if t1 and t2 else np.nan))
    return pd.DataFrame(rows,columns=cols)


def _figures(out, coverage, gradient):
    import matplotlib.pyplot as plt
    data = coverage[(coverage.scope == "buffered") & (coverage.assignment == "out_of_mask")]
    fig, ax = plt.subplots(figsize=(8,3.4),layout="constrained")
    x = np.arange(len(data))
    ax.bar(x-.18, data.square_fraction_of_assignment * 100, width=.36, color="#6f8e95", label="Squares")
    ax.bar(x+.18, data.UMI_fraction_of_assignment * 100, width=.36, color="#b46330", label="UMIs")
    ax.set(xticks=x, xticklabels=[s.replace("_", "\n") for s in data.qc_class], ylabel="Share (%)",
           title="Out-of-mask squares and raw RNA in the buffered region", ylim=(0,106))
    ax.legend(frameon=False)
    fig.savefig(out / "qc_class_coverage.png", dpi=165)
    plt.close(fig)
    tracks = ("Lyz2", "C1qa", "panT", "epithelial")
    fig, axes = plt.subplots(2,2,figsize=(9,6.6),layout="constrained",sharex=True)
    for ax, track in zip(axes.flat, tracks):
        for group, legend, color in [("tissue-low_all","Tissue-low","#a7522e"),
                                      ("tissue-high_all","Tissue-high","#276c79")]:
            tab = gradient[(gradient.marker == track) & (gradient.qc_group == group)].sort_values("band_order")
            ax.plot(tab.band_order, tab.marker_UMIs_per_square, color=color, marker="o", label=legend)
        series = gradient.loc[gradient.marker.eq(track), "n_squares"]
        if series.empty or series.sum() == 0:
            label = "Unassayed" if series.empty else "No eligible observations"
            ax.text(.5,.55,label,transform=ax.transAxes,ha="center")
        ax.set(title=f"{track} panel" if track in PANELS else track, xticks=range(len(DISTANCE_LABELS)),
               xticklabels=DISTANCE_LABELS, ylim=(0,None))
    axes[0,0].legend(frameon=False, fontsize=9)
    fig.supxlabel("Distance to nearest previously labeled macrophage/monocyte centroid (µm)")
    fig.supylabel("Observed marker UMIs per out-of-mask square")
    fig.suptitle("Native 2-µm squares in the 09B core")
    fig.savefig(out / "ghost_distance.png", dpi=165)
    plt.close(fig)


def analyze_ghost(tile, output_dir):
    """Write aggregate QC and source-distance evidence; per-square arrays are returned only in memory."""
    if tile["meta"]["domain"] != "all_in_tissue":
        raise ValueError("The four-class audit requires an all_in_tissue native tile.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels, class_column = _read_classes(tile)
    spots = tile["spots"]
    xy = spots[["x_um","y_um"]].to_numpy(float)
    core = spots.in_core.to_numpy(bool)
    outside = spots.cell_index.to_numpy() < 0
    totals = np.asarray(tile["counts"].sum(axis=0)).ravel().astype(np.int64)
    markers, measured = _marker_counts(tile)
    coverage, by_marker = _coverage(labels,core,outside,totals,markers)
    coverage.to_csv(out / "qc_class_coverage.csv", index=False)
    by_marker.to_csv(out / "qc_class_marker_counts.csv", index=False)
    mac_xy, src = _source_centroids(tile)
    distance = cKDTree(mac_xy).query(xy)[0] if len(mac_xy) else np.full(len(xy), np.nan)
    gradient = _gradient(labels,core & outside,distance,totals,markers)
    gradient.to_csv(out / "ghost_distance.csv", index=False)
    he_low = np.isin(labels,QC_CLASSES[2:])
    blocks = _block_comparison(xy, core & outside & he_low, distance, totals, markers)
    blocks.to_csv(out / "ghost_within_blocks.csv", index=False)
    _figures(out, coverage, gradient)
    audit = coverage[(coverage.scope=="buffered") & coverage.qc_class.ne("unassessed")]
    outside_rows = audit[audit.assignment=="out_of_mask"].set_index("qc_class")
    candidate_umis = int(outside_rows.loc["tissue-low_transcript-high", "UMIs"])
    outside_umis = int(totals[outside].sum())
    info = dict(
        qc_column=class_column, aligned_qc=tile["meta"]["inputs"]["aligned_qc"],
        qc_classes=list(QC_CLASSES), n_unassessed=int(np.count_nonzero(labels=="unassessed")),
        n_buffered_classified=int(audit.n_squares.sum()), n_raw_cutoff_disagreements=int(audit.n_disagree_raw_1p6.sum()),
        cutoff_check="Observed all-Gene-Expression UMIs <=1 vs >=2 against archived transcript-low/high.",
        n_buffered_out_of_mask= int(outside.sum()), buffered_out_of_mask_UMIs=outside_umis,
        buffered_out_of_mask_tissue_low_transcript_high_UMIs=candidate_umis,
        fraction_of_buffered_out_of_mask_UMIs_in_candidate_class=candidate_umis/outside_umis if outside_umis else None,
        marker_coverage=measured, centroid_source=str(src), section=tile["meta"]["sample"],
        n_same_section_macrophage_centroids=len(mac_xy), n_core_out_of_mask_squares=int(np.sum(core & outside)),
        gradient_status="available" if len(mac_xy) and np.any(core & outside) and markers else "no_sources_squares_or_assayed_markers",
        distance_reference="All prior macrophage/monocyte-labeled centroids in the prepared 09B section; no tile edge truncation.",
        primary_distance_domain="Core, out of mask, tissue-low_all including zero counts. Tissue-high_all is the comparison.",
        distance_bands_um=list(DISTANCE_LABELS), block_width_um=200, block_origin_Proseg_xy_um=[0,0],
        block_distance_definition_um="near <=25; far >50 through 100 inclusive. Both transcript levels pooled in tissue-low.",
        block_scope="Descriptive same-block differences. Each block with only one distance group has an empty contrast.",
    )
    (out / "ghost_qc.json").write_text(json.dumps(info, indent=2, allow_nan=False)+"\n")
    return dict(qc_class=labels, markers=markers, total_umis=totals, macrophage_xy=mac_xy,
                distance_um=distance, coverage=coverage, gradient=gradient, info=info)
