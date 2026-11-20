"""Compact disk-level review figures for the fixed-window 09H readout."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENDPOINTS = {
    ("near_panT", "myeloid"): "Myeloid in nearby T-supported cells",
    ("macrophage", "myeloid"): "Myeloid in macrophage/monocyte labels",
    ("panT_supported", "panT"): "Pan-T in T-supported cells",
    ("T_label", "Cd4"): "Cd4 in prior T labels",
    ("T_label", "CD8"): "CD8 in prior T labels",
}
COLORS = {"SPARKLE": "#00766e", "SpotClean": "#ae5d2f"}


def _retention(folder, frame, scope, methods):
    path = folder / f"{scope.lower()}_marker_retention.png"
    part = frame.loc[frame.scope.eq(scope)].copy()
    keys = list(zip(part.group, part.panel))
    labels = [ENDPOINTS.get(k, " / ".join(k)) for k in keys]
    part["label"] = labels
    repeated = part.label.duplicated(keep=False)
    part.loc[repeated, "label"] += " [" + part.loc[repeated, "genes"].fillna("") + "]"
    ordering = {k: i for i, k in enumerate(ENDPOINTS)}
    part["order"] = [ordering.get(k, len(ordering)) for k in keys]
    part.sort_values(["order", "label"], inplace=True)
    raw = pd.to_numeric(part.raw_UMIs, errors="coerce")
    estimates = pd.DataFrame(index=part.index)
    for method in methods:
        corrected = pd.to_numeric(part[method.lower() + "_counts"], errors="coerce")
        estimates[method] = 100 * corrected / raw.where(raw.gt(0))
    if estimates.empty or not estimates.notna().any().any():
        path.unlink(missing_ok=True)
        return
    title = ("SPARKLE on its verified footprints" if scope == "SPARKLE"
             else "SPARKLE and SpotClean on identical footprints")
    n = len(part)
    y = np.arange(n)
    fig, (ax, counts) = plt.subplots(1, 2, figsize=(9.7, 3.45 + .34 * max(0, n-5)),
                                     gridspec_kw={"width_ratios": [3, 1]}, layout="constrained")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for method, marker in zip(methods, ["o", "s"]):
        values = estimates[method].to_numpy()
        use = np.isfinite(values)
        offset = ((-.13 if method == "SPARKLE" else .13) if len(methods) == 2 else 0)
        ax.scatter(values[use], y[use] + offset, marker=marker, color=COLORS[method], s=46,
                   label=method, zorder=3)
    ax.axvline(100, color="#657987", ls=":", label="Raw reference", zorder=1)
    ax.set(xlabel="Corrected / observed marker counts (%)", yticks=y, yticklabels=part.label,
           xlim=(0, max(108, 1.08 * np.nanmax(estimates.to_numpy()))), ylim=(n-.5, -.5))
    ax.grid(axis="x", alpha=.18)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=3, frameon=False)
    counts.set(xlim=(0,1), ylim=(n-.5, -.5))
    counts.set_axis_off()
    for col, header, xpos in (("cells", "Cells", .31), ("raw_UMIs", "Raw UMIs", .97)):
        counts.text(xpos, 1.03, header, ha="right", va="bottom", weight="bold",
                    transform=counts.transAxes, fontsize=10)
        for i, val in enumerate(part[col]):
            label = "—" if pd.isna(val) else f"{val:,.0f}"
            counts.text(xpos, i, label, ha="right", va="center", fontsize=10)
    fig.savefig(path, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _coreceptors(folder):
    path = folder / "raw_coreceptor_masks.png"
    source = folder / "coreceptors_disk_by_score.csv"
    if not source.is_file():
        path.unlink(missing_ok=True)
        return
    values = pd.read_csv(source)
    known = values.assayed.astype("string").str.strip().str.lower().isin(["true", "1", "1.0"])
    data = values.loc[known & values.gene.isin(["Cd4", "Cd8a", "Cd8b1"])]
    by_role = data.groupby(["gene", "assignment"]).raw_marker_UMIs.sum(min_count=1)
    counts = by_role.unstack().reindex(index=["Cd4", "Cd8a", "Cd8b1"],
                                       columns=["assigned", "out_of_mask"])
    total = counts.sum(axis=1, min_count=2)
    frac = (100 * counts.out_of_mask / total.where(total.gt(0))).dropna()
    if frac.empty:
        path.unlink(missing_ok=True)
        return
    y = np.arange(len(frac))
    fig, (ax, table) = plt.subplots(1, 2, figsize=(8.5, 2.8), layout="constrained",
                                    gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle("Observed coreceptor RNA across the original disk", fontsize=14, fontweight="bold")
    ax.barh(y, frac, height=.40, color="#397995")
    ax.set(yticks=y, yticklabels=frac.index, ylim=(len(frac)-.5, -.5),
           xlabel="Share of measured gene UMIs outside Proseg masks (%)",
           xlim=(0, max(3., min(112., 1.25*frac.max()))))
    for i, v in enumerate(frac):
        ax.annotate(f"{v:.1f}%", (v, i), xytext=(3,0), textcoords="offset points", va="center")
    ax.spines[["right", "top"]].set_visible(False)
    table.set(xlim=(0,1), ylim=(len(frac)-.5, -.5))
    table.set_axis_off()
    for column, xpos, title in (("assigned", .42, "Inside UMIs"), ("out_of_mask", .98, "Outside UMIs")):
        table.text(xpos, 1.02, title, transform=table.transAxes, ha="right", va="bottom",
                   fontsize=9, weight="bold")
        for i, val in enumerate(counts.loc[frac.index,column]):
            table.text(xpos, i, f"{val:,.0f}", ha="right", va="center")
    fig.savefig(path, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_review_plots(comparison_dir):
    """Read aggregate counts and draw only panels with an estimable denominator."""
    folder = Path(comparison_dir)
    markers = pd.read_csv(folder / "method_coverage.csv")
    for scope, methods in (("SPARKLE", ["SPARKLE"]), ("paired", ["SPARKLE", "SpotClean"])):
        _retention(folder, markers, scope, methods)
    _coreceptors(folder)
