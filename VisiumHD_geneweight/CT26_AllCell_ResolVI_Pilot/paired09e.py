"""Compare the two 09D SPARKLE domain fits on identical original 09B core cells."""
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bins09d import locate_paths
from compare09d import PANELS, TARGETS, _original
from run09d import run_pilot


DOMAINS = ("all_in_tissue", "qc_retained_in_tissue")
DOMAIN_LABELS = ("All in-tissue", "H&E-high in-tissue")


def _read_json(path):
    return json.loads(path.read_text())


def _groups(original):
    t = original.T_label_09B.fillna(False)
    mac = original.macrophage_label_09B.fillna(False)
    supported = original.panT_09B_observed.fillna(False)
    near = original.distance_to_mac_um.le(25)
    far = original.distance_to_mac_um.ge(50)
    return {
        "all_core": np.ones(len(original), dtype=bool),
        "T_label": t,
        "B_label": original.cell_type.str.contains(r"^B[ _-]*cell", case=False, na=False),
        "macrophage_label": mac,
        "panT09C": supported,
        "panT09C_nonmac_near25": supported & ~mac & near,
        "panT09C_nonmac_far50": supported & ~mac & far,
        "T_panT09C": t & supported,
        "T_panT09C_near25": t & supported & near,
        "T_panT09C_far50": t & supported & far,
    }


def _load_fit(folder, domain, spec, original, geometry, inputs, package_version):
    status = _read_json(folder / "status.json")
    if status.get("state") != "complete" or status.get("run_09B") != spec["run_dir"]:
        return None
    settings = _read_json(folder / "settings.json")
    fields = dict(domain=domain, run_sparkle=True, padding_um=spec["padding_um"],
                  background_genes=spec["background_genes"], sparkle_radius_um=200, sparkle_background_bin_um=50)
    if any(settings.get(k) != v for k, v in fields.items()):
        return None
    source = _read_json(folder / "source.json")
    if (status.get("sample") != geometry["section"] or source.get("geometry_09b") != geometry
            or source.get("domain") != domain or source.get("sample") != geometry["section"]):
        return None
    # Compare resolved native features, positions and segmentation independently of mount aliases.
    for k, path in inputs.items():
        if Path(source["inputs"][k]).expanduser().resolve() != path.resolve():
            return None
    expected = spec.get("expected_background", {}).get(domain)
    if expected is not None:
        measured = pd.read_csv(folder / "background_coverage.csv",
                               usecols=["unassigned_squares", "unassigned_UMIs"]).sum()
        if (int(measured.unassigned_squares) != expected["squares"]
                or int(measured.unassigned_UMIs) != expected["UMIs"]):
            return None
    timestamp = (folder / "status.json").stat().st_mtime_ns
    raw_source = Path(source["raw_source"])
    raw_inputs = [raw_source] if raw_source.suffix == ".h5" else list(raw_source.parent.glob("*"))
    run = Path(spec["run_dir"])
    for path in [*raw_inputs, raw_source, *inputs.values(), run / "complete.json",
                 run / "settings.json", run / "local_cell_assignments.csv.gz"]:
        if path.is_file() and path.stat().st_mtime_ns > timestamp:
            return None
    diag = _read_json(folder / "sparkle_diagnostics.json")
    if (package_version and diag["version"] != package_version) or diag["r2_threshold"] != 0.01:
        return None
    coverage = _read_json(folder / "coverage_summary.json")
    cells = pd.read_csv(folder / "local_matched_cells.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    genes = pd.read_csv(folder / "sparkle_gene_fits.csv", dtype={"gene_id": str})
    if (not cells.index.is_unique or cells.index.hasnans or not cells.index.isin(original.index).all()
            or len(cells) != coverage["n_matched_core"] or coverage["n_original_core"] != len(original)
            or not genes.gene_id.is_unique or len(genes) != diag["n_fit_features"]):
        raise ValueError("Core-cell or gene axes disagree with the 09D receipts.")
    check = original.loc[cells.index]
    for key in ["cell_type", "T_label_09B", "macrophage_label_09B", "panT_09B_detected_symbols"]:
        if not np.array_equal(cells[key].to_numpy(), check[key].to_numpy()):
            raise ValueError(f"{key} differs from the original 09B core.")
    for key in ["09B_x_um", "09B_y_um", "09B_UMIs", "distance_to_mac_um"]:
        if not np.allclose(cells[key], check[key], rtol=1e-7, atol=1e-5, equal_nan=True):
            raise ValueError(f"{key} differs from the original 09B core.")
    for panel in PANELS:
        a, b = cells["hard_mask_" + panel], cells["sparkle_" + panel]
        if (not a.isna().equals(b.isna()) or np.isinf(a).any() or np.isinf(b).any()
                or (a < 0).any() or (b < 0).any() or (b > a + 1e-6).any()):
            raise ValueError(f"Invalid paired marker estimates for {panel}.")
    # Check the local CSV against the independent 09D aggregate used for the return ZIP.
    totals = pd.read_csv(folder / "panel_comparisons.csv").query("group == 'all_matched'")
    for method, prefix in [("observed_same_hard_mask", "hard_mask_"), ("SPARKLE_same_hard_mask", "sparkle_")]:
        records = totals.loc[totals.method.eq(method)].set_index("panel").reindex(PANELS)
        values = cells[[prefix + panel for panel in PANELS]]
        if (not np.allclose(values.sum(min_count=1), records.total_panel_counts, equal_nan=True)
                or not np.array_equal(values.count(), records.n_measured)
                or not records.n_cells.eq(len(cells)).all()):
            raise ValueError(f"Local panel table disagrees with the 09D aggregate for {method}.")
    return dict(folder=folder, cells=cells, genes=genes, diag=diag, source=source)


def _compare_groups(fits, original):
    wide, restricted = (fits[d]["cells"] for d in DOMAINS)
    common = original.index[original.index.isin(wide.index) & original.index.isin(restricted.index)]
    if not len(common):
        raise ValueError("No original evaluation-core cell has measured square centers in both runs.")
    marker_rows, coverage_rows, anchor_rows = [], [], []
    for group, within in _groups(original).items():
        requested = original.index[np.asarray(within, dtype=bool)]
        shared = common[common.isin(requested)]
        for domain in DOMAINS:
            cells = fits[domain]["cells"]
            rows = cells.loc[shared]
            n_represented = int(requested.isin(cells.index).sum())
            coverage_rows.append(dict(
                group=group, domain=domain, n_original_core=len(requested),
                n_represented=n_represented, n_absent=len(requested)-n_represented,
                n_shared=len(rows), n_only_in_this_domain=n_represented-len(rows),
                shared_raw_full_assay_UMIs=int(rows.hard_UMIs.sum()),
                shared_covered_area_um2=float(rows.represented_area_um2.sum()),
                median_shared_raw_full_assay_UMIs=rows.hard_UMIs.median(),
                median_shared_covered_area_um2=rows.represented_area_um2.median()))
            for panel in PANELS:
                raw = rows["hard_mask_" + panel].dropna()
                corrected = rows.loc[raw.index, "sparkle_" + panel]
                observed = raw.sum(min_count=1)
                adjusted = corrected.sum(min_count=1)
                marker_rows.append(dict(
                    group=group, domain=domain, panel=panel, n_shared=len(rows), n_assayed=len(raw),
                    raw_panel_UMIs=observed, corrected_panel_counts=adjusted,
                    fraction_counts_retained=adjusted / observed if observed > 0 else np.nan,
                    mean_raw_counts=raw.mean(), mean_corrected_counts=corrected.mean(),
                    cells_positive_raw=int(raw.gt(0).sum()), cells_positive_corrected=int(corrected.gt(0).sum())))
            if group in ("T_label", "panT09C", "T_panT09C"):
                for method in ("hard_mask", "sparkle"):
                    cd4, cd8 = rows[method+"_Cd4"], rows[method+"_CD8"]
                    assayed = cd4.notna() & cd8.notna()
                    four, eight = cd4[assayed].gt(0), cd8[assayed].gt(0)
                    anchor_rows.append(dict(
                        group=group, domain=domain, method=method, n_shared=len(rows), n_assayed=int(assayed.sum()),
                        Cd4_only=int((four & ~eight).sum()), CD8_only=int((~four & eight).sum()),
                        both=int((four & eight).sum()), neither=int((~four & ~eight).sum())))
    markers = pd.DataFrame(marker_rows)
    assessed = markers.n_assayed.replace(0, np.nan)
    markers["fraction_cells_positive_raw"] = markers.cells_positive_raw / assessed
    markers["fraction_cells_positive_corrected"] = markers.cells_positive_corrected / assessed
    # These fractions have different denominators from the within-group correction/observation ratio.
    all_shared = markers.loc[markers.group.eq("all_core")].set_index(["domain", "panel"])
    for measure, label in [("raw_panel_UMIs", "raw"), ("corrected_panel_counts", "corrected")]:
        total = all_shared[measure].reindex(pd.MultiIndex.from_frame(markers[["domain", "panel"]]))
        markers[f"share_of_all_shared_{label}_panel_counts"] = markers[measure] / total.replace(0, np.nan).to_numpy()
    return markers, pd.DataFrame(coverage_rows), pd.DataFrame(anchor_rows), common


def _plot(markers, target):
    import matplotlib.pyplot as plt

    groups = [("panT09C_nonmac_near25", "myeloid", "Pan-T +, near macrophages · myeloid"),
              ("panT09C_nonmac_far50", "myeloid", "Pan-T +, far from macrophages · myeloid"),
              ("macrophage_label", "myeloid", "Macrophage label · myeloid"),
              ("panT09C", "panT", "Pan-T + · pan-T markers"),
              ("T_label", "Cd4", "T label · Cd4"), ("T_label", "CD8", "T label · Cd8a / Cd8b1"),
              ("B_label", "B", "B label · B markers")]
    fig, ax = plt.subplots(figsize=(10.9, 5.8), layout="constrained")
    records = markers.set_index(["group", "panel", "domain"])
    ypos = np.arange(len(groups))
    for i, (domain, label) in enumerate(zip(DOMAINS, DOMAIN_LABELS)):
        fractions = np.array([records.at[(g,p,domain),"fraction_counts_retained"] for g,p,_ in groups])
        y = ypos + (-.14 if i == 0 else .14)
        valid = np.isfinite(fractions)
        ax.scatter(fractions[valid] * 100, y[valid], s=62, marker=("o","^")[i],
                   color=("#21646a","#b45c27")[i], label=label)
        for absent in y[~valid]:
            ax.text(3 if i == 0 else 27, absent, "no raw signal", va="center", fontsize=8, color=("#21646a","#b45c27")[i])
    labels = [f"{label}  (n={records.at[(g,p,DOMAINS[0]),'n_shared']:,})" for g,p,label in groups]
    ax.set(yticks=ypos, yticklabels=labels, xlim=(-3,105), ylim=(len(groups)-.47,-.47),
           xlabel="Corrected counts as a percentage of raw counts in the same domain",
           title="SPARKLE marker retention in matched core cells")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.13), ncols=2, frameon=False)
    fig.savefig(target, dpi=165)
    plt.close(fig)


def compare_domains(out, *, result_root, run_dir, study_root, temp_project_root, overrides=None,
                    padding_um=400., background_genes=128, display_progress=True, expected_background=None):
    """Reuse compatible full 09D folders, fit missing domains, and write aggregate comparisons."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = Path(run_dir).resolve()
    root = Path(result_root).resolve()
    spec = dict(run_dir=str(run), padding_um=padding_um, background_genes=background_genes)
    if expected_background is not None:
        spec["expected_background"] = expected_background
    geometry = _read_json(run / "complete.json")["geometry"]
    inputs = locate_paths(geometry["section"], study_root, temp_project_root, overrides)
    original, _ = _original(run)
    original = original.set_index("cell_id")
    try:
        package_version = version("stambient")
    except PackageNotFoundError:
        package_version = None
    provenance = dict(parameters=spec, fits={}, skipped_incompatible=[], fit_scope="buffered 09B region")
    receipt = out / "paired_fits.json"
    def checkpoint():
        receipt.write_text(json.dumps(provenance, indent=2) + "\n")

    fits = {}
    candidates = {p.parent for base in (root,run.parent) for p in base.glob("09D_*/status.json")}
    candidates = sorted(candidates,key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)
    for domain in DOMAINS:
        fit = None
        for candidate in candidates:
            try:
                fit = _load_fit(candidate, domain, spec, original, geometry, inputs, package_version)
            except (OSError, KeyError, ValueError, EOFError) as exc:
                provenance["skipped_incompatible"].append({"folder":str(candidate), "reason":str(exc)})
            if fit is not None:
                break
        reused = fit is not None
        if not reused:
            path, archive, status = run_pilot(result_root=root, run_dir=run, study_root=study_root,
                temp_project_root=temp_project_root, overrides=overrides, domain=domain,
                background_genes=background_genes, padding_um=padding_um, display_progress=display_progress)
            if status["state"] != "complete":
                provenance["fits"][domain] = dict(status=status, folder=str(path), review_zip=str(archive))
                checkpoint()
                raise RuntimeError(f"09D {domain} fit did not complete. {status.get('reason','')} See {path}.")
            fit = _load_fit(path, domain, spec, original, geometry, inputs, package_version)
            if fit is None:
                raise ValueError(f"The new 09D fit does not match the requested settings or input provenance: {path}")
        fits[domain] = fit
        diag = fit["diag"]
        provenance["fits"][domain] = dict(
            folder=str(fit["folder"]), reused=reused, assignment_method=fit["source"]["assignment_method"],
            n_buffered_squares=int(fit["source"]["n_tile_bins"]), n_buffered_cells=diag["n_cells"],
            n_matched_core=len(fit["cells"]), n_background_tiles=diag["n_empty_bins"],
            background_tiles_with_counts=diag["n_model_background_bins_with_selected_UMIs"],
            spatial_decay_um=diag["lambda_estimated"], r2_threshold=diag["r2_threshold"], version=diag["version"],
            n_genes_in_fit=diag["n_fit_features"], n_genes_with_reduced_counts=diag["n_genes_with_count_reduction"])
        checkpoint()
        print(f"{domain}: {'reused' if reused else 'fitted'} {fit['folder']}", flush=True)

    if fits[DOMAINS[0]]["diag"]["version"] != fits[DOMAINS[1]]["diag"]["version"]:
        raise ValueError("SPARKLE versions differ between the completed fits.")
    markers, coverage, anchors, common = _compare_groups(fits,original)
    files = ["paired_fits.json","paired_markers.csv","paired_cell_coverage.csv","paired_coreceptors.csv",
             "paired_gene_fits.csv","paired_marker_retention.png"]
    for frame, name in zip((markers,coverage,anchors),files[1:4]):
        frame.to_csv(out / name,index=False)
    frames = [fits[domain]["genes"].assign(domain=domain) for domain in DOMAINS]
    pooled = pd.concat(frames,ignore_index=True)
    pooled["prespecified_marker"] = pooled.gene_symbol.isin(TARGETS)
    pooled.to_csv(out / "paired_gene_fits.csv",index=False)
    gene_sets = [set(frame.gene_id) for frame in frames]
    shared = gene_sets[0] & gene_sets[1]
    full, restricted = (fits[d]["cells"].loc[common] for d in DOMAINS)
    provenance.update(n_original_core=len(original), n_shared_core=len(common), n_shared_fit_features=len(shared),
                      n_union_fit_features=len(gene_sets[0] | gene_sets[1]),
                      n_shared_cells_with_area_increase_in_restricted=int((restricted.square_count > full.square_count).sum()),
                      n_shared_cells_with_raw_depth_increase_in_restricted=int((restricted.hard_UMIs > full.hard_UMIs).sum()))
    source_control = markers.loc[markers.group.eq("macrophage_label") & markers.panel.eq("myeloid")]
    for row in source_control.itertuples():
        provenance["fits"][row.domain]["myeloid_fraction_retained_in_macrophage_labels"] = (
            float(row.fraction_counts_retained) if np.isfinite(row.fraction_counts_retained) else None)
    checkpoint()
    _plot(markers, out / "paired_marker_retention.png")
    return {"files":files, "provenance":provenance}
