"""Run the four-class QC audit and paired SPARKLE comparison for one completed 09B tile."""
from datetime import datetime, timezone
from importlib import metadata
import gc
import json
from pathlib import Path
import tempfile
import traceback
import zipfile

from run09d import select_run, _write_json


def _capture_failure(out, stages, stage, exc):
    name = f"{stage}_error.txt"
    (out / name).write_text(traceback.format_exc(), encoding="utf-8")
    stages[stage] = {"state":"failed", "reason":f"{type(exc).__name__}: {exc}", "detail":name}


def _capture_observations(tile, ghost):
    import numpy as np

    classes = np.asarray(ghost["qc_class"])
    total = np.asarray(ghost["total_umis"])
    spots = tile["spots"]
    outside = spots.cell_index.to_numpy() < 0
    counts = {}
    for domain, subset in (("all_in_tissue", np.ones(len(spots), bool)),
                           ("qc_retained_in_tissue", spots.qc_keep.to_numpy(dtype=bool))):
        mask = outside & subset
        counts[domain] = {"squares":int(mask.sum()), "UMIs":int(total[mask].sum())}
    candidate = outside & (classes == "tissue-low_transcript-high")
    return {
        "domain_background":counts,
        "tissue_low_transcript_high_outside_squares":int(candidate.sum()),
        "tissue_low_transcript_high_outside_UMIs":int(total[candidate].sum()),
        "all_outside_squares_with_QC_record":int((outside & spots.qc_record_present.to_numpy(bool)).sum()),
    }


def _reconcile(out, observations, paths):
    import csv
    import pandas as pd

    rows = []
    for domain, expected in observations["domain_background"].items():
        reference = pd.read_csv(Path(paths[domain]) / "background_coverage.csv")
        measured = {"squares":int(reference.unassigned_squares.sum()),
                    "UMIs":int(reference.unassigned_UMIs.sum())}
        for name, value in measured.items():
            rows.append(dict(domain=domain, measure=name, native_09E=expected[name], fit_09D=value,
                             agrees=expected[name] == value))
    with (out / "background_reconciliation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if any(not x["agrees"] for x in rows):
        raise ValueError("Native and fitted out-of-mask totals differ. Inspect background_reconciliation.csv and verify the source inputs.")


def _write_readout(out, observations, stages, sample):
    """Present aggregate denominators alongside the two fits for quick review."""
    lines = [f"# {sample} · Tissue QC and local RNA"]
    if observations is not None:
        all_umis = observations["domain_background"]["all_in_tissue"]["UMIs"]
        candidate = observations["tissue_low_transcript_high_outside_UMIs"]
        n = observations["tissue_low_transcript_high_outside_squares"]
        if all_umis:
            lines += ["", f"**{candidate/all_umis:.1%} of out-of-mask RNA** in the buffered region "
                      f"was observed in tissue-low/transcript-high squares "
                      f"({candidate:,} of {all_umis:,} UMIs across {n:,} candidate squares)."]
        else:
            lines += ["", "The buffered region had zero measured out-of-mask UMIs."]
    if stages.get("paired_SPARKLE", {}).get("state") == "complete":
        import pandas as pd
        records = pd.read_csv(out / "paired_markers.csv").set_index(["group","panel","domain"])
        lines += ["", "Each percentage below is corrected panel counts divided by raw panel UMIs "
                  "in the same domain and shared cells.", "",
                  "| Shared core group and marker panel | Cells | All in-tissue | H&E-high |",
                  "| --- | ---: | ---: | ---: |"]
        for group, panel, label in [
            ("panT09C_nonmac_near25","myeloid","Pan-T-supported, near macrophages · myeloid"),
            ("panT09C_nonmac_far50","myeloid","Pan-T-supported, far from macrophages · myeloid"),
            ("macrophage_label","myeloid","Macrophage/monocyte label · myeloid"),
            ("panT09C","panT","Pan-T-supported · pan-T"),
            ("T_label","Cd4","T label · Cd4"), ("T_label","CD8","T label · Cd8a/Cd8b1"),
            ("B_label","B","B label · B markers")]:
            row = records.loc[(group,panel,"all_in_tissue")]
            def value(domain):
                item = records.loc[(group,panel,domain)]
                rate = item.fraction_counts_retained
                if pd.isna(rate):
                    return "No measured raw signal"
                return f"{rate:.1%} of {int(item.raw_panel_UMIs):,} UMIs"
            lines.append(f"| {label} | {int(row.n_shared):,} | {value('all_in_tissue')} | {value('qc_retained_in_tissue')} |")
        lines += ["", "The near/far pan-T groups exclude previous macrophage/monocyte labels. "
                  "See paired_coreceptors.csv for the number of cells positive for either, both, or neither coreceptor."]
    incomplete = [k for k,v in stages.items() if v["state"] != "complete"]
    if incomplete:
        lines += ["", "Stages requiring review: " + ", ".join(incomplete) + ". See status.json for details."]
    (out/"readout.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_09e(*, result_root, run_dir, study_root, temp_project_root, overrides=None,
            padding_um=400., background_genes=128, image_path=None, centers_um=None,
            width_um=160., display_progress=True):
    """Return (local result directory, aggregate review ZIP, stage status)."""
    root = Path(result_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="09E_", dir=root))
    stages = {}
    status = {"started_utc":datetime.now(timezone.utc).isoformat(), "stages":stages}
    observations = None
    run = None
    try:
        _write_json(out/"settings.json", dict(run_dir=run_dir, result_root=root, study_root=study_root,
                    temp_project_root=temp_project_root, overrides=overrides, padding_um=padding_um,
                    background_genes=background_genes, image_path=image_path, centers_um=centers_um,
                    overlay_width_um=width_um))
        versions = {}
        for pkg in ("stambient","numpy","pandas","scipy","anndata","pyarrow","shapely","Pillow","openslide-python","tifffile"):
            try:
                versions[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                versions[pkg] = None
        _write_json(out/"versions.json", versions)
        run = select_run(root, run_dir)
        status["run_09B"] = str(run)
        receipt = json.loads((run/"complete.json").read_text())
        status["sample"] = receipt["geometry"]["section"]
        from bins09d import locate_paths
        paths = locate_paths(status["sample"], study_root, temp_project_root, overrides)
        _write_json(out/"path_inventory.json",{k:{"path":str(p),"exists":p.exists()} for k,p in paths.items()})
        stages["inputs"] = {"state":"complete"}
    except Exception as exc:
        _capture_failure(out, stages, "inputs", exc)

    if stages["inputs"]["state"] == "complete":
        tile = ghost = None
        try:
            from bins09d import prepare_tile
            from ghost09e import analyze_ghost
            print("Reading the native squares and the four saved H&E/transcript classes.", flush=True)
            tile = prepare_tile(run, study_root, temp_project_root, padding_um=padding_um,
                                overrides=overrides, domain="all_in_tissue")
            _write_json(out/"native_source.json", tile["meta"])
            ghost = analyze_ghost(tile, out)
            observations = _capture_observations(tile, ghost)
            _write_json(out/"native_background_summary.json", observations)
            available = ghost["info"]["gradient_status"] == "available"
            stages["native_QC_and_distance"] = {
                "state":"complete" if available else "partial",
                "detail":"ghost_qc.json", "gradient_status":ghost["info"]["gradient_status"]}
        except Exception as exc:
            _capture_failure(out, stages, "native_QC_and_distance", exc)

        if ghost is not None:
            try:
                from he09e import write_he_overlays
                print("Rendering registered H&E crops and raw square counts.", flush=True)
                info = write_he_overlays(tile, ghost["qc_class"], ghost["macrophage_xy"], out,
                                         image_path=image_path, centers_um=centers_um, width_um=width_um)
                stages["histology"] = {"state":info["state"],"detail":"he_overlay_receipt.json"}
            except Exception as exc:
                _capture_failure(out, stages, "histology", exc)
        else:
            stages["histology"] = {"state":"unavailable","reason":"Native QC inputs could not be aligned."}
        del tile, ghost
        gc.collect()

        try:
            from paired09e import compare_domains
            print("Comparing the all-tissue and QC-retained fits in shared evaluation cells.",flush=True)
            paired = compare_domains(out, result_root=root, run_dir=run, study_root=study_root,
                                     temp_project_root=temp_project_root, overrides=overrides,
                                     padding_um=padding_um, background_genes=background_genes,
                                     display_progress=display_progress,
                                     expected_background=observations["domain_background"] if observations else None)
            stages["paired_SPARKLE"] = {"state":"complete","detail":"paired_fits.json"}
        except Exception as exc:
            _capture_failure(out, stages, "paired_SPARKLE", exc)
        if observations is not None and stages["paired_SPARKLE"]["state"] == "complete":
            try:
                fit_paths = {k:v["folder"] for k,v in paired["provenance"]["fits"].items()}
                _reconcile(out, observations, fit_paths)
                stages["background_reconciliation"] = {"state":"complete"}
            except Exception as exc:
                _capture_failure(out, stages, "background_reconciliation", exc)

    try:
        _write_readout(out, observations, stages, status.get("sample", "09E"))
    except Exception as exc:
        _capture_failure(out, stages, "readout", exc)
    if all(s["state"] == "complete" for s in stages.values()):
        status["state"] = "complete"
    elif any(s["state"] in {"complete","partial"} for k,s in stages.items() if k != "inputs"):
        status["state"] = "partial"
    else:
        status["state"] = "diagnostic"
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(out/"status.json", status)
    # Only 09E review products live at this level. 09D per-cell matrices are in their own fit folders.
    archive = out / f"09E_return_to_chat_{run.name if run else '09B'}_{out.name[4:]}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix in {".csv",".json",".png",".txt",".md"} and not p.name.startswith("local_"):
                z.write(p, arcname=p.name)
    return out, archive, status
