"""Coordinate matched native-window fits and write an aggregate review archive."""
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import time
import traceback
import zipfile

from run09d import _write_json, select_run
from run09f import _plot_score_map, _run_r, find_rscript


def _resolve_r(explicit, run, root):
    if explicit:
        return find_rscript(explicit)
    receipts = {p for base in (root, run.parent) for p in base.glob("09F_*/status.json")}
    for path in sorted(receipts, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            status = json.loads(path.read_text())
            step = status["stages"]["R_setup_and_smoke"]
            if step["state"] == "complete" and Path(status["run_09B"]).resolve() == run:
                return find_rscript(step["Rscript"])
        except (OSError, KeyError, ValueError):
            continue
    return find_rscript()


def _archive(out, archive):
    """Exclude native matrices and per-cell files. Replace the current checkpoint atomically."""
    temporary = archive.with_suffix(".zip.tmp")
    keep = set(out.glob("*.json")) | set(out.glob("*_error.txt"))
    keep.update(p for p in (out / "comparison").glob("*")
                if p.is_file() and p.suffix in {".md", ".csv", ".png", ".json", ".tsv"})
    for name in ("selection.json", "candidates.csv", "score_search.csv"):
        keep.add(out / "regions" / name)
    for folder in (out / "regions").glob("region_*"):
        for name in ("roi_receipt.json", "score_and_roles.png"):
            keep.add(folder / name)
    for folder in (out / "models").glob("region_*/*"):
        for name in ("fit.json", "fit.tsv", "gene_fits.csv", "likelihood.tsv", "runtime.json", "shared_plan.tsv"):
            keep.add(folder / name)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(p for p in keep if p.is_file()):
            bundle.write(path, path.relative_to(out))
        for path in sorted(out.rglob("*.log")):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 32000))
                tail = stream.read().decode(errors="replace")
            tail = re.sub(r"(https?://)[^\s/@]+@", r"\1[redacted]@", tail)
            bundle.writestr(str(path.relative_to(out)), tail)
    os.replace(temporary, archive)


def run_09g(*, result_root, run_dir=None, study_root, temp_project_root, overrides=None, rscript=None,
            n_regions=2, side_bins=64, halo_bins=6, score_max=.10, min_background_fraction=.25,
            require_sr_background=False, min_mac_cells=3, min_near_t_cells=3, min_panel_umis=5,
            zarr_table="square_002um", fit_minutes=60, sparkle_bin_um=12., sparkle_radius_um=50., stride_bins=None):
    """Return the run directory, review ZIP and status. A failed fit leaves the other results available."""
    root = Path(result_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="09G_", dir=root))
    archive = out / f"09G_return_to_chat_{out.name[4:]}.zip"
    scripts = Path(__file__).resolve().parent
    settings = dict(run_dir=run_dir, study_root=study_root, temp_project_root=temp_project_root,
                    overrides=overrides, n_regions=n_regions, side_bins=side_bins, halo_bins=halo_bins,
                    score_max=score_max, min_background_fraction=min_background_fraction,
                    require_sr_background=require_sr_background, min_mac_cells=min_mac_cells,
                    min_near_t_cells=min_near_t_cells, min_panel_umis=min_panel_umis,
                    zarr_table=zarr_table, fit_minutes=fit_minutes, sparkle_bin_um=sparkle_bin_um,
                    sparkle_radius_um=sparkle_radius_um, stride_bins=stride_bins)
    _write_json(out / "settings.json", settings)
    versions = {}
    for pkg in ("stambient", "numpy", "pandas", "scipy", "zarr", "pyarrow", "shapely", "h5py", "anndata"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    files = ("regions09g.py", "models09g.py", "compare09g.py", "run09g.py", "spotclean09f.R",
             "setup_09f_spotclean.R", "roi09f.py", "readout09f.py", "compare09d.py", "raw09d.py", "bins09d.py")
    _write_json(out / "environment.json", dict(packages=versions, source_sha256={
        name: hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in files if (scripts/name).is_file()}))
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "stages": {},
              "requested": n_regions, "selected": 0, "paired_fits": 0, "verified_pairs": 0}
    stages = status["stages"]

    def checkpoint():
        _write_json(out / "status.json", status)
        _archive(out, archive)

    def stage(name, action):
        record = stages[name] = {"state": "running"}
        print(f"\n{name}", flush=True)
        start = time.monotonic()
        try:
            value = action(record)
            record["state"] = "complete"
            return value
        except KeyboardInterrupt:
            record["state"] = "interrupted"
            raise
        except Exception as exc:
            detail = name + "_error.txt"
            (out / detail).write_text(traceback.format_exc())
            record.update(state="failed", reason=f"{type(exc).__name__}: {exc}", detail=detail)
            print(record["reason"], flush=True)
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - start, 3)
            checkpoint()

    def prepare(record):
        nonlocal run
        from regions09g import prepare_regions
        if not 0 < float(fit_minutes) < float("inf"):
            raise ValueError("Choose a positive finite fit_minutes for each SpotClean process.")
        run = select_run(root, run_dir)
        status["run_09B"] = str(run)
        inputs = prepare_regions(out / "regions", run_dir=run, study_root=study_root,
                       temp_project_root=temp_project_root, overrides=overrides, n_regions=n_regions,
                       side_bins=side_bins, halo_bins=halo_bins, score_max=score_max,
                       min_background_fraction=min_background_fraction, require_sr_background=require_sr_background,
                       min_mac_cells=min_mac_cells, min_near_t_cells=min_near_t_cells,
                       min_panel_umis=min_panel_umis, zarr_table=zarr_table, stride_bins=stride_bins)
        status["selected"] = len(inputs)
        print(f"{len(inputs)} of {n_regions} requested regions qualified. Candidate counts are in regions/candidates.csv.",
              flush=True)
        for folder in inputs:
            receipt = json.loads((Path(folder)/"roi_receipt.json").read_text())
            check = receipt["primary_readout_preflight"]
            print(f"{Path(folder).name}: {check['n_mac']} macrophage/monocyte cells "
                  f"({check['myeloid_in_mac']} myeloid UMIs), {check['n_near_t']} nearby pan-T-supported cells "
                  f"({check['myeloid_in_near_t']} myeloid UMIs).", flush=True)
        return inputs

    def setup_r(record):
        exe, version = _resolve_r(rscript, run, root)
        record.update(executable=exe, version=version)
        _run_r([exe, "--vanilla", str(scripts / "setup_09f_spotclean.R")], out / "r_setup.log", record)
        return exe

    def fit_one(record, method, folder, destination):
        from models09g import fit_sparkle, fit_spotclean
        if method == "sparkle":
            fit_sparkle(folder, destination, bin_um=sparkle_bin_um, radius_um=sparkle_radius_um)
        else:
            fit_spotclean(folder, destination, rscript=exe, fit_minutes=fit_minutes)
        if not all((destination / name).is_file() for name in ("cell_counts.npz", "fit.json", "gene_fits.csv")):
            raise RuntimeError("The model did not produce its counts and fit receipts.")
        return destination

    run = None
    model_runs = []
    try:
        selected = stage("select_regions", prepare)
        if selected:
            exe = stage("R_setup", setup_r)
            for folder in selected:
                folder = Path(folder)
                label = folder.name
                entry = {"name": label, "input": folder, "sparkle": None, "spotclean": None}
                stage(label + "_score_map", lambda _: _plot_score_map(folder, folder / "score_and_roles.png"))
                for method in ("sparkle", "spotclean"):
                    name = label + "_" + method
                    if method == "spotclean" and not exe:
                        stages[name] = {"state": "skipped", "reason": "R setup did not complete."}
                        continue
                    dest = out / "models" / label / method
                    dest.mkdir(parents=True, exist_ok=True)
                    entry[method] = stage(name, lambda rec: fit_one(rec, method, folder, dest))
                model_runs.append(entry)
                status["paired_fits"] += int(entry["sparkle"] is not None and entry["spotclean"] is not None)
            def compare(_):
                from compare09g import compare_regions
                compared = compare_regions(model_runs, out / "comparison")
                status["verified_pairs"] = compared["n_verified_pairs"]
                return compared
            stage("matched_comparison", compare)
        if stages["select_regions"]["state"] != "complete":
            status["state"] = "diagnostic"
        elif not selected:
            status["state"] = "no_qualifying_regions"
        elif status["verified_pairs"] == n_regions and all(s["state"] == "complete" for s in stages.values()):
            status["state"] = "complete"
        else:
            status["state"] = "partial"
    except KeyboardInterrupt:
        status["state"] = "interrupted"
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint()
        print(f"\n{status['state']}: {out}", flush=True)
    return out, archive, status
