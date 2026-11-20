"""Run both native-resolution methods in the saved 09B core, with restartable tile fits."""
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import fcntl
import hashlib
import json
import os
import re
import tempfile
import traceback
import zipfile

import pandas as pd

from run09d import select_run, _write_json
from run09f import _run_r
from run09g import _resolve_r

METHODS = ("sparkle", "spotclean")


def _read(path):
    return json.loads(path.read_text())


def _atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    _write_json(tmp, data)
    os.replace(tmp, path)


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _log_tail(path, limit=16000):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return re.sub(r"(https?://)[^\s/@]+@", r"\1[redacted]@", stream.read().decode(errors="replace"))


def _review(out):
    """Keep the aggregate ZIP small even for hundreds of local model fits."""
    archive = out / f"09H_return_to_chat_{out.name}.zip"
    files = [out / s for s in ("settings.json", "status.json", "progress.csv")]
    status = _read(out/"status.json")
    if status.get("state") == "diagnostic":
        files.append(out/"run_error.txt")
    if status.get("r_setup_error"):
        files.append(out/"r_setup_error.txt")
    files.extend(p for p in (out/"regions").glob("*") if p.suffix in (".csv", ".json"))
    if status.get("verified_pairs") is not None:
        files.append(out/"disk_footprint.png")
        files.extend(p for p in (out/"comparison").glob("*") if p.suffix in (".csv", ".md", ".png", ".json"))
    tmp = archive.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_file():
                z.write(p, p.relative_to(out))
        if (status.get("r_setup") or status.get("r_setup_error")) and (out / "r_setup.log").is_file():
            z.writestr("r_setup_tail.log", _log_tail(out/"r_setup.log"))
        for failure in status.get("failed_attempts", []):
            folder = out / failure
            for name in ("error.txt", "runtime.json", "spotclean.log"):
                if (folder/name).is_file():
                    z.writestr(str(Path(failure)/name), _log_tail(folder/name))
    os.replace(tmp, archive)
    return archive


def _cached(directory, method, digest):
    for done in sorted(directory.glob("try_*/checkpoint.json"), reverse=True):
        try:
            data = _read(done)
            folder = done.parent
            if data["method"] == method and data["input_sha256"] == digest and all(
                _hash(folder / f) == data["outputs"][f] for f in ("cell_counts.npz", "gene_fits.csv", "fit.json")
            ):
                return folder
        except (OSError, ValueError, KeyError):
            pass
    return None


def _public_receipt(folder, method, digest):
    if _read(folder / "fit.json")["input_sha256"] != digest:
        raise ValueError("The fitted input fingerprint differs from the original tile.")
    _atomic_json(folder/"checkpoint.json", dict(method=method, input_sha256=digest, outputs={
        f: _hash(folder/f) for f in ("cell_counts.npz", "gene_fits.csv", "fit.json")}))


def _progress(out, tasks):
    rows = []
    for item in tasks:
        row = {"tile": item["name"], **item["coverage"]}
        for m in METHODS:
            row[m] = item["state"][m]
            row[m+"_reason"] = item["reason"].get(m, "")
            fit = item[m]
            if fit:
                doc = _read(fit / "fit.json")
                row[m+"_seconds"] = doc.get("wall_seconds")
                row[m+"_version"] = doc.get("version")
                if m == "spotclean":
                    row["spotclean_peak_R_MiB"] = doc.get("peak_R_RSS_MiB")
                    fields = ("distal_rate", "bleeding_rate", "iterations_reached_max",
                              "distal_rate_at_constraint", "radius_at_search_boundary")
                else:
                    fields = ("lambda_um", "lambda_at_search_boundary", "targets_passing_gate",
                              "targets_subtracted", "background_bins_with_counts")
                row.update({m + "_" + f: doc.get(f) for f in fields})
        rows.append(row)
    temp = out / "progress.csv.tmp"
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, out/"progress.csv")


def _fit_one(task, method, out, exe, minutes, bin_um, radius_um):
    from models09g import fit_sparkle, fit_spotclean
    base = out / "fits" / task["name"] / method
    base.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="try_", dir=base))
    try:
        if method == "sparkle":
            fit_sparkle(task["input"], attempt, bin_um=bin_um, radius_um=radius_um)
        else:
            fit_spotclean(task["input"], attempt, rscript=exe, fit_minutes=minutes)
        _public_receipt(attempt, method, task["digest"])
        task[method] = attempt
        task["state"][method] = "complete"
        task["reason"].pop(method, None)
    except Exception as exc:
        (attempt/"error.txt").write_text(traceback.format_exc())
        task["failed_attempts"].append(str(attempt.relative_to(out)))
        task["state"][method] = "failed"
        task["reason"][method] = f"{type(exc).__name__}: {exc}"
        print(f"  {method}: {exc}", flush=True)


def _tasks(out, folders):
    from models09g import input_digest
    entries = []
    for folder in folders:
        rec = _read(folder/"roi_receipt.json")
        digest, _ = input_digest(folder)
        item = {"name": folder.name, "input": folder, "digest": digest, "reason": {}, "state": {}, "failed_attempts": [],
                "coverage": {k: rec.get(k) for k in ("n_owned_disk_bins","n_positions","n_background",
                        "n_complete_cells","n_owned_target_cells","center_distance_um","origin_rc")}, "tries": 0}
        for m in METHODS:
            base = out / "fits" / folder.name / m
            fit = _cached(base, m, digest)
            item[m] = fit
            readiness = rec[m+"_readiness"]
            if fit:
                item["state"][m] = "cached"
            elif readiness["ready"]:
                item["state"][m] = "pending"
                item["tries"] += sum(1 for _ in base.glob("try_*"))
            else:
                item["state"][m] = "insufficient_input"
                item["reason"][m] = str(readiness.get("reason") or readiness.get("reasons") or "See tile receipt.")
        entries.append(item)
    return entries


def _draw_coverage(out, tasks):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np

    if not tasks:
        return
    frames = []
    for item in tasks:
        pos = pd.read_csv(item["input"]/"positions.tsv", sep="\t", usecols=["array_row","array_col","in_core","he_tissue_score"])
        pos = pos.loc[pos.in_core.eq(1)].copy()
        pos["availability"] = int(bool(item["sparkle"])) + 2 * int(bool(item["spotclean"]))
        frames.append(pos)
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        return
    if data.duplicated(["array_row","array_col"]).any():
        raise ValueError("A disk position was assigned to more than one tile.")
    r, c = data.array_row.to_numpy(), data.array_col.to_numpy()
    r, c = r-r.min(), c-c.min()
    shape = (r.max()+1, c.max()+1)
    def raster(values):
        grid = np.full(shape, np.nan, float)
        grid[r,c] = values
        return grid
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), layout="constrained")
    extent = (-1, 2*shape[1]-1, 2*shape[0]-1, -1)
    colors = ["#dedbd4", "#338376", "#d79353", "#354f78"]
    stain = axes[0].imshow(raster(data.he_tissue_score), extent=extent, cmap="viridis",
                           vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(stain, ax=axes[0], shrink=.73, label="Stored H&E tissue score")
    axes[0].set_title("Saved 09B evaluation core")
    axes[1].imshow(raster(data.availability), extent=extent, cmap=ListedColormap(colors),
                   vmin=-.5, vmax=3.5, interpolation="nearest")
    axes[1].set_title("Completed local fits")
    axes[1].legend(handles=[Patch(color=c, label=l) for c,l in zip(colors,
                    ["Neither", "SPARKLE", "SpotClean", "Both"])], ncols=2, frameon=False,
                    loc="upper center", bbox_to_anchor=(.5,-.17))
    for ax in axes:
        ax.set(xlabel="Native column offset (µm)", ylabel="Native row offset (µm)")
    fig.savefig(out/"disk_footprint.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def run_09h(*, result_root, study_root, temp_project_root, run_dir=None, overrides=None,
            rscript=None, side_bins=64, halo_bins=6, score_max=.10, require_sr_background=False,
            zarr_table="square_002um", max_new_tiles=None, fit_minutes=60,
            sparkle_bin_um=12., sparkle_radius_um=50., run_tag=""):
    """Evaluate the saved 09B core. Matching snapshots and completed estimates are reused."""
    if max_new_tiles is not None and (type(max_new_tiles) is not int or max_new_tiles < 1):
        raise ValueError("max_new_tiles must be None or a positive integer.")
    if not (0 < fit_minutes < float("inf")):
        raise ValueError("fit_minutes must be positive and finite.")
    root = Path(result_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    options = dict(side_bins=side_bins, halo_bins=halo_bins, score_max=score_max,
                   require_sr_background=require_sr_background, zarr_table=zarr_table)
    try:
        run = select_run(root, run_dir)
        fingerprints = {f: _hash(run/f) for f in ("complete.json","settings.json","local_cell_assignments.csv.gz")}
    except Exception as exc:
        out = Path(tempfile.mkdtemp(prefix="09H_diagnostic_", dir=root))
        (out/"run_error.txt").write_text(traceback.format_exc())
        status = {"state":"diagnostic","reason":str(exc)}
        _atomic_json(out/"status.json",status)
        return out, _review(out), status
    packages = {}
    for name in ("numpy","pandas","scipy","anndata","pyarrow","shapely","zarr","stambient"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    code = {f:_hash(scripts/f) for f in ("core09g.py","models09g.py","roi09f.py","regions09g.py","compare09d.py",
                                          "raw09d.py","bins09d.py","spotclean09f.R","readout09f.py")}
    settings = dict(run_dir=str(run), study_root=str(Path(study_root).expanduser().resolve()),
                    temp_project_root=str(Path(temp_project_root).expanduser().resolve()),
                    overrides={k:str(Path(v).expanduser().resolve()) for k,v in (overrides or {}).items()},
                    core=options, sparkle_bin_um=sparkle_bin_um, sparkle_radius_um=sparkle_radius_um,
                    input_receipts_sha256=fingerprints, model_source_sha256=code,
                    python_packages=packages, run_tag=str(run_tag))
    key = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:16]
    out = root / ("09H_fixed_" + key)
    out.mkdir(exist_ok=True)
    with (out/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another kernel is running this saved target in {out}.") from exc
        try:
            return _execute(out, root, run, settings, rscript, fit_minutes, max_new_tiles)
        except Exception as exc:
            (out/"run_error.txt").write_text(traceback.format_exc())
            status = {"state":"diagnostic","reason":str(exc)}
            _atomic_json(out/"status.json",status)
            return out, _review(out), status


def _execute(out, root, run, settings, rscript, minutes, limit):
    from core09g import prepare_09b_core
    from compare09g import compare_regions

    regions = out/"regions"
    status = {"state":"running","started_utc":datetime.now(timezone.utc).isoformat(),
              "per_tile_R_limit_minutes":minutes, "max_new_tiles":limit}
    _atomic_json(out/"settings.json",settings)
    _atomic_json(out/"status.json",status)
    tasks = []
    interrupted = False
    try:
        if not (regions/"selection.json").is_file():
            staging = Path(tempfile.mkdtemp(prefix="preparing_", dir=out))
            prepare_09b_core(staging, run_dir=run, study_root=settings["study_root"],
                                        temp_project_root=settings["temp_project_root"], overrides=settings["overrides"],
                                        **settings["core"])
            if not (staging/"selection.json").is_file():
                raise RuntimeError("The tile preparation did not produce a coverage receipt.")
            os.rename(staging, regions)
        manifest = _read(regions/"selection.json")
        folders = [regions / name for name in manifest["selected_regions"]]
        tasks = _tasks(out, folders)
        pending = [t for t in tasks if "pending" in t["state"].values()]
        # A timed first batch should include paired fits when both models have usable input.
        # This priority only changes processing order within the original disk.
        pending.sort(key=lambda item: (item["tries"], "insufficient_input" in item["state"].values(),
                                       item["coverage"]["center_distance_um"], item["name"]))
        batch = pending if limit is None else pending[:limit]
        status.update(total_tiles=len(tasks), pending_at_start=len(pending), tiles_in_this_batch=len(batch))
        print(f"Fixed 09B target · {len(tasks)} tiles, {len(pending)} with unfinished work. "
              f"This batch will process {len(batch)}.", flush=True)
        _progress(out,tasks)
        _atomic_json(out/"status.json",status)
        _review(out)
        exe = None
        if not tasks:
            status.update(state="no_computable_tiles",reason="See regions/coverage.json for the full disk denominator and unavailable tiles.")
        if any(x["state"]["spotclean"] == "pending" for x in batch):
            try:
                exe, version = _resolve_r(rscript, run, root)
                info = {"executable":exe,"version":version}
                _run_r([exe,"--vanilla",str(Path(__file__).with_name("setup_09f_spotclean.R"))],out/"r_setup.log",info)
                status["r_setup"] = info
            except Exception as exc:
                exe = None
                status["r_setup_error"] = str(exc)
                (out/"r_setup_error.txt").write_text(traceback.format_exc())
        for i,t in enumerate(batch,1):
            print(f"[{i}/{len(batch)}] {t['name']}",flush=True)
            for m in METHODS:
                if t["state"][m] != "pending":
                    continue
                if m == "spotclean" and not exe:
                    t["state"][m] = "setup_unavailable"
                    t["reason"][m] = status.get("r_setup_error","R unavailable")
                    continue
                _fit_one(t,m,out,exe,minutes,settings["sparkle_bin_um"],settings["sparkle_radius_um"])
            _progress(out,tasks)
            if t["spotclean"]:
                elapsed = _read(t["spotclean"]/"fit.json").get("wall_seconds")
                if elapsed is not None:
                    print(f"  SpotClean R wall time: {elapsed:.0f} s", flush=True)
            status["tiles_processed_this_invocation"] = i
            _atomic_json(out/"status.json",status)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        status.update(state="diagnostic",reason=str(exc))
        (out/"run_error.txt").write_text(traceback.format_exc())
    if tasks and status["state"] != "diagnostic":
        try:
            _progress(out,tasks)
            result = compare_regions(tasks,out/"comparison")
            _draw_coverage(out,tasks)
            status.update(verified_pairs=result["n_verified_pairs"], total_tiles=len(tasks),
                     estimable_pairs=sum(all(t["state"][m] != "insufficient_input" for m in METHODS) for t in tasks),
                     model_states={m:pd.Series([t["state"][m] for t in tasks]).value_counts().to_dict() for m in METHODS})
            unfinished = any(any(s in ("pending","setup_unavailable","failed") for s in t["state"].values()) for t in tasks)
            if interrupted:
                status["state"] = "interrupted"
            elif unfinished or status["verified_pairs"] < status["estimable_pairs"]:
                status["state"] = "partial"
            else:
                status["state"] = "finished_with_coverage_report"
        except (Exception, KeyboardInterrupt) as exc:
            status.update(state="diagnostic",reason=f"Readout: {exc}")
            (out/"run_error.txt").write_text(traceback.format_exc())
    elif interrupted:
        status["state"] = "interrupted"
    status["failed_attempts"] = [path for task in tasks for path in task["failed_attempts"]]
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out/"status.json",status)
    archive = _review(out)
    print(f"{status['state']} · Results: {out}",flush=True)
    return out, archive, status
