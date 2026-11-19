"""Run SpotClean in a separate R process and collect the 09F review products."""
from datetime import datetime, timezone
from pathlib import Path
import json
from importlib import metadata
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

from run09d import select_run, _write_json


def find_rscript(explicit=None):
    """Find the R 4.5 executable in the terminal environment or registered R kernels."""
    if explicit:
        choices = [shutil.which(str(explicit)) or str(Path(explicit).expanduser())]
    else:
        choices = [shutil.which("Rscript")]
        if os.environ.get("R_HOME"):
            choices.append(Path(os.environ["R_HOME"]) / "bin/Rscript")
        try:
            from jupyter_client.kernelspec import KernelSpecManager
            for spec in KernelSpecManager().get_all_specs().values():
                args = spec["spec"].get("argv", [])
                if args and Path(args[0]).name == "R":
                    path = shutil.which(args[0]) or args[0]
                    choices.append(Path(path).parent / "Rscript")
        except (ImportError, OSError):
            pass
    found = []
    for path in dict.fromkeys(str(x) for x in choices if x):
        if Path(path).is_file() and os.access(path, os.X_OK):
            version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
            info = (version.stdout + version.stderr).strip()
            if version.returncode == 0 and re.search(r"\b4\.5\.\d+", info):
                return path, info
            found.append(f"{path} ({info})")
    raise FileNotFoundError("Set RSCRIPT in the notebook to the Rscript next to your R 4.5 executable. "
                            "In your working R kernel, run file.path(R.home('bin'), 'Rscript'). "
                            f"Other versions found: {found}")


def _run_r(args, log, metrics, timeout=None):
    """Collect per-process wall time and POSIX high-water RSS, including fast exits."""
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    last_update = started
    with log.open("wb") as handle:
        proc = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = time.monotonic() - started
                if timeout is not None and elapsed > timeout:
                    raise TimeoutError(f"The R stage reached the configured {timeout / 60:g} minute limit.")
                if time.monotonic() - last_update >= 30:
                    print(f"R is running ({elapsed / 60:.1f} min). Log: {log}", flush=True)
                    last_update = time.monotonic()
                time.sleep(.25)
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
            metrics.update(wall_seconds=round(time.monotonic() - started, 3),
                           peak_R_RSS_MiB=round(usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024), 2),
                           exit_code=proc.returncode)
    if proc.returncode:
        raise RuntimeError(f"R exited with code {proc.returncode}. See {log.name}.")


def _plot_score_map(inputs, target):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    pos = pd.read_csv(inputs / "positions.tsv", sep="\t", dtype={"cell_id": str})
    pos["role"] = 0
    pos.loc[pos.in_tissue.eq(0), "role"] = 1
    pos.loc[pos.in_tissue.eq(1) & pos.cell_id.notna(), "role"] = 2
    def grid(column):
        return pos.pivot(index="array_row", columns="array_col", values=column).sort_index().sort_index(axis=1)

    score = grid("he_tissue_score")
    edge = [-1, 2 * score.shape[1] - 1, 2 * score.shape[0] - 1, -1]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), layout="constrained")
    palette = plt.get_cmap("viridis").copy()
    palette.set_bad("#cccccc")
    m = axes[0].imshow(score, extent=edge, vmin=0, vmax=1, cmap=palette, interpolation="nearest")
    fig.colorbar(m, ax=axes[0], shrink=.72, label="Combined staining score")
    axes[0].set_title("Stored H&E tissue score")
    if score.isna().any().any():
        axes[0].legend(handles=[Patch(color="#cccccc", label="Unmeasured")], frameon=False,
                       bbox_to_anchor=(.5, -.22), loc="upper center", fontsize=8)
    colors = ["#eadfc9", "#395e72", "#b46b43"]
    axes[1].imshow(grid("role"), extent=edge, cmap=ListedColormap(colors),
                   vmin=-.5, vmax=2.5, interpolation="nearest")
    axes[1].set_title("SpotClean inputs")
    names = ["Other possible tissue", "Provisional background", "Proseg assigned"]
    axes[1].legend(handles=[Patch(color=c, label=l) for c,l in zip(colors, names)],
                   bbox_to_anchor=(.5, -.22), loc="upper center", ncols=2, frameon=False, fontsize=8)
    for ax in axes:
        ax.set(xlabel="Native column offset (µm)", ylabel="Native row offset (µm)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _latest_paired(run):
    for path in sorted(run.parent.glob("09E_*/paired_fits.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            recorded = json.loads(path.read_text())["parameters"]["run_dir"]
            if Path(recorded).expanduser().resolve() == run.resolve():
                return path
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return None


def run_09f(*, result_root, study_root, temp_project_root, run_dir=None, rscript=None, overrides=None,
            side_bins=40, halo_bins=4, score_max=.1, min_background_fraction=.25, origin_rc=None,
            fit_minutes=30, paired_fits=None, require_sr_background=False, zarr_table="square_002um"):
    """Return the result folder, compact ZIP and stage status, retaining partial diagnostics."""
    root = Path(result_root).expanduser().resolve()
    root.mkdir(exist_ok=True, parents=True)
    out = Path(tempfile.mkdtemp(dir=root, prefix="09F_"))
    inputs, fit, summary = (out / s for s in ("input", "fit", "summary"))
    for folder in (inputs, fit, summary):
        folder.mkdir()
    scripts = Path(__file__).resolve().parent
    stats = {"started_utc": datetime.now(timezone.utc).isoformat(), "stages": {}}
    stages = stats["stages"]
    _write_json(out / "settings.json", dict(side_bins=side_bins, halo_bins=halo_bins, score_max=score_max,
                min_background_fraction=min_background_fraction, origin_rc=origin_rc, fit_minutes=fit_minutes,
                require_sr_background=require_sr_background, zarr_table=zarr_table,
                run_dir=run_dir, study_root=study_root, temp_project_root=temp_project_root, overrides=overrides))
    packages = {}
    for pkg in ("anndata", "zarr", "shapely", "numpy", "pandas", "scipy", "h5py", "pyarrow", "matplotlib"):
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None
    files = ("run09f.py", "roi09f.py", "readout09f.py", "spotclean09f.R", "setup_09f_spotclean.R")
    _write_json(out / "environment.json", dict(python=sys.version, packages=packages,
                source_sha256={f: hashlib.sha256((scripts/f).read_bytes()).hexdigest()
                               for f in files if (scripts/f).is_file()}))
    run = None

    def stage(name, fn):
        print(f"\n{name}", flush=True)
        record = stages[name] = {"state": "running"}
        start = time.monotonic()
        try:
            result = fn(record)
            record["state"] = "complete"
            return result
        except KeyboardInterrupt:
            record["state"] = "cancelled"
            raise
        except Exception as exc:
            detail = name + "_error.txt"
            (out / detail).write_text(traceback.format_exc())
            record.update(state="failed", reason=str(exc), detail=detail)
            print(str(exc), flush=True)
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - start, 3)
            _write_json(out / "status.json", stats)

    def install(record):
        exe, version = find_rscript(rscript)
        record.update(Rscript=exe, R_version=version)
        _run_r([exe, "--vanilla", str(scripts / "setup_09f_spotclean.R")],
               out / "install.log", record)
        return exe

    def prepare(record):
        nonlocal run
        from roi09f import prepare_roi
        run = select_run(root, run_dir)
        stats["run_09B"] = str(run)
        receipt = prepare_roi(inputs, run_dir=run, study_root=study_root, temp_project_root=temp_project_root,
                           overrides=overrides, side_bins=side_bins, halo_bins=halo_bins, score_max=score_max,
                           min_background_fraction=min_background_fraction, origin_rc=origin_rc,
                           require_sr_background=require_sr_background, zarr_table=zarr_table)
        print(f"Selected {receipt['n_positions']:,} squares, {receipt['n_background']:,} provisional background, "
              f"{receipt['n_zero_positions']:,} with zero measured UMIs.", flush=True)
        return receipt

    def run_fit(record):
        if not 0 < float(fit_minutes) < float("inf"):
            raise ValueError("Choose a finite positive FIT_MINUTES.")
        _run_r([exe, "--vanilla", str(scripts / "spotclean09f.R"), str(inputs), str(fit)],
               out / "fit.log", record, timeout=60 * float(fit_minutes))

    def summarize(record):
        from readout09f import make_readout
        paired = Path(paired_fits).expanduser() if paired_fits else _latest_paired(run)
        result = make_readout(inputs, fit if fit_ok else None, summary, paired_fits=paired)
        if isinstance(result, dict):
            _write_json(summary / "readout_summary.json", result)
        return result

    try:
        exe = stage("R_setup_and_smoke", install)
        stage("score_and_native_counts", prepare)
        roi_ok = stages["score_and_native_counts"]["state"] == "complete"
        if exe and roi_ok:
            stage("SpotClean_fit", run_fit)
        else:
            stages["SpotClean_fit"] = {"state": "skipped", "reason": "Requires R setup and a usable native window."}
        fit_ok = stages["SpotClean_fit"]["state"] == "complete"
        if roi_ok:
            stage("score_map", lambda _: _plot_score_map(inputs, summary / "score_and_roles.png"))
            stage("marker_readout", summarize)
        else:
            stages["marker_readout"] = {"state": "skipped", "reason": "Native window unavailable."}
    except KeyboardInterrupt:
        stages["interruption"] = {"state": "cancelled", "reason": "Interrupted in the notebook."}
    states = [item["state"] for item in stages.values()]
    stats["state"] = "complete" if all(s == "complete" for s in states) else (
        "partial" if "complete" in states else "diagnostic")
    stats["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(out / "status.json", stats)
    archive = out / f"09F_return_to_chat_{out.name[4:]}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(out.glob("*.json")) + sorted(out.glob("*_error.txt")):
            bundle.write(file, file.name)
        for file in sorted(out.glob("*.log")):
            # Store only recent diagnostics; a source installation can log thousands of lines.
            with file.open("rb") as stream:
                stream.seek(max(file.stat().st_size - 64000, 0))
                logtail = stream.read().decode(errors="replace")
            logtail = re.sub(r"(https?://)[^\s/@]+@", r"\1[redacted]@", logtail)
            bundle.writestr(file.name, logtail)
        for file in sorted(summary.iterdir()):
            if file.is_file() and file.suffix in {".json", ".csv", ".tsv", ".md", ".png"}:
                bundle.write(file, "summary/" + file.name)
        for file in [inputs / "roi_receipt.json", inputs / "score_search.csv", fit / "fit.tsv",
                     fit / "gene_status.tsv", fit / "likelihood.tsv", fit / "sessionInfo.txt"]:
            if file.is_file():
                bundle.write(file, file.name)
    return out, archive, stats
