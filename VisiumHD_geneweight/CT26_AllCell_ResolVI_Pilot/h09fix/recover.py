"""Reuse a saved 09H disk and its SPARKLE checkpoints; fit SpotClean in a separate directory."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback


def tile_budget(value):
    if value.lower() == "all":
        return None
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("Choose 0, a positive number of tiles, or all.")
    return count


def tile_priority(item, original):
    """This orders pending fits only. Every eligible tile remains in the full run."""
    stats = original.loc[item["name"]]
    support = min(stats["near_panT_cells"], stats["macrophage_cells"])
    return (item["tries"] > 0, item["sparkle"] is None, support < 3, support < 1,
            -item["coverage"]["n_background"], -item["coverage"]["n_complete_cells"], item["name"])


def failure_message(out, task):
    path = out / task["failed_attempts"][-1] / "spotclean.log"
    if path.exists():
        from run09h import _log_tail
        lines = [line.strip() for line in _log_tail(path).splitlines() if line.strip()]
        for i, line in enumerate(lines):
            if line.lower().startswith(("error", "fatal error")):
                # R can put the useful part of its error on the following line.
                return " ".join(lines[i:i+2])[:700]
    return task["reason"]["spotclean"]


def should_pause(message, previous):
    immediate = ("usage: spotclean09f.r", "could not load in ", "there is no package called",
                 "fatal error: cannot open file")
    return any(text in message.lower() for text in immediate) or message == previous


def verified_tasks(source, out, manifest):
    import pandas as pd
    from run09h import _cached, _tasks

    names = manifest["selected_regions"]
    if (manifest["analysis_mode"] != "09B_fixed_core" or not names or len(set(names)) != len(names)
            or any(Path(n).name != n or n in (".", "..", "") for n in names)):
        raise ValueError("Expected the saved 09H fixed disk and its unique prepared tile names.")
    original = pd.read_csv(source/"comparison"/"region_status.csv").set_index("region", verify_integrity=True)
    if set(original.index) != set(names):
        raise ValueError("Original comparison and prepared manifest disagree on the tile names.")

    tasks = _tasks(out, [source/"regions"/name for name in names])
    for t in tasks:
        name, digest = t["name"], t["digest"]
        if original.loc[name, "input_sha256"] != digest:
            raise ValueError(f"{name}: native matrix, axes or cell metadata changed since the original analysis.")
        for method in ("sparkle", "spotclean"):
            if t[method] is None:
                t[method] = _cached(source/"fits"/name/method, method, digest)
            if t[method] is not None:
                t["state"][method] = "cached"
                t["reason"].pop(method, None)
            elif original.loc[name, method] == "verified":
                raise ValueError(f"{name}: previously verified {method} checkpoint is missing or has changed.")
            elif method == "sparkle" and t["state"][method] == "pending":
                t["state"][method] = "missing_checkpoint"
                t["reason"][method] = "No valid saved SPARKLE checkpoint."
    return tasks, original


def repair(source, out, rscript, limit, minutes):
    from compare09g import compare_regions
    from run09f import find_rscript
    from run09h import (_atomic_json, _draw_coverage, _fit_one, _hash, _progress, _read, _review)

    original_status = _read(source/"status.json")
    manifest = _read(source/"regions"/"selection.json")
    bundle = Path(__file__).resolve().parent
    status = dict(state="running", source_run=str(source), max_new_tiles=limit, failed_attempts=[],
                  started_utc=datetime.now(timezone.utc).isoformat())
    # Protect other existing results if a similarly named directory already exists.
    settings_path = out/"settings.json"
    settings = {"source_09h": str(source), "source_selection_sha256": _hash(source/"regions"/"selection.json"),
                "source_status_table_sha256": _hash(source/"comparison"/"region_status.csv"),
                "model_code_sha256": {n: _hash(bundle/n) for n in ("models09g.py","spotclean09f.R")}}
    if settings_path.exists() and _read(settings_path) != settings:
        raise ValueError(f"Different source inputs or model version already occupy {out}.")
    if not settings_path.exists() and any(p.name != ".run.lock" for p in out.iterdir()):
        raise ValueError(f"The proposed recovery output already contains unrelated files: {out}")
    _atomic_json(settings_path, settings)
    _atomic_json(out/"status.json", status)
    tasks = []
    try:
        inputs = out/"regions"
        inputs.mkdir(exist_ok=True)
        for name in ("coverage.json","selection.json","candidates.csv"):
            original = source/"regions"/name
            if original.is_file():
                shutil.copyfile(original, inputs/name)
        tasks, original = verified_tasks(source, out, manifest)
        pending = sorted((t for t in tasks if t["state"]["spotclean"] == "pending"),
                         key=lambda t: tile_priority(t, original))
        batch = pending if limit is None else pending[:limit]
        status.update(total_tiles=len(tasks), pending_at_start=len(pending), planned_this_pass=len(batch),
                      planned_tiles=[t["name"] for t in batch], tiles_attempted=0)
        print(f"{len(tasks)} tiles · {len(pending)} SpotClean fits pending · {len(batch)} planned this pass.",
              flush=True)
        _progress(out, tasks)
        _atomic_json(out/"status.json", status)
        if batch:
            prior = original_status.get("r_setup", {}).get("executable")
            try:
                exe, version = find_rscript(rscript or (prior if prior and Path(prior).is_file() else None))
                status["r_setup"] = dict(executable=exe, version=version)
            except Exception as exc:
                status["r_setup_error"] = str(exc)
                (out/"r_setup_error.txt").write_text(traceback.format_exc())
                batch = []
        last_failure = None
        for t in batch:
            print(f"Fitting SpotClean on {t['name']} · {t['coverage']['n_complete_cells']} complete footprints · "
                  f"{t['coverage']['n_background']} background squares", flush=True)
            _fit_one(t, "spotclean", out, exe, minutes, None, None)
            status["tiles_attempted"] += 1
            _progress(out, tasks)
            if t["state"]["spotclean"] == "failed":
                msg = failure_message(out, t)
                print(msg, flush=True)
                if should_pause(msg, last_failure):
                    status["pause_reason"] = f"Stopped the remaining fits after a shared runtime error: {msg}"
                    break
                last_failure = msg
            else:
                last_failure = None
            _atomic_json(out/"status.json", status)
    except KeyboardInterrupt:
        status["pause_reason"] = "Interrupted; completed tile checkpoints are available."
    except Exception as exc:
        status.update(state="diagnostic", reason=str(exc))
        (out/"run_error.txt").write_text(traceback.format_exc())

    if tasks and status["state"] != "diagnostic":
        try:
            _progress(out, tasks)
            report = compare_regions(tasks, out/"comparison")
            from plots import write_review_plots
            write_review_plots(out/"comparison")
            _draw_coverage(out, tasks)
            import pandas as pd
            validated = pd.read_csv(out/"comparison"/"region_status.csv")
            invalid = [f"{row.region} {m}" for row in validated.itertuples() for m in ("spotclean","sparkle")
                       if str(getattr(row,m)).startswith("failed validation")]
            status.update(verified_pairs=report["n_verified_pairs"], readout_validation_failures=invalid,
                          model_states={m: dict(Counter(t["state"][m] for t in tasks))
                                        for m in ("sparkle", "spotclean")})
            outstanding = any(t["state"]["spotclean"] in ("pending","failed")
                              or t["state"]["sparkle"]=="missing_checkpoint" for t in tasks)
            status["state"] = "partial" if outstanding or invalid else "finished_with_coverage_report"
        except (Exception, KeyboardInterrupt) as exc:
            status.update(state="diagnostic",reason=f"Readout: {exc}")
            (out/"run_error.txt").write_text(traceback.format_exc())
    if status["state"] == "running":
        status["state"] = "interrupted"
    status["failed_attempts"] = [f for t in tasks for f in t["failed_attempts"]]
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out/"status.json", status)
    archive = _review(out)
    print(f"{status['state']} · verified pairs: {status.get('verified_pairs', 0)}", flush=True)
    for key in ("reason","pause_reason","r_setup_error"):
        if status.get(key):
            print(status[key], flush=True)
    print(f"Return ZIP: {archive}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--rscript")
    parser.add_argument("--tiles", type=tile_budget, default=1)
    parser.add_argument("--minutes", type=float, default=60.)
    args = parser.parse_args()
    if not (args.minutes > 0 and math.isfinite(args.minutes)):
        parser.error("--minutes must be a positive finite number.")
    bundle = Path(__file__).resolve().parent
    helpers = args.helpers.expanduser().resolve()
    source = args.run_dir.expanduser().resolve()
    required = [source/"regions"/"selection.json", source/"comparison"/"region_status.csv", source/"status.json"]
    if not all(p.is_file() for p in required):
        parser.error("Point --run-dir at the full saved 09H results directory, including fits/ and regions/.")
    if not (helpers/"readout09f.py").is_file():
        parser.error("Point --helpers at the folder containing the working 09G/09H helper modules.")
    isolated = ("compare09g.py", "models09g.py", "run09h.py", "spotclean09f.R", "plots.py")
    if not all((bundle/n).is_file() for n in isolated):
        parser.error("Extract the complete recovery package. A bundled model or readout helper is missing.")
    sys.path[:0] = [str(bundle), str(helpers)]
    os.environ.setdefault("MPLBACKEND", "Agg")
    out = source.with_name(source.name + "_recovered")
    out.mkdir(exist_ok=True)
    with ExitStack() as stack:
        if (source/".run.lock").is_file():
            source_lock = stack.enter_context((source/".run.lock").open("rb"))
            fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        dest_lock = stack.enter_context((out/".run.lock").open("a"))
        fcntl.flock(dest_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = repair(source, out, args.rscript, args.tiles, args.minutes)
    return int(status["state"] == "diagnostic")


if __name__ == "__main__":
    sys.exit(main())
