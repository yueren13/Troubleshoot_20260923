"""Create a compact return ZIP from the current or an existing completed run."""
from html import escape
import json
import os
from pathlib import Path
import tempfile
import zipfile

Y_FILES = (
    "observed_coreceptors_by_mouse.csv", "discovery_mouse_counts.csv",
    "reference_detection_by_mouse.csv", "discovery_settings.json", "signatures.json",
    "gene_map.png", "gene_map_evidence.png", "gene_module_summary.csv",
    "anchor_neighbors_top50.csv", "lineage_candidate_audit.csv.gz", "gene_map.csv.gz",
)
Z_FILES = (
    "08Z_complete.json", "frozen_thresholds.json", "panel_coverage.csv",
    "anchor_validation_by_mouse_depth.csv", "depth_matched_validation.csv",
    "comparator_auc_by_mouse.csv", "lineage_coverage_by_mouse.csv", "state_summary_by_mouse.csv",
    "calibration_grid.csv", "heldout_anchor_performance.png", "lineage_coverage_by_mouse.png",
)


def _completed(root, stage, exact):
    if exact is None:
        pointer = root / f"latest_{stage}.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"Missing {pointer}. Set the results directory to the one containing your completed run.")
        path = root / json.loads(pointer.read_text())["directory"]
    else:
        path = Path(exact).expanduser().resolve()
    receipt = json.loads((path / "complete.json").read_text())
    if receipt.get("stage") != stage:
        raise ValueError(f"Expected a completed {stage} result in {path}.")
    return path, receipt


def return_zip(results, *, stage, destination=None, ydir=None, zdir=None):
    """Write a portable ZIP beside the notebook. For 08Z find its particular 08Y run."""
    root = Path(results).expanduser().resolve()
    destination = Path(destination or Path.cwd()).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if stage == "08Z":
        zdir, z = _completed(root, "08Z", zdir)
        yrun = z["key"]["discovery_run"]
        ydir, y = _completed(root, "08Y", ydir or root / f"08Y_{yrun}")
        if y["run"] != yrun:
            raise ValueError("Selected discovery run differs from the one used for this 08Z run.")
        items = [("08Y", ydir, Y_FILES), ("08Z", zdir, Z_FILES)]
        stem = f"08YZ_return_to_chat_{z['run']}"
    elif stage == "08Y":
        ydir, y = _completed(root, "08Y", ydir)
        items = [("08Y", ydir, Y_FILES)]
        stem = f"08Y_return_to_chat_{y['run']}"
    else:
        raise ValueError("stage must be 08Y or 08Z")
    target = destination / (stem + ".zip")
    handle, tempname = tempfile.mkstemp(prefix="."+stem, dir=destination)
    os.close(handle)
    try:
        with zipfile.ZipFile(tempname, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
            html = ["<!doctype html><meta charset=utf-8><title>CT26 T-cell results</title>",
                    "<style>body{font:16px system-ui;max-width:1100px;margin:3em auto}img{max-width:100%}</style>",
                    "<h1>CT26 T-cell results</h1>"]
            manifest = []
            for tag, directory, files in items:
                html.append(f"<h2>{tag}</h2>")
                for name in files:
                    path = directory / name
                    if not path.is_file():
                        # Plotting can be disabled for a custom 08Z execution.
                        if tag == "08Z" and name.endswith(".png"):
                            continue
                        raise FileNotFoundError(f"Expected aggregate output {path}")
                    member = f"{tag}/{name}"
                    bundle.write(path, member)
                    html.append(f'<p><a href="{escape(member)}">{escape(name)}</a></p>')
                    if name.endswith(".png"):
                        html.append(f'<img alt="{escape(name)}" src="{escape(member)}">')
                    manifest.append(member)
            bundle.writestr("index.html", "\n".join(html))
            bundle.writestr("contents.json", json.dumps(manifest, indent=2))
        os.replace(tempname, target)
    finally:
        Path(tempname).unlink(missing_ok=True)
    return target
