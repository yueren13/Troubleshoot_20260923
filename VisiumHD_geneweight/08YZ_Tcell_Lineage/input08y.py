"""Read observed T-cell counts, mouse splits and 08O/08Q evidence for 08Y/08Z."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def ordered_hash(values):
    """Match the length-prefixed UTF-8 axis fingerprint used in 08V/W/X."""
    digest = hashlib.sha256()
    for value in values:
        data = str(value).encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _locate_root(location):
    start = Path(location or os.getenv("CT26_RUN_ROOT") or Path.cwd()).expanduser().resolve()
    if start.is_file():
        start = start.parent
    for parent in (start, *start.parents):
        if (parent / "14_tcell_ratio_subclustering/08O_latest_result.json").is_file():
            return parent
        config = parent / "ct26_analysis_config.json"
        pipeline = parent / "inputs/pipeline_config.json"
        if config.is_file() and pipeline.is_file():
            derived = _relative(_json(pipeline)["paths"]["derived_root"], parent)
            return derived / "08_ct26_joint_scvi" / _json(config)["run_name"]
    raise FileNotFoundError("Set run_root to the original CT26 code directory (containing ct26_analysis_config.json) or the run directory containing 14_tcell_ratio_subclustering.")


def _project_inputs(root):
    folder = root / "21_tcell_feature_qc_dynamic_weights"
    pinned = folder / "de6ba21e9e5acf0656c8/runs/4fb5d551e1fe0447"
    complete = sorted(folder.glob("*/runs/*/comparison_complete.json"))
    if (pinned / "comparison_complete.json").is_file():
        complete = [pinned / "comparison_complete.json"]
    plans = []
    for file in complete:
        plan = _json(file.parent / "plan.json")
        if str(plan.get("version", "")).startswith("08V-preparation-") and not _json(file).get("failures"):
            plans.append((file.parent, plan))
    if len(plans) != 1:
        raise ValueError(f"Expected one completed 08V reference or the reviewed run; found {len(plans)}. Use counts_h5ad with design_csv for an explicit alternate cohort.")
    run, plan = plans[0]
    data = plan["inputs"]
    path = Path(data["baseline_path"])
    ratio_file = Path(data["ratio_path"])
    index = data["index"]
    screen_dir = _relative(index["ratio_dir"], Path(data["pointer_path"]).parent) if index.get("ratio_dir") else None
    return run, plan, path, ratio_file, screen_dir


def _sample_design(obs, design_csv, strict_count=False):
    design = obs[["sample", "condition"]].drop_duplicates()
    if design["sample"].duplicated().any():
        raise ValueError("Each mouse must have one condition.")
    design = design.set_index("sample").sort_index()
    if design_csv is not None:
        source = pd.read_csv(design_csv, dtype=str)
        source = source.rename(columns={"sample_id": "sample", "sample_split": "split"})
        if not {"sample", "condition", "split"} <= set(source) or source["sample"].duplicated().any():
            raise ValueError("design_csv needs a unique row per mouse with sample, condition and split.")
        source = source.set_index("sample")
        if set(design.index) != set(source.index):
            raise ValueError("The sample set differs from the supplied design CSV.")
        source = source.reindex(design.index)
        if not source.condition.equals(design.condition) or not source.split.isin(["discovery", "evaluation"]).all():
            raise ValueError("Design conditions or split labels disagree with the expression data.")
        if strict_count and "n_cells" in source and not np.array_equal(source.n_cells.astype(int), obs["sample"].value_counts().reindex(design.index)):
            raise ValueError("The 08X design cell counts disagree with the input cohort.")
        design["split"] = source.split
    else:
        # This is the exact 08X algorithm with its original seed.
        for condition, rows in design.groupby("condition"):
            mice = sorted(rows.index, key=lambda s: hashlib.sha256(f"17|{condition}|{s}".encode()).digest())
            if len(mice) < 2:
                raise ValueError(f"Condition {condition!r} needs at least two mice for the 08X split.")
            design.loc[mice, "split"] = ["evaluation"] + ["discovery"] * (len(mice) - 1)
    if set(design.split) != {"discovery", "evaluation"}:
        raise ValueError("Both discovery and evaluation mice are required.")
    return design


def _flags(series):
    values = series.astype("string").str.strip().str.lower()
    valid = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    invalid = set(values.dropna()) - set(valid)
    if invalid:
        raise ValueError(f"Unknown mixed-flag values {sorted(invalid)[:5]}.")
    return values.map(valid).astype("boolean")


def _flag_file(csv, obs, mixed_col):
    cols = pd.read_csv(csv, nrows=0).columns
    col = mixed_col or next((k for k in ("cellassign_mixed_signature", "mixed_signature", "mixed") if k in cols), None)
    id_col = next((k for k in ("obs_name", "source_obs_name", "cell_id") if k in cols), None)
    sample_col = next((k for k in ("sample", "sample_id") if k in cols), None)
    if col not in cols or id_col is None:
        raise ValueError("mixed_csv needs an ID (08Q obs_name or cell_id) and a boolean mixed flag.")
    needed = [id_col, col] + ([sample_col] if sample_col else [])
    table = pd.read_csv(csv, dtype=str, usecols=needed)
    if table[id_col].isna().any():
        raise ValueError("Missing cell identifiers in mixed_csv.")
    source = _flags(table[col])
    ids = pd.Index(table[id_col])
    if id_col == "cell_id" and sample_col and not ids.isin(obs.index).all():
        keys = pd.MultiIndex.from_frame(table[[sample_col, id_col]])
        target = pd.MultiIndex.from_frame(obs[["sample", "source_cell_id"]])
    else:
        keys, target = ids, obs.index
    if not keys.is_unique or not target.is_unique or not keys.isin(target).all():
        raise ValueError("08Q mixed flags do not align uniquely to the loaded cohort by cell identifier.")
    source.index = keys
    data = source.reindex(target)
    if sample_col:
        samples = pd.Series(table[sample_col].to_numpy(), index=keys)
        expected = pd.Series(obs["sample"].to_numpy(), index=target)
        if not np.array_equal(samples.to_numpy(), expected.reindex(keys).to_numpy()):
            raise ValueError("Mouse identifiers attached to 08Q mixed flags differ from the H5AD.")
    return pd.Series(data.array, index=obs.index)


def _read_ratios(path, genes):
    frame = pd.read_csv(path, dtype={"gene": str})
    if not {"ratio", "gene"}.issubset(frame) or frame.gene.isna().any() or frame.gene.duplicated().any():
        raise ValueError("Expected unique genes and 08O ratio in the pooled ratio audit.")
    t = frame.set_index("gene").reindex(genes)
    r = pd.to_numeric(t["ratio"], errors="raise")
    if (r < 0).any():
        raise ValueError("A neighbor/focal ratio cannot be negative.")
    r = r.where(np.isfinite(r))
    for col in ("n_covered", "coverage_fraction", "focal_mean"):
        if col in t:
            r = r.where(pd.to_numeric(t[col], errors="raise") > 0)
    if "ratio_available" in t:
        r = r.where(_flags(t.ratio_available).fillna(False))
    return r


def _environment(metadata, design, ratio_csv, pooled_csv):
    if pooled_csv is not None:
        metadata["env_ratio_pooled"] = _read_ratios(pooled_csv, metadata.index)
    if ratio_csv is None:
        warnings.warn("No per-mouse 08O environment table was found. env_ratio stays missing; pooled ratios are descriptive only.", stacklevel=2)
        return
    required = ["sample", "gene", "n_focal", "n_covered", "focal_mean", "neighbor_mean"]
    frame = pd.read_csv(ratio_csv, usecols=required, dtype={"sample": str, "gene": str})
    discovery_ids = design.index[design.split == "discovery"]
    frame = frame[frame["sample"].isin(discovery_ids)]
    if set(frame["sample"]) != set(discovery_ids):
        raise ValueError("The per-sample environment table is missing discovery mice.")
    frame = frame[frame.gene.isin(metadata.index)].copy()
    if frame.duplicated(["sample", "gene"]).any() or frame.empty:
        raise ValueError("Environment counts need distinct mouse/gene keys matching the assay symbols.")
    for col in required[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if (frame[col] < 0).any():
            raise ValueError(f"Negative {col} in environment data.")
    # Zero focal expression is real information. Retain it in the sums. Exclude
    # absent measurements or mice with no eligible neighborhood at all.
    valid = (np.isfinite(frame[required[2:]]).all(axis=1) & (frame.n_focal > 0)
             & (frame.n_covered > 0) & (frame.n_covered <= frame.n_focal))
    frame = frame.loc[valid].copy()
    frame["focal"] = frame.n_focal * frame.focal_mean
    frame["neighbors"] = frame.n_focal * frame.neighbor_mean
    sums = frame.groupby("gene")[["focal", "neighbors"]].sum()
    metadata["env_ratio"] = (sums.neighbors / sums.focal.where(sums.focal > 0)).reindex(metadata.index)
    metadata["env_ratio_discovery_mice"] = frame.groupby("gene")["sample"].nunique().reindex(metadata.index, fill_value=0)


def _baseline_fingerprint(plan, path):
    """Check the lightweight 08V content fingerprint when its receipt supplies one."""
    match = next((item for item in plan["inputs"].get("source_signatures", [])
                  if Path(item["path"]).resolve() == path.resolve()), None)
    if match is None or not match.get("head_tail_sha256"):
        return None
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk_size = 1024 * 1024
        digest.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(size - chunk_size)
            digest.update(handle.read(chunk_size))
    result = digest.hexdigest()
    if size != match["size"] or result != match["head_tail_sha256"]:
        raise ValueError("08O count-file size or sampled content changed since the 08V run.")
    return result


def _source_stamp(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_manifest(path, data):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path, delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        name = handle.name
    os.replace(name, path / "input_manifest.json")


def open_bundle(*, run_root=None, counts_h5ad=None, design_csv=None, mixed_csv=None, ratio_csv=None,
                out=None, layer="counts", sample_col=None, condition_col="condition", mixed_col=None,
                gene_col=None, pooled_ratio_csv=None):
    """Return (observed CSR, obs, gene index, gene metadata), reading only required H5AD groups.

    run_root accepts the 08O/08V CT26 code or run directory. counts_h5ad is an explicit
    alternative, pre-subset to the intended T-cell cohort, with the complete assay gene
    axis. Raw count X requires explicitly setting layer='X'. ratio_csv is the per-sample
    08O gene_ratios_by_sample_linear.csv.gz; pooled_ratio_csv is descriptive only.
    """
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem

    root = None
    plan = None
    path = Path(counts_h5ad).expanduser().resolve() if counts_h5ad is not None else None
    run = None
    if path is None:
        if layer != "counts" or gene_col is not None:
            raise ValueError("Original project inputs must use layers/counts and the original gene axis.")
        root = _locate_root(run_root)
        run, plan, path, pooled, screen_dir = _project_inputs(root)
        pooled_ratio_csv = pooled_ratio_csv or pooled
        if ratio_csv is None and screen_dir:
            candidate = screen_dir / "gene_ratios_by_sample_linear.csv.gz"
            if candidate.is_file():
                ratio_csv = candidate
        pointer = root / "16_tcell_residuals_cellassign/08Q_latest_result.json"
        if mixed_csv is None and pointer.is_file():
            old = _json(pointer)
            if _relative(old["baseline_h5ad"], pointer.parent) != path.resolve():
                raise ValueError("Latest 08Q used a different baseline H5AD. Set mixed_csv to the matching export.")
            qdir = _relative(old["result_dir"], pointer.parent)
            candidates = [qdir / "cell_metadata.csv.gz", qdir / "cellassign/predictions.csv.gz"]
            mixed_csv = next((p for p in candidates if p.is_file()), None)
    stamp = _source_stamp(path)
    reference_fingerprint = _baseline_fingerprint(plan, path) if plan else None
    key = "X" if layer == "X" else "layers/" + layer
    with h5py.File(path, "r") as h:
        if key not in h:
            raise KeyError(f"Missing {key!r} in {path}. Set layer explicitly for an alternate observed-count assay.")
        obs0 = read_elem(h["obs"])
        var = read_elem(h["var"])
        if not {"data", "indices", "indptr"} <= set(h[key].keys() if isinstance(h[key], h5py.Group) else []):
            raise ValueError("This workflow requires sparse observed counts in the H5AD.")
        X = read_elem(h[key]).tocsr()
        if plan:
            receipt = _json(plan["inputs"]["receipt_path"])
            if "uns/08m/analysis_id" not in h or read_elem(h["uns/08m/analysis_id"]) != receipt.get("analysis_id"):
                raise ValueError("08O count source does not match its completion receipt.")
    original_ids = obs0.index.astype(str)
    genes = pd.Index(var[gene_col] if gene_col else var.index, name="gene")
    if genes.hasnans or not all(isinstance(g, str) and g.strip() for g in genes) or not genes.is_unique:
        raise ValueError("Supply a unique, nonempty gene symbol per column (gene_col if required).")
    if X.shape != (len(obs0), len(genes)) or obs0.index.hasnans:
        raise ValueError("Count dimensions or cell identifiers are invalid.")
    for start in range(0, X.nnz, 3_000_000):
        a = X.data[start:start + 3_000_000]
        if not np.isfinite(a).all() or (a < 0).any() or not (a == np.rint(a)).all():
            raise ValueError("Expected finite, nonnegative integer-valued observed molecule counts.")
    X.sum_duplicates()
    X.eliminate_zeros()
    X.sort_indices()
    sample_col = sample_col or next((col for col in ("sample", "sample_id") if col in obs0), None)
    if sample_col not in obs0 or condition_col not in obs0 or obs0[[sample_col, condition_col]].isna().any().any():
        raise ValueError("Every observation needs sample and condition metadata.")
    obs = pd.DataFrame(index=original_ids)
    obs.index.name = "cell_id"
    obs["sample"] = obs0[sample_col].astype(str).to_numpy()
    obs["condition"] = obs0[condition_col].astype(str).to_numpy()
    obs["source_cell_id"] = obs0["cell_id"].astype(str).to_numpy() if "cell_id" in obs0 else original_ids
    if not original_ids.is_unique:
        obs.index = pd.Index([f"{len(s)}:{s}{i}" for s, i in zip(obs["sample"], original_ids)], name="cell_id")
        if not obs.index.is_unique:
            raise ValueError("Cell IDs repeat within a mouse.")
    if "population" in obs0:
        obs["population"] = obs0.population.astype(str).to_numpy()
    if plan:
        if (len(obs) != int(plan["n_cells"]) or ordered_hash(original_ids) != plan["cell_order_hash"]
                or not original_ids.is_unique or not obs["population"].isin(["T cell 1", "T cell 2"]).all()):
            raise ValueError("T-cell population identity or order differs from the 08V reference.")
    common = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
    full = obs0.total_counts.to_numpy(dtype=float) if "total_counts" in obs0 else common
    if not np.isfinite(full).all() or (full < common).any():
        raise ValueError("Full-library depth is missing or smaller than measured count totals.")
    obs["total_counts"] = full
    obs["n_genes"] = np.diff(X.indptr)
    if plan and not np.array_equal(obs.n_genes.to_numpy(), obs0.n_genes_by_counts.to_numpy()):
        raise ValueError("08O detected-gene counts do not match original CSR.")
    design = _sample_design(obs, design_csv, strict_count=plan is not None)
    obs["split"] = obs["sample"].map(design["split"])
    obs["mixed"] = pd.array([pd.NA] * len(obs), dtype="boolean")
    col = mixed_col or next((k for k in ("cellassign_mixed_signature", "mixed_signature", "mixed") if k in obs0), None)
    if col in obs0:
        obs["mixed"] = _flags(obs0[col]).array
    if mixed_csv is not None:
        flags = _flag_file(mixed_csv, obs, mixed_col)
        conflict = obs.mixed.notna() & flags.notna() & obs.mixed.ne(flags)
        if conflict.any():
            raise ValueError("08Q mixed evidence disagrees between the H5AD and keyed CSV.")
        obs["mixed"] = obs.mixed.fillna(flags)
    elif mixed_col is not None and mixed_col not in obs0:
        raise KeyError(f"mixed_col {mixed_col!r} not found.")
    if obs.mixed.isna().any():
        warnings.warn(f"08Q mixed status unknown for {obs.mixed.isna().sum():,} cells. Unknown is preserved.", stacklevel=2)
    meta = pd.DataFrame({"env_ratio": np.nan, "env_ratio_pooled": np.nan, "env_ratio_discovery_mice": 0}, index=genes)
    _environment(meta, design, ratio_csv, pooled_ratio_csv)
    if stamp != _source_stamp(path):
        raise RuntimeError("Count source changed while the data were being read.")
    manifest = {
        "counts": stamp, "08v_count_file_head_tail_sha256": reference_fingerprint, "assay": key, "input_mode": "08O via 08V" if plan else "explicit pre-subset H5AD",
        "source_08v_run": str(run) if run else None, "source_08v_plan_id": plan["plan_id"] if plan else None,
        "n_cells": len(obs), "n_genes": len(genes), "gene_order_sha256": ordered_hash(genes),
        "cell_order_sha256": ordered_hash(obs.index), "mixed_known": int(obs.mixed.notna().sum()),
        "mixed_unknown": int(obs.mixed.isna().sum()), "mixed_csv": _source_stamp(mixed_csv) if mixed_csv else None,
        "mixed_obs_column": col if col in obs0 else None, "sample_design": design.reset_index().to_dict("records"),
        "per_mouse_ratio_file": _source_stamp(ratio_csv) if ratio_csv else None,
        "pooled_ratio_file": _source_stamp(pooled_ratio_csv) if pooled_ratio_csv else None,
        "environment_definition": "50-um non-T-neighbor/focal-T normalized expression; env_ratio from discovery mice only",
        "depth_definition": "obs.total_counts" if "total_counts" in obs0 else "sum of supplied raw assay",
    }
    obs.attrs["input_manifest"] = manifest
    if out is not None:
        _write_manifest(out, manifest)
    print(f"{len(obs):,} cells × {len(genes):,} genes; {design.split.eq('discovery').sum()} discovery and {design.split.eq('evaluation').sum()} evaluation mice; {obs.mixed.notna().sum():,} known 08Q flags; {meta.env_ratio.notna().sum():,} discovery environment ratios.")
    return X, obs, genes, meta
