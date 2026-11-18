"""Run 08Y and 08Z reproducibly from two separate Jupyter kernels."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def source_arguments(**values):
    """Keep the same source usable when 08Z runs in a separate kernel."""
    return {key: str(Path(value).expanduser().resolve()) for key, value in values.items()
            if value is not None and value != ""}


def _include(h, tag, raw):
    h.update(tag.encode() + b"\0")
    h.update(str(len(raw)).encode() + b":")
    h.update(raw)


def input_identity(X, obs, genes, gmeta):
    """Hash the existing sparse arrays without making another counts matrix."""
    if getattr(X, "format", None) != "csr":
        raise TypeError("The input adapter must return a CSR matrix.")
    if X.shape != (len(obs), len(genes)):
        raise ValueError("Input cell or gene dimensions disagree.")
    h = hashlib.sha256()
    for name in ("indptr", "indices", "data"):
        data = np.ascontiguousarray(getattr(X, name))
        h.update((name + str(data.dtype) + str(data.shape)).encode())
        h.update(memoryview(data).cast("B"))
    _include(h, "genes", json_text(list(map(str, genes))).encode())
    # Hash each field separately so metadata (including upstream depth) and ordering are pinned.
    for label, frame in (("obs", obs), ("var", gmeta)):
        h.update(label.encode())
        for key in [None] + list(frame.columns):
            value = frame.index.to_series() if key is None else frame[key]
            hashed = pd.util.hash_pandas_object(value, index=False, categorize=True).to_numpy()
            _include(h, "__index__" if key is None else str(key), memoryview(hashed).cast("B"))
    return h.hexdigest()


def marker_audit(X, obs, genes):
    """Per-sample direct RNA categories. Missing assay genes are explicit."""
    lookup = {str(g).casefold(): i for i, g in enumerate(genes)}
    want = ["Cd4", "Cd8a", "Cd8b1"]
    absent = [g for g in want if g.casefold() not in lookup]
    if absent:
        raise ValueError(f"Missing coreceptor assay genes: {absent}. Check the full observed matrix.")
    values = X[:, [lookup[g.casefold()] for g in want]].toarray() > 0
    cd4, cd8 = values[:, 0], values[:, 1:].any(axis=1)
    labels = np.select([cd4 & cd8, cd4, cd8], ["both", "CD4_only", "CD8_only"], default="neither")
    table = pd.crosstab([obs["sample"], obs["condition"], obs["split"]], labels)
    table = table.reindex(columns=["CD4_only", "CD8_only", "both", "neither"], fill_value=0)
    table["cells"] = table.sum(axis=1)
    table["neither_pct"] = 100 * table["neither"] / table["cells"]
    return table.reset_index()


def _versions():
    names = ["numpy", "scipy", "pandas", "scikit-learn", "anndata", "h5py", "umap-learn",
             "igraph", "leidenalg", "networkx", "matplotlib"]
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _code_sha(names):
    h = hashlib.sha256()
    for name in names:
        _include(h, name, Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def _atomic_json(path, payload):
    handle, raw = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(handle, "w", encoding="utf8") as out:
            out.write(json_text(payload))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _check(directory):
    receipt_path = directory / "complete.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing completed run receipt: {receipt_path}.")
    receipt = json.loads(receipt_path.read_text())
    for item in receipt["files"]:
        file = directory / item["path"]
        if not file.is_file() or file.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Run has a missing or incomplete file: {file}. "
                               "Choose a fresh output directory or restore the missing file.")
    for filename, digest in receipt.get("parameter_file_sha256", {}).items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen parameters changed in {directory / filename}.")
    return receipt


def _stage(root, stage, key, source_args, input_manifest, function):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = hashlib.sha256(json_text(key).encode()).hexdigest()[:20]
    dest = root / f"{stage}_{run}"
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(prefix=f".{stage}_{run}_", dir=root))
        try:
            function(tmp)
            _atomic_json(tmp / "input_manifest.json", input_manifest)
            files = [{"path": p.relative_to(tmp).as_posix(), "bytes": p.stat().st_size}
                     for p in sorted(tmp.rglob("*")) if p.is_file()]
            parameter_names = ["signatures.json", "discovery_settings.json", "frozen_thresholds.json"]
            digests = {name: hashlib.sha256((tmp / name).read_bytes()).hexdigest()
                       for name in parameter_names if (tmp / name).is_file()}
            _atomic_json(tmp / "complete.json", {"stage": stage, "run": run, "key": key,
                         "source_args": source_args, "files": files, "parameter_file_sha256": digests})
            try:
                tmp.rename(dest)
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not dest.is_dir():
                    raise
        finally:
            if tmp.is_dir():
                rmtree(tmp)
    receipt = _check(dest)
    if receipt["key"] != key:
        raise RuntimeError(f"Identity mismatch in {dest}.")
    _atomic_json(root / f"latest_{stage}.json", {"directory": dest.name, "stage": stage})
    return dest


def start_discovery(root, data, source_args, options):
    from genes08y import discover
    X, obs, genes, gmeta = data
    key = {"counts_and_metadata": input_identity(*data), "settings": options,
           "code": _code_sha(["input08y.py", "genes08y.py", "workflow08yz.py"]), "versions": _versions()}
    def execute(path):
        marker_audit(X, obs, genes).to_csv(path / "observed_coreceptors_by_mouse.csv", index=False)
        discover(X, obs, genes, gmeta, path, **options)
        summary = []
        for chunk in pd.read_csv(path / "anchor_rankings.csv.gz", chunksize=50_000):
            summary.append(chunk.loc[chunk["neighbor_rank"] <= 50])
        pd.concat(summary, ignore_index=True).to_csv(path / "anchor_neighbors_top50.csv", index=False)
        if not (path / "signatures.json").is_file():
            raise RuntimeError("No discovery signature specification was produced.")
    return _stage(root, "08Y", key, source_args, obs.attrs.get("input_manifest", {}), execute)


def find_discovery(root, exact=None):
    root = Path(root).expanduser().resolve()
    if exact is None:
        pointer = root / "latest_08Y.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"Run 08Y first or set DISCOVERY_DIR. Missing {pointer}")
        path = root / json.loads(pointer.read_text())["directory"]
    else:
        path = Path(exact).expanduser().resolve()
    receipt = _check(path)
    if receipt["stage"] != "08Y":
        raise ValueError(f"Expected an 08Y result in {path}")
    return path, receipt


def start_scoring(root, data, ydir, yreceipt, options):
    from lineage08z import run
    X, obs, genes, _ = data
    identity = input_identity(*data)
    if identity != yreceipt["key"]["counts_and_metadata"]:
        raise ValueError("Observed counts, row order, gene names, or metadata changed since 08Y. "
                         "Restore the source used in 08Y or rerun discovery on the intended data.")
    raw = (ydir / "signatures.json").read_bytes()
    key = {"counts_and_metadata": identity, "discovery_run": yreceipt["run"],
           "signature_sha256": hashlib.sha256(raw).hexdigest(), "settings": options,
           "code": _code_sha(["input08y.py", "lineage08z.py", "workflow08yz.py"]), "versions": _versions()}
    def execute(path):
        run(X, obs, genes, json.loads(raw), path, **options)
    return _stage(root, "08Z", key, yreceipt["source_args"], obs.attrs.get("input_manifest", {}), execute)


def make_review_zip(ydir, zdir, aggregate_files):
    """Include only explicitly named summary files from the two completed stages."""
    target = zdir.parent / f"08YZ_review_{zdir.name[4:]}.zip"
    handle, name = tempfile.mkstemp(prefix=".review-", dir=target.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for stage, folder in [("08Y", ydir), ("08Z", zdir)]:
                for filename in aggregate_files[stage]:
                    path = folder / filename
                    if path.is_file():
                        archive.write(path, f"{stage}/{filename}")
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return target
