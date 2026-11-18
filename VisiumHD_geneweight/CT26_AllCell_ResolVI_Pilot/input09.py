"""Read one CT26 sample from 08A/08F project records or an explicit all-cell H5AD."""
from pathlib import Path
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem


COORDS = (("_proseg_x_um", "_proseg_y_um"), ("spatial_x_um", "spatial_y_um"))
SECTIONS = ("section_id", "tissue_section", "section")

def _json(path):
    return json.loads(Path(path).read_text())

def _path(value, root):
    return (root / Path(value).expanduser()).resolve()

def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def _stamp(path):
    file = Path(path)
    s = file.stat()
    return [str(file.resolve()), s.st_size, s.st_mtime_ns]

def _source(sample, project_root):
    start = Path(project_root or os.environ.get("CT26_PROJECT_ROOT") or Path.cwd()).expanduser().resolve()
    for root in (start, *start.parents):
        cfg, inp = root/"ct26_analysis_config.json", root/"inputs/pipeline_config.json"
        if cfg.is_file() and inp.is_file():
            run = _path(_json(inp)["paths"]["derived_root"], root)/"08_ct26_joint_scvi"/_json(cfg)["run_name"]
            break
        if (root/"01_density_qc/prepared_samples.csv").is_file():
            run = root
            break
    else:
        raise FileNotFoundError("Set PROJECT_ROOT to the CT26 project code folder or to its 08 analysis run root.")
    prepared = run/"01_density_qc/prepared_samples.csv"
    table = pd.read_csv(prepared, dtype=str, keep_default_na=False)
    match = table.loc[table["sample"] == sample]
    if len(match) != 1:
        raise ValueError(f"Expected one prepared-sample row for {sample!r}. Available samples: {table['sample'].tolist()}")
    row = match.iloc[0]
    raw = _path(row.raw_h5ad, prepared.parent)
    shared = sum(_path(p, prepared.parent) == raw for p in table.raw_h5ad) > 1
    genes = _path(row.genes_parquet, prepared.parent)
    pointer = run/"08FG_v2_annotation_pointer.json"
    annot = _path(_json(pointer)["annotation_root"], run)
    side = annot/"full_cell_metadata"/f"{sample}_all_cells_preliminary_annotation.parquet"
    man = annot/"annotation_manifest.json"
    if _digest(_json(man)) != _json(side.with_suffix(".json")).get("spec", {}).get("annotation_id"):
        raise ValueError(f"The 08F sidecar receipt for {sample} has a different annotation ID.")
    key = str(row.raw_matrix_key).strip("/")
    return raw, key, side, genes, shared, [raw, genes, side, man, side.with_suffix(".json"), prepared, pointer]

def _index(group):
    key = group.attrs.get("_index", "_index")
    return pd.Index(read_elem(group[key]), dtype=str)

def _metadata_frame(group, columns):
    index = _index(group)
    if not index.is_unique:
        raise ValueError(f"Repeated cell IDs in {group.name}.")
    df = pd.DataFrame(index=index)
    for col in columns:
        if col in group:
            df[col] = read_elem(group[col])
    return df

def _coords(frame):
    xy = np.full((len(frame), 2), np.nan)
    for pair in COORDS:
        if set(pair) <= set(frame):
            v = frame[list(pair)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            missing = ~np.isfinite(xy).all(axis=1) & np.isfinite(v).all(axis=1)
            xy[missing] = v[missing]
    return xy

def _section(frame):
    key = next((key for key in SECTIONS if key in frame), None)
    if key is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[key].astype("string").str.strip().replace("",pd.NA)

def _cells(h, sample, side, shared):
    extra = ["sample", "sample_id", "cell_type", "preliminary_annotation", *SECTIONS]
    cols = [*extra, *(c for pair in COORDS for c in pair)]
    raw = _metadata_frame(h["obs"], cols)
    original_n = len(raw)
    side_frame = None
    if side is not None:
        import pyarrow.parquet as pq
        wanted = {"source_cell_id", *cols}
        selected = [k for k in pq.read_schema(side).names if k in wanted]
        side_frame = pd.read_parquet(side, columns=selected)
        side_frame = side_frame.loc[side_frame["sample"].astype(str) == sample].copy()
        if side_frame.empty or side_frame["source_cell_id"].isna().any():
            raise ValueError(f"No valid 08F sidecar identifiers for {sample}.")
        side_frame.index = pd.Index(side_frame["source_cell_id"], dtype=str)
        if not side_frame.index.is_unique or not side_frame.index.isin(raw.index).all():
            raise ValueError("Duplicate 08F cell IDs or unmatched source cells in the selected sample.")
    sample_key = next((k for k in ("sample", "sample_id") if k in raw), None)
    if sample_key:
        member = raw[sample_key].astype("string").eq(sample).fillna(False).to_numpy(dtype=bool)
    elif shared:
        member = raw.index.isin(side_frame.index)
    else:
        member = np.ones(len(raw), bool)
    if side_frame is not None and not side_frame.index.isin(raw.index[member]).all():
        raise ValueError("Raw sample identifiers conflict with matched 08F cells.")
    positions = np.flatnonzero(member)
    raw = raw.iloc[positions]
    if not len(raw):
        raise ValueError(f"No rows for {sample}. Check the sample ID in the input H5AD.")
    xy = _coords(raw)
    if side is None:
        if "obsm/spatial" in h:
            # The explicit standard input route declares obsm['spatial'] to be micrometres.
            node = h["obsm/spatial"]
            if not isinstance(node,h5py.Dataset) or node.shape[0] != original_n or node.shape[1] < 2:
                raise ValueError("Expected a dense, cell-aligned obsm/spatial with two columns in micrometres.")
            xy = np.asarray(node[:, :2], dtype=float)[positions]
        label_key = next((k for k in ("cell_type", "preliminary_annotation") if k in raw), None)
        labels = raw[label_key].astype("string") if label_key else pd.Series("Unknown",index=raw.index,dtype="string")
        sections = _section(raw)
        n_side = 0
        info = "explicit micron-coordinate H5AD"
    else:
        n_side = len(side_frame)
        aligned = side_frame.reindex(raw.index)
        z = _coords(aligned)
        use = np.isfinite(z).all(axis=1)
        xy[use] = z[use]
        labels = aligned["preliminary_annotation"].astype("string")
        sections = _section(aligned).fillna(_section(raw))
        info = "08F source_cell_id matched to raw obs index; 08F/raw micron columns"
    valid = np.isfinite(xy).all(axis=1)
    # Rows with unassigned labels stay in the spatial context if they have a physical centroid.
    if not valid.any():
        raise ValueError("No finite physical x/y coordinates. Use 08F micron columns or a standardized micron-coordinate H5AD.")
    if sections.notna().any() and sections[valid].isna().any():
        raise ValueError("Physical section IDs are missing for a subset of valid cells. Supply a complete section_id column.")
    obs = pd.DataFrame(index=raw.index[valid].rename("cell_id"))
    obs["sample"] = pd.Categorical([sample]*len(obs))
    obs["cell_type"] = pd.Categorical(labels.str.strip().replace("",pd.NA).fillna("Unknown").iloc[valid].astype(str))
    obs["section_id"] = pd.Categorical(sections.fillna(sample).iloc[valid].astype(str))
    audit = {"rows_in_source_h5ad": original_n,"rows_for_sample":len(raw),"rows_in_08F_sidecar":n_side,
             "excluded_invalid_coordinates":int((~valid).sum()),"coordinate_source":info,
             "sample_is_section_fallback":bool(sections[valid].isna().all())}
    return positions[valid], xy[valid], obs, audit

def _genes(h, key, table_path):
    v = h["raw/var" if key.startswith("raw/") else "var"]
    ids = _index(v)
    if table_path is not None:
        frame = pd.read_parquet(table_path, columns=["gene_symbol"])
        frame.index = frame.index.astype(str)
        if not ids.is_unique or not frame.index.is_unique or len(frame) != len(ids) or not ids.isin(frame.index).all():
            raise ValueError("Gene sidecar and count feature IDs differ. Use the raw feature-axis table from 08A.")
        symbols = frame["gene_symbol"].reindex(ids)
    else:
        symbols = pd.Series(read_elem(v["gene_symbol"]) if "gene_symbol" in v else ids)
    symbols = symbols.astype("string").str.strip()
    if symbols.isna().any() or symbols.eq("").any():
        raise ValueError("All measured source features need a nonempty gene symbol.")
    mapping, unique = pd.factorize(symbols, sort=False)
    return mapping, pd.Index(unique.astype(str), name="gene")

def _read_block(node, left, right, n_genes):
    if isinstance(node,h5py.Dataset):
        return sparse.csr_matrix(node[left:right,:])
    ptr = node["indptr"][left:right+1].astype(np.int64)
    a,b = int(ptr[0]),int(ptr[-1])
    return sparse.csr_matrix((node["data"][a:b],node["indices"][a:b],ptr-a),shape=(right-left,n_genes))

def _write_matrix(handle, node, positions, mapping, n_genes, nvar, source_shape):
    # Empty CSR scaffold was written with AnnData. Append cell blocks directly into
    # HDF5 to avoid constructing all of X and a second full CSR during file writing.
    group = handle["X"]
    for key in ("data","indices","indptr"):
        del group[key]
    data = group.create_dataset("data",shape=(0,),maxshape=(None,),dtype="int32",chunks=(131072,),compression="lzf")
    col = group.create_dataset("indices",shape=(0,),maxshape=(None,),dtype="int32",chunks=(131072,),compression="lzf")
    ptr = group.create_dataset("indptr",shape=(len(positions)+1,),dtype="int64",chunks=True,compression="lzf")
    totals = np.empty(len(positions), dtype=np.int64)
    nonzeros = np.empty(len(positions), dtype=np.int32)
    step = 4096 if not isinstance(node,h5py.Dataset) else max(1,min(4096,4_000_000//n_genes))
    offset=0
    start=0
    largest_block_bytes=0
    is_identity = np.array_equal(mapping, np.arange(n_genes))
    while start < len(positions):
        left = int(positions[start])
        right = min(source_shape[0],left + step)
        end = int(np.searchsorted(positions, right))
        q = _read_block(node,left,right,n_genes)[positions[start:end]-left]
        values=q.data
        if not np.isfinite(values).all() or (values<0).any() or (values>np.iinfo(np.int32).max).any() or not np.equal(values,np.rint(values)).all():
            raise ValueError("Input includes negative, fractional or nonfinite values. Supply raw observed molecule counts.")
        q=q.astype("int64",copy=False)
        if not is_identity:
            q=sparse.csr_matrix((q.data,mapping[q.indices],q.indptr),shape=(q.shape[0],nvar))
        q.sum_duplicates();q.eliminate_zeros();q.sort_indices()
        if q.nnz and q.data.max() > np.iinfo(np.int32).max:
            raise OverflowError("A collapsed count exceeds the int32 output range.")
        through=offset+q.nnz
        data.resize((through,));col.resize((through,))
        data[offset:through]=q.data
        col[offset:through]=q.indices
        ptr[start:end+1]=offset+q.indptr.astype(np.int64)
        totals[start:end]=np.asarray(q.sum(axis=1,dtype=np.int64)).ravel()
        nonzeros[start:end]=np.diff(q.indptr)
        largest_block_bytes=max(largest_block_bytes,q.data.nbytes+q.indices.nbytes+q.indptr.nbytes)
        offset=through;start=end
    return totals, nonzeros, largest_block_bytes

def prepare_sample(sample, output_root, project_root=None, sample_h5ad=None):
    """Return the prepared observed-count H5AD for one physical sample.

    An explicit all-cell H5AD must have micron coordinates in obsm['spatial'] or
    named micron obs columns, and integer observed counts in layers/counts or X.
    """
    sample=str(sample)
    if not sample or "/" in sample or "\\" in sample or sample in (".",".."):
        raise ValueError("Provide a single valid source sample ID.")
    if sample_h5ad is None:
        raw,key,side,features,shared,files=_source(sample,project_root)
    else:
        raw=Path(sample_h5ad).expanduser().resolve()
        with h5py.File(raw,"r") as h:
            key="layers/counts" if "layers/counts" in h else "X"
        side=features=None;shared=False;files=[raw]
    sources=[_stamp(p) for p in files]
    spec={"sample":sample,"sources":sources,"matrix_key":key,"adapter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    code=_digest(spec)
    output=Path(output_root).expanduser().resolve()/"inputs"
    output.mkdir(parents=True,exist_ok=True)
    final=output/(re.sub(r"[^A-Za-z0-9_.-]","_",sample)+"_"+code[:12])
    def cached():
        receipt=_json(final/"input_summary.json")
        result=final/"all_cells.h5ad"
        if receipt.get("input_identity") != code or result.stat().st_size != receipt.get("output_bytes"):
            raise RuntimeError(f"Incomplete cached sample at {final}. Choose a fresh output directory or restore the files.")
        print(f"{sample}: {receipt['n_cells']:,} cells × {receipt['n_genes']:,} genes; {receipt['total_observed_counts']:,} UMIs; CSR arrays {receipt['csr_gib']:.2f} GiB.")
        return result
    if final.exists():
        return cached()
    tmp=Path(tempfile.mkdtemp(prefix=f".{final.name}-",dir=output))
    try:
        with h5py.File(raw,"r") as h:
            node=h[key]
            if not isinstance(node,h5py.Dataset) and node.attrs.get("encoding-type") not in ("csr_matrix",b"csr_matrix"):
                raise ValueError("Provide CSR or dense H5AD counts. Convert CSC to CSR before this row-streamed preparation.")
            shape=node.shape if isinstance(node,h5py.Dataset) else tuple(node.attrs["shape"])
            mapping,genes=_genes(h,key,features)
            positions,xy,obs,audit=_cells(h,sample,side,shared)
            if shape != (audit["rows_in_source_h5ad"],len(mapping)):
                raise ValueError("Matrix shape differs from its associated cell or feature axis.")
            out=tmp/"all_cells.h5ad"
            a=ad.AnnData(X=sparse.csr_matrix((len(obs),len(genes)),dtype="int32"),obs=obs,var=pd.DataFrame(index=genes))
            a.obsm["spatial"]=xy.astype(np.float32)
            a.uns["spatial_units"]="micrometres"
            a.write_h5ad(out)
            del a
            with h5py.File(out,"r+") as to:
                totals,nonzeros,block_bytes=_write_matrix(to,node,positions,mapping,len(mapping),len(genes),shape)
                # Add compact numeric obs columns without rewriting the complete sparse matrix.
                for key_col,values in {"total_counts":totals,"n_genes_by_counts":nonzeros}.items():
                    ds=to["obs"].create_dataset(key_col,data=values,compression="lzf")
                    ds.attrs["encoding-type"]="array";ds.attrs["encoding-version"]="0.2.0"
                obs_group=to["obs"]
                order=list(obs_group.attrs["column-order"])+["total_counts","n_genes_by_counts"]
                del obs_group.attrs["column-order"]
                obs_group.attrs.create("column-order",order,dtype=h5py.string_dtype())
        if sources != [_stamp(p) for p in files]:
            raise RuntimeError("An upstream input changed during preparation.")
        nnz=int(nonzeros.sum())
        csr_bytes=8*nnz+8*(len(obs)+1)
        receipt=dict(input_identity=code,specification=spec,n_cells=len(obs),n_genes=len(genes),
                     n_original_features=len(mapping),n_collapsed_features=len(mapping)-len(genes),
                     n_sections=int(obs.section_id.nunique()),n_zero_count_cells=int((totals==0).sum()),
                     total_observed_counts=int(totals.sum()),nnz=nnz,output_bytes=out.stat().st_size,
                     csr_gib=csr_bytes/(2**30),estimated_three_csr_copies_gib=3*csr_bytes/(2**30),
                     dense_float32_gib=len(obs)*len(genes)*4/(2**30),
                     largest_collapsed_csr_block_mib=block_bytes/(2**20),**audit)
        (tmp/"input_summary.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        try: tmp.rename(final)
        except OSError as e:
            if e.errno not in (errno.EEXIST,errno.ENOTEMPTY) or not final.is_dir():raise
    finally:
        if tmp.is_dir():shutil.rmtree(tmp)
    return cached()
