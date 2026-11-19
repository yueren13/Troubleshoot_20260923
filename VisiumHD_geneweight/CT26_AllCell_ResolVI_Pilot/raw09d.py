"""Read native, unfiltered Space Ranger gene counts for a chosen set of 2-µm barcodes."""

import gzip
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def _raw_file(folder, *names):
    for name in names:
        for suffix in (".gz", ""):
            candidate = folder / (name + suffix)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Missing {names} (.gz or plain) in {folder}")


def _open_binary(path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _match_barcodes(barcode_blocks, wanted):
    """Map the requested headers to file columns, retaining only matches across blocks."""
    found = np.full(len(wanted), -1, dtype=np.int64)
    offset = 0
    for headers in barcode_blocks:
        dest = wanted.get_indexer(headers)
        hit = np.flatnonzero(dest >= 0)
        ix = dest[hit]
        if len(ix) != len(np.unique(ix)) or np.any(found[ix] >= 0):
            raise ValueError("A requested barcode occurs more than once in the raw matrix.")
        found[ix] = offset + hit
        offset += len(headers)
    missing = found < 0
    if missing.any():
        sample = wanted[missing][:3].tolist()
        raise ValueError(f"{missing.sum()} of {len(wanted)} requested squares lack raw-matrix "
                         f"barcode headers (examples: {sample}). Confirm barcode namespace, "
                         "coverage and sample; absence cannot establish a measured zero.")
    return found, offset


def _gene_axis(ids, symbols, types):
    mask = np.asarray(types) == "Gene Expression"
    genes = pd.DataFrame({"gene_id": ids, "gene_symbol": symbols}).loc[mask].reset_index(drop=True)
    if genes.empty or genes.gene_id.eq("").any() or not genes.gene_id.is_unique:
        raise ValueError("Expected at least one Gene Expression feature with distinct, nonempty IDs.")
    if genes.gene_symbol.eq("").any():
        raise ValueError("The Gene Expression feature table has an empty gene symbol.")
    return genes, np.flatnonzero(mask)


def _integer_counts(values):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError("Expected numeric raw counts.")
    if values.size and (values.min() < 0 or values.max() >= 2**63):
        raise ValueError("Raw counts must be nonnegative and fit in int64.")
    if values.dtype.kind == "f" and not (np.isfinite(values).all() and np.equal(values,np.floor(values)).all()):
        raise ValueError("Raw counts must be finite, integer-valued observations.")
    return values.astype(np.int64, copy=False)


def _h5_counts(path, wanted):
    with h5py.File(path, "r") as f:
        mat = f["matrix"]
        n_genes, n_spots = map(int, mat["shape"][:])
        feat = mat["features"]
        genes, use = _gene_axis(feat["id"].asstr()[:], feat["name"].asstr()[:],
                                feat["feature_type"].asstr()[:])
        barcodes = mat["barcodes"].asstr()
        blocks = (barcodes[start:start + 131072] for start in range(0, n_spots, 131072))
        cols, total = _match_barcodes(blocks, wanted)
        ptr, ids, data = mat["indptr"], mat["indices"], mat["data"]
        if (total != n_spots or len(mat["barcodes"]) != n_spots or len(feat["id"]) != n_genes
                or len(ptr) != n_spots + 1 or ptr[0] != 0 or ptr[-1] != len(data)
                or len(ids) != len(data) or ptr.dtype.kind not in "iu" or ids.dtype.kind not in "iu"):
            raise ValueError(f"Raw matrix dimensions disagree with its headers or CSC arrays in {path}.")
        # Read pointers in nearby windows so scattered barcodes do not require one HDF5 call each.
        order = np.argsort(cols)
        sorted_cols = cols[order]
        starts = np.empty(len(cols), dtype=np.int64)
        ends = np.empty_like(starts)
        for pos in np.split(np.arange(len(cols)), np.flatnonzero(np.diff(sorted_cols // 32768)) + 1):
            c = sorted_cols[pos]
            p = np.asarray(ptr[c[0]:c[-1]+2], dtype=np.int64)
            if np.any(np.diff(p) < 0):
                raise ValueError("Raw CSC pointers must be nondecreasing.")
            local = c - c[0]
            starts[pos], ends[pos] = p[local], p[local+1]
        widths = ends - starts
        if np.any(starts < 0) or np.any(ends > len(data)) or np.any(widths < 0):
            raise ValueError("Raw CSC pointers fall outside the count vectors.")
        roi_ptr = np.r_[0, np.cumsum(widths, dtype=np.int64)]
        roi_ids = np.empty(roi_ptr[-1], dtype=np.int64)
        roi_data = np.empty(roi_ptr[-1], dtype=np.int64)
        # Coalesce adjacent compressed reads, then keep only requested nonzero entries.
        for pos in np.split(np.arange(len(cols)), np.flatnonzero(np.diff(starts // 131072)) + 1):
            first, last = pos[0], pos[-1]
            src_lo, src_hi = starts[first], ends[last]
            dst_lo, dst_hi = roi_ptr[first], roi_ptr[last + 1]
            if dst_hi == dst_lo:
                continue
            take = np.arange(dst_hi-dst_lo) + np.repeat(starts[pos]-src_lo-(roi_ptr[pos]-dst_lo), widths[pos])
            roi_ids[dst_lo:dst_hi] = ids[src_lo:src_hi][take]
            roi_data[dst_lo:dst_hi] = _integer_counts(data[src_lo:src_hi][take])
        if np.any(roi_ids < 0) or np.any(roi_ids >= n_genes):
            raise ValueError("Raw CSC contains an invalid gene index.")
        raw = sparse.csc_matrix((roi_data, roi_ids, roi_ptr), shape=(n_genes,len(cols)))
        return raw[:, np.argsort(order)][use].tocsr(), genes


def _mex_counts(folder, wanted, chunk_rows):
    features = _raw_file(folder, "features.tsv", "genes.tsv")
    ft = pd.read_csv(features, sep="\t", header=None, dtype=str, na_filter=False)
    if ft.shape[1] < 2:
        raise ValueError(f"Expected stable IDs and gene symbols in {features}.")
    types = ft[2] if 2 in ft else np.full(len(ft), "Gene Expression")
    genes, use = _gene_axis(ft[0], ft[1], types)
    lookup_gene = np.full(len(ft), -1, dtype=np.int64)
    lookup_gene[use] = np.arange(len(use))
    barcode_path = _raw_file(folder, "barcodes.tsv")
    with pd.read_csv(barcode_path, sep="\t", header=None, usecols=[0], dtype=str, na_filter=False,
                     chunksize=131072) as reader:
        cols, n_barcodes = _match_barcodes((chunk[0] for chunk in reader), wanted)
    order = np.argsort(cols)
    src_cols = cols[order]
    mat_path = _raw_file(folder, "matrix.mtx")
    out_i, out_j, out_v = [], [], []
    with _open_binary(mat_path) as stream:
        header = stream.readline().lower().split()
        if header not in ([b"%%matrixmarket",b"matrix",b"coordinate",kind,b"general"]
                          for kind in (b"integer", b"real")):
            raise ValueError(f"Expected a general Matrix Market coordinate count matrix in {mat_path}.")
        dimensions = next((line.split() for line in stream if line.strip() and not line.lstrip().startswith(b"%")), [])
        if len(dimensions) != 3:
            raise ValueError("Missing Matrix Market coordinate dimensions.")
        nrow, ncol, expected_nnz = map(int, dimensions)
        if (nrow,ncol) != (len(ft),n_barcodes) or expected_nnz < 0:
            raise ValueError("MEX dimensions disagree with feature and barcode headers.")
        seen = 0
        # A standard MEX file can have zero stored entries.
        if expected_nnz:
            with pd.read_csv(stream, sep=r"\s+", names=["gene","spot","count"], header=None,
                             comment="%", chunksize=chunk_rows) as chunks:
                for chunk in chunks:
                    seen += len(chunk)
                    ii, jj = chunk.gene.to_numpy(), chunk.spot.to_numpy()
                    if (ii.dtype.kind not in "iu" or jj.dtype.kind not in "iu" or
                            ii.min() < 1 or ii.max() > nrow or jj.min() < 1 or jj.max() > ncol):
                        raise ValueError("Invalid gene or barcode index in raw MEX.")
                    values = _integer_counts(chunk["count"].to_numpy())
                    idx = np.searchsorted(src_cols, jj-1).clip(max=len(cols)-1)
                    match = (src_cols[idx] == jj-1) & (lookup_gene[ii-1] >= 0)
                    if match.any():
                        out_i.append(lookup_gene[ii[match]-1])
                        out_j.append(order[idx[match]])
                        out_v.append(values[match])
        elif any(line.strip() and not line.lstrip().startswith(b"%") for line in stream):
            raise ValueError("MEX contains entries despite an nnz of zero.")
        if seen != expected_nnz:
            raise ValueError(f"MEX declares {expected_nnz} entries but contains {seen}.")
    shape = (len(genes),len(cols))
    if not out_v:
        return sparse.csr_matrix(shape, dtype=np.int64), genes, mat_path
    roi = sparse.coo_matrix((np.concatenate(out_v),(np.concatenate(out_i),np.concatenate(out_j))),shape=shape)
    return roi.tocsr(), genes, mat_path


def read_square_counts(square_dir, barcodes, *, mex_chunk_rows=1_000_000):
    """Return (CSR genes × requested squares, gene_id/gene_symbol frame, source path).

    square_dir is Space Ranger's binned_outputs/square_002um directory. Only raw
    Gene Expression counts enter the result. Columns preserve requested order and
    zero-count columns. Stable feature IDs stay distinct even if symbols repeat.
    A barcode absent from the raw header raises an error. Memory use scales with
    the returned ROI, feature table, and bounded input chunks.
    """
    square_dir = Path(square_dir)
    wanted = pd.Index(barcodes)
    if (wanted.empty or not wanted.is_unique or
            any(not isinstance(b, str) or not b.strip() for b in wanted)):
        raise ValueError("Provide unique, nonempty text barcodes in the desired output order.")
    if not isinstance(mex_chunk_rows, (int, np.integer)) or mex_chunk_rows < 1:
        raise ValueError("mex_chunk_rows must be a positive integer.")
    path = square_dir / "raw_feature_bc_matrix.h5"
    if path.is_file():
        counts, genes = _h5_counts(path, wanted)
    else:
        counts, genes, path = _mex_counts(square_dir/"raw_feature_bc_matrix", wanted, mex_chunk_rows)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    return counts, genes, str(path)
