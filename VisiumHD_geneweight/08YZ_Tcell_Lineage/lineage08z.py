"""Observed T-cell programs, masked UCell scores and mouse-held-out lineage validation.

Positive signatures use the UCell >= 2.7.6 normalization, average tied ranks and
missing_genes='skip'. Unobserved genes receive max_rank. Source and equations:
https://github.com/carmonalab/UCell and https://doi.org/10.1093/bioinformatics/btag055
"""

import json
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


CORE = {"cd4", "cd8a", "cd8b1"}
PROGRAMS = {
    "panT": ["Cd3d", "Cd3e", "Cd3g", "Trac", "Trbc1", "Trbc2"],
    "cytotoxic": ["Prf1", "Nkg7", "Gzma", "Gzmb", "Gzmk", "Ctsw"],
    "naive_memory": ["Tcf7", "Lef1", "Ccr7", "Sell", "Il7r", "Klf2"],
    "regulatory": ["Foxp3", "Il2ra", "Ctla4", "Ikzf2", "Tnfrsf18"],
    "checkpoint": ["Pdcd1", "Havcr2", "Lag3", "Tigit", "Tox", "Entpd1"],
    "interferon": ["Isg15", "Ifit1", "Ifit3", "Ifitm3", "Mx1", "Stat1", "Irf7"],
    "cycling": ["Mki67", "Top2a", "Cenpf", "Ube2c", "Ccnb1", "Pclaf"],
    "TCR_activation": ["Nr4a1", "Nr4a2", "Egr1", "Egr2", "Cd69", "Icos"],
    "gamma_delta": ["Trdc", "Trgc1", "Trgc2"],
    "NK_associated": ["Ncr1", "Klrd1", "Klrb1c", "Tyrobp"],
    "myeloid_context": ["Lyz2", "Csf1r", "Adgre1", "C1qa", "C1qb", "Lst1"],
    "epithelial_context": ["Epcam", "Krt8", "Krt18", "Krt19"],
}
LINEAGES = ("CD4", "CD8")


def _write_csv(frame, path, index=False):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=index, compression="gzip" if path.suffix == ".gz" else None)
    tmp.replace(path)


def _write_json(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ucell(X, panels, rank_cap, excluded, chunk_size):
    """CSR positive-entry ranks; cap-average ties after ranking across all available columns."""
    names = list(panels)
    sizes = np.array([len(panels[k]) for k in names])
    cols = [j for k in names for j in panels[k]]
    signature_index = np.repeat(np.arange(len(names)), sizes)
    membership = sparse.csr_matrix(
        (np.ones(len(cols), np.float32), (cols, signature_index)),
        shape=(X.shape[1], len(names)),
    )
    sum_min = sizes * (sizes + 1.) / 2
    denom = sizes * rank_cap - sum_min
    score = np.full((X.shape[0], len(names)), np.nan, np.float32)
    hits = np.zeros(score.shape, np.int16)
    if not len(cols):
        return score, hits
    for lo in range(0, X.shape[0], chunk_size):
        block = X[lo:lo + chunk_size]
        rows = np.repeat(np.arange(block.shape[0], dtype=np.int32), np.diff(block.indptr))
        good = (block.data > 0) & ~excluded[block.indices]
        rr, cc, vv = rows[good], block.indices[good], block.data[good]
        n = len(rr)
        if n == 0:
            score[lo:lo + block.shape[0], sizes > 0] = 0
            continue
        # Ascending counts within each row. For a tie [a,b), its descending
        # average rank is row_end - (a+b)/2 + 0.5 in zero-based flat indices.
        order = np.lexsort((vv, rr))
        rs, values = rr[order], vv[order]
        starts = np.r_[0, np.flatnonzero((np.diff(rs) != 0) | (values[1:] != values[:-1])) + 1]
        ends = np.r_[starts[1:], n]
        row_end = np.cumsum(np.bincount(rs, minlength=block.shape[0]))
        avg_rank = row_end[rs[starts]] - (starts + ends) / 2. + .5
        weights = np.repeat(np.maximum(0, rank_cap - avg_rank), ends - starts)
        ranked = sparse.csr_matrix((weights.astype(np.float32), (rs, cc[order])), shape=block.shape)
        sub = (ranked @ membership).toarray()
        valid = sizes > 0
        score[lo:lo + block.shape[0], valid] = (sub[:, valid] / denom[valid]).astype(np.float32)
        ranked.data[:] = 1  # Includes positive transcripts whose rank was capped.
        hits[lo:lo + block.shape[0]] = (ranked @ membership).toarray().astype(np.int16)
    return score, hits


def verify_rank_math():
    """Independent dense average-rank reference, including cap-crossing ties and missing sets."""
    a = np.array([[3, 2, 2, 2, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 1, 1, 0, 1]])
    panels = {"a": [0, 3], "b": [4], "absent": []}
    excluded = np.array([False, True, False, False, False, False, False])
    scores, hits = _ucell(sparse.csr_matrix(a), panels, 3, excluded, 1)
    for row, cell in enumerate(a):
        pos = (cell > 0) & ~excluded
        ranks = np.full(len(cell), 3.)
        ranks[pos] = np.minimum(rankdata(-cell[pos], method="average"), 3)
        for k, ix in enumerate(list(panels.values())[:2]):
            n = len(ix)
            want = 1 - (ranks[ix].sum() - n*(n+1)/2) / (3*n - n*(n+1)/2)
            np.testing.assert_allclose(scores[row, k], want, atol=1e-7)
            assert hits[row, k] == np.sum(cell[ix] > 0)
    assert np.isnan(scores[:, 2]).all()


def _wilson_upper(k, n, confidence=.95):
    if n == 0:
        return np.nan
    z = NormalDist().inv_cdf(confidence)
    p = k / n
    return (p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def _safe_auc(y, values):
    good = np.isfinite(values)
    if np.unique(y[good]).size != 2:
        return np.nan
    return roc_auc_score(y[good], values[good])


def _metrics(y, hit):
    pos, neg = y, ~y
    tp, fp, np_, nn = int(np.sum(pos & hit)), int(np.sum(neg & hit)), int(pos.sum()), int(neg.sum())
    tpr, fpr = (tp / np_ if np_ else np.nan), (fp / nn if nn else np.nan)
    return dict(n_target=np_, n_opposite=nn, correct_calls=tp, opposite_calls=fp,
                recovery=tpr, opposite_call_rate=fpr,
                balanced_precision=tpr/(tpr+fpr) if tpr+fpr > 0 else np.nan)


def _pooled_rule(rows, min_anchors, min_mice, target_error, min_balanced_precision, min_correct):
    good = [r for r in rows if min(r["n_target"], r["n_opposite"]) >= min_anchors]
    if len(good) < min_mice:
        return False, "insufficient eligible mice"
    nwrong = sum(r["opposite_calls"] for r in good)
    nopp = sum(r["n_opposite"] for r in good)
    sensitivity = float(np.mean([r["recovery"] for r in good]))
    error = float(np.mean([r["opposite_call_rate"] for r in good]))
    bprec = sensitivity / (sensitivity + error) if sensitivity+error else 0
    if error > target_error or _wilson_upper(nwrong, nopp) > target_error:
        return False, "opposite calls exceed target or insufficient negative support"
    if max(r["opposite_call_rate"] for r in good) > 2 * target_error:
        return False, "an eligible mouse exceeds twice the error target"
    if bprec < min_balanced_precision:
        return False, "low equal-prior anchor precision"
    if sum(r["correct_calls"] for r in good) < min_correct or sum(r["correct_calls"] > 0 for r in good) < min_mice:
        return False, "too few replicated correct calls"
    return True, "passed"


def _depth_bins(depth, discovery_neither):
    source = depth[discovery_neither]
    if not len(source):
        return np.zeros(len(depth), dtype=int), []
    edges = np.unique(np.quantile(source, [.25, .50, .75])).tolist()
    return np.searchsorted(edges, depth, side="right"), edges


def _anchor_tables(index, labels, depth_bins, groups, samples, condition, unresolved, score, depth, eligible, calls, lin):
    rows, targets = [], []
    for mouse in groups:
        mi = samples == mouse
        im = np.where(index & mi)[0]
        ui = np.where(unresolved & mi & eligible)[0]
        for b in [-1, *range(int(depth_bins.max()) + 1)]:
            ix = im if b == -1 else im[depth_bins[im] == b]
            y = labels[ix] == lin
            pop = len(ui) if b == -1 else int((depth_bins[ui] == b).sum())
            metrics = _metrics(y, calls[ix])
            rows.append(dict(sample=mouse, condition=condition[mi][0], lineage=lin,
                             depth_bin="all" if b == -1 else str(b), unresolved_n=pop,
                             auc_margin=_safe_auc(y, score[ix]), **metrics))
            if b != -1 and pop:
                targets.append((mouse, b, pop, metrics))
    return rows, targets


def _plot(metrics, coverage, out):
    shown = metrics[(metrics.depth_bin == "all") & (metrics.split == "evaluation")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    names = list(dict.fromkeys(shown["sample"].astype(str)))
    position = {name: i for i, name in enumerate(names)}
    for lin, marker in zip(LINEAGES, ["o", "^"]):
        m = shown[shown.lineage == lin]
        x = m["sample"].map(position).to_numpy(dtype=float)
        axes[0].scatter(x, m.recovery, marker=marker, label=lin, s=55)
        axes[1].scatter(x, m.opposite_call_rate, marker=marker, label=lin, s=55)
    for ax, title in zip(axes, ["Recovery of masked RNA anchors", "Opposite-lineage calls on masked anchors"]):
        ax.set_title(title)
        ax.set_ylabel("Fraction of corresponding anchors")
        ax.set_ylim(bottom=-.015)
        ax.set_xticks(np.arange(len(names)), names, rotation=35)
        ax.legend(loc="upper right")
    fig.savefig(out/"heldout_anchor_performance.png", dpi=180)
    plt.close(fig)
    post = coverage.pivot_table(index="sample", columns="deployed_lineage", values="cells", aggfunc="sum", fill_value=0, observed=True)
    ax = post.div(post.sum(axis=1), axis=0).plot.bar(stacked=True, figsize=(11, 5), width=.82)
    ax.set_ylabel("Fraction of original T-cell compartment")
    ax.set_title("Direct evidence, validated inferred calls and unresolved cells")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.figure.tight_layout()
    ax.figure.savefig(out/"lineage_coverage_by_mouse.png", dpi=180)
    plt.close(ax.figure)


def run(X, obs, genes, signatures, out, *, max_rank=None, chunk_size=1024,
        min_hits=2, min_panT=1, target_error=.02, min_balanced_precision=.80,
        min_anchors_per_class=20, min_discovery_mice=3, min_evaluation_mice=2,
        min_correct=15, allow_unknown_mixed=False, make_plots=True):
    """Score all cells, calibrate on discovery mice, evaluate frozen thresholds, then gate deployment.

    Opposite-call rate = opposite-class anchors called target / all eligible opposite
    anchors. Equal-prior precision uses mouse-balanced recovery and opposite-call rate.
    The pooled one-sided Wilson bound is a guard on the observed anchor counts, not a
    between-mouse confidence interval. Subtype program scores are continuous.
    """
    verify_rank_math()
    if not sparse.issparse(X) or X.format != "csr":
        raise ValueError("Pass a CSR matrix of nonnegative observed full-assay counts.")
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
        X.sort_indices()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "08Z_complete.json"
    receipt.unlink(missing_ok=True)
    cols = pd.Index([str(g).casefold() for g in genes])
    gene_names = np.asarray(genes, dtype=str)
    if not cols.is_unique or not obs.index.is_unique or X.shape != (len(obs), len(cols)):
        raise ValueError("Gene symbols and cell IDs must be unique and aligned with the matrix.")
    if not len(obs) or obs[["sample", "condition", "split"]].isna().any().any():
        raise ValueError("Sample, condition and split must be present for every cell.")
    if min(min_hits, min_panT, chunk_size, min_discovery_mice, min_evaluation_mice, min_correct, min_anchors_per_class) < 1:
        raise ValueError("Size, coverage and replication parameters must be positive.")
    if not 0 < target_error < .5 or not .5 <= min_balanced_precision <= 1:
        raise ValueError("Check target_error and min_balanced_precision.")
    if not CORE.issubset(set(cols)):
        raise ValueError("Cd4, Cd8a and Cd8b1 must all be present in the measured gene universe.")
    meta = obs[["sample", "condition", "split"]].astype(str).drop_duplicates()
    if not meta["sample"].is_unique or set(meta["split"]) != {"discovery", "evaluation"}:
        raise ValueError("Use disjoint discovery/evaluation mice with one condition per mouse.")
    sample = obs["sample"].astype(str).to_numpy()
    condition = obs["condition"].astype(str).to_numpy()
    split = obs["split"].astype(str).to_numpy()
    discovery = split == "discovery"
    groups = {s: sorted(meta.loc[meta.split == s, "sample"]) for s in ["discovery", "evaluation"]}
    mixed = obs.get("mixed", pd.Series(pd.NA, index=obs.index)).astype("boolean")
    allowed_mixture = (~mixed.fillna(not allow_unknown_mixed)).to_numpy(dtype=bool)

    requested = dict(PROGRAMS)
    for lin in LINEAGES:
        key = lin + "_support"
        if key not in signatures or not isinstance(signatures[key], (list, tuple)):
            raise ValueError(f"08Y signatures.json lacks a gene list for {key}.")
        requested[key] = list(dict.fromkeys(str(g) for g in signatures[key]))
        if CORE.intersection(g.casefold() for g in requested[key]):
            raise ValueError(f"Remove coreceptor genes from learned panel {key}.")
    if set(g.casefold() for g in requested["CD4_support"]) & set(g.casefold() for g in requested["CD8_support"]):
        raise ValueError("The two lineage-specific support panels must be disjoint.")
    panels = {}
    for k, names in requested.items():
        cols_found = cols.get_indexer([g.casefold() for g in names])
        panels[k] = list(dict.fromkeys(int(j) for j in cols_found if j >= 0))
    core_cols = dict(zip(["Cd4", "Cd8a", "Cd8b1"], cols.get_indexer(["cd4", "cd8a", "cd8b1"])))
    core_mask = np.asarray(cols.isin(CORE))
    n_detected = np.zeros(X.shape[0], dtype=np.int32)
    masked_ngenes = n_detected.copy()
    depth = np.empty(X.shape[0], dtype=float)
    masked_depth = np.empty_like(depth)
    measured = np.empty((X.shape[0], 3), dtype=np.float32)
    for start in range(0, X.shape[0], chunk_size):
        sl = slice(start, min(start+chunk_size, X.shape[0]))
        b = X[sl]
        if not np.isfinite(b.data).all() or np.any(b.data < 0):
            raise ValueError("Nonfinite/negative matrix values encountered. Supply observed counts.")
        b.eliminate_zeros()
        measured[sl] = b[:, list(core_cols.values())].toarray()
        n_detected[sl] = np.diff(b.indptr)
        masked_ngenes[sl] = n_detected[sl] - (measured[sl] > 0).sum(axis=1)
        depth[sl] = np.asarray(b.sum(axis=1, dtype=np.float64)).ravel()
        masked_depth[sl] = np.maximum(0, depth[sl] - measured[sl].sum(axis=1))
    max_len = max(len(v) for v in panels.values())
    max_possible = X.shape[1] - 3
    auto_cap = max(max_len + 1, int(np.rint(np.median(masked_ngenes[discovery]))), 2)
    cap = min(max_possible, auto_cap) if max_rank is None else int(max_rank)
    if not max_len < cap <= max_possible:
        raise ValueError(f"max_rank must exceed largest measured signature ({max_len}) and be <= {max_possible}.")
    _write_csv(pd.DataFrame([{"panel": k, "requested": ",".join(requested[k]),
                              "measured": ",".join(gene_names[panels[k]]),
                              "n_requested": len(requested[k]), "n_measured": len(panels[k]),
                              "effective_max_rank": cap} for k in panels]), out/"panel_coverage.csv")

    states = {k: panels[k] for k in PROGRAMS}
    st_scores, st_hits = _ucell(X, states, cap, np.zeros(len(cols), bool), chunk_size)
    state = {k: st_scores[:, i] for i, k in enumerate(states)}
    nhits = {k: st_hits[:, i] for i, k in enumerate(states)}
    # Re-rank after withholding all three coreceptors. Their expression cannot change
    # a non-coreceptor rank, the margin or the masked-depth baseline.
    masked_panels = {k: panels[k] for k in ["CD4_support", "CD8_support", "cytotoxic", "naive_memory"]}
    ms, mh = _ucell(X, masked_panels, cap, core_mask, chunk_size)
    ranked_support = {lin: ms[:, i] for i, lin in enumerate(LINEAGES)}
    ndetected = {lin: mh[:, i] for i, lin in enumerate(LINEAGES)}
    margin = np.nan_to_num(ms[:, 0], nan=0.) - np.nan_to_num(ms[:, 1], nan=0.)
    margins = {"CD4": margin, "CD8": -margin}
    cd4 = measured[:, 0] > 0
    cd8 = np.any(measured[:, 1:] > 0, axis=1)
    raw = np.select([cd4 & cd8, cd4, cd8], ["both", "CD4", "CD8"], default="neither")
    eligible = (nhits["panT"] >= min_panT) & allowed_mixture
    anchors = eligible & ((raw == "CD4") | (raw == "CD8"))
    neither = raw == "neither"
    depth_bins, boundaries = _depth_bins(masked_ngenes, discovery & neither)
    condition_lookup = dict(zip(meta["sample"], meta["condition"]))
    search_grid = np.linspace(.001, 1., 201)
    thresholds, deployed, gate, cal_rows = {}, {}, {}, []
    bench_rows, weighted_rows, auc_rows = [], [], []
    pass_flags, query_calls = {}, {}
    assay_depth = np.log1p(masked_depth)
    # Learn direction for the univariate depth comparator in the discovery split only.
    tr = discovery & anchors
    au = _safe_auc(raw[tr] == "CD4", assay_depth[tr])
    direction = -1 if np.isfinite(au) and au < .5 else 1

    for lin in LINEAGES:
        side = (ndetected[lin] >= min_hits) & (margins[lin] > 0) & eligible
        enough_panel = len(panels[lin+"_support"]) >= min_hits
        folds = []
        for mouse in groups["discovery"]:
            ids = np.flatnonzero(anchors & (sample == mouse))
            y = raw[ids] == lin
            folds.append((mouse, int(y.sum()), int((~y).sum()),
                          np.sort(margins[lin][ids[side[ids] & y]]),
                          np.sort(margins[lin][ids[side[ids] & ~y]])))
        cutoff = None
        # On this nested score grid, the first acceptable threshold maximizes recovery.
        for th in search_grid if enough_panel else []:
            rows = []
            for mouse, npos, nneg, vp, vn in folds:
                tp = len(vp) - int(np.searchsorted(vp, th))
                fp = len(vn) - int(np.searchsorted(vn, th))
                rows.append(dict(n_target=npos, n_opposite=nneg, correct_calls=tp, opposite_calls=fp,
                                 recovery=tp/npos if npos else np.nan,
                                 opposite_call_rate=fp/nneg if nneg else np.nan))
            ok, why = _pooled_rule(rows, min_anchors_per_class, min_discovery_mice,
                                   target_error, min_balanced_precision, min_correct)
            usable = [r for r in rows if min(r["n_target"], r["n_opposite"]) >= min_anchors_per_class]
            cal_rows.append(dict(lineage=lin, threshold=th, passed=ok, reason=why,
                                 eligible_mice=len(usable),
                                 macro_recovery=np.mean([r["recovery"] for r in usable]) if usable else np.nan,
                                 macro_opposite_rate=np.mean([r["opposite_call_rate"] for r in usable]) if usable else np.nan))
            if ok and cutoff is None:
                cutoff = float(th)
        thresholds[lin] = cutoff
        # The second split only measures the frozen classifier. Release decisions below
        # can veto a lineage; they never refit the scores, thresholds or candidate genes.
        calls = side & (margins[lin] >= (np.inf if cutoff is None else cutoff))
        query_calls[lin] = calls
        rows, targets = _anchor_tables(anchors, raw, depth_bins, groups["evaluation"], sample,
                                       condition, neither, margins[lin], masked_depth, eligible, calls, lin)
        erows = [r for r in rows if r["depth_bin"] == "all"]
        ok, why = _pooled_rule(erows, min_anchors_per_class, min_evaluation_mice, target_error,
                               min_balanced_precision, min_correct)
        pass_flags[lin] = bool(cutoff is not None and ok)
        gate[lin] = (why if cutoff is not None else
                     ("insufficient measured independent panel genes" if not enough_panel else "no discovery threshold passed"))
        deployed[lin] = calls & pass_flags[lin]
        for s in groups:
            if s == "evaluation":
                rm, wt = rows, targets
            else:
                rm, wt = _anchor_tables(anchors, raw, depth_bins, groups[s], sample,
                                        condition, neither, margins[lin], masked_depth, eligible, calls, lin)
            for r in rm:
                r.update(split=s, discovery_threshold=cutoff, passed_evaluation=pass_flags[lin])
            bench_rows.extend(rm)
            for mouse in groups[s]:
                bins_pop = [(pop, d) for m, b, pop, d in wt if m == mouse]
                good = [(pop, d) for pop, d in bins_pop if min(d["n_opposite"], d["n_target"]) >= min_anchors_per_class]
                total_pop, supported = sum(pop for pop, _ in bins_pop), sum(pop for pop, _ in good)
                weighted_rows.append(dict(sample=mouse, condition=condition_lookup[mouse], split=s, lineage=lin,
                                          eligible_neither=total_pop, supported_neither=supported,
                                          supported_depth_fraction=supported/total_pop if total_pop else np.nan,
                                          recovery=sum(pop*d["recovery"] for pop,d in good)/supported if supported else np.nan,
                                          opposite_call_rate=sum(pop*d["opposite_call_rate"] for pop,d in good)/supported if supported else np.nan))
                ids = np.flatnonzero(anchors & (sample == mouse))
                y = raw[ids] == lin
                sign = 1 if lin == "CD4" else -1
                for name, pred in [("masked_direct_only", np.zeros(len(ids))),
                                   ("masked_depth_only", direction*sign*assay_depth[ids]),
                                   ("naive_memory_minus_cytotoxic", sign*(ms[ids,3]-ms[ids,2])),
                                   ("learned_masked_margin", margins[lin][ids])]:
                    auc_rows.append(dict(sample=mouse, condition=condition_lookup[mouse], split=s, lineage=lin,
                                         model=name, n_target=int(y.sum()), n_opposite=int((~y).sum()),
                                         auc=_safe_auc(y, pred)))
    model = dict(engine="sparse UCell positive scores, average ties, missing genes skipped, no smoothing",
                 normalization="sum(max_rank - capped_rank) / (n * max_rank - n*(n+1)/2)",
                 rank_cap=cap, rank_cap_selection="discovery median masked detected genes" if max_rank is None else "explicit",
                 coreceptors_withheld_from_lineage_ranks=sorted(CORE), full_universe_columns=int(X.shape[1]),
                 masked_universe_columns=int(X.shape[1]-3), split_mice=groups, thresholds=thresholds,
                 measured_lineage_genes={lin: gene_names[panels[lin+"_support"]].tolist() for lin in LINEAGES},
                 rank_coverage_rule="The scored denominator contains measured panel genes; per-cell nondetections receive max_rank.",
                 validation="Exclusive raw coreceptor, panT and mixture-eligible RNA anchors. Proxy labels.",
                 ranking_depth_bin_edges=boundaries, negative_error_target=target_error,
                 threshold_grid={"start":.001, "stop":1., "points":201}, min_hits=min_hits,
                 min_panT=min_panT, min_anchors_per_class=min_anchors_per_class,
                 min_discovery_mice=min_discovery_mice, min_evaluation_mice=min_evaluation_mice,
                 min_correct=min_correct, min_balanced_precision=min_balanced_precision,
                 allow_unknown_mixed=allow_unknown_mixed,
                 deployment_gate="Mouse-mean FPR and pooled one-sided 95% Wilson upper bound <= target, each eligible mouse <= 2*target; equal-prior mouse-balanced precision and replicated recovery floors. Gate applied once per lineage; no held-out retuning.",
                 depth_standardization="Per-mouse target weights from eligible neither-detected cells in discovery-defined n_detected quartiles; bins below per-class anchor minimum have no estimate.",
                 absent_panel_margin="If one lineage has no measured candidates, its margin contribution is zero; its own calls are disabled.")
    _write_json(model, out/"frozen_thresholds.json")

    id_columns = [name for name in ("source_cell_id", "population") if name in obs]
    cell = obs[["sample", "condition", "split", *id_columns]].copy()
    cell.index.name = "cell_id"
    cell["mixed_flag"] = mixed
    cell["observed_total"] = depth
    cell["observed_n_genes"] = n_detected
    cell["masked_n_genes"] = masked_ngenes
    cell["masked_UMIs"] = masked_depth
    cell["depth_bin"] = depth_bins
    for j, g in enumerate(core_cols):
        cell[g+"_observed_count"] = measured[:, j]
    cell["T_identity_evidence"] = pd.Categorical(np.where(
        nhits["panT"] >= 2, "multiple_panT", np.where(nhits["panT"] > 0, "single_panT", "no_panT_detected")))
    cell["eligible_panT_mixture"] = eligible
    cell["observed_coreceptors"] = pd.Categorical(raw)
    label = np.full(len(cell), "unresolved", dtype="U25")
    label[raw == "both"] = "both_detected"
    for lin in LINEAGES:
        label[(raw == lin) & eligible] = lin + "_observed"
        label[neither & deployed[lin]] = lin + "_inferred"
        cell[lin+"_support_UCell"] = ranked_support[lin]
        cell[lin+"_support_n"] = ndetected[lin]
        cell[lin+"_margin"] = margins[lin]
        cell[lin+"_frozen_call"] = query_calls[lin]
    cell["deployed_lineage"] = pd.Categorical(label)
    for name in states:
        cell[name+"_UCell"] = state[name]
        cell[name+"_n"] = nhits[name]
    _write_csv(cell, out/"per_cell_annotations.csv.gz", index=True)
    coverage = (cell.groupby(["sample", "condition", "split", "observed_coreceptors", "deployed_lineage"],
                             observed=True).size().rename("cells").reset_index())
    _write_csv(coverage, out/"lineage_coverage_by_mouse.csv")
    state_rows = []
    for mouse in sorted(meta["sample"]):
        mi = sample == mouse
        for nm in states:
            for group, mask in [("all", mi), ("coreceptor_neither", mi & neither)]:
                a = state[nm][mask]
                h = nhits[nm][mask]
                finite = np.isfinite(a)
                state_rows.append(dict(sample=mouse, condition=condition_lookup[mouse], population=group,
                                       program=nm, n_cells=len(a), n_scored=int(finite.sum()),
                                       mean_UCell=float(a[finite].mean()) if finite.any() else np.nan,
                                       fraction_one_or_more=float((h>=1).mean()) if len(h) and len(panels[nm]) else np.nan,
                                       fraction_two_or_more=float((h>=2).mean()) if len(h) and len(panels[nm]) else np.nan))
    _write_csv(pd.DataFrame(state_rows), out/"state_summary_by_mouse.csv")
    _write_csv(pd.DataFrame(cal_rows, columns=["lineage", "threshold", "passed", "reason", "eligible_mice",
                                                "macro_recovery", "macro_opposite_rate"]), out/"calibration_grid.csv")
    anchor_summary = pd.DataFrame(bench_rows)
    _write_csv(anchor_summary, out/"anchor_validation_by_mouse_depth.csv")
    _write_csv(pd.DataFrame(weighted_rows), out/"depth_matched_validation.csv")
    _write_csv(pd.DataFrame(auc_rows), out/"comparator_auc_by_mouse.csv")
    if make_plots:
        _plot(anchor_summary, coverage, out)
    summary = dict(thresholds=thresholds, passed_evaluation=pass_flags, reason=gate, rank_cap=cap,
                   n_cells=len(obs), n_anchor=int(anchors.sum()), n_no_coreceptor=int(neither.sum()),
                   n_eligible_neither=int(np.sum(neither & eligible)),
                   n_inferred={lin:int(np.sum(neither & deployed[lin])) for lin in LINEAGES},
                   per_cell="per_cell_annotations.csv.gz",
                   evaluate="anchor_validation_by_mouse_depth.csv",
                   model="frozen_thresholds.json")
    _write_json(summary, receipt)
    return summary
