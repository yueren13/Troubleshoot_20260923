"""Run the Ada-14R native-square audit and package aggregate evidence for review."""
from pathlib import Path
from importlib import metadata
import json
import tempfile
import traceback
import zipfile


def select_run(result_root, explicit=None):
    if explicit is not None:
        run = Path(explicit).expanduser().resolve()
        if not (run / "complete.json").is_file():
            raise FileNotFoundError(f"Expected the completed 09B receipt at {run / 'complete.json'}")
        return run
    existing = sorted(p.parent for p in Path(result_root).expanduser().glob("09B_*/complete.json"))
    preferred = Path(result_root).expanduser() / "09B_4403598b0ac3"
    if preferred in existing:
        return preferred.resolve()
    if len(existing) == 1:
        return existing[0].resolve()
    historical = Path("/home/domino/reny28/Projects/Visium_projects/TBIO-8110/code/python/"
                      "09CT26_AllCell_ResolVI_Pilot/09_pilot_results/09B_4403598b0ac3")
    if not existing and (historical / "complete.json").is_file():
        return historical
    raise FileNotFoundError(f"Set RUN_DIR to the original completed 09B run directory. Found {existing}")


def _write_json(path, doc):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    path.write_text(json.dumps(doc, indent=2, default=convert) + "\n")


def run_pilot(*, result_root, run_dir=None, study_root, temp_project_root, overrides=None,
              domain="all_in_tissue", run_sparkle=True, padding_um=400., background_genes=128,
              display_progress=True):
    """Return the output directory, archive and status, preserving completed stages on error."""
    outroot = Path(result_root).expanduser().resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="09D_", dir=outroot))
    review_files = []
    def record(name):
        review_files.append(name)
        return out / name

    stage = "configuration"
    status = {"state":"started", "stage":stage}
    try:
        _write_json(record("settings.json"), dict(result_root=outroot, run_dir=run_dir, study_root=study_root,
                    temp_project_root=temp_project_root, overrides=overrides, domain=domain, padding_um=padding_um,
                    run_sparkle=run_sparkle, background_genes=background_genes, sparkle_radius_um=200,
                    sparkle_background_bin_um=50))
        run = select_run(outroot, run_dir)
        receipt = json.loads((run / "complete.json").read_text())
        status.update(run_09B=str(run), sample=receipt["geometry"]["section"])
        stage = "environment and inputs"
        from bins09d import locate_paths, prepare_tile
        from compare09d import summarize, comparisons, fit_panel, plot_audit, PANELS
        import matplotlib.pyplot as plt
        import pandas as pd

        versions = {}
        for p in ("stambient","numpy","scipy","anndata","pandas","h5py","pyarrow","shapely"):
            try:
                versions[p] = metadata.version(p)
            except metadata.PackageNotFoundError:
                versions[p] = None
        _write_json(record("versions.json"), versions)
        paths = locate_paths(status["sample"], study_root, temp_project_root, overrides)
        _write_json(record("path_inventory.json"), {k:{"path":str(p),"exists":p.exists()} for k,p in paths.items()})
        stage = "native counts and segmentation registration"
        print("Reading the native 2-µm counts and registering square centers to the original cells.", flush=True)
        tile = prepare_tile(run, study_root, temp_project_root, padding_um=padding_um,
                            overrides=overrides, domain=domain)
        _write_json(record("source.json"),tile["meta"])
        stage = "sparsity and observed markers"
        summary = summarize(tile)
        _write_json(record("coverage_summary.json"),summary["info"])
        summary["coverage"].to_csv(record("sparsity.csv"), index=False)
        summary["background"].to_csv(record("background_coverage.csv"), index=False)
        observed = comparisons(summary)
        observed.to_csv(record("panel_comparisons.csv"),index=False)
        cells = summary["cells"].copy()
        for name, frame in (("proseg",summary["proseg_panel"]),("hard_mask",summary["raw_panel"])):
            cells = cells.join(frame.add_prefix(name+"_"))
        cells.to_csv(out / "local_matched_cells.csv.gz",index=False)
        gene_symbols = set(tile["genes"].gene_symbol)
        pd.DataFrame([{"panel":name,"requested":",".join(genes),
                       "native_symbols_present":",".join(g for g in genes if g in gene_symbols)}
                      for name,genes in PANELS.items()]).to_csv(record("marker_panel_coverage.csv"),index=False)
        fig = plot_audit(tile,summary)
        fig.savefig(record("native_sparsity_and_background.png"),dpi=155,bbox_inches="tight")
        plt.close(fig)
        print("Native-square comparisons saved.",flush=True)
        if not run_sparkle:
            status.update(state="audit_only", reason="RUN_SPARKLE was disabled.")
        elif versions["stambient"] is None:
            status.update(state="audit_only", reason="stambient is absent. Select the installed CT26 SPARKLE kernel to fit.")
        elif not len(cells):
            status.update(state="audit_only",reason="No 09B evaluation-core cells have native square-center assignments.")
        else:
            stage = "SPARKLE targeted CPU fit"
            print("Fitting SPARKLE in the buffered region. Reporting paired results in the 09B core.",flush=True)
            panel, genes, diag = fit_panel(tile, summary, background_genes=background_genes, radius=200.,
                                            display_progress=display_progress,output_dir=out)
            _write_json(record("sparkle_diagnostics.json"),diag)
            genes.to_csv(record("sparkle_gene_fits.csv"),index=False)
            compared = comparisons(summary,panel)
            compared.to_csv(out / "panel_comparisons.csv",index=False)
            cells.join(panel.add_prefix("sparkle_")).to_csv(out / "local_matched_cells.csv.gz",index=False)
            stage = "marker retention plot"
            _plot_retention(compared, record("marker_retention.png"))
            status.update(state="complete")
        status.update(stage="finished")
    except Exception as error:
        status.update(state="diagnostic", stage=stage,
                      reason=f"{type(error).__name__}: {error}")
        record("traceback.txt").write_text(traceback.format_exc())
    _write_json(record("status.json"),status)
    label = Path(status.get("run_09B","09B")).name
    archive = out / f"09D_return_to_chat_{label}_{out.name[4:]}.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as package:
        for name in review_files:
            file = out / name
            if file.is_file():
                package.write(file, arcname=name)
    return out, archive, status


def _plot_retention(table, path):
    import numpy as np
    import matplotlib.pyplot as plt
    groups = [("panT09C","myeloid","Myeloid in\npan-T cells"),
              ("panT09C","panT","Pan-T in\npan-T cells"),
              ("macrophage_label","myeloid","Myeloid in\nmacrophages"),
              ("B_label","B","B markers in\nB-labeled cells")]
    fig,ax = plt.subplots(figsize=(7.7,3.8), layout="constrained")
    for i,(group,panel,label) in enumerate(groups):
        vals = table[(table.group==group) & (table.panel==panel)].set_index("method")
        if not vals.empty:
            raw = vals.loc["observed_same_hard_mask","total_panel_counts"]
            corrected = vals.loc["SPARKLE_same_hard_mask","total_panel_counts"]
            if np.isfinite(raw) and raw > 0 and np.isfinite(corrected):
                ax.bar(i,corrected/raw,color="#317e82")
                ax.text(i,corrected/raw+.025,f"{corrected/raw:.0%}",ha="center")
                continue
        ax.text(i,.04,"No raw signal",ha="center",fontsize=8)
    ax.set(xticks=range(len(groups)),xticklabels=[v[2] for v in groups],ylim=(0,1.13),
           ylabel="Fraction of hard-mask counts retained",title="Marker retention in the same cells and masks")
    fig.savefig(path,dpi=160)
    plt.close(fig)
