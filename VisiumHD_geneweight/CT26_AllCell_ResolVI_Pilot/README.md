# Whole-tissue gene neighborhoods and a one-mouse ResolVI pilot

## Run on the Adagrasib project

1. Extract **CT26_AllCell_ResolVI_Pilot.zip** inside the CT26 project used for 08V. Keep the notebooks, Python modules, and setup script together. If the project is elsewhere, set `PROJECT_ROOT` in the first code cell.
2. Run **09A_All_Cell_Gene_Map.ipynb** for **Ada-14R** in the project's Python kernel. This mouse had 284,041 retained tissue cells in the earlier 08U analysis. The input receipt gives the count under the current pilot's coordinate and sample rules. For a second mouse, set `SAMPLES = ["Ada-14R", "Ada-9"]`. Previously completed runs are reused.
3. Open **09B_ResolVI_One_Sample.ipynb** in the scvi-tools kernel. Use the same output root as 09A so the prepared count input can be reused. The two notebooks can also be run independently. The first 09B fit uses a contiguous region with all available cell types. To fit the entire mouse, change `MODE` to `"full"`. The full and tile runs have different output directories.
4. Each notebook creates a **downloadable return-to-chat ZIP beside the notebook**. 09A includes gene maps and predeclared marker-pair results; 09B includes clustering, marker specificity, spatial controls, model-mixture summaries, and run-size information. The input H5AD, per-cell assignments and latent matrices stay in the working directory.

## Audit the 09B PCA clusters

Run **09C_PCA_Cluster_Audit.ipynb** beside 09B in the same Python kernel. It reads the saved PCA coordinates, cell assignments and prepared count H5AD. It selects the completed Ada-14R pilot `09B_4403598b0ac3` when available. Set `RUN_DIR` to select another completed run. The audit shows the size, RNA depth, detected genes, marker support, and tissue distribution of each original PCA cluster in the evaluation core. It also checks how cluster membership changes at lower Leiden resolutions on a graph rebuilt from the saved coordinates. It creates a downloadable `09C_return_to_chat_*.zip` beside the notebook. Both model fits and the observed cell assignments are reused.

## Native squares and SPARKLE in the 09B region

**Use your TBIO-8057 spatial environment as the starting point.** In a JupyterLab terminal, change to the extracted package folder and run

```bash
bash setup_09d_sparkle.sh --from-python \
  /home/domino/reny28/Projects/Visium_projects/TBIO-8057/env/gpu_env/.venv/bin/python
```

Select **CT26 SPARKLE 0.1.4 (spatial CPU)** in **09D_Native_Bins_and_SPARKLE.ipynb** and choose **Run All**. The installer reads the Python build and versions of the scientific and Jupyter packages used by 09D from the supplied environment. It installs those versions, adds the pilot's tested versions for missing dependencies, and fits a small synthetic dataset with `stambient==0.1.4` before registering the kernel. The new environment has its own packages. The GPU project's other tools, such as CUDA-enabled PyTorch and scVI, remain available through your usual kernel.

**Use the pilot's pinned environment.** Run `bash setup_09d_sparkle.sh` and select **CT26 SPARKLE 0.1.4 (CPU)**. To choose the Python build while using the pilot's package versions, run `CT26_PYTHON=/path/to/python bash setup_09d_sparkle.sh`. This also provides an independent setup if the spatial project's package versions are incompatible with 09D.

The default analysis uses the completed Ada-14R tile `09B_4403598b0ac3` and its prepared H5AD. If you moved the pilot, set `RUN_DIR` to its completed directory and keep the sibling `inputs/` folder with the prepared H5AD.

Download `09D_return_to_chat_*.zip` from the link beside the notebook. It contains the source and status receipts, native coverage and marker tables, two figures after a successful fit, and a traceback for a recoverable error. Cell-level output stays under the new `09_pilot_results/09D_*` directory. The local `sparkle_targeted_counts.npz` stores the targeted observed and corrected gene-by-cell matrices, feature and cell IDs, and a core-cell flag.

### Where the original Ada-14R inputs were recorded

The shared study root is `/stash/data/nonclin/TBIO-8110_VisiumHD-Adagrasib-mouseTumor`. The following paths are relative to it.

| Input | Recorded location |
| --- | --- |
| Native SpatialData | `zarr_files/Ada-14R.zarr`, table `square_002um` |
| Native raw count matrix | `source_files/Ada-14R/outs/binned_outputs/square_002um/raw_feature_bc_matrix/` |
| Native grid and scale | `source_files/Ada-14R/outs/binned_outputs/square_002um/spatial/` |
| Step 00 barcode-aligned QC | `derived_files/tbio8110_stardist_proseg_resolvi_v1/00_config/Ada-14R/Ada-14R_aligned_qc.parquet` |
| Step 02 Proseg cell IDs and polygons | `derived_files/tbio8110_stardist_proseg_resolvi_v1/02_proseg/Ada-14R/` |
| Step 03 assignment receipt | `derived_files/tbio8110_stardist_proseg_resolvi_v1/03_cell_qc_obs_transfer/Ada-14R/Ada-14R_bin_to_cell_assignment_summary.json` |

The Step 03 assignment array was cached at `/host_root/nethome/reny28/Projects/Visium_projects/TBIO-8110/tmp/tbio8110_stardist_proseg_resolvi_v1/03_cell_qc_obs_transfer/Ada-14R/Ada-14R_bin_to_proseg_cell_index.npy`. The notebook verifies its scope and receipt. If it was cleared or covers only QC-retained squares while the selected domain includes additional squares, the notebook rebuilds assignments in the region from the Step 02 polygons. The native input uses raw Space Ranger counts and the exported Step 00 barcode alignment. The historical Zarr location is recorded for inspection. Under a different mount, set `STUDY_ROOT`, `TEMP_PROJECT_ROOT` or individual `PATH_OVERRIDES`.

### How to read 09D

The primary sparsity and mixing summaries use the original 09B core. They include zero-count native in-tissue positions and compare complete native 4×4 blocks with the same sixteen constituent squares. The region for source/background fitting adds 400 µm around the original core. `coverage_summary.json` records multi-cell 8‑µm blocks, including blocks touching previous T and B labels, plus lost core cells and offsets between square-based and original centroids. The QC-restricted sensitivity option is `DOMAIN = "qc_retained_in_tissue"`. Background coverage by QC status matters because Step 02 used QC-retained transcripts for segmentation.

The targeted pilot uses prespecified lineage genes and up to 128 features with the highest out-of-mask UMI counts, fitting a region-specific decay scale with 50‑µm background tiles and a 200‑µm neighborhood radius. Source centers and output cells share the same hard square-to-polygon assignment. Compare `observed_same_hard_mask` with `SPARKLE_same_hard_mask` for the effect of correction. Compare `observed_09B_Proseg` with `observed_same_hard_mask` for the effect of changing the assignment. CD4/CD8-negative cells keep their prior evidence status. Inspect background coverage, gene-specific fit, depth/area strata, and source-marker retention before extending the fit to a whole mouse.

[Original Step 00 preflight](https://github.com/yueren13/Troubleshoot_20260923/blob/96f6b7a18c18c8287b7bd99a1cc58e08aa427bd0/project_1/00_TBIO8110_Config_Preflight.ipynb) · [Original Step 03 assignment](https://github.com/yueren13/Troubleshoot_20260923/blob/96f6b7a18c18c8287b7bd99a1cc58e08aa427bd0/project_1/TBIO8110_03_CellQC_ObsTransfer_CellCycle_h5ad_uns_fix.ipynb) · [SPARKLE](https://github.com/WangShuai-3/SPARKLE)

## Four-class tissue QC and paired SPARKLE fits

Run **09E_Tissue_QC_and_Ghost_RNA.ipynb** beside 09D using the existing **CT26 SPARKLE 0.1.4 (spatial CPU)** or **CT26 SPARKLE 0.1.4 (CPU)** kernel. Keep the original local `09_pilot_results` folder available. The default uses the Ada-14R 09B core and 400 µm of padding. Run all cells once. A new 09E result directory and a downloadable `09E_return_to_chat_*.zip` are generated for each execution.

- **Four classes.** The notebook joins the archived Step 00 QC labels by native barcode. It counts assigned and out-of-mask squares, zeros, singletons, ≥2 UMIs and total RNA for each H&E/transcript combination, and reports disagreement with the proposed raw-count cutoff of 1.6 UMIs. Start with `qc_class_coverage.csv` for the fraction of the out-of-mask RNA observed in tissue-low/transcript-high squares.
- **Source proximity.** `ghost_distance.csv` compares native out-of-mask *Lyz2*, *C1qa*, *C1qb* and other measured markers with reference panels across distances to previously labeled macrophage/monocyte centroids in the prepared section. The primary comparison pools the two tissue-low classes so zero-count positions stay in the denominator. `ghost_within_blocks.csv` provides near/far contrasts within 200-µm blocks. Counts per square and per 1,000 total UMIs help distinguish local depth from marker enrichment.
- **Morphology.** The figures show H&E, the four QC classes, candidate out-of-mask squares, macrophage/monocyte locations and three myeloid markers in the same physical fields. The notebook uses the registered Step 00 slide with OpenSlide when available, or a registered Space Ranger hires PNG and its own scale. `he_overlay_receipt.json` identifies the source and crop selection. Set `IMAGE_PATH` for a relocated registered slide or `CROP_CENTERS_UM` for a specific site. The numerical analyses still run if the registered image source is unavailable.
- **Shared-cell comparison.** 09E checks the original local 09D fits and computes any missing matching domain. It compares all Space Ranger in-tissue squares with the subset classified tissue-high by H&E. Both transcript levels are included. The notebook evaluates shared original core cells against the raw hard-mask counts in each domain. `paired_markers.csv` records the number of cells, raw and corrected counts, and positive cells. `paired_coreceptors.csv` reports the four direct CD4/CD8 detection combinations. `paired_cell_coverage.csv`, `paired_gene_fits.csv` and `paired_fits.json` document changes in area, depth, included genes and estimated decay. `background_reconciliation.csv` checks the out-of-mask UMI and square totals against the fits.

The notebook displays a short numerical summary and includes it as `readout.md` in the return archive, together with cropped H&E review figures. Full images, square-level records and cell-level matrices stay on the study system. The aggregate ZIP from 09D can be retained for reference. Reusing a fit requires the original local 09D folder and its per-cell results.

## Inputs and environment

The default adapter reads the selected mouse's observed molecule-count H5AD and existing 08A/08F gene and cell metadata. It aligns input by identifier, sums duplicate gene symbols, writes CSR in chunks, and retains unassigned cells with usable coordinates. Existing upstream objects are opened read-only. A standardized one-mouse H5AD with observed counts, gene symbols, cell-type labels and physical centroids in micrometres can also be supplied. A T-cell-only input would omit the source populations needed in both experiments. The local input summary includes the number of selected cells, nonzero counts, invalid coordinates, and working-size estimates.

09A uses NumPy, pandas, SciPy, scikit-learn, anndata, h5py, pyarrow, matplotlib, umap-learn, igraph and leidenalg. 09B additionally uses Scanpy, scvi-tools with the RESOLVI model, Pyro and PyTorch. Use the existing CUDA-enabled kernel on EC2. The pilot was exercised on CPU with scvi-tools **1.3.3** and **1.5.0.post1**. CPU smoke runs verify the execution path only. On a g5.24xlarge, the default fit uses **one A10G**. The notebook first runs a small calculation on the first visible GPU. It records the PyTorch and CUDA versions, device visibility and probe outcome in `runtime.json`. Training continues if the probe succeeds and only the optional peak-memory counter fails. The memory report gives peak allocated GPU memory when measured. The dense matrix sizes give planning references. Do not scale the tile's memory estimate by four GPUs.

## How to read the comparisons

**09A compares gene neighborhoods.** The first map uses the full prepared mouse. The second and third use equal numbers of whole-tissue and annotated T cells matched by section and depth stratum, with the same gene axis. A streamed covariance calculation refines the gene PCs; absolute coordinates can differ from 08Y. Check the signed co-detection and gene-PC ranks for the specified CD8, pan-T, myeloid, endothelial, B-cell, stromal and epithelial controls. Compartment markers can grow closer as cell-type variation increases even when CD4/CD8 distinctions within T cells remain weak. A repeat in Ada-9 allows mouse-level comparison of the same marker pairs.

**09B compares cell embeddings.** The default requests a core of about 12,000 cells plus a 100 µm halo. Every method receives the same model-eligible cells and feature set. The default uses about 1,200 selected genes plus available control genes, eight physical neighbors within section, a maximum training batch size of 64 and 40 epochs. After count filtering, 09B chooses a shared batch size for scVI and ResolVI that includes every eligible cell and avoids a one-cell training batch. The fit summary reports the eligible cell count and batch sizes. The three coreceptor genes are withheld before feature selection. Raw-count log-normalized PCA, nonspatial scVI and unsupervised ResolVI each receive a gene-expression neighbor graph and Leiden partition. The initial region is centered near the tissue median; choose a different `center_um` if the interior has insufficient T or myeloid cells for evaluation. Raw counts and metadata are never replaced by predictions.

Start with `specificity_and_distance.csv`, `marker_controls.csv`, and `embedding_comparison.csv`. Look for reduced myeloid signal inside pan-T-supported T cells, especially near macrophages, while myeloid markers remain concentrated in macrophages and TCR genes in T cells. Compare the change with the nonspatial scVI decoder. Marker abundance is reported per 10,000 counts relative to the modeled feature set. The CD4/CD8 test measures prediction of direct RNA anchors across spatial blocks of the same mouse. It uses its own eligibility definition and provides internal evidence. Model-estimated components, a visually cleaner Leiden map, or an estimated molecule in a cell do not independently establish lineage.

**If the full-mouse run exceeds available memory**, use the tile diagnostics to distinguish GPU allocations from CPU graph or count processing. A smaller training minibatch can reduce GPU demand. A lower feature count changes the comparison and should be applied to all three methods. Every change gets a distinct run directory. If a 09B fit fails, its diagnostic ZIP includes the runtime information, settings and traceback. The notebook displays a download link. To retry an earlier failed 09B with this update, copy `09B_ResolVI_One_Sample.ipynb` and `resolvi09.py` into your existing notebook folder, restart the GPU kernel, and rerun 09B with the same output root. A completed 09A run and its prepared input can be reused.

[ResolVI model and assumptions](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/resolvi.html) · [ResolVI class reference](https://docs.scvi-tools.org/en/1.3.3/api/reference/scvi.external.RESOLVI.html)
