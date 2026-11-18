# 08Y and 08Z · T-cell gene neighborhoods and lineage scoring

Two Jupyter notebooks continue the Adagrasib mouse Visium HD T-cell analysis from the count source used by 08W/08X. 08Y discovers expression partners of lineage markers. 08Z tests those partners against masked RNA anchors and writes lineage and functional-state annotations for every cell.

## Run on EC2

1. Download **08YZ_Tcell_Lineage_Notebooks.zip** from the parent GitHub folder. Extract the whole package inside the CT26 project where 08V ran. Keep both notebooks and the four Python modules together.
2. Open **08Y_Gene_Neighborhoods.ipynb** in the existing project Python kernel and use **Run All**. If the project is elsewhere, set `PROJECT_ROOT` in its first code cell to the code directory containing `ct26_analysis_config.json` and `inputs/pipeline_config.json`, or to the analysis directory containing `14_tcell_ratio_subclustering`.
3. Open **08Z_UCell_Lineage_and_State.ipynb** in a fresh kernel and use **Run All**. It locates the completed discovery run under the same `OUTPUT_ROOT` and reopens its recorded inputs. If you changed `OUTPUT_ROOT` in 08Y, use that path in 08Z as well. Set `DISCOVERY_DIR` to select a specific older run.
4. Open the 08Z per-mouse validation and coverage tables. The last cell writes a compact `08YZ_review_*.zip` containing aggregate tables and figures for follow-up. The full `per_cell_annotations.csv.gz` stays in the 08Z output directory.

The CPU path uses Python, NumPy, pandas, SciPy, scikit-learn, h5py, anndata, matplotlib and IPython. The gene UMAP and Leiden partition additionally use `umap-learn`, `igraph` and `leidenalg`. If those three packages are missing, install them with `%pip install umap-learn igraph leidenalg` in the selected notebook kernel and restart it. The discovery report identifies the projection and clustering algorithms actually used. A fallback uses the first two gene PCs or graph-constrained Ward clustering when the corresponding optional dependency is unavailable. The scoring module implements the positive UCell statistic on sparse matrix chunks.

## Project inputs

The default route follows the reviewed 08V plan to the completed 08O `analysis.h5ad`, verifies cohort and count provenance, reads the full common-gene `layers["counts"]`, and aligns 08Q's mixed-signature flag by cell identifier. It restores the 08X split of eight discovery and four evaluation mice. For the original cohort, expect 590,247 cells. The evaluation mice are Ada-1, Ada-12, Ada-15 and Ada-6. The notebooks print the direct coreceptor counts by mouse for comparison with the 08WX summary.

If the analysis moved or an auxiliary table was exported separately, 08Y exposes paths for a T-cell-only count H5AD, the 08X sample-design CSV, the 08Q mixed-flag CSV and the 08O per-mouse ratio CSV. A custom count input must include the original measured gene universe, sample and condition, unique gene symbols and sparse, nonnegative integer-valued observed molecule counts. Use `sample_col`, `condition_col`, `gene_col` or an explicit `layer="X"` only for an alternate count input. The project route resolves the 08O count layer directly. The imported 08Q flag can contain unknown values. Unknown flags retain their own status, and the default lineage rules require a known negative mixed flag. If all flags are missing, check the 08Q pointer or set `MIXED_CSV` to its matching `cell_metadata.csv.gz`.

## Outputs to inspect

| Question | File |
|---|---|
| Which genes associate with `Cd4`, `Cd8a`, `Cd8b1`, `Foxp3`, or a state program across discovery mice? | `anchor_neighbors_top50.csv`, `anchor_rankings.csv.gz` and `per_mouse_gene_associations.csv.gz` |
| Which partners were retained, and why were others excluded? | `lineage_candidate_audit.csv.gz` and `signatures.json` |
| Which genes cluster in gene space? | `gene_map.png`, `gene_map_evidence.png`, `gene_module_summary.csv`, `gene_map.csv.gz` and `gene_map_neighbors.csv.gz` |
| How well did each frozen signature recover masked coreceptor labels? | `anchor_validation_by_mouse_depth.csv`, `depth_matched_validation.csv`, `comparator_auc_by_mouse.csv` and `frozen_thresholds.json` |
| How many additional cells received CD4-like or CD8-like labels? | `08Z_complete.json` and `lineage_coverage_by_mouse.csv` |
| What evidence is available for a single cell, including a lineage-unresolved cell? | `per_cell_annotations.csv.gz` includes direct marker counts, mixture status, pan-T evidence, observed and inferred labels, score margins, state scores and detected signature-member counts. |

Results default to `08YZ_results/` under the notebook's working directory. A completed stage is keyed to the input data, parameters, code and dependency versions. The separate output folders and file receipts support reruns. Use a matching Python environment in both notebooks.

## Interpretation

**08Y.** Gene associations are signed correlations of binary detection after within-mouse library-depth stratification. Up to 30,000 discovery cells per mouse enter the default model. Equal-mouse summaries rank the partners. The map uses sketched, standardized discovery-mouse profiles, PCA and a gene-neighbor graph. The top-ranked anchor table uses the direct correlations. The 50 µm non-T-neighbor/focal-T expression ratio describes spatial context. When per-mouse measurements exist, 08Y recomputes it on discovery mice for feature selection. Whole-cohort ratios are descriptive. Non-T and state gene controls help identify alternative explanations for co-detection.

**08Z.** Positive UCell scores use average tied ranks, observed count data, the measured members of each signature, and a discovery-derived rank cap. Three coreceptor genes are withheld throughout the lineage predictor, including from the rank universe. Fixed state programs are scored on the full measured gene axis. For each lineage independently, the discovery mice determine a score-margin threshold with an empirical 2% opposite-anchor call-rate target. The four evaluation mice assess the frozen rule with per-mouse, pooled-count and minimum-support checks. The default rule requires at least two evaluation mice with sufficient anchors. A passing rule can assign additional cells with neither coreceptor detected, at least two observed support genes, pan-T support and an admissible mixed flag. A failed rule leaves those cells unresolved. The two lineages can have different outcomes.

The 2% target applies to eligible opposite-lineage RNA anchors. The true error rate among naturally coreceptor-undetected cells is unknown. Depth-standardized results show how much of each mouse's unresolved population is represented by adequately supported anchor-depth strata. State scores describe programs such as cytotoxicity, memory, regulatory activity, interferon response and cycling. A score depends on the panel and its detected members.

**Methods.** [UCell and pyUCell](https://doi.org/10.1093/bioinformatics/btag055) · [UCell rank and parameter documentation](https://bioconductor.org/packages/release/bioc/vignettes/UCell/inst/doc/UCell_parameters.html). The gene graph is a custom discovery analysis. [hdWGCNA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10326379/) provides related background on gene-centered networks.

## Verification before release

Both notebook workflows were exercised cell by cell in separate Python/IPython processes on a synthetic 12-mouse project with sparse H5AD counts and the 08O, 08Q and 08V source layout. Checks covered input preservation, frozen panels, gene UMAP, Leiden modules, per-mouse validation and cell-level outputs. Additional tests checked that replacing evaluation expression leaves discovery unchanged, that empty panels yield zero inferred calls, and that sparse rank scores match official pyUCell 0.8.0 CPU scores in a numerical comparison. The original 08X sample-design export was checked independently. Full-cohort biological performance and EC2 runtime require the project count data.
