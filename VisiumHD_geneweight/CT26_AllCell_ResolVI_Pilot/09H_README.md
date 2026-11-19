# 09H · SPARKLE and SpotClean across the original 09B region

Unzip the package into the folder containing the working 09G notebook. Keep the bundled helper files together and use the same **CT26 SPARKLE spatial CPU** kernel. Restart the kernel after updating the files. Open **09H_Original_09B_Region.ipynb** and choose **Run All**. The notebook locates the completed 09B run and uses its saved section, center, inner evaluation radius and original core cell IDs.

The first code cell follows the earlier notebooks. Set `RUN_DIR` to the original completed 09B directory if it moved, and keep its prepared H5AD accessible at the saved location or in the sibling `inputs/` directory. `PATH_OVERRIDES` accepts the same native count, segmentation, aligned QC and SpatialData paths as 09G. The workflow reads the stored `he_tissue_score` from the native-bin annotations. A prior working 09F R executable can be reused automatically, or you can set `RSCRIPT` to its path.

## Full run or timed first pass

The default `TILES_PER_RUN = None` processes all remaining fit-ready tiles. For a timed first pass, set `TILES_PER_RUN = 2`. Then change it back to `None` and rerun with the same `RUN_TAG` and inputs. The initial preparation covers the entire disk even for a short batch. The runner prioritizes untried tiles, then tiles with usable input for both methods, then distance to the disk center. An early batch can therefore have better support for fitting than the rest of the disk. The full-disk runtime for the study specimen is unknown.

Each R fit runs sequentially and has a 60-minute limit by default, excluding R setup. The notebook shows elapsed R time and memory for completed fits and separate status counts for each method. Valid completed fits are reused. A failed or interrupted method can be retried by rerunning with the same settings. Set a new `RUN_TAG` for a fresh set of fits if an input was replaced in place or the R installation changed.

## What goes into the comparison

The saved disk sets the target. Every observed native 2 µm bin center inside it enters the reference denominator, including zero-count positions. 09H applies no marker-abundance, immune-cell-count or 25% background threshold to select the region. `regions/coverage.json` gives the original geometry, the number of measured positions and the number of original evaluation cells. It records any positions that could not be included in a prepared fit grid.

Both methods receive one raw count matrix per 64 × 64 native grid (128 × 128 µm). The default six-bin margin gives adjacent fits overlapping context. Every disk square has one owner for raw bin totals. An original core cell contributes once if its owner contains the whole measurable Proseg footprint. All bins of that footprint count toward its marker total, even when a cell crosses an ownership boundary. This produces local model estimates across the disk. The cropped grids limit evidence from distant sources.

SPARKLE fits leakage using counts outside Proseg masks. SpotClean treats measured, unassigned bins with a finite staining score ≤ 0.10 as provisional background. By default those bins may lie inside or outside the Space Ranger tissue boundary. Set `REQUIRE_SR_BACKGROUND = True` to restrict candidates to outside that boundary. A tile with few background bins can still be fit if it has the required evidence, including measurable background RNA. Such estimates are especially dependent on the chosen background. If a method lacks the input it needs, its reason is recorded. Raw data and a successful fit from the other method remain available.

## Read and return the results

Start with the disk map and the `paired` row in `comparison/domain_coverage.csv`. It shows how much of the original domain has estimates from both methods. Then read the myeloid rows together in `pooled_retention.csv`. Lower myeloid retention in nearby pan-T-supported cells is more persuasive when myeloid markers remain detectable in labeled macrophages/monocytes. Each marker row gives its actual paired cell count and observed UMIs. The pooled ratio divides summed corrected counts by summed observed counts on those same footprints. A small raw denominator gives a fragile ratio. The tiles come from one section.

| File | What to inspect |
| --- | --- |
| `comparison/method_coverage.csv` | The raw observations across reviewed cells, each method on its own available subset, and both methods on shared support. |
| `comparison/background_sensitivity.csv` | Matched retention for tiles below and at or above 25% provisional SpotClean background. Compare the UMI and cell denominators in each band. |
| `comparison/coreceptors_disk_by_score.csv` | Observed Cd4, Cd8a and Cd8b1 in uniquely owned disk bins, by assignment and H&E score. These are counts of measured RNA and do not assign missing coreceptor identity to a T cell. |
| `comparison/edge_sensitivity.csv` | Marker estimates grouped by whether the cell center is within one SPARKLE donor radius of the fit edge. |
| `progress.csv` and `comparison/region_status.csv` | Method status, reasons for unavailable fits, elapsed time where measured, and final readout verification. |
| `regions/local/cell_coverage.csv.gz` | Local ledger with the owner and inclusion or exclusion reason for each original 09B core cell. |

Use the **09H_return_to_chat_*.zip** link shown in the notebook to return results, including a partial run or diagnostic. The ZIP includes aggregate tables, available figures, progress and coverage. Native matrices, individual cell records and full model output stay in the local results directory.
