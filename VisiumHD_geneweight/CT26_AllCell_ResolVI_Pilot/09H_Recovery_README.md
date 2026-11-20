# 09H recovery and review

Extract the full add-on **beside the working 09H notebook**. Keep the `h09fix/` subfolder intact. Open **09H_Recover_and_Review.ipynb** using the same spatial CPU kernel. The recovery needs the local 09H results directory with the original prepared native tiles and saved fit checkpoints.

## Run the first tile

Choose **Run All** with `TILES = 1`. The notebook locates a single original `09H_fixed_*` directory in `09_pilot_results/`. If several runs exist, set `RUN_DIR` in the first code cell to the intended original directory. The new output is its sibling with the suffix `_recovered`. The notebook starts a fresh Python process for each invocation.

| Setting | Result |
| --- | --- |
| `TILES = 1` | Attempt one eligible SpotClean tile and regenerate the disk readout. |
| `TILES = 0` | Recalculate coverage and figures from verified saved estimates. This runs without a new R fit. |
| `TILES = "all"` | Continue through the remaining eligible tiles using the same run. |
| `RSCRIPT`, `MINUTES` | Override the R executable or change the per-fit time limit. The default limit is 60 minutes. |

A pilot can have a successful paired tile and still have too few target cells or zero observed marker counts to estimate a particular retention ratio. Inspect `domain_coverage.csv` and the row-specific denominators before interpreting that ratio.

## Read the results

The coverage table reconciles the numbers of measured disk squares and complete cell footprints in the reviewed, single-method and paired subsets. Among the marker tables, **SPARKLE** shows that method across its available footprints. **Paired** uses identical cell footprints, gene panels and raw UMI denominators for the two methods. One specimen can contribute many tiles. The plots aggregate within the disk.

Read myeloid retention in nearby pan-T-supported cells together with retention in macrophage/monocyte labels. The raw coreceptor plot reports inside- and outside-mask counts over uniquely owned squares in the disk. Inside-mask counts include all cell types. Counts below the footprint level cannot establish which T cell produced a transcript.

The notebook displays the compact figures and gives a direct link to a return ZIP beside it. The archive contains the same figures and tables, available model status and diagnostic information. Typical figure filenames are `comparison/sparkle_marker_retention.png`, `comparison/paired_marker_retention.png` and `comparison/raw_coreceptor_masks.png`. Each appears when its corresponding measurements support a plot.
