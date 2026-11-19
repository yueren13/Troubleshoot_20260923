# 09F · Score-based SpotClean pilot

Extract this add-on into the existing folder containing 09E, keeping the R and Python files beside the notebooks. Select the Python kernel used for 09E, open **09F_Score_Based_SpotClean.ipynb** and choose **Run All**. The notebook invokes the local R executable. An R notebook kernel or a new Python environment is optional.

## Installation and timing

09F targets R 4.5.x and reuses a loadable SpotClean installation. When installation is needed, its destination defaults to `~/.local/share/ct26/R/4.5/spotclean`. It uses an existing Bioconductor 3.21 or 3.22 release when detected, otherwise 3.22. The package repositories and compiled dependencies available to the Domino session determine the first-run install time. Each setup finishes with a small synthetic spatial fit.

To run only installation and the smoke test from the same folder:

```bash
Rscript --vanilla setup_09f_spotclean.R
```

If the notebook cannot locate your R 4.5 executable, run `file.path(R.home("bin"), "Rscript")` in your existing R session and paste the result into the notebook’s `RSCRIPT` setting. `SPOTCLEAN_R_LIB`, `CT26_CRAN_REPO` and `CT26_BIOC_MIRROR` are optional settings for managed installations.

The default biological fit covers 40 × 40 native squares at 2 µm, including a four-square halo. Its wall-time limit is 30 minutes. The installer, input preparation and fit are timed separately. The fit receipt records the candidate radii, iterations, elapsed time and process peak RSS. A 20 × 20 window provides a smaller timing test if needed. Elapsed times from either window apply to that window and its data.

## Background and biological readout

The notebook joins measured H&E scores by barcode, using the aligned table or the native SpatialData table. The pilot threshold of 0.10 and the minimum background fraction of 0.25 are configurable. The first background definition combines low score, absence from Proseg masks and absence from Space Ranger tissue. The default allows low-score, unassigned squares within Space Ranger tissue when a window with enough of the first kind is unavailable. The receipt records which rule ran. A missing score never qualifies a position as background. The continuous score and QC class remain available alongside those model assignments. A failed search produces the observed score distribution for choosing another window or threshold.

The results report source and neighboring-cell marker retention, and observed Cd4, Cd8a and Cd8b1 counts inside and outside masks by score. Cell-level comparisons use complete footprints inside the inner window. Marker coverage, group size and raw UMI denominators accompany the results. A matching 09E receipt enables an additional same-cell comparison when the SPARKLE and SpotClean input supports agree. Edge windows can contain few previously labeled immune cells.

Download the **09F_return_to_chat_*.zip** beside the notebook. It contains human-readable and machine-readable summaries, timing, model and setup diagnostics. The working folder retains the native matrices and cell-level records.

## Validation and package references

The notebook and native alignment were tested on synthetic native counts, Proseg footprints and a scored AnnData Zarr table. The R scripts were syntax checked and their calls were reconciled with the package release source. The included installer runs the executable smoke test in the target R environment. The storage and pipeline defaults from earlier notebooks are used for path discovery.

- [Bioconductor releases and corresponding R versions](https://bioconductor.org/about/release-announcements/)
- [SpotClean package and vignette](https://bioconductor.org/packages/SpotClean/)
- [Bioconductor 3.22 SpotClean source](https://github.com/bioc/SpotClean/tree/RELEASE_3_22)
