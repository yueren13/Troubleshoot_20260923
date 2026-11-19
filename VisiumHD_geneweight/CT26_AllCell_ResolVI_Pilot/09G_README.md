# 09G · Matched SPARKLE and SpotClean regions

Unzip the add-on into the folder containing the working 09E and 09F notebooks. Keep the bundled version of `spotclean09f.R` in that folder. Its new 09G option accepts the larger input and shared spatial gene plan. Its two-argument 09F entry point retains its previous behavior.

Open **09G_SPARKLE_SpotClean_Matched_Regions.ipynb** in the CT26 SPARKLE spatial CPU kernel and choose **Run All**. The notebook reuses the earlier paths, scores, segmentation and 09B labels. It looks up the working R 4.5 executable from the prior 09F receipt and uses the same SpotClean setup and smoke test. If the project moved, set `RUN_DIR`, `PATH_OVERRIDES` or `RSCRIPT` in the first code cell.

## What changed after 09F

The search targets up to two nonoverlapping native windows per section. The 64 × 64 default provides 128 µm on each side. A six-square halo leaves a 104 × 104 µm core. All core bin centers must be in the earlier evaluation disk. Only complete Proseg footprints inside a core enter the readout.

Selection starts with a regular lattice of potential window origins. Each window needs a complete measured grid, sufficient provisional background and higher-scoring tissue, and enough complete cells with each prior label or marker rule. The defaults require at least three macrophage/monocyte-labeled cells, three nearby pan-T-supported non-macrophages and three T-labeled cells. Nearby means within 25 µm of a labeled macrophage or monocyte in the earlier analysis. Those groups may overlap. A candidate needs at least five myeloid UMIs in each primary group and five pan-T UMIs in the nearby pan-T group, summed over each group. Cd4 and CD8 abundance do not influence selection. The candidate table retains observed denominators and exclusion reasons.

Provisional background requires measured `he_tissue_score` ≤ 0.10 and absence from all Proseg masks. The default seeks ≥25% such squares outside Space Ranger tissue and allows candidates inside Space Ranger tissue when needed. The chosen rule and its counts are recorded for each region. Unmeasured scores remain unassessed. Histology images are optional for other analyses and are not an input to 09G.

## Reading the results

- Each method receives the same native square counts, array geometry, spatial gene plan and Proseg assignments. SPARKLE estimates gene-specific local leakage using out-of-mask squares. SpotClean fits on the native grid with score-defined background and its corrected tissue positions are summed back to the original footprints.
- 09G includes raw and corrected marker sums for each region. Each method must reproduce the raw counts for every target and assigned cell. Only verified pairs can contribute to a pooled estimate. The input fingerprint and gene fits can be inspected for either method.
- Read myeloid retention in macrophage/monocyte-labeled cells alongside retention in nearby T-supported cells. The observation and cell counts give the strength of each comparison. Ratios above 100% are possible with SpotClean. An unchanged SPARKLE marker can indicate insufficient evidence for subtraction.
- `edge_sensitivity.csv` splits cells by whether their centroids are at least one SPARKLE donor radius from the crop edge. `coreceptors_by_score.csv` reports **observed** Cd4, Cd8a and Cd8b1 RNA inside and outside masks by score. Multiple regions from one section are descriptive.
- SpotClean runs sequentially with a configurable 60-minute limit per region. Its resource cost grows with grid size. Fit receipts record elapsed time and peak R memory. The SPARKLE fit records the regional decay scale, background support and the number of markers that passed its fit threshold.

The notebook writes a **09G_return_to_chat_*.zip** beside it. Full native and cell-level matrices stay in the numbered local results folder. On a partial run the ZIP includes completed stages and diagnostics. If no window qualifies, inspect `regions/candidates.csv` and `regions/selection.json` for the limiting criterion. `SEARCH_STRIDE_BINS` controls the spacing of candidate origins. The pipeline does not silently relax the selection rules.

## Verification

The Python stages were exercised with synthetic native counts, measured zero-count positions, stored barcode-aligned tissue scores and original-format Proseg polygons. The real CPU SPARKLE package was tested on a 64 × 64 fixture. The two-method readout was tested using known counts, shuffled axes, missing fits and a deliberately mismatched matrix. The R code was syntax checked. A paired biological run requires the study data and installed R environment.
