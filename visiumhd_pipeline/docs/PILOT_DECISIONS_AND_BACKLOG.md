# Agreed pipeline decisions and open work

**Snapshot: September 29, 2026.** This records the project conversation and inspected artifacts; it does not change defaults or implement features. [Lessons/evidence](PILOT_LESSONS_20260929.md) distinguish local deliveries, source findings and confirmed repository state.

## Keep these invariants

| Decision | Meaning |
|---|---|
| Flexible midway entry | Raw data, accepted QC bins, existing segmented AnnData and export-only products remain distinct routes. |
| Explicit count source | Preserve raw integer counts; do not fabricate them from corrected/normalized expression. |
| Full-resolution coordinate authority | Bin x/y refer to Space Ranger full-resolution column/row pixels. Smaller display images carry transforms; they do not redefine cell/bin coordinates. |
| Separate tables | Bins and cells are separately loadable. Loading a selected table may still require RAM; this is not a claim of fully backed analysis. |
| Honest geometry | Existing polygons remain real boundaries; centroid glyphs are not invented segmentations. |
| Fixed-reference tissue evidence | Extract H/E against the declared reference, use fixed divisors, combine at image pixels, then sample at bin centers. Keep display normalization separate. |
| Review before spatial transfer | Save source polygons/images and transforms, inspect overlays, then assign metadata. Unknown coverage is not negative tissue. |
| Accepted QC preservation | Keep both accepted tissue-high classes, including low-transcript tissue. New analysis policies do not silently overwrite historic QC. |
| Selective persistence | Distinguish new elements, changed data, transformations and table metadata; avoid rewriting counts for small annotation changes. |
| Storage roles, not assumptions | Mounted object storage and fast work storage are configured paths. Retained work is not disposable scratch. Never manipulate mounts. |
| Explicit model grouping | Sample is batch; treatment is metadata. Integration groups are deliberate, and clustering is not cell-type annotation. |
| Source/control separation | Public repo contains sanitized source and documentation, not real deployment manifests, raw logs or corporate assets. |

## Tracked work and acceptance checkpoints

| Tracker | Scope | Status at this review |
|---|---|---|
| [#2](https://github.com/yueren13/Troubleshoot_20260923/issues/2) / [PR #3](https://github.com/yueren13/Troubleshoot_20260923/pull/3) | Saved-model-to-inference handoff | Committed draft correction; unmerged; real train/infer and DDP validation still separate. |
| [#4](https://github.com/yueren13/Troubleshoot_20260923/issues/4) | Consolidated v3 release and onboarding | Local fixes delivered in several artifacts; reconciliation, reproducible setup and version-aware docs remain work. |
| [#5](https://github.com/yueren13/Troubleshoot_20260923/issues/5) | Compact master images | Open requirement, not implemented in reviewed v3. |
| [#6](https://github.com/yueren13/Troubleshoot_20260923/issues/6) | Analysis QC before integration | Complete requested policy not implemented in reviewed v3. |
| [#7](https://github.com/yueren13/Troubleshoot_20260923/issues/7) | Readable count plots and replot-only route | Open presentation-layer enhancement. |
| [#8](https://github.com/yueren13/Troubleshoot_20260923/issues/8) | H&E/annotation/persistence promotion and validation | Review implementation delivered; real scientific/storage validation and repository promotion remain separate. |

### Compact master policy

The requested working master stores hires/lowres images with explicit transforms and retains the original image's external location and identity metadata. Full-resolution embedding should remain an explicit export option. The current v3 writer still materializes original-image chunks and pyramid levels. Disabling normalized H&E or changing report thumbnail size does not remove the original image.

An external reference is not a self-contained slide; high-resolution access still requires that source. Existing completed stores need a deliberate compaction path, not deletion underneath a running writer. Quantitative H&E/signal maps must retain their values and provenance independently from display images.

### Analysis-QC policy

The owner's requested starting settings are:

| Quantity | Requested criterion |
|---|---|
| Human mitochondrial percentage | Strictly less than 20% |
| Mouse mitochondrial percentage | Strictly less than 10% |
| Raw counts per cell | At least 3; configurable to 16 or another reviewed value |
| Cells detecting a gene | At least 3 retained cells |

These are user-selected criteria, not universal tissue/chemistry-valid thresholds. The proposed gene-detection scope is the selected integration group's retained cells; that scope and per-sample detection summaries must be explicit. It is not a requirement for three genes per cell or detection in every sample.

Preserve all master cells/counts and historic QC. Create an analysis flag with failure reasons and a reproducible profile. Compute mitochondrial percentage and totals on the declared full count-feature set, check feature recognition and invalid metrics, select cells, filter genes, then choose HVGs and final nonzero training inputs. Changing analysis thresholds should not require resegmentation. scVI still receives counts, not plotted log10 values.

### QC display policy

Use log10(1+count) spacing and original-count labels such as 0, 10, 100, 1,000 and 10,000. Position the 100 tick at log10(101), not a guessed integer position. Retain zeros, comparable axes and original-unit cutoff labels. Percentages and fixed 0-1 H&E scores remain linear; spatial/UMAP axes remain their coordinates. Replot from metadata without changing expression, QC decisions or model input.

## Evidence checkpoints before a baseline release

1. **Packaging:** one identified code/patch version; notebook-first setup works from a clean kernel; current docs match the installed source.
2. **CPU/control:** test discovery, actual worker imports, preview/execute, sample selection, prerequisite gating, failure reporting and safe reruns.
3. **Scientific storage:** counts and IDs preserved across import/readback; verified coordinate overlays; real table-metadata update and interruption recovery on approved storage.
4. **Analysis:** intended QC exclusions quantified by sample; feature rules tested; actual scVI save/load/infer; clustering/resolution outputs and downstream reload.
5. **Performance:** measured worker memory and throughput for representative inputs; keep provisional estimates labeled; validate DDP/RAPIDS modes independently.

No checkpoint is inferred merely from successful syntax validation. Do not combine historical test counts from different archives as if they were one end-to-end run. Do not promise a one-hour cohort completion without measured successful timings.

## Remaining uncertainties

The exact cause of the unfinished historical annotation mismatch, complete original Step03 classification settings, real impact of XML exclusions, and full-cohort scientific/GPU completion are not established by the available record. One pasted log's scope inconsistency also remains unresolved. Keep these distinctions visible in future reviews.

## How to use this record

Open a focused issue or PR, link its affected invariant and regression test, record the deployed version and observed result, and update the status after verification. The repository owner decides merges. This documentation change leaves `main`, historical sources, running notebooks and biological data untouched; it does not consolidate the code by itself.
