# Start here: four datasets at different stages

## 1. Separate sample routing from analysis parameters

`config/samples.csv` is the sample routing table: where inputs are, what already exists, where each output category must be stored, and which samples belong together for integration.

`config/settings.json` is the shared execution and analysis configuration: interpreter paths, GPU IDs, CPU/RAM limits, policy allowlists, QC thresholds, scVI settings and clustering backend. It avoids repeating dozens of identical model settings in every CSV row. An optional local `overrides_json` file supplies a reviewed per-sample subset of analysis settings.

Examples are placeholders, not guessed paths for your other three datasets. The legacy-study CSV carries forward the 12 source paths from the v1 bundle, but its destination roots are new placeholders. Check all source paths and count-layer semantics before using it.

## 2. Start with one already-segmented sample

Copy `samples.cells_only.example.csv` to `samples.csv`. Keep one row for the first migration pilot (integration needs two later). Fill `cell_input`, `counts_source`, coordinate mode and output paths. Existing cell AnnData does not require `bin_input` or `spaceranger_outs`.

- If X contains integer raw counts, use `counts_source=X`.
- If X is corrected or log-normalized while raw counts are `adata.layers['counts']`, use `counts_source=layers:counts`.
- If no raw count matrix remains, use `cell_annotated + export_only`. The migration/scVI route must not invent raw counts from corrected values.

Do not put multiple already-pooled samples into one row and call that one batch. Each row must describe one biological/technical sample. Split a previously pooled AnnData by its actual sample column into separate inputs first; otherwise batch labels would be wrong.

Run notebook 00 and the spatial-environment smoke tests in 99. Run 00B or 05 for the import. Review cell-count distributions, QC and any image overlay. Set `coordinates_reviewed=true` only after real registration review; unregistered cell-only coordinates do not require a false sign-off. Then run 06 and inspect 07.

The v2 count-first product retains annotations and embeddings but not unselected expression layers or neighbor graphs. The source object is never changed. To preserve every byte of the original annotated object, use export-only for a distinct job/run instead of assuming all corrected matrices are duplicated into the new cells table.

## 3. Add boundaries or bins only when available

`cell_boundaries` accepts GeoParquet or supported GeoJSON (including gzip); identify matching IDs using `boundary_id_column`. Use `__index__` for a GeoParquet index. Cell and boundary IDs must be the identical set. Multiple polygons for one cell must be intentionally dissolved before import; mismatches are not silently dropped.

Both centroids and polygons must be in the same declared input coordinate system. `cell_to_fullres_affine` uses the same order as `tenx_to_image_affine`: `a;b;c;d;e;f`, meaning x'=a*x+b*y+c, y'=d*x+e*y+f. It is NOT Shapely's coefficient ordering. `legacy_sr_row_col_um` converts the previous Proseg convention (first coordinate row*source_mpp, second column*source_mpp).

With no boundary file, the SpatialData shape element is `cell_centroids`; radius=1 is a rendering glyph only. There is no measured whole-cell segmentation mask, derived area, or reconstructed transcript coordinate layer. Existing measured area annotations can remain as source metadata.

Supplying optional `bin_input` adds a second expression table. Use `bin_input_kind=spatialdata` and an explicit `bin_table` such as `square_002um`, or use standalone AnnData bins. It must contain original full-resolution coordinates (or be joined to Space Ranger positions) and actual counts in `bin_counts_source`. Stage cell_segmented plus bins implies existing bin QC is being reused; the declared QC column is required.

## 4. Scale only after the pilot

Choose GPU IDs available to you; the pipeline does not reserve hardware against unrelated jobs outside its own launch calls. Each phase is synchronous and returns before the next phase begins. Do not start simultaneous StarDist, scVI and RAPIDS notebooks on the same assigned GPUs.

`worker_*.resources.json` records elapsed seconds and sampled summed RSS of the process tree. Use a conservative estimate based on large representative slides, with headroom, for `execution.stardist.estimated_ram_gib_per_worker` and `execution.proseg.estimated_ram_gib_per_worker`. Unknown values keep one worker even though the g5 profile requests four. These sampled measurements can miss short peaks and summed RSS can count shared memory twice; they are not a precise unique-memory profiler.

## 5. Integration is explicit, not automatic across the four datasets

Assign the same `integration_group` to compatible rows. Leave it blank for exports, bins-only results or studies that should not be pooled. At least two cell-producing rows are required. Gene features are intersected; missing genes are not zero-filled to force compatibility.

Run 08, 09, 10 and 11 for each selected group. The ordinary profile uses single-GPU scVI and RAPIDS. The opt-in profile selects DDP, multi-GPU neighbors and Dask Leiden. Review the compute guide before choosing it. `cell_annotated` is an input state, not a promise that this pipeline performs cell-type annotation.

All ten Leiden resolutions (0.1–1.0) use one common scVI-derived UMAP. microViz brewerPlus provides the first 41 colors; additional colors are explicitly marked as extensions. Results include sample composition and QC diagnostics; inspect possible biological overcorrection rather than optimizing batch overlap alone.

## 6. Publish only finished products

Complete annotation writeback before notebook 12. Active processing occurs in approved local scratch; finalized object, plot and data categories are published to their own CSV destinations. An S3 publication is valid only after its `_SUCCESS.json` marker exists. A top-level local publication receipt lists the category destinations.

Keep source data and scratch until publication and downstream load checks are verified. There is no automatic deletion. A new `run_id` creates a separate output; an already-published run is immutable. Re-running computations may change logs/metadata, so do not expect those changes to overwrite a committed publication under the same ID.

Review `docs/VALIDATION_SUMMARY.md`: local tests cover the control layer; target scientific/GPU/S3 execution remains necessary before scaling.
