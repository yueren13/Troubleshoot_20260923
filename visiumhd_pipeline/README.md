# Visium HD pipeline v2 — CSV entry stages, independent storage, isolated compute

**Review release 0.2.0, September 24, 2026.** This replaces the v1 notebook entry points while retaining the reusable `vhd/` helper package and the original scientific engines. The new control, adapter and compute modules are separate subpackages.

Start with **[START_HERE](docs/START_HERE.md)**, then **[CSV schema](docs/CSV_SCHEMA.md)**. The runtime and testing limits are in **[Validation](docs/VALIDATION_SUMMARY.md)**. No full-slide, S3, TensorFlow, Proseg, scVI-DDP or RAPIDS execution has been performed in the delivery environment.

## Core decisions

- One fixed-schema CSV row = one sample. Multiple datasets may have different entry stages, image formats and output roots.
- Entry stages: `raw`, `bin_imported` (also accepts `bin_inputed`), `bin_QCed`, `cell_segmented`, `cell_annotated`.
- Entry stage describes supplied input, not a claim that files are valid. Every applicable action validates its inputs. The CSV is never auto-edited to claim progress.
- Standalone AnnData `.h5ad` and `.zarr` are valid cell inputs. They do not need to be SpatialData or include the original bins.
- `counts_source=X` or `layers:counts` is explicit. The count-first migration does not normalize counts, round corrected expression to integers, or load unselected expression layers. Existing annotations/embeddings are kept; full source objects remain untouched.
- `cell_annotated + export_only` instead preserves the complete source object byte-for-byte, including corrected X, all layers, graphs, embeddings and annotations. No QC or integration is run. This route preserves source format, rather than pretending an AnnData file becomes SpatialData by renaming it.
- Cell boundaries are optional. Missing polygons yield **centroid point shapes**, not invented cell boundaries or inferred cell area. Unregistered centroids remain in `sample_native`; no H&E co-registration is implied.
- Only rows assigned the same **explicit** `integration_group` are combined. Four studies are not automatically one integration. The sample batch is the unique `dataset_id__sample_id` key.

## Folder organization

```text
config/                  fixed CSV templates/examples + environment/storage/analysis settings
notebooks/               15 short notebooks: orchestration, review, export
vhd/
  control/               manifest validation, action routing, scratch/S3 publication
  adapters/              standalone AnnData, existing bins/cells, SpatialData assembly
  compute/               worker processes, CPU/GPU scheduling, scVI-DDP, RAPIDS/Dask
  core.py                coordinates, counts, serialization, fingerprints, locks
  io.py                  Space Ranger import and selective table loading
  images.py, qc.py       original H&E, normalization and bin QC
  segmentation.py        StarDist prior + CPU Proseg engine
  cells.py               Proseg reconstruction, metrics and geometric annotation transfer
  storage.py             spatial models and transforms
  integration.py         shared-HVG preparation, results, writeback
  plots.py, palette.py   figures and microViz-based categorical colors
scripts/                 optional command-line controller
tests/                   local tests plus optional actual AnnData/SpatialData/scVI smoke tests
env/                     dependency guidance, not verified GPU lockfiles
docs/                    schema, routes, storage, compute, limitations and provenance
```

Active CSV/JSON examples are kept in `config/`. No original biological inputs or generated example biological results are shipped.

## Minimal start

```bash
# From the extracted folder. Preserve your established scientific environments.
cp config/samples.cells_only.example.csv config/samples.csv
cp config/settings.g5_24xlarge.example.json config/settings.json
# Edit BOTH files: input paths, output/scratch allowlists and interpreter paths.
# Then launch Jupyter from this folder or its notebooks subdirectory.
```

The controller package may be installed with `python -m pip install -e . --no-deps` in existing environments after their dependencies are present. Launchers also set `PYTHONPATH`, so an editable install is not mandatory. Use the configured full Python executable path, not a shell activation command.

Notebooks default to **dry run**. Set the notebook's `EXECUTE=True` after reviewing its plan, or explicitly set `VHD_EXECUTE=1` in the controller process environment. No S3 transfer is allowed until the separate storage policy flags permit it.

## Common routes

| Supplied data | Notebooks after 00/99 |
|---|---|
| Raw Space Ranger -> QC bins only | 01, 02, review, 06, 07, 12 |
| Raw Space Ranger -> new segmented cells | 01, 02, review, 03, 04, 05, 06, 07 |
| Existing imported bins, not QCed | 01, 02, review, then 03–07 as needed |
| Existing QCed bins | 01, 02 (reuse only), review, 03–07 |
| Existing segmented AnnData | 00B (or 05; optionally 01/02 when also supplying bins), review, 06, 07 |
| Existing annotated AnnData, export only | 12 |
| Prepared compatible cell stores -> integration | 08, 09, 10, 11, then 12 |

Notebook 00B does not assemble or resegment. The `stage` stays at its entry value; completion markers decide safe reuse. Setting a stage never fakes a completed prerequisite.

## Storage contract

For each row, all active writes go beneath:

```text
<tmp_dir>/vhd/<dataset_id>/<sample_id>/<run_id>/
    work/                  staged matrices, masks, Proseg scratch and checkpoints
    reports/               QC plots, tables, logs and status
    objects/<sample>.zarr  active per-sample product
    publish_plots/         publication payload assembled from figures
    publish_data/          publication payload assembled from tables/logs/provenance
    runtime/               framework temporary and cache paths
```

Publishing separates these products:

```text
<output_object_dir>/<dataset>/<sample>/<run_id>/objects/
<output_plot_dir>/<dataset>/<sample>/<run_id>/plots/
<output_data_dir>/<dataset>/<sample>/<run_id>/data/
```

Integration has its own approved roots in settings; it is not silently assigned to the first sample's destination. See [Storage policy](docs/STORAGE_POLICY.md) before using S3 or institutional data.

## Compute contract

The g5 example exposes GPU IDs 0–3, requests four StarDist sample workers and four Proseg workers (20 threads each), but **unknown RAM/job forces one pilot**. Review recorded `*.resources.json`, then provide a conservative measured RAM estimate per worker. The launcher applies both CPU and currently available host-RAM limits.

The ordinary profile uses single-GPU scVI and single-GPU RAPIDS. `settings.multigpu_optin.example.json` explicitly enables four-rank scVI DDP plus Dask Leiden / `mg_ivfflat` neighbors. These are alternative profiles; do not run them concurrently on the same GPUs. One slide is not distributed across all GPUs by the StarDist adapter; four GPUs run four independent slides. Proseg remains CPU. UMAP fitting remains single GPU even in the Dask profile.

Dask temporary spill is routed to the integration scratch. Not every CUDA allocation/algorithm can spill. Multi-GPU does not guarantee a faster run. See [Compute](docs/COMPUTE_AND_ENVIRONMENTS.md).

## Output tables and loading

For full combined output: `tables['bins_2um']` contains raw bin counts, `tables['cells']` contains raw cell counts, and optional `tables['cells_scvi']` contains zero-gene integration annotations. Images are multiscale; final boundaries are shapes; bins are compact labeled rasters. Cell-only output simply omits absent bin components.

```python
from vhd.io import load_table, load_spatial_with_table, table_metadata
cells = load_table('/approved/local/sample.zarr', 'cells')
# Add Leiden/UMAP/latent annotations after notebook 11:
cells = load_table('/approved/local/sample.zarr', 'cells', merge_scvi=True)
# Inspect metadata without reading the selected table's expression:
obs, var = table_metadata('/approved/local/sample.zarr', 'cells')
```

The selected expression table still loads into RAM; the other expression table does not. This is not a fully backed multi-table AnnData implementation. S3 input is explicitly staged under policy rather than handed to HDF5 as a remote URL.

## Scientific scope retained from v1

Original Space Ranger `pxl_col_in_fullres` and `pxl_row_in_fullres` remain the bin coordinate authority. Legacy swapped Proseg row/column microns have an explicit conversion. Both high-tissue QC classes are retained. Geometric bin annotation transfer is not a claim of molecule-level Proseg assignment.

The fresh H&E normalization / bin-QC baseline remains **not an exact replica** of the user's original Step01–03 helpers, whose source was not available in the v1 audit. Reusing an existing `qc_class` is the migration path that preserves accepted decisions. Cell QC defaults are technical minima, not a validated biological filter. scVI does not make sample/condition confounding identifiable. Leiden labels are not biological cell-type annotation.

See [API sources and provenance](docs/API_SOURCES_AND_PROVENANCE.md) and [Validation](docs/VALIDATION_SUMMARY.md) for precisely what was implemented versus executed.
