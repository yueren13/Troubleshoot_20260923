# Fixed CSV schema (41 columns)

Every supplied CSV has the same headers. Keep optional columns, leaving their cells blank; unknown/missing/duplicate headers cause an error. CSV values are parsed as text, preserving leading-zero sample IDs. Quote fields containing commas. Relative paths are resolved against the CSV's directory, never the notebook's current directory. Only local/POSIX paths and `s3://bucket/prefix` are accepted; scratch must be local.

## Identity and execution

| Column | Values / purpose |
|---|---|
| enabled | true/false; blank means enabled. Disabled rows are skipped. |
| dataset_id | Filesystem-safe study/project identifier. |
| sample_id | One sample, unique within the dataset. Combined key: dataset_id__sample_id. |
| stage | raw, bin_imported, bin_QCed, cell_segmented, cell_annotated. Case-insensitive; bin_inputed accepted as alias. |
| goal | bin_QCed, cell_annotated, export_only. Blank defaults to export_only for annotated entry, otherwise cell_annotated. |
| integration_group | Explicit group to combine; blank = no pooled integration. Only cell-producing goals may join a group. |

## Source data and image

| Column | Purpose |
|---|---|
| spaceranger_outs | Directory containing binned_outputs/square_002um. Raw import expects unfiltered matrix, spatial/tissue_positions.parquet and scalefactors_json.json. Optional at cell entry. |
| segmentation_image | Original uint8 RGB/RGBA H&E level-0 image path. It is not a resized hires.png substitute. Optional for cell-only native-coordinate output. |
| image_type | auto, tif/TIF/tiff, btf alias or svs. Normalized case-insensitively. Actual file content is inspected by OpenSlide/tifffile; declaring a type cannot make an invalid file valid. |
| bin_input | Existing AnnData h5ad/zarr or SpatialData zarr containing the bins. Optional for cell-only import. |
| bin_input_kind | auto, anndata, spatialdata. AnnData root encoding is inspected; a .zarr suffix is not treated as proof of SpatialData. |
| bin_table | Required when the bin input is SpatialData; e.g. square_002um or bins_2um. Blank for standalone AnnData. |
| bin_counts_source | X or layers:<name>; blank defaults to X. Values must be nonnegative finite integer counts. |
| bin_qc_column | Existing accepted class annotation; default qc_class. Used when entry stage says QC is complete. |
| bin_feature_id_column | Optional var column to become feature IDs; otherwise preserve bin var_names. |
| bin_gene_symbol_column | Optional bin var gene-symbol column. Otherwise use gene_symbol when present, or var_names. |
| cell_input | Existing cell AnnData h5ad/zarr (or explicit SpatialData table). Required for cell entry stages. |
| cell_input_kind | auto, anndata, spatialdata. |
| cell_table | Required for SpatialData input, blank for standalone AnnData. |
| counts_source | REQUIRED for count-first cell import: X or layers:<name>. Not required for export_only. No normalized-matrix inference. |
| feature_id_column | Optional cell var column to become feature IDs; otherwise preserve var_names. No silent duplicate suffixing. |
| gene_symbol_column | Optional cell var symbol field; otherwise gene_symbol or var_names. Supply real symbols for mt/ribosomal QC when var_names are Ensembl IDs. |
| cell_id_column | Optional obs column matching boundary IDs; otherwise use obs_names. IDs must be nonmissing and unique. |
| spatial_key | Cell obsm coordinate key, default spatial. Must be finite N×2. Not required for export_only. |
| cell_qc_column | Optional existing Boolean QC-pass column to intersect with the new technical eligibility mask. Without this column, original QC is preserved as metadata but not automatically applied. |

## Coordinate and geometry declarations

| Column | Purpose |
|---|---|
| cell_coordinate_mode | fullres_xy_pixels; image_xy_um; legacy_sr_row_col_um; affine; unregistered. Required for count-first cell import. |
| cell_to_fullres_affine | For affine mode, six semicolon-separated a;b;c;d;e;f coefficients or a quoted JSON 3×3 matrix. |
| cell_boundaries | Optional local/S3 GeoParquet or supported GeoJSON boundaries. Same coordinate system as cell centroids is REQUIRED. |
| boundary_id_column | Column matching cell_id_column/obs_names, or __index__ for a GeoParquet index. |
| tenx_to_image_affine | Original Space Ranger fullres x/y -> supplied image level-0 x/y. Blank means identity. Six coefficients a;b;c;d;e;f, not Shapely ordering. |
| image_mpp_x, image_mpp_y | Both positive microns/pixel axes, or both blank to use trustworthy OpenSlide metadata. Ordinary TIFF fallback requires explicit MPP. |
| source_mpp | Space Ranger coordinate calibration, not image/hires scale. Required for legacy swapped Proseg coordinates unless available from supplied Space Ranger outputs; bins also require it. |
| coordinates_reviewed | true only after reviewing image registration. Unregistered cell-only import does not imply registration. |
| fresh_qc_reviewed | true only after reviewing new H&E/UMI QC thresholds and plots. Not needed when accepted bin classes are imported. |

`fullres_xy_pixels` means x=original image column and y=original image row. `image_xy_um` uses supplied image MPP and the configured image registration. `legacy_sr_row_col_um` means old Proseg first coordinate is SR row*source_mpp, second coordinate SR column*source_mpp. `unregistered` preserves native units without implying microns or full-resolution pixels. Such cells cannot share a combined bin store until registration is supplied.

## Storage and optional metadata

| Column | Purpose |
|---|---|
| output_object_dir | Approved local/S3 durable object root, separate from source and scratch. |
| output_plot_dir | Approved local/S3 figure root. |
| output_data_dir | Approved local/S3 table, log, status and provenance root. |
| tmp_dir | Approved local/POSIX scratch. Active expression/masks/models and staged source copies are stored here, not just small temporaries. |
| metadata_json | Optional LOCAL JSON map of sample metadata to add to obs. Collisions with existing cell annotations are rejected. |
| overrides_json | Optional LOCAL JSON analysis overrides. Routing and policy cannot be changed here. Use CSV/settings for those. |

All four storage fields are required for a uniform fixed schema, even if a particular stage produces no plots. Policy allowlists in settings must explicitly permit these paths; they do not replace IAM or institutional storage approval.

## Minimum entry requirements

| Entry | Required beyond identity/storage |
|---|---|
| raw | spaceranger_outs + segmentation_image, image calibration |
| bin_imported | bin_input (+ bin_table if SpatialData), segmentation_image, original bin coordinates or SR positions, count source/calibration |
| bin_QCed | Same as bin_imported, plus the accepted bin_qc_column |
| cell_segmented | cell_input + explicit counts_source + spatial_key + cell_coordinate_mode; polygons/image/bins optional |
| cell_annotated / export_only | cell_input (+ cell_table if source is SpatialData); no raw counts or coordinates required |

`cell_annotated` with goal=cell_annotated intentionally uses the count-first migration route, recalculating technical QC while preserving source annotations. Use export_only for a no-transformation full-object export. Nonspatial AnnData is exportable, but it cannot become a spatially registered cell product without coordinates.
