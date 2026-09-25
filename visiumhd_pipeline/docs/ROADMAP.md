# Validation-first roadmap

This is a development checklist, not a claim that the pipeline is production ready.

## First milestone: one existing segmented sample

- [ ] Run notebook 00 with a private local CSV and approved storage roots.
- [ ] Run notebook 99; verify scientific tests execute rather than skip.
- [ ] Import a known raw-count source and confirm cell/gene IDs and annotation preservation.
- [ ] Validate centroid/polygon transforms on H&E, or explicitly retain an unregistered frame.
- [ ] Assemble and read back cells-only and, when available, dual-table products.
- [ ] Confirm selecting cells does not read the bin expression matrix.
- [ ] Validate export-only preservation and destination separation.

## Scientific/runtime coverage

- [ ] Retrieve output-stripped original early preprocessing notebooks/helper modules and audit fresh QC equivalence.
- [ ] Add deterministic synthetic AnnData/SpatialData roundtrip tests to a pinned scientific CI environment.
- [ ] Capture successful CPU, StarDist, scVI and RAPIDS environment inventories/lockfiles separately.
- [ ] Measure one representative StarDist/Proseg sample before choosing parallel-worker RAM limits.
- [ ] Compare small single-GPU and optional DDP scVI runs; document training/validation differences.
- [ ] Validate RAPIDS neighbor/Leiden/UMAP APIs and Dask behavior in the installed environment.
- [ ] Validate S3 publication and readback only on approved infrastructure.

## Release readiness

- [ ] Record pilot outcomes and known limitations without publishing sensitive data.
- [ ] Review private-data exposure in historical `project_1/` separately; this foundation does not rewrite history.
- [ ] Decide licensing and publication permissions explicitly.
- [ ] Create a versioned release only after the intended entry routes and invariants are validated.
