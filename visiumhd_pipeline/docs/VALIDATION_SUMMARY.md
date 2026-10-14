# v2 validation summary — September 24, 2026

## Executed in the delivery environment

- **66 tests passed; 3 skipped.** Full output: `docs/PYTEST_OUTPUT.txt`.
- **15 notebooks** passed nbformat validation.
- **38 code cells** compiled successfully.
- **35 Python files** (package, tests, CLI and utility) passed syntax parsing.
- Active example JSON files parsed, and the four-study, cell-only and legacy-study CSV examples passed schema/configuration tests.

The executable tests include: exact CSV stage routing, case/alias handling, missing/duplicate ID rejection, explicit counts-source rules, source/output separation, policy prefix boundaries, optional remote-transfer denials, immutable local publication, real SHA256 readback/tamper detection, no automatic overwrite of partial prefixes, hardware concurrency arithmetic, unknown-RAM serial pilot behavior, per-process CUDA/cache environment, torchrun command construction, old/new seed-parameter adaptation, recursive directory fingerprints, real subprocess success/failure reporting, and inherited coordinate/count/QC/palette tests.

These are actual tests of the control layer and tested CPU helpers; they are not all end-to-end biological pipeline tests.

## Not executed here

The delivery environment did not have the full scientific packages, data, NVIDIA GPUs, Proseg binary or access to the user's S3 policy/IAM environment. Therefore the following are **not validated runtime claims**:

- Real AnnData/SpatialData import, encoding roundtrips, cells-only/dual-table assembly and annotation writeback.
- Whole-slide StarDist inference or multi-sample GPU throughput.
- Actual Proseg execution, recovery or performance on the user's files.
- scVI single-GPU training, multi-GPU DDP, model loading and latent inference.
- RAPIDS-singlecell neighbors, UMAP, cuGraph/Dask Leiden and GPU spill behavior.
- S3 staging, upload, IAM access, encryption/retention requirements or downstream remote loading.
- Exact reproduction of the original Step01–03 helper methods.

The three skipped items are the two scientific test modules gated by missing dependencies (the SpatialData module and the new resume-adapter module) and the opt-in scVI test. One module-level skip can represent several unexecuted scientific test cases; “3 skipped” is not evidence that only three scientific assertions remain untested.

`tests/test_resume_optional.py` supplies real tiny h5ad/AnnData-zarr selected-count tests, a cells-only centroid SpatialData roundtrip, and a byte-preserving export test. `tests/test_spatial_optional.py` supplies the two-table coordinate/label/image test and annotation-only writeback test. They must run in the target spatial environment. Notebook 99 launches those tests with the configured scratch path; packages reported as skipped must not be counted as passed.

## Required pilot before scaling

1. Edit real interpreter paths, input paths, storage roots and allowlists. Run notebook 00.
2. Run notebook 99 in the relevant target environments. Confirm expected scientific tests actually execute, not skip.
3. Migrate one representative already-segmented sample with notebook 00B or 05. Check raw count source, feature IDs, cell IDs, cell annotations and coordinate/boundary overlay.
4. Assemble/read the product in 06/07. Confirm the cell-only route does not load or reconstruct missing bins.
5. Run one local and, when permitted, one S3 publication pilot; verify `_SUCCESS.json`, destination categories and a downstream read.
6. Benchmark one representative StarDist and Proseg sample before increasing workers; inspect recorded host-RAM use.
7. For integration, compare a small single-GPU model and optional DDP/RAPIDS-Dask runs in the exact installed environments. API availability and syntax checks alone do not establish biological correctness or speedup.

The supplied environment files are unpinned inputs/guidance, not proven lockfiles. Capture a resolved lockfile and environment inventory after target tests succeed. Preserve the working v1/legacy environments and source objects throughout migration.
