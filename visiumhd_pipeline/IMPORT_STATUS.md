# Foundation source import: complete

Updated September 25, 2026. Review the changes in [draft PR #1](https://github.com/yueren13/Troubleshoot_20260923/pull/1).

The interrupted upload has been recovered. The development branch now contains the complete v2 helper package, 15 source-only notebooks, five test files, fixed-format CSV examples, environment guidance, documentation, the notebook-audit utility, and the repository validation tool. CPU-only GitHub Actions checks have also been configured.

**Source-import completion is not production or biological validation.** The pull request remains a draft; it has not been merged into `main`.

## Integrity and intentional changes

The entire `vhd/` directory matches the original supplied v2 ZIP byte-for-byte. Its Git tree SHA is `e8ccb93e6323590aefe5c29b152474ec3def4c87` (28 Python files and three helper READMEs). No scientific algorithms were changed during the foundation import.

The repository version replaces the study-specific migration example with `config/samples.legacy_migration.example.csv`, using generic study/sample identifiers and `/EDIT/` paths. Corresponding notebook example comments and the test filename reference were updated. Contributor guidance, source-completeness checks and lightweight CI are repository additions.

The original `project_1/` tree remains `0715d1d3a04ff67313100974c91428917ac95f6c`. No original notebooks were overwritten and no changes were pushed to `main`.

## Validation performed during recovery

On the local review copy matching the committed helper, notebook and test sources:

- CPU/synthetic tests: **66 passed**.
- Full available test collection: **66 passed, 3 skipped**. Missing scientific dependencies and the opt-in scVI test account for the skips; skipped modules are not successful runtime validation.
- Static validation: **36 Python files, 15 notebooks, 38 notebook code cells, and four sample CSV templates** passed.
- No notebook was executed against a biological dataset.

Reproduce the lightweight checks from `visiumhd_pipeline/`:

```bash
python tools/validate_repository.py
python -m pytest -q tests/test_core.py tests/test_control.py
```

The live GitHub Actions result is available on PR #1 and the repository Actions page; configuring the workflow does not itself establish a passing run.

## Known blocker before scVI training

[Issue #2](https://github.com/yueren13/Troubleshoot_20260923/issues/2) records a reproduced state-guard problem: a valid saved model awaiting inference is incorrectly treated as an untracked partial output. Fix and regression-test that handoff before using notebook 09 for a training pilot. Preserving the original source does not imply that every runtime path is correct.

## Still requires target-environment validation

AnnData/SpatialData roundtrips, original-image registration, real-sample StarDist and Proseg, scVI single-GPU/DDP execution, RAPIDS/Dask behavior, S3 publication, and institutional storage-policy approval remain separate pilot tasks. Start with one existing-cell sample in approved storage, not a full production run.

This repository is public. Commit source and sanitized examples only. Existing public history is not scrubbed by the newly added ignore rules. Repository visibility, branch protection, and software licensing were not changed.
