# Changelog

## Unreleased — repository foundation (2026-09-25)

- Import the v2 review pipeline into `visiumhd_pipeline/`, separate from the preserved historical notebooks.
- Retain 15 notebooks, structured helpers, CSV midway entry, selective table loading, independent storage and optional parallel-compute launchers.
- Replace the study-specific migration CSV with a generic legacy-study example; omit study-specific historical archive material.
- Add contributor/assistant guidance, notebook hygiene checks, lightweight CI, issue/PR templates and ignore rules.
- No intentional scientific algorithm changes. Package version remains 0.2.0 review.
- Scientific roundtrips, whole-slide segmentation, scVI/DDP, RAPIDS/Dask and S3 runtime validation remain pilot work.

## 0.2.0 review — previous delivery (2026-09-24)

CSV-driven entry stages, standalone AnnData adapters, policy-aware publication and isolated CPU/GPU launchers.
