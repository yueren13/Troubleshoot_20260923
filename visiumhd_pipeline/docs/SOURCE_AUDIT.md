# Source audit and scope

This repository foundation imports the reusable v2 review bundle supplied on September 24, 2026. It preserves the scientific implementation of that bundle; repository setup is not a new biological validation.

The historical workflow used three early preprocessing notebooks and later StarDist/Proseg/postprocessing notebooks. The previous review inspected portions of the later notebooks but could not retrieve complete source for the large early notebooks or the two original helper modules. Fresh H&E normalization and bin QC are therefore explicit baselines, not a verified exact reproduction of those early methods. Importing existing accepted QC is the preservation route.

Study-specific paths and sample manifests are deliberately excluded from the public foundation. A generic legacy-migration example replaces the study-specific CSV. The earlier archive is not republished. Original files already present under `project_1/` are unchanged; this addition is not a privacy audit or history cleanup of those files.

See `BASELINE_PROVENANCE.md`, `VALIDATION_SUMMARY.md`, and `ROADMAP.md` for provenance, runtime limits, and follow-up work.
