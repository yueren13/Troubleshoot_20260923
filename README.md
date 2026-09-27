# Troubleshoot_20260923
Codes for trouble shooting

## Maintained Visium HD pipeline

The reusable pipeline now lives in **[visiumhd_pipeline/](visiumhd_pipeline/README.md)**.
Start with its [run guide](visiumhd_pipeline/docs/START_HERE.md),
[CSV schema](visiumhd_pipeline/docs/CSV_SCHEMA.md), and
[validation limits](visiumhd_pipeline/docs/VALIDATION_SUMMARY.md).

- `visiumhd_pipeline/`: v2 review foundation, helper package, 15 notebooks, tests, generic examples and environment guidance.
- `project_1/`: original troubleshooting notebooks, preserved unchanged. Do not use these as the maintained entry points or overwrite them during migration.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch/pull-request workflow,
[AGENTS.md](AGENTS.md) for engineering constraints, and the
[roadmap](visiumhd_pipeline/docs/ROADMAP.md) for validation priorities.

**The foundation source import is complete.** Its [import and validation record](visiumhd_pipeline/IMPORT_STATUS.md) distinguishes source completeness from scientific runtime validation. The work is in [draft PR #1](https://github.com/yueren13/Troubleshoot_20260923/pull/1), not merged into `main`. Resolve the [scVI training-to-inference guard issue](https://github.com/yueren13/Troubleshoot_20260923/issues/2) before a training pilot.

**Review release, not production validation.** CPU unit tests do not validate whole-slide processing, biological accuracy, GPU workflows or institutional storage compliance.

This repository is public. Commit source, synthetic tests and sanitized examples only.
Keep real sample manifests, credentials, images, matrices and executed notebook outputs outside version control.
The addition of ignore rules does not remove information already present in `project_1/` or Git history.
No software license is selected by this foundation commit; preserve third-party attribution and review distribution rights before release.
