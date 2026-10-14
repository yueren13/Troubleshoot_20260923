# Instructions for contributors and coding assistants

## Source of truth

Use `visiumhd_pipeline/` as the maintained pipeline. Read its README, START_HERE,
CSV_SCHEMA, VALIDATION_SUMMARY, BASELINE_PROVENANCE and ROADMAP before changes.
Preserve `project_1/` as historical source unless the owner explicitly requests an edit.
Read the current branch and files before proposing a fix; do not rely on remembered ZIP contents.

## Review workflow

Work on a focused feature/fix branch. Open a pull request with changes, tests,
compatibility effects and unexecuted checks. Do not push to main, auto-merge,
force-push, delete data or relax storage policy without explicit authorization.
Do not change the software license or publication visibility on the owner's behalf.

## Scientific invariants

- Canonical bin x/y are Space Ranger full-resolution column/row pixels.
  Keep full-resolution coordinates and image transforms explicit.
- Raw counts remain raw. Require an explicit count source; do not round corrected
  expression or replace Proseg counts with geometric bin-center assignments.
- Keep both accepted tissue-high QC classes. Missing annotations are unassessed.
- Standalone AnnData cells can enter midway without Space Ranger or bins.
  Missing boundaries must not become invented whole-cell masks or areas.
- Cells and bins remain separate loadable tables. A selected table may still be
  in-memory; do not advertise fully backed loading without implementing/testing it.
- Integration groups are explicit. Sample is batch, not a substitute for biological
  design; clustering is not cell-type annotation.
- Do not silently change segmentation/QC/integration defaults during maintenance.

## Data and safety

This repository is public. Use synthetic fixtures and placeholder paths only.
Never commit credentials, real sample manifests, images, expression objects,
private paths, identifiers, or notebook outputs. Inspect staged changes; .gitignore
is not a privacy guarantee and cannot remove historical exposures.
Do not execute imported notebooks automatically. Keep dry runs, review gates,
storage allowlists and no-overwrite behavior intact.

## Testing and reporting

From `visiumhd_pipeline/`:

```bash
python tools/validate_repository.py
python -m pytest -q tests/test_core.py tests/test_control.py
python -m pytest -q -rs
```

Use `env/ci.requirements.in` for lightweight checks. Scientific roundtrip and
GPU tests belong in approved target environments. Count skips separately and
state what was not run. Never claim GPU speedup or biological validity from
syntax checks or control-layer tests. Update CHANGELOG and add a regression test
for behavior changes. Record the commit SHA, configuration and environment for runs.
