# Pilot troubleshooting runbook

**Scope: v3 review bundle plus the locally delivered execution/validator corrections.** The open v2 foundation does not contain every interface described here. Confirm the installed version and patch list before copying commands. See [pilot evidence and incidents](PILOT_LESSONS_20260929.md).

## The six facts to check before a write

1. **Code:** the imported `vhd` package comes from the intended checkout and its patches are recorded.
2. **Configuration:** the exact CSV and JSON paths, resolved roots and run ID match the intended scenario. Relative sidecars are relative to the CSV.
3. **Interpreter:** each `VHD_PY_*` points to a Python executable in the intended environment, not `lib/python...` or a package `__file__`. Do not resolve away a virtual-environment symlink.
4. **Scope:** selected sample IDs are explicit; enabled rows, sample count, worker count and GPU count are different controls.
5. **State:** required imported intermediates or validated prior products exist in this run's workspace. An input-stage label is not proof of preparation.
6. **Permission to proceed:** execution mode, coordinate review, single-writer acknowledgement and storage allowlists are separately satisfied. Do not turn review flags true without the review.

## Symptom -> interpretation -> safe next action

| Symptom | Interpretation | Next action |
|---|---|---|
| `Unset path variable` | The current kernel cannot expand a referenced variable. | Set approved paths in that kernel/setup, or use explicit configuration paths. Do not create a directory named after the literal placeholder. |
| Sidecar metadata missing after copying CSV | Relative paths now start from a different folder. | Keep the example hierarchy intact or fix the metadata references. Preserve sample IDs and values. |
| Package path ends in `site-packages/.../__init__.py` | This identifies a package, not the worker executable. | Inspect `sys.executable` in the intended environment; retain that executable path. |
| `No module named numpy` in a base-interpreter worker | The worker may not be using the configured venv. | Compare command, `sys.executable`, `sys.prefix`, and `sys.base_prefix`. Correct launcher path handling; do not install packages into shared base Python to conceal the bug. |
| `Executed notebook outputs are not allowed` | Repository hygiene is being applied to active notebooks. | With the corrected validator, use `--mode working`; retain strict default mode for clean-source publication. |
| `No module named pytest` | Tests did not start. | Install the test runner in the exact test interpreter; do not report failed scientific tests. |
| Only `CalledProcessError` is visible | Captured output was hidden by the wrapper. | Print captured stdout and stderr before raising. Preserve the exit status. |
| Notebook finishes, but no output exists | It may have been a preview or only printed planned paths. | Inspect mode and worker/status records; verify products and readback, not merely a printed destination. |
| `No prepared bins or cells` | Assembly lacks current-run intermediates. | Execute the existing-cell/bin import for the same scenario and workspace, review it, then assemble. Do not rename an original bin table to `cells`. |
| `single_writer_ack` PermissionError | Deliberate pipeline gate, not necessarily an OS permission failure. | Review destination and concurrent access; update the active setting only after acknowledgement, then reload configuration. |
| Source images missing | A configured path is unavailable to this process. | Compare with a current source manifest and check visibility. An earlier directory listing is evidence of a recorded location, not present-day access. |
| One worker although more requested | RAM/CPU scheduling constrained the request; unknown RAM deliberately limits concurrency. | Inspect the reason and a genuine completed run's resource report. Do not treat a guessed estimate as measured or a hard cap. |
| Partial store or fingerprint mismatch | An interrupted or conflicting state requires inspection. | Stop; retain data and logs. Use the documented recovery procedure, never broad deletion or force overwrite. |

## Preview and execution: identify the installed interface

The original notebooks derive `EXECUTE` from `VHD_EXECUTE` when the initialization cell runs. Updating the environment later does not retroactively change that Python variable. Separate kernels do not inherit variables assigned inside another kernel.

The execution-hotfix notebooks use a **Run controls** cell instead:

```python
RUN_MODE = "preview"  # change to "execute" only for reviewed actions
SAMPLE_KEYS = ["StudyExample__Sample01"]
ALLOW_ALL_SAMPLES = False
```

Rerun the controls and initialization cells after editing. In the patched sample launchers, selecting all enabled rows is explicit:

```python
SAMPLE_KEYS = None
ALLOW_ALL_SAMPLES = True
```

This selects a queue, not unlimited workers. Integration is grouped separately; do not assume a per-sample selection automatically subsets a pooled integration group. Keep stage-specific reviews and readback checkpoints; running every cell blindly is not the intended review workflow.

## Existing-cell route: the actual prerequisites

```text
Existing annotated H5AD / AnnData Zarr
    -> existing-cell import (00B, or the route-specific recovery notebook)
    -> prepared cells and geometry metadata in the current workspace
    -> review H&E overlay when registered / preserve explicit native frame otherwise
    -> assemble (06, or the recovery notebook's assembly phase)
    -> verify the new cells table and read it (07)
    -> analysis preparation and integration (08-11), after agreed QC work is available
```

No new StarDist/Proseg run is required to repackage existing segmented cells. Real polygons are optional; centroid glyphs are not whole-cell masks. Keep the later QC-annotated expression object rather than silently reverting to an earlier Proseg matrix. Adding bins is a separate route choice. The count-first import is not an archival copy of every layer, graph or source image.

## Where to look

Use the loaded configuration rather than reconstructing long paths by hand:

```python
from vhd.control.manifest import sample_layout
from pathlib import Path

for row in PROJECT["rows"]:
    if row["goal"] == "export_only":
        continue
    paths = sample_layout(PROJECT, row)
    print(row["sample_key"])
    print("Work:", paths["work"])
    print("Master:", paths["store"])
    print("Reports:", paths["report"])
    print("H&E overlay:", Path(paths["report"]) / "cells_qc_on_HE.png")
```

This reports expected locations, not completion. For existing-cell imports, inspect the import log/completion record and prepared `cells_qc.h5ad`; for assembly, inspect the worker status, `storage_manifest.json`, and the actual selected table. Publication destinations are distinct from retained working products. Never assume two institutional mount aliases refer to the same storage without checking in the approved environment.

## Rerunning safely

Keep a run's identity and scientific configuration stable while completing it. Matching completed outputs may be reused only through the stage's validation logic. Changed scientific inputs, incomplete outputs and published snapshots require the documented new-run or deliberate recovery path. Do not change the run ID merely to increase worker count, and do not mix pilot and full-cohort integration outputs.

Concurrent workers may write different sample stores; do not run overlapping writers on one master. A single-writer acknowledgement is not an automatic lock against every external process. A recoverable multi-file replacement is not a claim of atomicity on an object-storage mount. No pipeline step should mount, unmount or repair institutional storage.

## A useful issue report

Include the action and exact error tail, installed source/patch version, effective interpreter, selected sample count, config/run fingerprint, expected product and observed status, plus whether the run was preview or execute. Redact private roots, sample identifiers and sensitive values before posting. Report a passed syntax check, a successful import, a read-back-valid master and a trained model as separate outcomes.

For the full incident history, see [PILOT_LESSONS_20260929.md](PILOT_LESSONS_20260929.md). For work that remains unimplemented, see [PILOT_DECISIONS_AND_BACKLOG.md](PILOT_DECISIONS_AND_BACKLOG.md). This runbook does not install fixes or authorize changes to active data.
