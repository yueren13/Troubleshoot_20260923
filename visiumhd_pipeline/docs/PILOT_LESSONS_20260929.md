# Visium HD pilot: bugs, setup pitfalls and lessons learned

**Review date: September 29, 2026.** This is a retrospective, not a production-readiness certificate. It documents the owner's workflow, supplied code and pilot logs, and our joint troubleshooting. Deployment paths, sample identities, treatment data and raw logs are deliberately omitted.

Read alongside [the troubleshooting runbook](TROUBLESHOOTING_RUNBOOK.md) and [decisions / backlog](PILOT_DECISIONS_AND_BACKLOG.md). Original `project_1/` sources and biological inputs are not changed by this documentation.

## 1. Status is not one Boolean

| Status used here | What it establishes |
|---|---|
| Confirmed observation | A supplied log or inspected source supports the specific behavior. It may not establish its cause. |
| Local correction delivered | A review ZIP/hotfix was supplied. This does not mean it is in the repository or installed everywhere. |
| User-reported progress | The owner reported a check or stage working. Do not infer completion of all samples or all downstream steps. |
| Committed draft fix | Code exists in a review branch/PR; it is not necessarily merged or scientifically validated. |
| Open requirement | Agreed desired behavior not implemented in the reviewed version. |

At this review, [PR #1](https://github.com/yueren13/Troubleshoot_20260923/pull/1) is the open, draft, unmerged v2 foundation at `f706a5bfb63f6256ddf8e08c1497725f37c50249`. [PR #3](https://github.com/yueren13/Troubleshoot_20260923/pull/3) is an open, draft, unmerged scVI handoff fix at `6de1ffcfc2471b9f5125033493ab15a090f41be9`. The complete v3 review bundle and later local hotfixes are not yet that merged baseline. A branch name, ZIP delivery, passing syntax check, and completed biological run are different milestones.

The owner reported notebook 99 passing after the validator/test-runner corrections and later inspected generated report images. The provided record does not establish a completed, read-back-validated full-cohort scVI/GPU analysis. This review does not run one.

## 2. Onboarding and execution incidents

These are recorded as reproducible interactions between operator actions and the interface, not as a judgment of the operator's ability. The maintainer should make the correct operation discoverable.

| ID | Trigger and observed symptom | Finding and corrective lesson | Status / evidence |
|---|---|---|---|
| OP-01 | CSV/settings copied into `config/`; `Unset path variable: VHD_WORK_ROOT`. | The examples depended on environment variables in the current kernel. Copying two files did not set them. Provide one explicit notebook-first setup path and list every resolved dependency. | Confirmed log; setup guidance delivered. E01, E02. |
| OP-02 | CSV moved while retaining `../../metadata/...`. | Relative sidecars are resolved from the CSV folder. Relocation changes their meaning. Keep the bundle intact or update references; never infer missing sample metadata. | Source/configuration review; not asserted as a separately confirmed user failure. E02. |
| OP-03 | Package `__file__` location used to infer a Python library-directory setting. | A `VHD_PY_*` field needs the intended environment's executable, not its package/library folder. Inspect `sys.executable` in that environment. A different kernel gives a different answer. | User question resolved by clarification. E01. |
| BUG-01 | Configured venv Python; workers used base Python and could not import NumPy. | Executable-path `Path.resolve()` followed the venv symlink. Preserve the invocation path for executables; retain normal safety handling for data paths. Check imports in the actual worker before batch submission. | Log/source-confirmed; local execution hotfix delivered; later preflight used the venv. E03, E05. |
| BUG-02 | Notebook 99 rejected its own saved execution count/output. | A repository hygiene rule was incorrectly used as a working-notebook prerequisite. Separate strict repository validation from working mode; neither should execute notebooks. | Reproduced and local fix delivered; user reported static validation passed. E01, E04. |
| OP-04 | Outer `CalledProcessError` hid the failing command's explanation. | Capture plus `check=True` raised before printing. Print stdout/stderr before stopping; do not suppress failures. | Diagnostic reporting corrected. E01, E04. |
| OP-05 | `No module named pytest` after static validation passed. | The test runner was absent from the selected environment; tests had not started. Install the test dependency there, not scientific packages in the base interpreter. | User reported notebook 99 passed after installation. E01. |
| UX-01 | Notebook 06 ran without error, but no new `tables/cells` existed. | Preview is not execution. The final cell printed expected paths without checking products. Display mode, scope and effective configuration; verify existence and readback separately. | Source-confirmed reporting weakness; run-controls hotfix delivered. E03, E06. |
| OP-06 | Execution enabled, then `storage.single_writer_ack` raised `PermissionError`. | An intentional pipeline acknowledgement was mistaken for OS permissions. Execution, coordinate approval and writer acknowledgement are distinct. Check once before launching a batch; do not auto-approve unknown conditions. | Confirmed guard/log; configuration corrected. E05. |
| OP-07 | Assembly failed with `No prepared bins or cells`. | Source H5ADs were not prepared v3 intermediates in this run's workspace. Run import, inspect outputs, then assemble with the same configuration/run ID. A CSV stage does not manufacture prerequisites. | Confirmed missing-prepared-input state; exact prior execution history remains uncertain. Recovery notebook delivered. E06. |
| OP-08 | All configured source images were reported missing. | Initial path information conflicted with the later authoritative legacy manifest. Correct the sample image-path field from the manifest, verify visibility and preserve other fields. Do not relabel a segmentation report as the original image. | Cross-source conflict resolved; path-repair utility delivered. E02, E07. |
| UX-02 | Report images could not be found in historical shared folders. | New reports lived in per-sample work/run directories, not old outputs or publication destinations. Print full paths and offer an output index / inline image display. | Owner subsequently reported inspecting the reports. E01, E06. |
| OP-09 | One worker confused with one sample; planned scope unclear. | Sample selection and concurrency are independent. Unknown RAM forces a serial pilot; a provisional estimate is not a measured peak or hard cap. A single pasted record mixed one planned sample with twelve failures; do not invent a root cause. | Semantics clarified; scope discrepancy needs reproducible evidence. E03, E05. |

**Follow-up:** [Issue #4](https://github.com/yueren13/Troubleshoot_20260923/issues/4) consolidates the local fixes, setup contract, error reporting, output index and rerun tests into a reviewable release. None of the observations above justifies deleting partial stores, disabling signature checks, or changing the user's mounted infrastructure.

## 3. Scientific/helper findings

| ID | Confirmed code behavior or limitation | Required contract | Status / evidence |
|---|---|---|---|
| SCI-01 | Legacy XML import subtracts negative regions, then rebuilds a single Polygon from its exterior only. Interior exclusions can disappear. | Preserve holes and multipart geometry. Synthetic outer area 100 minus inner area 4 must remain 96, not 100. Plot cleanup cannot restore a discarded hole. | Synthetic reproduction recorded; real annotation impact not established. E08. |
| SCI-02 | Annotation reruns initialize only absent columns, then add true memberships. | Recompute and replace source-owned memberships. Moving/shrinking/deleting a polygon must remove obsolete positives. Preserve unknown coverage separately. | Synthetic/stub-query reproduction recorded; not a real spatial I/O test. E08. |
| SCI-03 | Whole-object extent was used for image scale; separate X/Y scales were later averaged; missing-frame Identity was assumed in legacy code. | Named reference frame, units, offsets and transforms must be explicit. Match sample sources by IDs, not list order. Approval follows overlays, not an inferred identity. | Source-confirmed risk; exact historical mismatch cause unresolved. E08. |
| SCI-04 | Fixed-basis Macenko could still apply per-image percentile concentration amplification. | Tissue evidence is not aesthetic stain normalization. Preserve fixed-reference extraction, fixed concentration divisors, explicit white reference and numerical values. | Quantitative objective clarified; fixed-score implementation supplied separately in v3. E09. |
| SCI-05 | Fixed-score threshold rules could overwrite each other; dtype/layout and invalid-value edge cases were underspecified. | Reject conflicting/incomplete thresholds, declare intensity range/channel axes, distinguish in-bounds from valid measurements, and freeze optional white calibration per source rather than per tile. | Synthetic checks and v3 review changes delivered; real-slide calibration not claimed. E09. |
| SCI-06 | Combining image H/E then interpolating is not generally equivalent to interpolating H/E then combining. | Preserve the requested operation order. Adjacent H-only/E-only pixels yield midpoint score 1.00 in the former order versus 0.75 in the latter synthetic example. | Mathematical/synthetic regression case. E09. |
| IO-01 | Legacy persistence writes the updated table under a backup name, deletes old data, then writes the updated table again. Newer non-table defaults can save transformations without changed vertices. | Distinguish add, data replacement, transform-only and metadata-only operations. Recovery content and prior on-disk data are different. Preserve counts; validate reopened metadata. | Source-confirmed; selective persistence supplied in v3 review, actual mount recovery still requires testing. E08, E10. |
| SCI-07 | Bin-center categorical/continuous transfer and prior gating have limited spatial semantics. | Majority agreement and annotation coverage need separate denominators. Independent Boolean memberships are not exclusive classes. Center-sampled sums are not physical integrals. Seed filtering is not input-bin filtering or a hard Proseg boundary. | Explicit scope retained; do not expand claims beyond implemented measurements. E10. |
| BUG-03 | The v2 final completion guard rejected a valid saved scVI model awaiting separate inference. | Distinguish fresh, verified model-ready, complete and inconsistent states without weakening fingerprints/overwrite protection. | [Issue #2](https://github.com/yueren13/Troubleshoot_20260923/issues/2), draft PR #3; actual training/DDP handoff not established here. E11. |

**Follow-up:** [Issue #8](https://github.com/yueren13/Troubleshoot_20260923/issues/8) covers promotion and real-environment validation of H&E, annotation and selective-persistence contracts. Its existence does not mean the changes are merged.

## 4. Latest agreed improvements remain open

- [Issue #5](https://github.com/yueren13/Troubleshoot_20260923/issues/5): compact hires/lowres master images and an external original-image reference, while preserving full-resolution coordinates. Current reviewed v3 still embeds original-resolution H&E. Lazy reads are not zero-cost writes.
- [Issue #6](https://github.com/yueren13/Troubleshoot_20260923/issues/6): explicit analysis QC before integration. Owner-selected strict mitochondrial thresholds are human <20% and mouse <10%; cell total raw counts >=3, adjustable to 16; gene detection in >=3 retained cells. Counts per cell, genes per cell and cells per gene are different quantities. Gene-filter scope must be declared.
- [Issue #7](https://github.com/yueren13/Troubleshoot_20260923/issues/7): log10(1+count) spacing with original-count tick labels 0, 10, 100, 1,000, 10,000; metadata-only replot. Display changes must not change RNA counts, H&E optical density, QC flags or scVI inputs.

## 5. Evidence index and limits

| Evidence | Basis available during this review |
|---|---|
| E01 | This project conversation: user code/errors, clarifications, patch application and reported progress. No transcript or personal statements are republished. |
| E02 | Supplied scenario configurations and sidecar setup; initial sample/path description. |
| E03 | Supplied worker log showing base-Python invocation and missing NumPy; `VisiumHD_v3_Execution_Hotfix_20260928.zip`. |
| E04 | Saved-notebook validator traceback; `VisiumHD_v3_Validation_Fix.zip` and its diagnostic results. |
| E05 | Supplied log showing successful venv import preflight followed by the writer-acknowledgement guard. |
| E06 | Supplied missing-prepared-input tracebacks; v3 notebook/source review; recovery notebook. |
| E07 | Legacy `sample_manifest.csv`, `pipeline_config(1).json`, and stage summaries. Exact deployment values remain private. |
| E08 | Attached `cfs_202412.py`, `cfs_spatialdata_202412.py`, annotation notebooks and helper-review evidence ZIP. |
| E09 | Attached `cfs_normhe_202605.py`, pasted fixed-score functions, and fixed-score review evidence ZIP. |
| E10 | `VisiumHD_Pipeline_v3_Review_20260928.zip`, implementation/validation records and image/integration-QC source audit. |
| E11 | GitHub PR #1, issue #2, PR #3 and their exact refs, read during this review. |

Historical test totals belong to the named archive/commit and scope. They are not cumulative evidence that a later mixed installation passed every test. This documentation pass did not rerun biological processing, perform GPU benchmarks, access institutional mounts or verify all cohort products. A successful legacy dataset does not prove the new pipeline is equivalent.

Future records should attach a redacted error, code version plus patch list, effective config fingerprint, sample/action scope, expected/observed outputs and verification result. Store sensitive logs/manifests in approved private storage, not this repository. Preserve the distinction between user intent, implementation, and observed execution.
