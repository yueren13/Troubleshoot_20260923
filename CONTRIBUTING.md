# Working on the Visium HD pipeline

## A normal change

1. Start from the current repository branch and create `fix/<topic>` or `feature/<topic>`.
2. Make the smallest testable change in `visiumhd_pipeline/`; keep notebooks thin and helpers structured.
3. Add a regression test and update the changelog and relevant documentation.
4. Run the lightweight checks and any applicable target-environment tests.
5. Open a pull request describing the change, compatibility, actual test results and remaining checks.
6. The owner reviews and merges. Do not auto-merge or treat the review release as production validated.

When asking for help, share the repository path, branch or commit, notebook/action,
redacted traceback, package versions and expected result. Use an issue for a tracked
bug or enhancement and refer to it from the pull request. A new chat should read the
current repository and AGENTS.md rather than assume a downloaded bundle is current.

## Local setup

```bash
git clone https://github.com/yueren13/Troubleshoot_20260923.git
cd Troubleshoot_20260923
# During initial review, checkout pipeline/visiumhd-v2-foundation.
cd visiumhd_pipeline
python -m venv .venv
. .venv/bin/activate
python -m pip install -r env/ci.requirements.in
python -m pip install -e . --no-deps
python tools/validate_repository.py
python -m pytest -q tests/test_core.py tests/test_control.py
```

The CI environment is not a StarDist/scVI/RAPIDS environment. Do not upgrade an
established scientific environment just to install the controller. Follow the
compute guide and notebook 99 for scientific tests. Optional tests that skip do
not establish runtime compatibility.

## Keep private configuration private

Copy a generic CSV to `config/samples.csv` and a settings example to
`config/settings.json`; both local paths are ignored. Preserve headers and edit
paths on approved infrastructure. Do not commit populated study manifests.
Notebook outputs must be cleared before committing. Inspect `git diff --cached`
and `git status` before every push. There is no automatic synchronization of your
EC2 filesystem, S3, chat attachments or repository.

## Reproducibility and releases

Record `git rev-parse HEAD`, the resolved config, selected count layer, environment
inventory and execution logs with each approved analysis run. Keep sensitive run
metadata outside this public repository. A release/tag is a code snapshot, not a
validation certificate. Create a stable release only after the pilot checklist is
satisfied. Do not commit GPU binaries, learned model weights or raw datasets.

No license is chosen on the owner's behalf; preserve upstream notices and settle
licensing/publication rights before distributing a stable public release.
