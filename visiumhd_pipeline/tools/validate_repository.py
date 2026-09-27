#!/usr/bin/env python3
"""Static source/notebook checks. Does not run notebooks or load biological datasets."""
from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = """
vhd/__init__.py vhd/core.py vhd/io.py vhd/images.py vhd/plots.py
vhd/palette.py vhd/qc.py vhd/preflight.py vhd/storage.py
vhd/segmentation.py vhd/cells.py vhd/integration.py
vhd/adapters/__init__.py vhd/adapters/anndata_input.py
vhd/adapters/assemble.py vhd/adapters/bins.py vhd/adapters/cells.py
vhd/control/__init__.py vhd/control/manifest.py vhd/control/runner.py
vhd/control/storage.py vhd/compute/__init__.py vhd/compute/launch.py
vhd/compute/worker.py vhd/compute/integration_worker.py
vhd/compute/scvi_worker.py vhd/compute/clustering.py vhd/compute/probe.py
vhd/adapters/README.md vhd/control/README.md vhd/compute/README.md
scripts/run_pipeline.py tools/strip_source_notebooks.py
tests/test_core.py tests/test_control.py tests/test_resume_optional.py
tests/test_spatial_optional.py tests/test_scvi_optional.py
config/samples.fixed_schema.template.csv config/samples.cells_only.example.csv
config/samples.four_studies.example.csv config/samples.legacy_migration.example.csv
config/settings.g5_24xlarge.example.json config/settings.multigpu_optin.example.json
env/ci.requirements.in pyproject.toml pytest.ini
""".split()


def validate(root: Path = ROOT) -> dict:
    errors = []
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        errors.append('Missing required paths: ' + ', '.join(missing))
    python_files = sorted(p for name in ('vhd', 'tests', 'tools', 'scripts')
                          for p in (root / name).rglob('*.py')
                          if '__pycache__' not in p.parts)
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
    notebooks = sorted((root / 'notebooks').glob('*.ipynb'))
    if len(notebooks) < 15:
        errors.append(f'Expected at least the 15 foundation notebooks; found {len(notebooks)}.')
    code_cells = 0
    for path in notebooks:
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            if notebook.metadata.get('widgets'):
                raise ValueError('Saved widget payload is not allowed.')
            for number, cell in enumerate(notebook.cells):
                if cell.get('attachments'):
                    raise ValueError(f'Cell {number} contains an attachment.')
                if cell.cell_type != 'code':
                    continue
                code_cells += 1
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    raise ValueError(f'Cell {number} contains saved execution state.')
                ast.parse(cell.source, filename=f'{path.name}:cell{number}')
        except Exception as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
    csv_files = sorted((root / 'config').glob('samples.*.csv'))
    try:
        module = ast.parse((root / 'vhd/control/manifest.py').read_text(encoding='utf-8'))
        assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'COLUMNS'
                                  for target in node.targets))
        headers = ast.literal_eval(assignment.value)
        for path in csv_files:
            with path.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != headers:
                    raise ValueError(f'{path.name}: fixed CSV headers/order differ from COLUMNS.')
                for line, row in enumerate(reader, 2):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f'{path.name}:{line}: inconsistent CSV field count.')
    except Exception as exc:
        errors.append(f'CSV schema: {exc}')
    for path in sorted((root / 'config').glob('*.example.json')):
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f'{path.name}: {exc}')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'python_files': len(python_files), 'notebooks': len(notebooks),
            'code_cells': code_cells, 'csv_templates': len(csv_files),
            'notebooks_executed': False, 'biological_validation': False}


if __name__ == '__main__':
    try:
        print(json.dumps(validate(), indent=2))
    except Exception as exc:
        print(f'Repository validation failed:\n{exc}', file=sys.stderr)
        raise SystemExit(1)
