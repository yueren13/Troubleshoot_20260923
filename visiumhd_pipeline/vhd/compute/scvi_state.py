"""Dependency-light stage decisions for Issue #2. No model is loaded here."""
from pathlib import Path
import json
from ..core import file_stamp, reusable, dump_json


def record_model_receipt(path, signature, **details):
    p = Path(path)
    model = p / 'model' / 'model.pt'
    if not model.is_file() or model.stat().st_size == 0:
        raise RuntimeError('Model save did not produce a nonempty model.pt.')
    dump_json({**details, 'signature': signature,
               'model_stamp': file_stamp(model, full_hash=True)}, p / '09_model_saved.json')


def classify_training_state(path, signature, phase):
    """Fresh, saved-model-ready, completed, or invalid partial. Never overwrite.

    Partial inference is rejected without destroying the model. Legacy receipts
    without fingerprints need explicit verification/migration, not blind adoption.
    """
    if phase not in ('train', 'infer'):
        raise ValueError('phase must be train or infer.')
    p = Path(path)
    model = p / 'model' / 'model.pt'
    receipt = p / '09_model_saved.json'
    inference = [p / 'latent.npy', p / 'training_obs.parquet', p / 'training_var.csv']
    marker = p / '09_train_complete.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get('signature') != signature:
            raise RuntimeError('Saved model configuration mismatch; use a new run_id.')
        if not model.is_file() or model.stat().st_size == 0:
            raise RuntimeError('Model receipt exists but model.pt is missing/empty.')
        if 'model_stamp' not in record:
            raise RuntimeError('Legacy unverified model receipt; verify and migrate it explicitly or use a new run_id.')
        if record['model_stamp'] != file_stamp(model, full_hash=True):
            raise RuntimeError('Saved model changed since its receipt.')
    elif marker.exists():
        raise RuntimeError('Completed run lacks a verified model receipt; inspect before reuse.')
    if marker.exists():
        reusable(marker, signature, [model, *inference])
        return 'complete'
    if receipt.exists():
        if any(x.exists() for x in inference):
            raise RuntimeError('Incomplete inference outputs exist. Retain model/receipt; inspect and quarantine only inference outputs before retrying.')
        return 'model_ready'
    if (p / 'model').exists() or any(x.exists() for x in inference):
        raise RuntimeError('Untracked partial model/inference outputs; no automatic overwrite or deletion.')
    if phase == 'infer':
        raise RuntimeError('Inference requires a matching trained-model receipt.')
    return 'fresh'
