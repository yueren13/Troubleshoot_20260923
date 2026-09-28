"""CPU-only stage-transition tests for issue #2; no model training is implied."""
import os,json
import pytest
from vhd.compute.scvi_state import classify_training_state,record_model_receipt
from vhd.core import completed


def ready(p):
    (p/'model').mkdir();(p/'model/model.pt').write_bytes(b'synthetic model, not a real artifact');record_model_receipt(p,'s')


def complete(p):
    paths=[p/'model/model.pt']
    for n in ('latent.npy','training_obs.parquet','training_var.csv'):
        f=p/n;f.write_bytes(b'synthetic');paths.append(f)
    completed(p/'09_train_complete.json','s',paths)


def test_fresh(tmp_path):
    assert classify_training_state(tmp_path,'s','train')=='fresh'
    with pytest.raises(RuntimeError):classify_training_state(tmp_path,'s','infer')


def test_saved_model_handoff_without_final_marker(tmp_path):
    ready(tmp_path)
    assert classify_training_state(tmp_path,'s','train')=='model_ready'
    assert classify_training_state(tmp_path,'s','infer')=='model_ready'


def test_completed_reuse(tmp_path):
    ready(tmp_path);complete(tmp_path)
    assert classify_training_state(tmp_path,'s','infer')=='complete'


@pytest.mark.parametrize('case',['signature','missing','empty','changed','partial','legacy'])
def test_invalid_state_rejected(tmp_path,case):
    ready(tmp_path);model=tmp_path/'model/model.pt'
    if case=='missing':model.unlink()
    elif case=='empty':model.write_bytes(b'')
    elif case=='changed':model.write_bytes(b'other')
    elif case=='partial':(tmp_path/'latent.npy').write_bytes(b'partial')
    elif case=='legacy':(tmp_path/'09_model_saved.json').write_text(json.dumps({'signature':'s'}))
    with pytest.raises(RuntimeError):classify_training_state(tmp_path,'wrong' if case=='signature' else 's','infer')
    assert (tmp_path/'09_model_saved.json').exists()


def test_untracked_model_rejected(tmp_path):
    (tmp_path/'model').mkdir()
    with pytest.raises(RuntimeError):classify_training_state(tmp_path,'s','train')


def test_same_size_mtime_tamper_detected(tmp_path):
    ready(tmp_path);complete(tmp_path);model=tmp_path/'model/model.pt';stat=model.stat()
    b=bytearray(model.read_bytes());b[0]^=1;model.write_bytes(b);os.utime(model,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(RuntimeError,match='changed'):classify_training_state(tmp_path,'s','infer')


def test_model_survives_diagnostic_failure(tmp_path):
    ready(tmp_path);record_model_receipt(tmp_path,'s',history_complete=False)
    assert classify_training_state(tmp_path,'s','infer')=='model_ready'
