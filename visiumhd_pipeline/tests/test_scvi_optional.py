"""Opt-in CPU API smoke test, not a biological or GPU performance benchmark."""
import os
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('VHD_TEST_SCVI')!='1',reason='Set VHD_TEST_SCVI=1 in the scVI environment.')


def test_scvi_count_model_roundtrip(tmp_path):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import anndata as ad
    import scvi
    x=np.random.default_rng(42).poisson(2,size=(80,30)).astype(np.int32)
    a=ad.AnnData(X=sparse.csr_matrix(x),obs=pd.DataFrame({'sample':['a']*40+['b']*40}))
    scvi.settings.seed=42
    scvi.model.SCVI.setup_anndata(a,batch_key='sample')
    model=scvi.model.SCVI(a,n_latent=3,gene_likelihood='nb')
    model.train(max_epochs=1,accelerator='cpu',devices=1,batch_size=32,train_size=.8,early_stopping=False)
    z=model.get_latent_representation()
    assert z.shape==(80,3) and np.isfinite(z).all()
    model.save(str(tmp_path/'model'),save_anndata=False)
    assert (tmp_path/'model'/'model.pt').exists()
