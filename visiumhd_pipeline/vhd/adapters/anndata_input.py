"""Read a named AnnData matrix without loading unrelated dense layers or SpatialData tables."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from ..core import integer_csr

@contextmanager
def open_anndata_group(path,kind='auto',table=''):
    p=Path(path)
    if p.suffix.lower()=='.h5ad':
        if kind=='spatialdata': raise ValueError('A .h5ad is AnnData, not SpatialData.')
        import h5py
        with h5py.File(p,'r') as root:
            if root.attrs.get('encoding-type') not in ('anndata',b'anndata'):
                raise ValueError('H5AD root is not encoded as AnnData.')
            yield root
        return
    import zarr
    root=zarr.open_group(str(p),mode='r')
    is_ad=root.attrs.get('encoding-type')=='anndata'
    if kind=='anndata' and not is_ad: raise ValueError('Declared AnnData Zarr has no AnnData root encoding.')
    if kind=='spatialdata' and is_ad: raise ValueError('Declared SpatialData input is standalone AnnData.')
    if not is_ad:
        if not table: raise ValueError('SpatialData input requires an explicit table name; no first-table guessing.')
        if 'tables' not in root or table not in root['tables']: raise KeyError(f'Missing tables/{table}')
        root=root['tables'][table]
    if root.attrs.get('encoding-type')!='anndata': raise ValueError('Selected Zarr group is not AnnData.')
    yield root


def metadata(path,kind='auto',table=''):
    from anndata.io import read_elem
    with open_anndata_group(path,kind,table) as r:
        return read_elem(r['obs']),read_elem(r['var'])


def matrix_element(root,source):
    if source=='X': return root['X']
    if source.startswith('layers:'): return root['layers'][source[7:]]
    raise ValueError('Explicit counts source must be X or layers:<name>.')


def read_selected(path,kind='auto',table='',source='X',*,max_matrix_gib=100,counts=True,embeddings=True):
    import anndata as ad
    from anndata.io import read_elem
    with open_anndata_group(path,kind,table) as r:
        xnode=matrix_element(r,source)
        # Do not instantiate an unrelated corrected dense X when counts are in layers:counts.
        encoding=str(xnode.attrs.get('encoding-type',''))
        if encoding in ('csr_matrix','csc_matrix'):
            size=sum(int(xnode[k].size)*np.dtype(xnode[k].dtype).itemsize for k in ('data','indices','indptr'))
        else: size=int(xnode.size)*np.dtype(xnode.dtype).itemsize
        if size>float(max_matrix_gib)*2**30:
            raise MemoryError(f'Selected matrix alone is {size/2**30:.1f} GiB; review the import RAM budget.')
        x=read_elem(xnode)
        if counts: x=integer_csr(x)
        a=ad.AnnData(X=x,obs=read_elem(r['obs']),var=read_elem(r['var']))
        if embeddings and 'obsm' in r:
            for key in r['obsm'].keys(): a.obsm[key]=read_elem(r['obsm'][key])
        if 'uns' in r: a.uns=read_elem(r['uns'])
    return a
