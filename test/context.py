import anndata as ad
import numpy as np
import scipy.sparse as sp


def get_base_anndata(n_rows: int = 10, n_genes: int = 10, sparse=False) -> ad.AnnData:
    x = np.eye(n_rows, n_genes).astype(np.float32)
    if sparse:
        x = sp.csr_matrix(x, dtype=np.float32)
    adata = ad.AnnData(X=x)
    return adata
