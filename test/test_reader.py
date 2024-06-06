import tempfile
import os

from cap_anndata.reader import read_h5ad, read_directly
from test.context import get_base_anndata


def test_read_anndata_file():
    adata = get_base_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_read_anndata_file.h5ad")
    adata.write_h5ad(file_path)
    del adata

    with read_h5ad(file_path=file_path) as cap_adata:
        assert cap_adata is not None, "CapAnnData file must be valid!"

    cap_adata = read_directly(file_path=file_path)
    assert cap_adata is not None, "CapAnnData file must be valid!"
    cap_adata._file.close()

    os.remove(file_path)
