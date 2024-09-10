import tempfile
import os

from cap_anndata.reader import read_h5ad, read_directly
from test.context import get_base_anndata
import pytest


def prepare_h5ad_file(name):
    adata = get_base_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, name)
    adata.write_h5ad(file_path)
    return file_path


@pytest.mark.parametrize("edit", [True, False])
def test_read_in_context(edit):
    file_path = prepare_h5ad_file("test_read_in_context.h5ad")

    with read_h5ad(file_path=file_path, edit=edit) as cap_adata:
        assert cap_adata is not None, "CapAnnData file must be valid!"
        cap_adata.read_obs()
        cap_adata.read_uns()
        if edit:
            cap_adata.overwrite()

    os.remove(file_path)


@pytest.mark.parametrize("edit", [True, False])
def test_read_as_function(edit):
    file_path = prepare_h5ad_file("test_read_as_function.h5ad")

    cap_adata = read_h5ad(file_path=file_path, edit=edit)
    assert cap_adata is not None, "CapAnnData file must be valid!"
    cap_adata.read_obs()
    cap_adata.read_uns()
    if edit:
        cap_adata.overwrite()

    cap_adata._file.close()  # TODO: create a file property
    os.remove(file_path)


@pytest.mark.parametrize("edit", [True, False])
def test_read_directly(edit):
    # TODO: remove deprecated function and unit test
    file_path = prepare_h5ad_file("test_read_directly.h5ad")

    cap_adata = read_directly(file_path=file_path, edit=edit)
    assert cap_adata is not None, "CapAnnData file must be valid!"
    cap_adata.read_obs()
    cap_adata.read_uns()
    if edit:
        cap_adata.overwrite()

    cap_adata._file.close()  # TODO: create a file property
    os.remove(file_path)
