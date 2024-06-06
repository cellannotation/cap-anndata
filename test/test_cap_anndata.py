import anndata as ad
import numpy as np
import tempfile
import os
import h5py
import pandas as pd
import pytest

from cap_anndata import CapAnnData
from test.context import get_base_anndata
from cap_anndata.reader import read_h5ad


def get_filled_anndata(n_rows: int = 10, n_genes: int = 10, sparse=False) -> ad.AnnData:
    adata = get_base_anndata(n_rows, n_genes, sparse)

    adata.obs["cell_type"] = [f"cell_{i%3}" for i in range(adata.shape[0])]
    adata.obs["number"] = [i / 10 for i in range(adata.shape[0])]
    adata.obs.index = [f"obs_{i}" for i in range(adata.shape[0])]

    adata.var.index = [f"gene_{i}" for i in range(adata.shape[1])]
    adata.var["filtered"] = [i > 4 for i in range(adata.shape[1])]
    adata.var["gene_names"] = [f"gene_name_{i}" for i in range(adata.shape[1])]
    adata.var["dispersion"] = [i / 100 for i in range(adata.shape[1])]

    adata.raw = adata
    return adata


def test_read_shape():
    n_rows = 10
    n_genes = 20
    adata = get_base_anndata(n_rows, n_genes)
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_read_shape.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path) as file:
        cap_adata = CapAnnData(file)
        shape = cap_adata.shape
    
    os.remove(file_path)

    assert shape == (n_rows, n_genes), "Shape axis size is incorrect!"

    for i in [0, 1]:
        assert type(shape[i]) == int, "Shape axis type is wrong!"


def test_read_df():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_read_obs.h5ad")

    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r') as file:
        cap_adata = CapAnnData(file)
        cap_adata.read_obs()
        cap_adata.read_var()
        cap_adata.read_var(raw=True)

    os.remove(file_path)
    pd.testing.assert_frame_equal(adata.obs, cap_adata.obs, check_frame_type=False)
    pd.testing.assert_frame_equal(adata.var, cap_adata.var, check_frame_type=False)
    pd.testing.assert_frame_equal(adata.raw.var, cap_adata.raw.var, check_frame_type=False)


def test_partial_read():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_partial_read.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r') as file:
        cap_adata = CapAnnData(file)
        cap_adata.read_obs(columns=['cell_type'])
        cap_adata.read_obs(columns=['cell_type'])
        cap_adata.read_var(columns=['dispersion'])
        cap_adata.read_var(columns=['dispersion'], raw=True)
    
    os.remove(file_path)

    assert len(adata.obs.columns) == len(cap_adata.obs.column_order)
    assert len(adata.var.columns) == len(cap_adata.var.column_order)
    assert len(adata.raw.var.columns) == len(cap_adata.raw.var.column_order)

    assert len(cap_adata.obs.columns) == 1
    assert len(cap_adata.var.columns) == 1
    assert len(cap_adata.raw.var.columns) == 1

    pd.testing.assert_index_equal(adata.obs.index, cap_adata.obs.index)
    pd.testing.assert_index_equal(adata.var.index, cap_adata.var.index)
    pd.testing.assert_index_equal(adata.raw.var.index, cap_adata.raw.var.index)


def test_overwrite_df():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_overwrite_df.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r+') as file:
        cap_adata = CapAnnData(file)
        cap_adata.read_obs(columns=["cell_type"])
        cap_adata.obs["cell_type"] = [f"new_cell_type_{i%2}" for i in range(cap_adata.shape[0])]
        cap_adata.obs["const_str"] = "some string"
        ref_obs = cap_adata.obs.copy()

        # Modify 'var'
        cap_adata.read_var()
        cap_adata.var["gene_names"] = [f"new_gene_{i}" for i in range(cap_adata.shape[1])]
        cap_adata.var["extra_info"] = np.random.rand(cap_adata.shape[1])
        ref_var = cap_adata.var.copy()

        # Modify 'raw.var', assuming 'raw' is also a CapAnnData
        cap_adata.read_var(raw=True)
        cap_adata.raw.var["gene_names"] = [f"raw_new_gene_{i}" for i in range(cap_adata.raw.shape[1])]
        cap_adata.raw.var["extra_info"] = np.random.rand(cap_adata.shape[1])
        ref_raw_var = cap_adata.raw.var.copy()

        cap_adata.overwrite(['obs', 'var', 'raw.var'])

    adata = ad.read_h5ad(file_path)
    os.remove(file_path)
    
    # Assert changes in 'obs'
    assert all([c in adata.obs.columns for c in ref_obs.columns])
    pd.testing.assert_frame_equal(ref_obs, adata.obs[ref_obs.columns.to_list()], check_frame_type=False)

    # Assert changes in 'var'
    assert all([c in adata.var.columns for c in ref_var.columns])
    pd.testing.assert_frame_equal(ref_var, adata.var[ref_var.columns.to_list()], check_frame_type=False)

    # Assert changes in 'raw.var'
    assert all([c in adata.raw.var.columns for c in ref_raw_var.columns])
    pd.testing.assert_frame_equal(ref_raw_var, adata.raw.var[ref_raw_var.columns.to_list()], check_frame_type=False)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("vertical_slice", [None, False, True, "mask"])
def test_link_x(sparse, vertical_slice):
    adata = get_filled_anndata(sparse=sparse)
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_link_x.h5ad")
    adata.write_h5ad(file_path)

    if vertical_slice is None:
        s_ = np.s_[:]
    elif vertical_slice == "mask":
        mask = np.array([i < 5 for i in range(adata.shape[0])])
        s_ = np.s_[mask, :5]
    else:
        # slice over var or obs
        s_ = np.s_[:, 0:5] if vertical_slice else np.s_[0:5, :]

    with h5py.File(file_path, 'r') as file:
        cap_adata = CapAnnData(file)
        x = cap_adata.X[s_]
        raw_x = cap_adata.raw.X[s_]
    
    os.remove(file_path)
    if sparse:
        assert np.allclose(adata.X.A[s_], x.A)
        assert np.allclose(adata.raw.X.A[s_], raw_x.A)
    else:
        assert np.allclose(adata.X[s_], x)
        assert np.allclose(adata.raw.X[s_], raw_x)


@pytest.mark.parametrize("sparse", [False, True])
def test_shape(sparse):
    n_rows = 15
    n_genes = 25

    adata = get_filled_anndata(n_rows, n_genes, sparse)
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_shape.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path) as file:
        cap_adata = CapAnnData(file)
        shape = cap_adata.shape
        shape_raw = cap_adata.raw.shape

    os.remove(file_path)
    for sh in [shape, shape_raw]:
        assert sh == (n_rows, n_genes)


def test_read_obsm():
    adata = get_filled_anndata()
    obsm_names = [f"X_test{i}" for i in range(2)]

    for emb in obsm_names:
        adata.obsm[emb] = np.random.random(size=(adata.shape[0], 2))

    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_read_obsm.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r') as f:
        cap_adata = CapAnnData(f)

        for emb in obsm_names:
            assert emb in cap_adata.obsm_keys()
            assert cap_adata.obsm[emb].shape == adata.obsm[emb].shape
        
        x_1 = cap_adata.obsm[obsm_names[0]][:]
        x_2 = cap_adata.obsm[obsm_names[1]][:]

    os.remove(file_path)
    assert np.allclose(adata.obsm[obsm_names[0]], x_1)
    assert np.allclose(adata.obsm[obsm_names[1]], x_2)


def test_read_uns():
    adata = get_base_anndata()
    key1, key2 = "key1", "key2"
    keys = (key1, key2)
    
    adata.uns = {k: {k: k} for k in keys}
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_read_uns.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r') as f:
        cap_adata = CapAnnData(f)

        for k in keys:
            assert k in cap_adata.uns
        
        cap_adata.read_uns(keys=[key1])

        assert cap_adata.uns[key1] == adata.uns[key1]  # connected
        assert cap_adata.uns[key2] != adata.uns[key2]  # not connected

    os.remove(file_path)


def test_modify_uns():
    adata = get_base_anndata()
    adata.uns = {
        "field_to_ingore": list(range(100)),
        "field_to_rename": "value",
        "field_to_expand": {"key1": {}},
        "field_to_modify": {"a": "b"}
    }
    new_name = "renamed_field"
    d_to_exp = {"sub_key1": "v1", "sub_key2": "v2"}
    v_to_mod = "value"

    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_modify_uns.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r+') as f:
        cap_adata = CapAnnData(f)

        cap_adata.read_uns(keys=["field_to_rename", "field_to_expand", "field_to_modify"])

        cap_adata.uns[new_name] = cap_adata.uns.pop("field_to_rename")
        cap_adata.uns["field_to_expand"]["key1"] = d_to_exp
        cap_adata.uns["field_to_modify"] = v_to_mod

        cap_adata.overwrite(['uns'])
    
    adata = ad.read_h5ad(file_path)

    assert adata.uns is not None
    assert len(adata.uns.keys()) == 4
    assert new_name in adata.uns.keys()
    assert adata.uns['field_to_expand']["key1"] == d_to_exp
    assert adata.uns['field_to_modify'] == v_to_mod


def test_empty_obs_override():
    """
    especially for solving the issue:
    https://github.com/cellannotation/cap-anndata/pull/5
    """
    adata = get_base_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_modify_uns.h5ad")
    adata.write_h5ad(file_path)

    with h5py.File(file_path, 'r+') as f:
        cap_adata = CapAnnData(f)
        cap_adata.read_obs()

        cap_adata.obs["cell_type_1"] = pd.Series(data=np.nan, index=cap_adata.obs.index, dtype="category")
        cap_adata.obs["cell_type_new"] = pd.Series(data=np.nan, index=cap_adata.obs.index, dtype="category")
        cap_adata.overwrite(fields=["obs"])


def test_obs_last_column_removal():
    col_name = 'cell_type'
    adata = ad.AnnData(X=np.ones(shape=(10,10), dtype=np.float32))
    adata.obs[col_name] = col_name

    temp_folder = tempfile.mkdtemp()
    file_path = f"{temp_folder}/test.h5ad"
    adata.write(filename=file_path)

    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.read_obs()
        cap_adata.obs.remove_column(col_name=col_name)
        cap_adata.overwrite(['obs'])

    os.remove(file_path)
