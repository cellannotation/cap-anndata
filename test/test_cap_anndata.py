import anndata as ad
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import tempfile
import os
import pandas as pd
import pytest

from cap_anndata import CapAnnDataDF
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


def save_filled_anndata(
    file_name: str, n_rows: int = 10, n_genes: int = 10, sparse=False
) -> str:
    adata = get_filled_anndata(n_rows, n_genes, sparse)
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, file_name)
    adata.write_h5ad(file_path)
    return file_path


def test_read_shape():
    n_rows = 10
    n_genes = 20
    adata = get_base_anndata(n_rows, n_genes)
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_read_shape.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path) as cap_adata:
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

    with read_h5ad(file_path) as cap_adata:
        cap_adata.read_obs()
        cap_adata.read_var()
        cap_adata.raw.read_var()

    os.remove(file_path)
    pd.testing.assert_frame_equal(adata.obs, cap_adata.obs, check_frame_type=False)
    pd.testing.assert_frame_equal(adata.var, cap_adata.var, check_frame_type=False)
    pd.testing.assert_frame_equal(
        adata.raw.var, cap_adata.raw.var, check_frame_type=False
    )


def test_partial_read():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_partial_read.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path) as cap_adata:
        cap_adata.read_obs(columns=["cell_type"])
        cap_adata.read_obs(columns=["cell_type"])
        cap_adata.read_var(columns=["dispersion"])
        cap_adata.raw.read_var(columns=["dispersion"])

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


@pytest.mark.parametrize("compression", ["gzip", "lzf"])
def test_overwrite_df(compression):
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_overwrite_df.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.read_obs(columns=["cell_type"])
        cap_adata.obs["cell_type"] = [
            f"new_cell_type_{i%2}" for i in range(cap_adata.shape[0])
        ]
        cap_adata.obs["const_str"] = "some string"
        ref_obs = cap_adata.obs.copy()

        # Modify 'var'
        cap_adata.read_var()
        cap_adata.var["gene_names"] = [
            f"new_gene_{i}" for i in range(cap_adata.shape[1])
        ]
        cap_adata.var["extra_info"] = np.random.rand(cap_adata.shape[1])
        ref_var = cap_adata.var.copy()

        # Modify 'raw.var', assuming 'raw' is also a CapAnnData
        cap_adata.raw.read_var()
        cap_adata.raw.var["gene_names"] = [
            f"raw_new_gene_{i}" for i in range(cap_adata.raw.shape[1])
        ]
        cap_adata.raw.var["extra_info"] = np.random.rand(cap_adata.shape[1])
        ref_raw_var = cap_adata.raw.var.copy()

        cap_adata.overwrite(["obs", "var", "raw.var"], compression=compression)

    adata = ad.read_h5ad(file_path)
    os.remove(file_path)

    # Assert changes in 'obs'
    assert all([c in adata.obs.columns for c in ref_obs.columns])
    pd.testing.assert_frame_equal(
        ref_obs, adata.obs[ref_obs.columns.to_list()], check_frame_type=False
    )

    # Assert changes in 'var'
    assert all([c in adata.var.columns for c in ref_var.columns])
    pd.testing.assert_frame_equal(
        ref_var, adata.var[ref_var.columns.to_list()], check_frame_type=False
    )

    # Assert changes in 'raw.var'
    assert all([c in adata.raw.var.columns for c in ref_raw_var.columns])
    pd.testing.assert_frame_equal(
        ref_raw_var,
        adata.raw.var[ref_raw_var.columns.to_list()],
        check_frame_type=False,
    )


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

    with read_h5ad(file_path) as cap_adata:
        x = cap_adata.X[s_]
        raw_x = cap_adata.raw.X[s_]

    os.remove(file_path)
    if sparse:
        assert np.allclose(adata.X.toarray()[s_], x.toarray())
        assert np.allclose(adata.raw.X.toarray()[s_], raw_x.toarray())
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

    with read_h5ad(file_path) as cap_adata:
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

    with read_h5ad(file_path) as cap_adata:
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

    with read_h5ad(file_path) as cap_adata:
        for k in keys:
            assert k in cap_adata.uns

        cap_adata.read_uns(keys=[key1])

        assert cap_adata.uns[key1] == adata.uns[key1]  # connected
        assert cap_adata.uns[key2] != adata.uns[key2]  # not connected

    os.remove(file_path)


@pytest.mark.parametrize("compression", ["gzip", "lzf"])
def test_modify_uns(compression):
    adata = get_base_anndata()
    adata.uns = {
        "field_to_ignore": list(range(100)),
        "field_to_rename": "value",
        "field_to_expand": {"key1": {}},
        "field_to_modify": {"a": "b"},
    }
    new_name = "renamed_field"
    d_to_exp = {"sub_key1": "v1", "sub_key2": "v2"}
    v_to_mod = "value"

    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_modify_uns.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.read_uns(
            keys=["field_to_rename", "field_to_expand", "field_to_modify"]
        )

        cap_adata.uns[new_name] = cap_adata.uns.pop("field_to_rename")
        cap_adata.uns["field_to_expand"]["key1"] = d_to_exp
        cap_adata.uns["field_to_modify"] = v_to_mod

        cap_adata.overwrite(["uns"], compression=compression)

    adata = ad.read_h5ad(file_path)

    assert adata.uns is not None
    assert len(adata.uns.keys()) == 4
    assert new_name in adata.uns.keys()
    assert adata.uns["field_to_expand"]["key1"] == d_to_exp
    assert adata.uns["field_to_modify"] == v_to_mod


def test_empty_obs_override():
    """
    especially for solving the issue:
    https://github.com/cellannotation/cap-anndata/pull/5
    """
    adata = get_base_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_modify_uns.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.read_obs()

        cap_adata.obs["cell_type_1"] = pd.Series(
            data=np.nan, index=cap_adata.obs.index, dtype="category"
        )
        cap_adata.obs["cell_type_new"] = pd.Series(
            data=np.nan, index=cap_adata.obs.index, dtype="category"
        )
        cap_adata.overwrite(fields=["obs"])


def test_obs_var_keys():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_obs_keys.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path) as cap_adata:
        obs_keys_before_read = cap_adata.obs_keys()
        var_keys_before_read = cap_adata.var_keys()
        cap_adata.read_obs()
        cap_adata.read_var()
        obs_keys_after_read = cap_adata.obs_keys()
        var_keys_after_read = cap_adata.var_keys()

    assert obs_keys_before_read == adata.obs_keys()
    assert obs_keys_after_read == adata.obs_keys()
    assert var_keys_before_read == adata.var_keys()
    assert var_keys_after_read == adata.var_keys()
    os.remove(file_path)


def test_reset_read():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_flush_read.h5ad")
    adata.write_h5ad(file_path)

    with read_h5ad(file_path) as cap_adata:
        cap_adata.read_obs(columns=["cell_type"])
        cap_adata.obs["cell_type"] = list(range(cap_adata.shape[0]))

        cap_adata.read_obs(columns=["number"])
        assert cap_adata.obs.columns.size == 2
        assert all(cap_adata.obs.cell_type.values == list(range(cap_adata.shape[0])))

        cap_adata.read_obs(columns=["cell_type"])
        pd.testing.assert_series_equal(cap_adata.obs.cell_type, adata.obs.cell_type)

        cap_adata.read_obs(columns=["number"], reset=True)
        assert cap_adata.obs.columns.size == 1
    os.remove(file_path)


def test_obs_last_column_removal():
    col_name = "cell_type"
    adata = ad.AnnData(X=np.ones(shape=(10, 10), dtype=np.float32))
    adata.obs[col_name] = col_name

    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_obs_last_column_removal.h5ad")
    adata.write(filename=file_path)

    # Remove last column
    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.read_obs()
        cap_adata.obs.remove_column(col_name=col_name)
        cap_adata.overwrite(["obs"])

    # Check no any issues on read with updated file
    with read_h5ad(file_path) as cap_adata:
        cap_adata.read_obs()
        assert col_name not in cap_adata.obs.columns

    # Check compatability with anndata
    adata = ad.read_h5ad(file_path)
    os.remove(file_path)


@pytest.mark.parametrize("field", ["obs", "var", "raw.var"])
def test_df_setter(field):
    def set_field(field, new_df, cap_adata):
        if field == "obs":
            cap_adata.obs = new_df
        elif field == "var":
            cap_adata.var = new_df
        else:
            cap_adata.raw.var = new_df

    n_rows, n_genes = 10, 5
    adata = get_filled_anndata(n_rows=n_rows, n_genes=n_genes)
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_obs_setter.h5ad")
    adata.write(filename=file_path)

    new_df = pd.DataFrame(
        index=range(n_rows if field == "obs" else n_genes), columns=["new_column"]
    )

    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.read_obs()
        cap_adata.read_var()
        cap_adata.raw.read_var()

        # test bad format
        try:
            set_field(field, new_df, cap_adata)
        except TypeError:
            pass
        except Exception as e:
            assert False, f"Unexpected exception: {e}"
        else:
            assert False, "Expected TypeError"

        # test good format
        new_df = CapAnnDataDF.from_df(new_df)
        set_field(field, new_df, cap_adata)

        # test bad shape
        new_df = CapAnnDataDF.from_df(new_df[: new_df.shape[0] // 2])

        try:
            set_field(field, new_df, cap_adata)
        except ValueError:
            pass
        except Exception as e:
            assert False, f"Unexpected exception: {e}"
        else:
            assert False, "Expected ValueError"


def test_layer_base_operations():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_layers.h5ad")

    adata.write_h5ad(file_path)

    layer_name_dense = "layer_dense"
    layer_name_sparse = "layer_sparse"

    shape = (10,10)
    data = np.random.random(shape)
    dense_array = data
    sparse_array = csr_matrix(data)

    # Test add layers
    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.create_layer(name=layer_name_dense, matrix=dense_array)
        cap_adata.create_layer(name=layer_name_sparse, matrix=sparse_array)

    with read_h5ad(file_path) as cap_adata:
        assert np.array_equal(dense_array, cap_adata.layers[layer_name_dense]), "Must be correct dense matrix!"
        assert np.array_equal(sparse_array.todense(), cap_adata.layers[layer_name_sparse][:].todense()), "Must be correct sparse matrix!"

    # Test remove layers
    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.remove_layer(layer_name_dense)
        cap_adata.remove_layer(layer_name_sparse)

    with read_h5ad(file_path) as cap_adata:
        assert layer_name_dense not in cap_adata.layers.keys(), "Dense matrix must be removed!"
        assert layer_name_sparse not in cap_adata.layers.keys(), "Sparse matrix must be removed!"

    # Test modify layers
    layer_name_edit = "layer_for_edit"
    data = np.ones(shape)
    with read_h5ad(file_path, edit=True) as cap_adata: # fill layer
        cap_adata.create_layer(name=layer_name_edit, matrix=data)
    with read_h5ad(file_path, edit=True) as cap_adata: # modify backed
        cap_adata.layers[layer_name_edit][0:1,0:1] = 0
    with read_h5ad(file_path) as cap_adata: # check is changed
        assert False == np.array_equal(data, cap_adata.layers[layer_name_edit][:]), "Layer matrix must be edited previously!"

    os.remove(file_path)


def test_layer_create_append():
    adata = get_filled_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, "test_layers.h5ad")

    adata.write_h5ad(file_path)

    shape = (10,10)
    data = np.random.random(shape)

    # Test add empty dense and sparse data layers and edit them
    layer_name_empty_dense = "layer_empty_dense"
    def layer_name_empty_sparse(format: str):
        return f"layer_empty_sparse_{format}"
    def sparse_array_class(format: str):
        return csr_matrix if format == "csr" else csc_matrix

    with read_h5ad(file_path, edit=True) as cap_adata:
        cap_adata.create_layer(
            name=layer_name_empty_dense,
            matrix=None,
            matrix_shape=shape,
            data_dtype=np.float32,
            format="dense",
        )
        for format in ["csr", "csc"]:
            array_class = sparse_array_class(format)
            sparse_array = array_class(data)
            name = layer_name_empty_sparse(format)
            cap_adata.create_layer(
                name=name,
                matrix=None,
                matrix_shape=shape,
                data_dtype=sparse_array.data.dtype,
                format=format,
            )
    sparse_info = {}
    with read_h5ad(file_path, edit=True) as cap_adata:
        # Modify dense dataset
        cap_adata.layers[layer_name_empty_dense][0, 0] = 1
        # Modify sparse datasets
        for format in ["csr", "csc"]:
            name = layer_name_empty_sparse(format)
            sparse_dataset = cap_adata.layers[name]
            chunk_shape = (1,10) if format == "csr" else (10,1)
            array_class = sparse_array_class(format)
            chunk_data = array_class(np.ones(chunk_shape))
            sparse_dataset.append(chunk_data)
            sparse_info[format] = {
                "shape": chunk_data.data.shape,
                "dtype": chunk_data.data.dtype,
            }
    with read_h5ad(file_path) as cap_adata: # check is changed
        assert np.any(cap_adata.layers[layer_name_empty_dense][:] == 1), "Dense layer is not changed!"
        for format in ["csr", "csc"]:
            name = layer_name_empty_sparse(format)
            matrix = cap_adata.layers[name][:]
            assert np.any(matrix.toarray() == 1), f"Layer {format} matrix must be edited previously!"
            assert matrix.data.shape == sparse_info[format]["shape"], "shape is incorrect!"
            assert matrix.data.dtype == sparse_info[format]["dtype"], "dtype is wrong!"

    os.remove(file_path)
