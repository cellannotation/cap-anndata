import pandas as pd
import numpy as np
from cap_anndata import CapAnnDataDF


def test_from_df():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_anndata_df = CapAnnDataDF.from_df(data)

    assert np.allclose(data.values, cap_anndata_df.values)
    assert all(data.columns == cap_anndata_df.column_order)


def test_create_column():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_anndata_df = CapAnnDataDF.from_df(data, column_order=["A", "B", "D"])
    cap_anndata_df["C"] = [7, 8, 9]

    assert 'C' in cap_anndata_df.columns
    assert 'C' in cap_anndata_df.column_order


def test_rename_column():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_anndata_df = CapAnnDataDF.from_df(data)
    cap_anndata_df.rename_column('A', 'A_renamed')

    assert 'A_renamed' in cap_anndata_df.columns
    assert 'A' not in cap_anndata_df.columns
    assert 'A_renamed' in cap_anndata_df.column_order


def test_remove_column():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_anndata_df = CapAnnDataDF.from_df(data)
    cap_anndata_df.remove_column('B')

    assert 'B' not in cap_anndata_df.columns
    assert 'B' not in cap_anndata_df.column_order


def test_from_df_class_method():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    new_df = CapAnnDataDF.from_df(data, ['B', 'A'])

    assert list(new_df.column_order) == ['B', 'A']


def test_column_order_integrity():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_anndata_df = CapAnnDataDF.from_df(data)
    cap_anndata_df["C"] = [7, 8, 9]
    cap_anndata_df.rename_column('A', 'A_renamed')
    cap_anndata_df.remove_column('B')

    expected_order = ['A_renamed', 'C']
    assert list(cap_anndata_df.column_order) == expected_order


def test_join():
    data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    data2 = pd.DataFrame({'D': [7, 8, 9], 'E': [10, 11, 12]})
    cap_anndata_df1 = CapAnnDataDF.from_df(data1, column_order=['A', 'B', 'C'])

    cap_anndata_df1 = cap_anndata_df1.join(data2, how='left')

    expected_order = ['A', 'B', 'C', 'D', 'E']
    assert list(cap_anndata_df1.column_order) == expected_order
    assert cap_anndata_df1.shape == (3, 4)


def test_merge():
    data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    data2 = pd.DataFrame({'A': [2, 3, 4], 'D': [10, 11, 12]})
    cap_anndata_df1 = CapAnnDataDF.from_df(data1, column_order=['A', 'B', 'C'])

    cap_anndata_df1 = cap_anndata_df1.merge(data2, how='inner', on='A')

    expected_order = ['A', 'B', 'C', 'D']
    assert list(cap_anndata_df1.column_order) == expected_order
    assert cap_anndata_df1.shape == (2, 3)
