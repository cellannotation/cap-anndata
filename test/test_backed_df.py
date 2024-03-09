import pandas as pd
import numpy as np
from cap_anndata import CapAnnDataDF  



def test_from_df():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_ann_data_df = CapAnnDataDF.from_df(data)

    assert np.allclose(data.values, cap_ann_data_df.values)
    assert all(data.columns == cap_ann_data_df.column_order)


def test_create_column():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_ann_data_df = CapAnnDataDF.from_df(data, column_order=["A", "B", "D"])
    cap_ann_data_df["C"] = [7, 8, 9]

    assert 'C' in cap_ann_data_df.columns
    assert 'C' in cap_ann_data_df.column_order


def test_rename_column():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_ann_data_df = CapAnnDataDF.from_df(data)
    cap_ann_data_df.rename_column('A', 'A_renamed')

    assert 'A_renamed' in cap_ann_data_df.columns
    assert 'A' not in cap_ann_data_df.columns
    assert 'A_renamed' in cap_ann_data_df.column_order


def test_remove_column():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_ann_data_df = CapAnnDataDF.from_df(data)
    cap_ann_data_df.remove_column('B')

    assert 'B' not in cap_ann_data_df.columns
    assert 'B' not in cap_ann_data_df.column_order


def test_from_df_class_method():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    new_df = CapAnnDataDF.from_df(data, ['B', 'A'])

    assert list(new_df.column_order) == ['B', 'A']


def test_column_order_integrity():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    cap_ann_data_df = CapAnnDataDF.from_df(data)
    cap_ann_data_df["C"] = [7, 8, 9]
    cap_ann_data_df.rename_column('A', 'A_renamed')
    cap_ann_data_df.remove_column('B')

    expected_order = ['A_renamed', 'C']
    assert list(cap_ann_data_df.column_order) == expected_order
