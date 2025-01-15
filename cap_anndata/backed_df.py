import pandas as pd
import numpy as np
from typing import List, Any, Union

from pandas._typing import Self
from pandas.core.generic import bool_t


class CapAnnDataDF(pd.DataFrame):
    """
    The class to expand the pandas DataFrame behaviour to support partial
    reading and writing of AnnData obs and var (raw.var) fields.
    The main feature of the class is handling <column-order> attribute
    which must be a copy of h5py.Group attribute
    """

    _metadata = ["column_order"]

    def rename_column(self, old_name: str, new_name: str) -> None:
        i = np.where(self.column_order == old_name)[0]
        tmp_array = self.column_order.copy()
        tmp_array[i] = new_name
        self.column_order = tmp_array.copy()
        self.rename(columns={old_name: new_name}, inplace=True)

    def remove_column(self, col_name: str) -> None:
        i = np.where(self.column_order == col_name)[0]
        self.column_order = np.delete(self.column_order, i)
        self.drop(columns=[col_name], inplace=True)

    def __setitem__(self, key, value) -> None:
        if key not in self.column_order:
            self.column_order = np.append(self.column_order, key)
        return super().__setitem__(key, value)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        new_inst = cls(df)
        new_inst.column_order = df.columns.to_numpy()
        return new_inst

    def join(self, other: Any, **kwargs) -> Self:
        result = super().join(other=other, **kwargs)
        if isinstance(other, CapAnnDataDF):
            new_columns = [
                col for col in other.column_order if col not in self.column_order
            ]
        else:
            new_columns = [col for col in other.columns if col not in self.column_order]
        df = self.from_df(result)
        df.column_order = np.append(self.column_order, new_columns)
        return df

    def merge(self, right, **kwargs) -> Self:
        result = super().merge(right=right, **kwargs)
        if isinstance(right, CapAnnDataDF):
            new_columns = [
                col for col in right.column_order if col not in self.column_order
            ]
        else:
            new_columns = [col for col in right.columns if col not in self.column_order]
        df = self.from_df(result)
        df.column_order = np.append(self.column_order, new_columns)
        return df

    def copy(self, deep: Union[bool_t, None] = True) -> Self:
        df = self.from_df(super().copy(deep=deep))
        df.column_order = self.column_order
        return df
