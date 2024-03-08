import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class CapAnnDataDF(pd.DataFrame):
    """
    The class to expand the pandas DataFrame behaviour to support partial
    reading and writing of AnnData obs and var (raw.var) fields.
    The main feature of the class is handling <column-order> attribute
    which must be a copy of h5py.Group attribute
    """
    _metadata = ['column_order']

    def rename_column(self, old_name: str, new_name: str) -> None:
        i = np.where(self.column_order == old_name)[0]
        self.column_order[i] = new_name
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
    def from_df(cls, df: pd.DataFrame, column_order: List[str] = None):
        if column_order is None:
            column_order = df.columns.to_numpy()

        new_inst = cls(df)
        new_inst.column_order = column_order
        return new_inst
