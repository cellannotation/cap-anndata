import pandas as pd
import numpy as np
from typing import List, Any
import logging

from pandas._typing import Self
from pandas.core.generic import bool_t

logger = logging.getLogger(__name__)


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
    def from_df(cls, df: pd.DataFrame, column_order: List[str] = None) -> Self:
        if column_order is None:
            column_order = df.columns.to_numpy()

        new_inst = cls(df)
        new_inst.column_order = column_order
        return new_inst

    def join(self, other: Any, on=None, how="left", lsuffix="", rsuffix="", sort=False):
        result = super().join(
            other, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort
        )
        if isinstance(other, CapAnnDataDF):
            new_columns = [
                col for col in other.column_order if col not in self.column_order
            ]
        else:
            new_columns = [col for col in other.columns if col not in self.column_order]
        column_order = np.append(self.column_order, new_columns)
        return self.from_df(result, column_order=column_order)

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    ):
        result = super().merge(
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )
        if isinstance(right, CapAnnDataDF):
            new_columns = [
                col for col in right.column_order if col not in self.column_order
            ]
        else:
            new_columns = [col for col in right.columns if col not in self.column_order]
        column_order = np.append(self.column_order, new_columns)
        return self.from_df(result, column_order=column_order)

    def copy(self, deep: bool_t | None = True) -> Self:
        return self.from_df(super().copy(deep=deep), column_order=self.column_order)
