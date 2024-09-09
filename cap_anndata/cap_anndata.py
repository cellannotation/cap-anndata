import logging
import anndata as ad
import numpy as np
import h5py
from typing import List, Union, Dict, Tuple, Final
import scipy.sparse as ss
from packaging import version

if version.parse(ad.__version__) < version.parse("0.11.0"):
    from anndata.experimental import sparse_dataset, read_elem, write_elem
else:
    from anndata import sparse_dataset, read_elem, write_elem

from cap_anndata import CapAnnDataDF, CapAnnDataDict


logger = logging.getLogger(__name__)

X_NOTATION = Union[h5py.Dataset, ad.experimental.CSRDataset, ad.experimental.CSCDataset]
OBSM_NOTATION = Dict[str, X_NOTATION]

NotLinkedObject: Final = "__NotLinkedObject"


class BaseLayerMatrixAndDf:
    def __init__(self, file: h5py.File, path_to_content: str = "/") -> None:
        self._file = file
        self._path_to_content = path_to_content
        self._X: X_NOTATION = None

    @property
    def X(self) -> X_NOTATION:
        if self._X is None:
            self._link_x()
        return self._X

    def _link_x(self) -> None:
        x = self._file[self._path_to_content + "X"]
        if isinstance(x, h5py.Dataset):
            # dense X
            self._X = x
        else:
            # sparse dataset
            self._X = sparse_dataset(x)

    @property
    def shape(self) -> Tuple[int, int]:
        if self.X is not None:
            shape = tuple(map(int, self.X.shape))
        else:
            shape = None
        return shape

    def _lazy_df_load(self, key: str) -> CapAnnDataDF:
        df = CapAnnDataDF()
        attribute = self._path_to_content + key
        column_order = self._read_attr(self._file[attribute], "column-order")
        df.column_order = column_order
        if df.column_order.dtype != object:
            # empty DataFrame will have column_order as float64
            # which leads to failure in overwrite method
            df.column_order = df.column_order.astype(object)
        return df

    @staticmethod
    def _read_attr(obj: Union[h5py.Group, h5py.Dataset], attr_name: str) -> any:
        attrs = dict(obj.attrs)
        if attr_name not in attrs.keys():
            raise KeyError(f"The {attr_name} doesn't exist!")
        return attrs[attr_name]

    def _read_df(self, key: str, columns: List[str]) -> CapAnnDataDF:
        group_path = self._path_to_content + key
        if group_path not in self._file.keys():
            raise ValueError(f"The group {group_path} doesn't exist in the file!")

        h5_group = self._file[group_path]

        column_order = self._read_attr(h5_group, "column-order")

        if columns is None:
            # read whole df
            df = CapAnnDataDF.from_df(read_elem(h5_group), column_order=column_order)
        else:
            cols_to_read = [c for c in columns if c in column_order]
            df = CapAnnDataDF()
            df.column_order = column_order
            index_col = self._read_attr(h5_group, "_index")
            df.index = read_elem(h5_group[index_col])

            for col in cols_to_read:
                df[col] = read_elem(h5_group[col])

        if df.column_order.dtype != object:
            # empty DataFrame will have column_order as float64
            # which leads to failure in overwrite method
            df.column_order = df.column_order.astype(object)
        return df

    def _write_elem(self, dest_key: str, elem: any, compression: str) -> None:
        write_elem(self._file, dest_key, elem, dataset_kwargs={"compression": compression})

    def _validate_cap_df(self, cap_df: CapAnnDataDF, axis: int) -> None:
        if not isinstance(cap_df, CapAnnDataDF):
            raise TypeError(
                f"The input should be an instance of CapAnnDataDF class but {type(cap_df)} given!"
            )

        if axis not in [0, 1]:
            raise ValueError("The axis should be either 0 or 1!")

        if cap_df.shape[0] != self.shape[axis]:
            items = "cells" if axis == 0 else "genes"
            raise ValueError(
                f"The number of rows in the input DataFrame should be equal to the number of {items} in the "
                "AnnData object!"
            )


class RawLayer(BaseLayerMatrixAndDf):
    def __init__(self, h5_file: h5py.File):
        super().__init__(h5_file, path_to_content="/raw/")
        self._var: CapAnnDataDF = None

    @property
    def var(self) -> CapAnnDataDF:
        if self._var is None:
            self._var = self._lazy_df_load("var")
        return self._var

    @var.setter
    def var(self, cap_df: CapAnnDataDF) -> None:
        self._validate_cap_df(cap_df, axis=1)
        self._var = cap_df

    def read_var(self, columns: List[str] = None, reset: bool = False) -> None:
        df = self._read_df(key="var", columns=columns)
        if self.var.empty or reset:
            self._var = df
        else:
            for col in df.columns:
                self._var[col] = df[col]


class CapAnnData(BaseLayerMatrixAndDf):
    def __init__(self, h5_file: h5py.File) -> None:
        super().__init__(h5_file, path_to_content="/")
        self._file: h5py.File = h5_file
        self._obs: CapAnnDataDF = None
        self._var: CapAnnDataDF = None
        self._X: X_NOTATION = None
        self._obsm: OBSM_NOTATION = None
        self._layers: CapAnnDataDict = None
        self._uns: CapAnnDataDict = None
        self._raw: RawLayer = None
        self._shape: Tuple[int, int] = None

    @property
    def obs(self) -> CapAnnDataDF:
        if self._obs is None:
            self._obs = self._lazy_df_load("obs")
        return self._obs

    @obs.setter
    def obs(self, cap_df: CapAnnDataDF) -> None:
        self._validate_cap_df(cap_df, axis=0)
        self._obs = cap_df

    @property
    def var(self) -> CapAnnDataDF:
        if self._var is None:
            self._var = self._lazy_df_load("var")
        return self._var

    @var.setter
    def var(self, cap_df: CapAnnDataDF) -> None:
        self._validate_cap_df(cap_df, axis=1)
        self._var = cap_df

    @property
    def raw(self) -> RawLayer:
        if self._raw is None:
            if "raw" not in self._file.keys():
                logger.warning("Can't read raw.var since raw layer doesn't exist!")
                return

            self._raw = RawLayer(self._file)
        return self._raw

    @property
    def uns(self) -> CapAnnDataDict:
        if self._uns is None:
            self._uns = CapAnnDataDict(
                {k: NotLinkedObject for k in self._file["uns"].keys()}
            )
        return self._uns

    @property
    def layers(self) -> CapAnnDataDict:
        if self._layers is None:
            self._link_layers()
        return self._layers

    @property
    def obsm(self) -> OBSM_NOTATION:
        if self._obsm is None:
            self._link_obsm()
        return self._obsm

    def read_obs(self, columns: List[str] = None, reset: bool = False) -> None:
        df = self._read_df("obs", columns=columns)
        if self.obs.empty or reset:
            self._obs = df
        else:
            for col in df.columns:
                self._obs[col] = df[col]

    def read_var(self, columns: List[str] = None, reset: bool = False) -> None:
        df = self._read_df("var", columns=columns)
        if self.var.empty or reset:
            self._var = df
        else:
            for col in df.columns:
                self._var[col] = df[col]

    def overwrite(self, fields: List[str] = None, compression: str = "lzf") -> None:
        field_to_entity = {
            "obs": self.obs,
            "var": self.var,
            "raw.var": self.raw.var if self.raw is not None else None,
            "uns": self.uns,
            "layers": self.layers,
        }

        if fields is None:
            fields = list(field_to_entity.keys())
        else:
            for f in fields:
                if f not in field_to_entity.keys():
                    raise KeyError(
                        f"The field {f} is not supported! The list of supported fields are equal to supported "
                        f"attributes of the CapAnnData class: obs, var, raw.var and uns."
                    )

        for key in ["obs", "var", "raw.var"]:
            if key in fields:
                entity: CapAnnDataDF = field_to_entity[key]
                if entity is None:
                    continue

                key = key.replace(".", "/") if key == "raw.var" else key

                for col in entity.columns:
                    self._write_elem(f"{key}/{col}", entity[col].values, compression=compression)

                column_order = entity.column_order
                if (
                    column_order.size == 0
                ):  # Refs https://github.com/cellannotation/cap-anndata/issues/6
                    column_order = np.array([], dtype=np.float64)
                self._file[key].attrs["column-order"] = column_order

        if "uns" in fields:
            for key in self.uns.keys():
                if self.uns[key] is not NotLinkedObject:
                    dest = f"uns/{key}"
                    self._write_elem(dest, self.uns[key], compression=compression)
            for key in self.uns.keys_to_remove:
                del self._file[f"uns/{key}"]

        if "layers" in fields:
            for key in self.layers.keys_to_add:
                dest = f"layers/{key}"
                self._write_elem(dest, self.layers[key], compression=compression)
            for key in self.layers.keys_to_remove:
                del self._file[f"layers/{key}"]

    def create_layer(self,
                     name: str,
                     matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
                     matrix_shape: Union[tuple[int, int], None] = None,
                     data_dtype: Union[np.dtype, None] = None,
                     indices_dtype: Union[np.dtype, None] = None,
                     indptr_dtype: Union[np.dtype, None] = None,
                     format: Union[str, None] = None,
                     compression: str = "lzf"
                    ) -> None:
        dest = f"layers/{name}"
        if name in self.layers.keys():
            raise ValueError(f"Please explicitly remove the old layer '{name}' before creating a new one!")

        if matrix is not None:
            self._write_elem(dest, matrix, compression=compression)
        else:
            if format == "dense":
                group = self._file.create_dataset(name=dest, shape=matrix_shape, dtype=data_dtype)
                # https://anndata.readthedocs.io/en/latest/fileformat-prose.html#dense-arrays-specification-v0-2-0
                group.attrs['encoding-type'] = 'array'
                group.attrs['encoding-version'] = '0.2.0'
            elif format in ["csr", "csc"]: # Based on https://github.com/appier/h5sparse/blob/master/h5sparse/h5sparse.py
                if data_dtype is None:
                    data_dtype = np.float64
                if matrix_shape is None:
                    matrix_shape = (0, 0)
                sparse_class = ss.csr_matrix if format == "csr" else ss.csc_matrix
                data = sparse_class(matrix_shape, dtype=data_dtype)
                group = self._file.create_group(dest)
                group.attrs['h5sparse_format'] = format
                group.attrs['h5sparse_shape'] = data.shape
                group.create_dataset('data', data=data.data, dtype=data_dtype, maxshape=(None,), chunks=True)
                group.create_dataset('indices', data=data.indices, dtype=indices_dtype, maxshape=(None,), chunks=True)
                group.create_dataset('indptr', data=data.indptr, dtype=indptr_dtype, maxshape=(None,), chunks=True)
            else:
                raise NotImplementedError

        self._link_layers()

    def read_uns(self, keys: List[str] = None) -> None:
        if keys is None:
            keys = list(self.uns.keys())

        for key in keys:
            existing_keys = self.uns.keys()
            if key in existing_keys:
                source = self._file[f"uns/{key}"]
                self.uns[key] = read_elem(source)

    def _link_layers(self) -> None:
        if self._layers is None:
            self._layers = CapAnnDataDict()
        if "layers" in self._file.keys():
            group = self._file["layers"]
            for entity_name in group.keys():
                entity = group[entity_name]
                if isinstance(entity, h5py.Dataset):
                    # dense array
                    self._layers[entity_name] = entity
                else:
                    # sparse array
                    self._layers[entity_name] = ad.experimental.sparse_dataset(entity)
        else:
            _ = self._file.create_group('layers')

    def _link_obsm(self) -> None:
        self._obsm = {}
        if "obsm" in self._file.keys():
            obsm_group = self._file["obsm"]
            for entity_name in obsm_group.keys():
                entity = obsm_group[entity_name]
                if isinstance(entity, h5py.Dataset):
                    # dense array
                    self._obsm[entity_name] = entity
                else:
                    # sparse array
                    self._obsm[entity_name] = sparse_dataset(entity)

    def obsm_keys(self) -> List[str]:
        return list(self.obsm.keys())

    def obs_keys(self) -> List[str]:
        return self.obs.column_order.tolist()

    def var_keys(self) -> List[str]:
        return self.var.column_order.tolist()
