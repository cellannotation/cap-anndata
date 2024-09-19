import logging
import anndata as ad
import numpy as np
import h5py
from typing import List, Union, Any, Tuple, Final
import scipy.sparse as ss
from packaging import version

if version.parse(ad.__version__) < version.parse("0.11.0"):
    from anndata.experimental import sparse_dataset, read_elem, write_elem
else:
    from anndata import sparse_dataset, read_elem, write_elem

from cap_anndata import CapAnnDataDF, CapAnnDataDict

logger = logging.getLogger(__name__)

X_NOTATION = Union[
    h5py.Dataset, ad.experimental.CSRDataset, ad.experimental.CSCDataset, None
]
ARRAY_MAPPING_NOTATION = CapAnnDataDict[str, X_NOTATION]

NotLinkedObject: Final = "__NotLinkedObject"


class BaseLayerMatrixAndDf:
    def __init__(self, file: h5py.File, path_to_content: str = "/") -> None:
        self._file = file
        self._path_to_content = path_to_content
        self._X: X_NOTATION = None

    @property
    def file(self) -> h5py.File:
        return self._file

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
            if isinstance(columns, str):
                # single column provided instead of list
                columns = [columns]
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
        write_elem(
            self._file, dest_key, elem, dataset_kwargs={"compression": compression}
        )

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

    def _link_array_mapping(self, cap_dict: CapAnnDataDict, key: str) -> None:
        """Method to update given cap_dict with backed array entities from the file."""
        if key not in self._file.keys():
            raise KeyError(f"The key {key} doesn't exist in the file! Ignore linking.")

        group = self._file[key]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"The object {key} must be a group!")

        for array_name in group.keys():
            array = group[array_name]
            if isinstance(array, h5py.Dataset):
                cap_dict[array_name] = array
            elif isinstance(array, h5py.Group):
                cap_dict[array_name] = sparse_dataset(array)
            else:
                raise ValueError(
                    f"Can't link array in {key} due to unsupported type of object: {type(array)}"
                )

    def _create_new_matrix(
        self,
        dest: str,
        matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
        matrix_shape: Union[tuple[int, int], None] = None,
        data_dtype: Union[np.dtype, None] = None,
        format: Union[str, None] = None,  # TODO: use Enum instead of str
        compression: str = "lzf",
    ) -> None:
        if matrix is not None:
            self._write_elem(dest, matrix, compression=compression)
        else:
            if format == "dense":
                group = self._file.create_dataset(
                    name=dest,
                    shape=matrix_shape,
                    dtype=data_dtype,
                    compression=compression,
                )
                # https://anndata.readthedocs.io/en/latest/fileformat-prose.html#dense-arrays-specification-v0-2-0
                group.attrs["encoding-type"] = "array"
                group.attrs["encoding-version"] = "0.2.0"
            elif format in [
                "csr",
                "csc",
            ]:  # Based on https://github.com/appier/h5sparse/blob/master/h5sparse/h5sparse.py
                if data_dtype is None:
                    data_dtype = np.float64
                if matrix_shape is None:
                    matrix_shape = (0, 0)
                sparse_class = ss.csr_matrix if format == "csr" else ss.csc_matrix
                data = sparse_class(matrix_shape, dtype=data_dtype)
                self._write_elem(dest, data, compression=compression)
            else:
                raise NotImplementedError(
                    f"Format must  be 'dense', 'csr' or 'csc' but {format} given!"
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
        self._obsm: CapAnnDataDict = None
        self._varm: CapAnnDataDict = None
        self._layers: CapAnnDataDict = None
        self._uns: CapAnnDataDict = None
        self._obsp: CapAnnDataDict = None
        self._varp: CapAnnDataDict = None
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

            if len(self._file["raw"].keys()) == 0:
                logger.warning("The raw layer is empty!")
                return

            self._raw = RawLayer(self._file)
        return self._raw

    @property
    def uns(self) -> CapAnnDataDict[str, Any]:
        if self._uns is None:
            self._uns = CapAnnDataDict(
                {k: NotLinkedObject for k in self._file["uns"].keys()}
            )
        return self._uns

    @property
    def layers(self) -> CapAnnDataDict[str, X_NOTATION]:
        if self._layers is None:
            self._link_layers()
        return self._layers

    @property
    def obsm(self) -> CapAnnDataDict[str, X_NOTATION]:
        if self._obsm is None:
            self._link_obsm()
        return self._obsm

    @property
    def varm(self) -> CapAnnDataDict[str, X_NOTATION]:
        if self._varm is None:
            self._link_varm()
        return self._varm

    @property
    def obsp(self) -> CapAnnDataDict[str, X_NOTATION]:
        if self._obsp is None:
            self._link_obsp()
        return self._obsp

    @property
    def varp(self) -> CapAnnDataDict[str, X_NOTATION]:
        if self._varp is None:
            self._link_varp()
        return self._varp

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
            self._link_array_mapping(cap_dict=self._layers, key="layers")

    def _link_obsm(self) -> None:
        key = "obsm"
        if self._obsm is None:
            self._obsm = CapAnnDataDict()
        if key in self._file.keys():
            self._link_array_mapping(cap_dict=self._obsm, key=key)

    def _link_varm(self) -> None:
        key = "varm"
        if self._varm is None:
            self._varm = CapAnnDataDict()
        if key in self._file.keys():
            self._link_array_mapping(cap_dict=self._varm, key=key)

    def _link_obsp(self):
        key = "obsp"
        if self._obsp is None:
            self._obsp = CapAnnDataDict()

        if key in self._file.keys():
            self._link_array_mapping(cap_dict=self._obsp, key=key)

    def _link_varp(self):
        key = "varp"
        if self._varp is None:
            self._varp = CapAnnDataDict()

        if key in self._file.keys():
            self._link_array_mapping(cap_dict=self._varp, key=key)

    def obsm_keys(self) -> List[str]:
        return list(self.obsm.keys())

    def obs_keys(self) -> List[str]:
        return self.obs.column_order.tolist()

    def var_keys(self) -> List[str]:
        return self.var.column_order.tolist()

    def overwrite(self, fields: List[str] = None, compression: str = "lzf") -> None:
        field_to_entity = {
            "obs": self.obs,
            "var": self.var,
            "raw.var": self.raw.var if self.raw is not None else None,
            "uns": self.uns,
            "layers": self.layers,
            "obsm": self.obsm,
            "varm": self.varm,
            "obsp": self.obsp,
            "varp": self.varp,
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
                    self._write_elem(
                        f"{key}/{col}", entity[col].values, compression=compression
                    )

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

        for field in ["layers", "obsm", "varm", "obsp", "varp"]:
            if field in fields:
                for key in field_to_entity[field].keys_to_remove:
                    del self._file[f"{field}/{key}"]

    def create_layer(
        self,
        name: str,
        matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
        matrix_shape: Union[tuple[int, int], None] = None,
        data_dtype: Union[np.dtype, None] = None,
        format: Union[str, None] = None,
        compression: str = "lzf",
    ) -> None:
        """
        The empty layer will be created in the case of `matrix` is None.
        """
        self._create_new_matrix_in_field(
            field="layers",
            name=name,
            matrix=matrix,
            matrix_shape=matrix_shape,
            data_dtype=data_dtype,
            format=format,
            compression=compression,
        )
        self._link_layers()

    def create_obsm(
        self,
        name: str,
        matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
        matrix_shape: Union[tuple[int, int], None] = None,
        data_dtype: Union[np.dtype, None] = None,
        format: Union[str, None] = None,
        compression: str = "lzf",
    ) -> None:
        self._create_new_matrix_in_field(
            field="obsm",
            name=name,
            matrix=matrix,
            matrix_shape=matrix_shape,
            data_dtype=data_dtype,
            format=format,
            compression=compression,
        )
        self._link_obsm()

    def create_varm(
        self,
        name: str,
        matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
        matrix_shape: Union[tuple[int, int], None] = None,
        data_dtype: Union[np.dtype, None] = None,
        format: Union[str, None] = None,
        compression: str = "lzf",
    ) -> None:
        self._create_new_matrix_in_field(
            field="varm",
            name=name,
            matrix=matrix,
            matrix_shape=matrix_shape,
            data_dtype=data_dtype,
            format=format,
            compression=compression,
        )
        self._link_varm()

    def create_obsp(
        self,
        name: str,
        matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
        matrix_shape: Union[tuple[int, int], None] = None,
        data_dtype: Union[np.dtype, None] = None,
        format: Union[str, None] = None,
        compression: str = "lzf",
    ) -> None:
        self._create_new_matrix_in_field(
            field="obsp",
            name=name,
            matrix=matrix,
            matrix_shape=matrix_shape,
            data_dtype=data_dtype,
            format=format,
            compression=compression,
        )
        self._link_obsp()

    def create_varp(
        self,
        name: str,
        matrix: Union[np.ndarray, ss.csr_matrix, ss.csc_matrix, None] = None,
        matrix_shape: Union[tuple[int, int], None] = None,
        data_dtype: Union[np.dtype, None] = None,
        format: Union[str, None] = None,
        compression: str = "lzf",
    ) -> None:

        self._create_new_matrix_in_field(
            field="varp",
            name=name,
            matrix=matrix,
            matrix_shape=matrix_shape,
            data_dtype=data_dtype,
            format=format,
            compression=compression,
        )
        self._link_varp()

    def _create_new_matrix_in_field(self, field, name, **kwargs):
        """**kwargs: matrix, matrix_shape, data_dtype, format, compression"""
        dest = f"{field}/{name}"
        field_entity = getattr(self, field)
        if name in field_entity.keys():
            raise ValueError(
                f"Please explicitly remove the existing '{name}' entity from {field} "
                f"before creating a new one!"
            )
        if field not in self._file.keys():
            self._file.create_group(field)
        self._create_new_matrix(dest=dest, **kwargs)

    def remove_layer(self, name: str) -> None:
        del self._file[f"layers/{name}"]
        self._link_layers()

    def remove_obsp(self, name: str) -> None:
        del self._file[f"obsp/{name}"]
        self._link_obsp()

    def remove_varp(self, name: str) -> None:
        del self._file[f"varp/{name}"]
        self._link_varp()

    def remove_obsm(self, name: str) -> None:
        del self._file[f"obsm/{name}"]
        self._link_obsm()

    def remove_varm(self, name: str) -> None:
        del self._file[f"varm/{name}"]
        self._link_varm()

    def create_repr(self) -> str:
        indent = " " * 4
        s = f"CapAnnData object"
        s += f"\n{indent}File: {self._file}"
        s += f"\n{indent}X shape: {self.shape}"
        s += f"\n{indent}Has raw X: {self.raw is not None}"
        for field in ["obs", "obsm", "var", "uns", "layers"]:
            if field in self._file:
                in_memory = set()
                if field in ["obs", "var", "uns"]:
                    attr = getattr(self, field)
                    if attr is not None:
                        in_memory = set(attr.keys())
                keys = list(self._file[field].keys())
                keys = [k for k in keys if k != "_index"]
                keys = [(k if k not in in_memory else f"{k}*") for k in keys]
                keys_str = str(keys).replace("*'", "'*")
                s += f"\n{indent}{field}: {keys_str}"
        s += f"\n{indent}Note: fields marked with * are in-memory objects."
        return s

    def __repr__(self) -> str:
        return self.create_repr()

    def __str__(self) -> str:
        return self.create_repr()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._file is not None:
            self._file.close()
        logger.debug("CapAnnData closed!")
