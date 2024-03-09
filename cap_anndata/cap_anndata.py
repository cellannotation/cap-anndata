import logging
import contextlib
import anndata as ad
import h5py
from typing import List, Union, Dict, Tuple, Final
from anndata._io.specs import read_elem, write_elem
from dataclasses import dataclass

from src.cap_anndata import CapAnnDataDF, CapAnnDataUns

logger = logging.getLogger(__name__)

X_NOTATION = Union[h5py.Dataset, ad.experimental.CSRDataset, ad.experimental.CSCDataset]
OBSM_NOTATION = Dict[str, X_NOTATION]

NotLinkedObject: Final = "__NotLinkedObject"


@dataclass
class RawLayer:
    var: CapAnnDataDF = None
    X: X_NOTATION = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.X.shape if self.X is not None else None


class CapAnnData:
    def __init__(self, h5_file: h5py.File) -> None:
        self._file: h5py.File = h5_file
        self.obs: CapAnnDataDF = None
        self.var: CapAnnDataDF = None
        self._X: X_NOTATION = None
        self._obsm: OBSM_NOTATION = None
        self._uns: CapAnnDataUns = None
        self._raw: RawLayer = None
        self._shape: Tuple[int, int] = None

    @property
    def X(self) -> X_NOTATION:
        if self._X is None:
            self._link_x()
        return self._X

    @property
    def obsm(self) -> OBSM_NOTATION:
        if self._obsm is None:
            self._link_obsm()
        return self._obsm

    @property
    def raw(self) -> RawLayer:
        if self._raw is None:
            self._link_raw_x()
        return self._raw

    @property
    def uns(self) -> CapAnnDataUns:
        if self._uns is None:
            self._uns = CapAnnDataUns({k: NotLinkedObject for k in self._file["uns"].keys()})
        return self._uns

    def read_obs(self, columns: List[str] = None) -> None:
        self.obs = self._read_df(self._file["obs"], columns=columns)

    def read_var(self, columns: List[str] = None, raw: bool = False) -> None:
        if raw:
            # Check if raw exists first
            if "raw" not in self._file.keys():
                logger.debug("Can't read raw.var since raw layer doesn't exist!")
                return

            if self._raw is None:
                self._raw = RawLayer()
                self._link_raw_x()

            key = "raw/var"
            self._raw.var = self._read_df(self._file[key], columns=columns)
        else:
            key = "var"
            self.var = self._read_df(self._file[key], columns=columns)

    def _read_df(self, h5_group: h5py.Group, columns: List[str]) -> CapAnnDataDF:
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
        return df

    @staticmethod
    def _read_attr(obj: Union[h5py.Group, h5py.Dataset], attr_name: str) -> any:
        attrs = dict(obj.attrs)
        if attr_name not in attrs.keys():
            raise KeyError(f"The {attr_name} doesn't exist!")
        return attrs[attr_name]

    def overwrite(self, fields: List[str] = None) -> None:
        field_to_entity = {
            "obs": self.obs,
            "var": self.var,
            "raw.var": self.raw.var if self.raw is not None else None,
            "uns": self.uns
        }

        if fields is None:
            fields = list(field_to_entity.keys())
        else:
            for f in fields:
                if f not in field_to_entity.keys():
                    raise KeyError(
                        f"The field {f} is not supported! The list of suported fields are equal to supported attributes of the CapAnnData class: obs, var, raw.var and uns.")

        for key in ["obs", "var", "raw.var"]:
            if key in fields:
                entity: CapAnnDataDF = field_to_entity[key]
                if entity is None:
                    continue

                key = key.replace(".", '/') if key == "raw.var" else key

                for col in entity.columns:
                    self._write_elem_lzf(f"{key}/{col}", entity[col].values)
                self._file[key].attrs['column-order'] = entity.column_order

        if "uns" in fields:
            for key in self.uns.keys():
                if self.uns[key] is not NotLinkedObject:
                    dest = f"uns/{key}"
                    self._write_elem_lzf(dest, self.uns[key])
            for key in self.uns.keys_to_remove:
                del self._file[f"uns/{key}"]

    def read_uns(self, keys: List[str] = None) -> None:
        if keys is None:
            keys = list(self.uns.keys())

        for key in keys:
            existing_keys = self.uns.keys()
            if key in existing_keys:
                sourse = self._file[f"uns/{key}"]
                self.uns[key] = read_elem(sourse)

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    def _link_x(self) -> None:
        x = self._file["X"]
        if isinstance(x, h5py.Dataset):
            # dense X
            self._X = x
        else:
            # sparse dataset
            self._X = ad.experimental.sparse_dataset(x)

    def _link_raw_x(self) -> None:
        if "raw" in self._file.keys():
            if self._raw is None:
                self._raw = RawLayer()

            raw_x = self._file["raw/X"]
            if isinstance(raw_x, h5py.Dataset):
                # dense X
                self._raw.X = raw_x
            else:
                # sparse dataset
                self._raw.X = ad.experimental.sparse_dataset(raw_x)

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
                    self._obsm[entity_name] = ad.experimental.sparse_dataset(entity)
        logger.debug(f"obsm={self._obsm}")

    def obsm_keys(self) -> List[str]:
        return list(self.obsm.keys())

    def _write_elem_lzf(self, dest_key: str, elem: any) -> None:
        write_elem(self._file, dest_key, elem, dataset_kwargs={"compression": "lzf"})

    @staticmethod
    @contextlib.contextmanager
    def read_anndata_file(file_path, backed='r'):
        """The method to read anndata file using original AnnData package"""
        logger.debug(f"Read file {file_path} in backed mode = {backed}...")

        adata = None
        try:
            adata = ad.read_h5ad(file_path, backed=backed)
            logger.debug(f"Successfully read anndata file path {file_path}")
            yield adata

        except Exception as error:
            logger.error(f"Error during read anndata file at path: {file_path}, error = {error}!")
            raise error

        finally:
            if adata is not None:
                if adata.isbacked:
                    adata.file.close()
                logger.debug("AnnData closed!")
