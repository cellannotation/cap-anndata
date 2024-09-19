# Changelog

## [Unreleased]

## [v0.3.0] - 2024-09-19
### Added
- **Support more AnnData fields**, additional API for accessing, modifying and deleting fields: `layers`, `obsm`, `varm`, `obsp`, `varp` was added.([#22](https://github.com/cellannotation/cap-anndata/pull/22), [#26](https://github.com/cellannotation/cap-anndata/pull/26)) 
- **String representation** for `CapAnnData` objects. ([#24](https://github.com/cellannotation/cap-anndata/pull/24))
- **Deprecation warnings** for `read_directly` function. ([#25](https://github.com/cellannotation/cap-anndata/pull/25)). It is recommended to use `read_h5ad` instead.

### Changed
- **Dependencies updated** in `setup.py` and `requirements.txt` to relax version constraints. ([#23](https://github.com/cellannotation/cap-anndata/pull/26))
- **Internal implementation** `CapAnnDataUns` was renamed to `CapAnnDataDict`.

### Fixed
- **Empty raw layer** fix. The situation when raw h5py.Group exists in the .h5ad file but is empty is recognized as no raw layer exists. ([#27](https://github.com/cellannotation/cap-anndata/pull/27))

## [v0.2.2] - 2024-08-28

### Added
- compression option in `overwrite` method to allow using different compression algorithms. Note: `lzf` is still default. 

### Changed

- `requirements.txt` updated to freeze package versions. Possible incompatibility with `numpy` v2+ was displayed. 

## [v0.2.1] - 2024-06-14

### Fixed
- **Python 3.9 Compatibility**: Fixed type notation in `CapAnnDataDF`.

## [v0.2.0] - 2024-06-12

### Added

- **Reader Module**: Introduced `reader.py` with context manager `read_h5ad` and  function `read_directly` for reading AnnData files.
- **Join and Merge Methods**: Enhanced methods for joining and merging dataframes in `CapAnnDataDF`.
- **New Methods for Accessing Keys**: Added `obs_keys` and `var_keys` methods to `CapAnnData` for retrieving column keys without load content of `.h5ad` file to memory.
- **Append columns from disc**: Added the possibility to join the new column from disc to in-memory DataFrame. Previously, it was only possible to read new DataFrame from disc inplace of in-memory one. 
- **DataFrame Setters**: Added validation in DataFrame setters to ensure correct data types and shapes.

### Changed

- **Refactored CapAnnData Class**:
  - Moved `RawLayer` to a separate class.
  - Introduced `BaseLayerMatrixAndDf` as a base class for `CapAnnData` and `RawLayer`.
  - Simplified `read_obs`, `read_var`, and `overwrite` methods.
  - Enhanced `_read_df` to handle column order and in-memory DataFrame updates.
  - Replaced `read_obs` and `read_var` methods to support resetting DataFrames and incremental reads.

### Fixed

- **Empty Column Order Handling**: Fixed issue with empty column order attribute in `CapAnnDataDF`.
- **Overwrite Method**: Corrected handling of column order in the `overwrite` method to avoid type mismatches.

### Removed

- **Context Manager for AnnData Files**: Removed context manager method `read_anndata_file` from `CapAnnData`.
## [v0.1.1] - 2024-04-09

### Fixed
- Bug in `overwrite` method for empty `obs`/`var` sections by ensuring that the `column_order` attribute is always of type `object`.  


## [v0.1.0] - 2024-03-09

### Added
- Initial release of CAP-AnnData.
- Introduced the `CapAnnData` class for managing AnnData files, with functionalities to:
  - Read and write specific columns of `obs` and `var`.
  - Handle the `X` and `raw.X` matrices efficiently.
  - Link and manage `obsm` and `uns` sections lazily, loading them only when accessed.
- Included the `CapAnnDataDF` class to extend `pandas.DataFrame` for handling partial reads and writes, with additional methods for renaming and removing columns.
- Provided the `CapAnnDataDict` class to manage the `uns` section of AnnData files, tracking keys to remove upon overwrite.
- Implemented methods to facilitate in-memory modifications and saving changes back to the file, including the `overwrite` method to selectively update sections of the AnnData file.