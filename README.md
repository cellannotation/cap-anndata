# CAP-AnnData: Enhanced Partial I/O for AnnData Files

## Overview
CAP-AnnData enriches the AnnData ecosystem by offering tailored functionalities for partial reading and writing of AnnData files. This enhancement allows for selective manipulation of sections such as `obs`, `var`, `X`, `raw.X`, `obsm`, and `uns` without the need for loading entire datasets into memory. Leveraging AnnData's native methods, CAP-AnnData aims to maintain backward compatibility while improving efficiency, especially useful for large-scale single-cell genomics data.

## Getting Started

### Installation
Install CAP-AnnData via pip:

```commandline
pip install -U cap-anndata
```

### Running Tests
Ensure the integrity and reliability of CAP-AnnData on your system by running the unit tests via `pytest` from the root of the repo.

```commandline
pip install pytest
pytest test
```

Make sure Python 3.9 or newer is used, along with all requirements specified in requirements.txt

## How-TO:

#### 1. Access AnnData File DataFrames

##### Basic Reading
By default, `CapAnnData` does not automatically read any data. To begin working with dataframes, you need to explicitly read the data from the AnnData file. You can read the entire dataframe or select specific columns. For partial reading, provide a list of column names.

```python
from cap_anndata import read_h5ad

file_path = "your_data.h5ad"
with read_h5ad(file_path=file_path, edit=False) as cap_adata:
    # Get the list of all obs columns in AnnData file
    cap_adata.obs_keys()  # ['a', 'b', 'c']
    # Read all columns of 'obs'
    cap_adata.read_obs()
    # Get the list of columns of DataFrame in memory
    cap_adata.obs.columns  # ['a', 'b', 'c']

    # Get the list of all var columns in AnnData file
    cap_adata.var_keys()  # ['d', 'e', 'f']
    # Read specific columns of 'var'
    cap_adata.read_var(columns=['d'])
    cap_adata.var.columns  # ['d']
    # Read additional column
    cap_adata.read_var(columns=['e'])
    cap_adata.var.columns  # ['d', 'e']

    # Read column and reset the in-memory DataFrame before that
    cap_adata.read_var(columns=['f'], reset=True)
    cap_adata.var.columns  # ['f']

    # Read no columns of raw.var (only the index)
    cap_adata.raw.read_var(columns=[])
```

##### Difference between `obs_keys()` and `obs.columns`
`obs_keys()` returns the list of columns in the on-disc AnnData file, while `obs.columns` returns the list of columns in the in-memory DataFrame. The two lists may differ if you read only specific columns. If you modify the in-memory DataFrame, the `obs_keys()` will reflect the changes. BTW it is recommended to check the `obs_keys()` before the `overwrite()` call to avoid the AnnData file damage.

If a column doesn't exist in the file, no error will be raised but the column will be missing in the resulting DataFrame. So, the list of columns saying more like "try to read this columns from the file". It is needed because we there is no way yet to check if the column exists before the read. Exactly the same behavior is for the `var_keys()` and `var.columns`. 

#### 2. Modify the AnnData File DataFrames In-Place

You can directly modify the dataframe by adding, renaming, or removing columns.

```python
# Create a new column
cap_adata.obs['new_col'] = [value1, value2, value3]

# Rename a column
cap_adata.obs.rename_column('old_col_name', 'new_col_name')

# Remove a column
cap_adata.obs.remove_column('col_to_remove')
```

After modifications, you can overwrite the changes back to the AnnData file. If a value doesn't exist, it will be created.
Note: `read_h5ad` must be called with `edit=True` argument to open `.h5ad` file in `r+` mode.

```python
# overwrite all values which were read
cap_adata.overwrite()

# overwrite choosen fields
cap_adata.overwrite(['obs', 'var'])
```

The full list of supported fields: `obs`, `var`, `raw.var`, `obsm`, `uns`.

#### 3. How to Read Few Columns but Overwrite One in a Dataframe

The only way yet to do that is to drop all columns from in-memory dataframe (with `pandas.drop`!) before the call of `overwrite` method.

```python
# Read specific columns
cap_adata.read_obs(columns=['cell_type', 'sample'])

# Drop a column in-memory
# DON'T USE remove_column here!
cap_adata.obs.drop(columns='sample', inplace=True)

# Overwrite changes
cap_adata.overwrite(['obs'])

# NOTE that the line 
# cap_adata.read_obs(columns=['sample'], reset=True)
# Will override in-memory changes with values from the AnnData file
```

#### 4. How to work with X and raw.X

The CapAnnData package won't read any field by default. However, the `X` and `raw.X` will be linked to the backed matrices automatically upon the first request to those fields. 
The X object will be returned as the `h5py.Dataset` or `AnnData.experimental.sparse_dataset`.

```python
with read_h5ad(file_path=file_path, edit=False) as cap_adata:
    # self.X is None here
    cap_adata = CapAnnData(file)  

    # will return the h5py.Dataset or CSRDataset
    x = cap_adata.X  

    # The same for raw.X
    raw_x = cap_adata.raw.X 

    # take whole matrix in memory
    x = cap_adata.X[:] 
```

The CapAnnData supports the standard `numpy`/`h5py` sclising rules

```python
# slice rows
s_ = np.s_[0:5]
# slice columns
s_ = np.s_[:, 0:5]
# boolean mask + slicing
mask = np.array([i < 5 for i in range(adata.shape[0])])
s_ = np.s_[mask, :5]
```

#### 5. How to handle obsm embeddings matrixes

By the default the CapAnnData will not read the embeddings matrix. 
The link to the h5py objects will be created upon the first call of the `.obsm` property. 
Alike the AnnData package the call like `cap_adata.obsm["X_tsne"]` will not return the in-memory matrix but will return the backed version instead. 
It is possible to get the information about the name and shape of the embeddings without taking the whole matrix in the memory.

```python
with read_h5ad(file_path=file_path, edit=False) as cap_adata:
    # will return the list of strings
    obsm_keys = cap_adata.obsm_keys()  

    # return the shape of the matrix in backed mode
    embeddings = obsm_keys[0]
    shape = cap_adata.obsm[embeddings].shape  

    # take the whole matrix in memory
    matrix = cap_adata.obsm[embeddings][:]
```

#### 6. How to read and modify uns section

The `CapAnnData` class will lazely link the uns section upon the first call but ***WILL NOT*** read it into memory. Instead, the dictionary of the pairs `{'key': "__NotLinkedObject"}` will be creted. It allow to get the list of keys before the actual read. To read the uns section in the memory the `.read_uns(keys)` method must be called.

```python
with read_h5ad(file_path=file_path, edit=True) as cap_adata:
    # will return the keys() object
    keys = cap_adata.uns.keys()  

    # read in memory the first key only
    cap_adata.read_uns([keys[0]])

    # read the whole uns section into memory
    cap_adata.read_uns()
```

Since the `.uns` section is in the memory (partially or completely) we can work with it as with the regular `dict()` python object. The main feature of the `CapAnnDataUns` class which inherited from `dict` is the tracking of the keys which must be removed from the `.h5ad` file upon overwrite. 

```python
# get the value
v = cap_adata.uns["key1"]
v = cap_adata.uns.get("key1")

# modify values
cap_adata.uns["key1"] = "new_value"

# create new keys
cap_adata.uns["new_key"] = "value"

# remove keys
cap_adata.uns.pop("key1")  # is recommended way
del cap_adata.uns.pop("key2")
cap_adata.uns.popitem()
```

To save `uns` changes the method `CapAnnData.overwrite()` must be called. 

```python
cap_adata.overwrite()  # all in-memory fields will be overwritten
cap_adata.overwrite(["uns"])  # overwrite the uns secion only
```

#### 7. Join and Merge DataFrames

Cap-AnnData provides enhanced methods for joining and merging dataframes, preserving column order and data integrity

```python
from cap_anndata import CapAnnDataDF
import pandas as pd

data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
data2 = pd.DataFrame({'D': [7, 8, 9], 'E': [10, 11, 12]})
cap_anndata_df1 = CapAnnDataDF.from_df(data1, column_order=['A', 'B', 'C'])

cap_df = cap_anndata_df1.join(data2, how='left')

cap_df.columns  # ['A', 'B', 'D', 'E']
cap_df.column_order  # ['A', 'B', 'C', 'D', 'E']

data3 = pd.DataFrame({'A': [2, 3, 4], 'D': [10, 11, 12]})
cap_df = cap_anndata_df1.merge(data3, on='A')

cap_df.columns  # ['A', 'B', 'D']
cap_df.column_order  # ['A', 'B', 'C', 'D']
cap_df.shape  # (2, 3)
```
