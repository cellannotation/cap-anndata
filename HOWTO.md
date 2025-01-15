# CAP-AnnData How-TO

## Content
1. [Open h5ad file](#1-open-h5ad-file)
2. [Read DataFrames: obs and var](#2-read-dataframes-obs-and-var)
3. [Modify DataFrames In-Place](#3-modify-dataframes-in-place)
4. [Read Few Columns but Overwrite One in a Dataframe](#4-read-few-columns-but-overwrite-one-in-a-dataframe)
5. [Work with **X** and **raw.X**](#5-work-with-x-and-rawx)
6. [Work with **layers**](#6-work-with-layers)
7. [Handle **obsm** and **varm** sections](#7-handle-obsm-and-varm-sections)
8. [Work with **obsp** and **varp**](#8-work-with-obsp-and-varp)
9. [Read and modify **uns** section](#9-read-and-modify-uns-section)
10. [Join and Merge CapAnnDataDF](#10-join-and-merge-capanndatadf)

## 1. Open h5ad file
To open .h5ad file use `read_h5ad` function. It could be used as a context manager or ordinary function. 

```python
from cap_anndata import read_h5ad

path = "path/to/file.h5ad"
with read_h5ad(path, edit=False) as cap_adata:
    print(cap_adata.file)  # h5py.File opened in 'r' mode
# Don't need to manually close the file
``` 
or 
```python
cap_adata = read_h5ad(path, edit=True)
print(cap_adata.file) # h5py.File opened in 'r+' mode 
cap_adata.file.close()  # Don't forget to close the file
```

## 2. Read DataFrames: obs and var

### Basic Reading
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

### Difference between `obs_keys()` and `obs.columns`
`obs_keys()` returns the list of columns in the on-disc AnnData file, while `obs.columns` returns the list of columns in the in-memory DataFrame. The two lists may differ if you read only specific columns. If you modify the in-memory DataFrame, the `obs_keys()` will reflect the changes. BTW it is recommended to check the `obs_keys()` before the `overwrite()` call to avoid the AnnData file damage.

If a column doesn't exist in the file, no error will be raised but the column will be missing in the resulting DataFrame. So, the list of columns saying more like "try to read that columns from the file". Exactly the same behavior is for the `var_keys()` and `var.columns`. 

## 3. Modify DataFrames In-Place

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

# overwrite chosen fields
cap_adata.overwrite(['obs', 'var'])
```

The full list of supported fields: `obs`, `var`, `raw.var`, `obsm`, `layers` and `uns`.

## 4. Read Few Columns but Overwrite One in a Dataframe

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

## 5. Work with **X** and **raw.X**

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

The CapAnnData supports the standard `numpy`/`h5py` slicing rules

```python
# slice rows
s_ = np.s_[0:5]
# slice columns
s_ = np.s_[:, 0:5]
# boolean mask + slicing
mask = np.array([i < 5 for i in range(adata.shape[0])])
s_ = np.s_[mask, :5]
```
## 6. Work with **layers**

By the default the CapAnnData will not read the layers.
Links to available layers will be created upon the first call of the `.layers` property.
Alike the AnnData package the call like `cap_adata.layers["some_layer"]` will not return the in-memory matrix but will return the backed version instead.

Base operations with layers:
```python
with read_h5ad(file_path=file_path, edit=True) as cap_adata:
    # Will return available layer names
    layer_names = cap_adata.layers.keys()

    # Return the matrix in backed mode
    my_layer = cap_adata.layers["some_layer"]

    # Take the whole matrix into memory
    my_layer = my_layer[:]

    # Create new layer from in-memory matrix
    # NOTE: The matrix will be propagated to the .h5ad file immediately
    cap_adata.create_layer(
        name="some_layer",
        matrix=array, # could be numpy dense array or csr sparse matrix
    )

    # Create empty layer for dense array
    # NOTE: After that in .h5ad file the empty h5py.Dataset will be created
    # so one can fill it later (for ex. in chunks)
    cap_adata.create_layer(
        name="empty_dense_array",
        matrix=None,
        matrix_shape=shape,
        data_dtype=dtype,
        format="dense",
    )
    # Create empty layer for sparse matrix
    # NOTE: it will create empty sparse_dataset in the .h5ad file.
    # One can fill it with AnnData.experimental.sparse_dataset.append() method
    cap_adata.create_layer(
        name="empty_sparse_matrix",
        matrix=None,
        matrix_shape=shape,
        data_dtype=dtype,
        format='csr', # or 'csc'
    )

# Add data with chunks
with read_h5ad(file_path, edit=True) as cap_adata:
    # Dense dataset
    cap_adata.layers["empty_dense_array"][0, 0] = 1
    # Sparse datasets
    sparse_dataset = cap_adata.layers["empty_sparse_matrix"]
    sparse_dataset.append(chunk_data) # chunk_data must be 'scipy.sparse.csr(c)_matrix'

# Remove layer
# NOTE: The layer will be removed from the .h5ad file immediately
cap_adata.remove_layer("dense_array")

# Delayed remove
# NOTE: the array mapping which is used for layers
# and similar entities behaves like a dict() object.
cap_adata.layers.pop("dense_array")  # removed only from in-memory object, still exists in the .h5ad file
cap_adata.overwrite(["layers"])  # propagate removal to the .h5ad file
```


## 7. Handle **obsm** and **varm** sections

By the default the CapAnnData will not read the embedding matrix. 
The link to the h5py objects will be created upon the first call of the `.obsm` property. 
Alike the AnnData package the call like `cap_adata.obsm["X_tsne"]` will not return the in-memory matrix but will return the backed version instead. 
It is possible to get the information about the name and shape of the embeddings without taking the whole matrix in the memory.

```python
with read_h5ad(file_path=file_path, edit=False) as cap_adata:
    obsm_keys = cap_adata.obsm_keys()  # ['X_tsne', 'X_umap']

    # return the shape of the matrix in backed mode
    embeddings = obsm_keys[0]
    shape = cap_adata.obsm[embeddings].shape  

    # take the whole matrix in memory
    matrix = cap_adata.obsm[embeddings][:]
```

One can create new embeddings in the `.h5ad` file by the `create_obsm` method and delete existing with the `remove_obsm` method.
The usage is the same to the methods described in [layers section](#6-work-with-layers). 
The same is for `varm` matrices, just replace `obsm` with `varm`.


## 8. Work with **obsp** and **varp**
The usage of the `obsp` and `varp` is quite the same as for the `obsm`, `varm` and `layers`. One can read entity of interest with `read_obsp(key)` and `read_varp(key)` methods. Create new objects with `create_obsp` and `create_varp`, remove existing with `remove_obsp` and `remove_varp`.
See [layers](#6-work-with-layers) and [obsm/varm](#7-handle-obsm-and-varm-matrices) sections for examples. 


## 9. Read and Modify **uns** Section

The `CapAnnData` class will lazily link the uns section upon the first call but ***WILL NOT*** read it into memory. Instead, the dictionary of the pairs `{'key': "__NotLinkedObject"}` will be created. It allows to get the list of keys before the actual read. To read the uns section in the memory the `.read_uns(keys)` method must be called.

```python
with read_h5ad(file_path=file_path, edit=True) as cap_adata:
    # will return the keys() object
    keys = cap_adata.uns.keys()  

    # read in memory the first key only
    cap_adata.read_uns([keys[0]])

    # read the whole uns section into memory
    cap_adata.read_uns()
```

Since the `.uns` section is in the memory (partially or completely) we can work with it as with the regular `dict()` python object. The main feature of the `CapAnnDataDict` class which inherited from `dict` is the tracking of the keys which must be removed from the `.h5ad` file upon overwrite. 

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
cap_adata.overwrite(["uns"])  # overwrite the uns section only
```


## 10. Join and Merge CapAnnDataDF

Cap-AnnData provides enhanced methods for joining and merging dataframes, preserving column order and data integrity

```python
from cap_anndata import CapAnnDataDF
import pandas as pd

data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
data2 = pd.DataFrame({'D': [7, 8, 9], 'E': [10, 11, 12]})
cap_anndata_df1 = CapAnnDataDF.from_df(data1)
cap_anndata_df1.column_order=['C', 'B', 'A'] # could be changed

cap_df = cap_anndata_df1.join(data2, how='left')

cap_df.columns  # ['A', 'B', 'D', 'E']
cap_df.column_order  # ['A', 'B', 'C', 'D', 'E']

data3 = pd.DataFrame({'A': [2, 3, 4], 'D': [10, 11, 12]})
cap_df = cap_anndata_df1.merge(data3, on='A')

cap_df.columns  # ['A', 'B', 'D']
cap_df.column_order  # ['A', 'B', 'C', 'D']
cap_df.shape  # (2, 3)
```
