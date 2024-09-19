# CAP-AnnData: Partial I/O for AnnData (.h5ad) Files

## Overview
CAP-AnnData offering functionalities for selective reading and writing of [AnnData](https://pypi.org/project/anndata/) 
file fields without the need for loading entire dataset (or even entire field) into memory. 
For example, it allows to read and modify the single `obs` column taking nothing into memory except the column itself.
Package eager to replicate the original AnnData API as much as possible, 
while providing additional features for efficient data manipulation for heavy datasets.

## Installation
Install CAP-AnnData via pip:

```commandline
pip install -U cap-anndata
```

## Basic Example

The example below displayes how to read a single `obs` column, create new obs column and propagate it to the `.h5ad` file.
```python
from cap_anndata import read_h5ad

file_path = "your_data.h5ad"
with read_h5ad(file_path=file_path, edit=True) as cap_adata:
    print(cap_adata.obs_keys())  # ['a', 'b', 'c']
    print(cap_adata.obs) # Empty DataFrame
    cap_adata.read_obs(columns=['a'])
    print(cap_adata.obs.columns) # ['a']
    cap_adata.obs['new_col'] = cap_adata.obs['a']
    cap_adata.overwrite(fields=['obs'])
```

More example can be found in the [How-TO](https://github.com/cellannotation/cap-anndata/blob/main/HOWTO.md) file. 
