import sys
import pytest


if __name__ == "__main__":
    files = [
        "test_backed_df.py",
        "test_backed_dict.py",
        "test_cap_anndata.py",
        "test_reader.py",
    ]
    sys.exit(pytest.main(files))
