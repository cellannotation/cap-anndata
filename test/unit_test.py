import logging
import sys
import pytest


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Start unit testing")

    files = [
        "test_backed_df.py",
        "test_backed_dict.py",
        "test_cap_anndata.py",
        "test_reader.py",
    ]
    sys.exit(pytest.main(files))
