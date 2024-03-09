import logging
import sys
import pytest


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Start unit testing")

    files = [
        "test_backed_df.py",
        "test_backed_uns.py",
        "test_cap_anndata.py",
    ]
    sys.exit(pytest.main(files))
    