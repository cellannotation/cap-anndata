import logging
import contextlib
import h5py

from cap_anndata import CapAnnData


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def read_h5ad(file_path: str, edit: bool = False):
    """
    This is the main read method for CapAnnData.
    Must be used in 'with' context.
    """
    mode = "r+" if edit else "r"
    logger.debug(f"Read file {file_path} mode={mode} in context...")

    try:
        file = h5py.File(file_path, mode)
        cap_adata = CapAnnData(file)
        logger.debug(f"Successfully read anndata file path {file_path}")
        yield cap_adata

    except Exception as error:
        logger.error(f"Error during read anndata file at path: {file_path}, error = {error}!")
        raise error

    finally:
        file.close()
        logger.debug("AnnData closed!")


def read_directly(file_path: str, edit: bool = False) -> CapAnnData:
    """
    Must be used only in specific cases.
    User is responsible to close the h5py file when the work with CapAnnData instance done.
    """
    mode = "r+" if edit else "r"
    logger.debug(f"Read file {file_path} mode={mode} directly...")
    file = h5py.File(file_path, mode)
    cap_adata = CapAnnData(file)
    return cap_adata
