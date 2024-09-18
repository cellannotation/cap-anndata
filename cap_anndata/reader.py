import logging
import h5py
import warnings

from cap_anndata import CapAnnData


logger = logging.getLogger(__name__)


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
        return cap_adata

    except Exception as error:
        logger.error(
            f"Error during read anndata file at path: {file_path}, error = {error}!"
        )
        raise error


def deprecated(message):
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function. {}".format(func.__name__, message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


# TODO: remove deprecated function
@deprecated(
    "It will be removed in the next version of package. Please replace it with `read_h5ad`."
)
def read_directly(file_path: str, edit: bool = False) -> CapAnnData:
    """
    Must be used only in specific cases.
    User is responsible to close the h5py file when the work with CapAnnData instance done.
    """
    return read_h5ad(file_path, edit)
