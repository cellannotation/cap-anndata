import tempfile
import os
from unittest.mock import call, patch

from cap_anndata.reader import read_h5ad, read_directly
from test.context import get_base_anndata
import pytest
import warnings


def prepare_h5ad_file(name):
    adata = get_base_anndata()
    temp_folder = tempfile.mkdtemp()
    file_path = os.path.join(temp_folder, name)
    adata.write_h5ad(file_path)
    return file_path


@pytest.mark.parametrize("edit", [True, False])
def test_read_in_context(edit):
    file_path = prepare_h5ad_file("test_read_in_context.h5ad")

    with read_h5ad(file_path=file_path, edit=edit) as cap_adata:
        assert cap_adata is not None, "CapAnnData file must be valid!"
        cap_adata.read_obs()
        cap_adata.read_uns()
        if edit:
            cap_adata.overwrite()

    os.remove(file_path)


@pytest.mark.parametrize("edit", [True, False])
def test_read_as_function(edit):
    file_path = prepare_h5ad_file("test_read_as_function.h5ad")

    cap_adata = read_h5ad(file_path=file_path, edit=edit)
    assert cap_adata is not None, "CapAnnData file must be valid!"
    cap_adata.read_obs()
    cap_adata.read_uns()
    if edit:
        cap_adata.overwrite()

    cap_adata.file.close()
    os.remove(file_path)


@pytest.mark.parametrize("edit", [True, False])
def test_read_directly(edit):
    # TODO: remove deprecated function and unit test
    file_path = prepare_h5ad_file("test_read_directly.h5ad")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cap_adata = read_directly(file_path=file_path, edit=edit)

    assert cap_adata is not None, "CapAnnData file must be valid!"
    cap_adata.read_obs()
    cap_adata.read_uns()
    if edit:
        cap_adata.overwrite()

    cap_adata.file.close()
    os.remove(file_path)


@pytest.mark.parametrize(
    "retries,retry_delay,lock_clears_after,expected_attempts,expected_sleeps",
    [
        # The file opens after two locked attempts.
        (2, 0.25, 2, 3, 2),
        # The file remains locked after all retries are exhausted.
        (2, 1, None, 3, 2),
        # Retry is disabled by default.
        (0, 1, None, 1, 0),
        # A negative retries value disables retry.
        (-1, 1, None, 1, 0),
        # A negative retry delay disables retry.
        (2, -1, None, 1, 0),
    ],
)
def test_read_retry_logic(
    retries,
    retry_delay,
    lock_clears_after,
    expected_attempts,
    expected_sleeps,
):
    """Test all retry outcomes.

    Args:
        retries: Number of additional read attempts passed to ``read_h5ad``.
        retry_delay: Delay in seconds passed to ``read_h5ad``.
        lock_clears_after: Number of locked attempts before a successful open;
            ``None`` means the file remains locked.
        expected_attempts: Expected total number of calls to ``h5py.File``.
        expected_sleeps: Expected number of delays between attempts.
    """
    file = object()
    side_effect = (
        BlockingIOError
        if lock_clears_after is None
        else [BlockingIOError] * lock_clears_after + [file]
    )

    with patch("cap_anndata.reader.h5py.File", side_effect=side_effect) as h5py_file:
        with patch("cap_anndata.reader.sleep") as sleep:
            if lock_clears_after is None:
                with pytest.raises(BlockingIOError):
                    read_h5ad(
                        "locked.h5ad",
                        retries=retries,
                        retry_delay=retry_delay,
                    )
            else:
                cap_adata = read_h5ad(
                    "locked.h5ad",
                    retries=retries,
                    retry_delay=retry_delay,
                )
                assert cap_adata.file is file, (
                    "read_h5ad must return CapAnnData backed by the successfully opened file"
                )

    assert h5py_file.call_count == expected_attempts, (
        f"Expected {expected_attempts} file open attempts, "
        f"but got {h5py_file.call_count}"
    )
    expected_sleep_calls = [call(retry_delay)] * expected_sleeps
    assert sleep.call_args_list == expected_sleep_calls, (
        f"Expected retry delays {expected_sleep_calls}, "
        f"but got {sleep.call_args_list}"
    )
