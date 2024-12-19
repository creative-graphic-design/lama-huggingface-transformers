import pytest

from lama import LamaPretrainedModel, convert_from_big_lama_zip


@pytest.fixture
def original_repo_id() -> str:
    return "smartywu/big-lama"


@pytest.fixture
def zip_filename() -> str:
    return "big-lama.zip"


@pytest.fixture
def checkpoint_path() -> str:
    return "big-lama/models/best.ckpt"


def test_lama(original_repo_id: str, zip_filename: str, checkpoint_path: str):
    model = convert_from_big_lama_zip(
        repo_id=original_repo_id,
        zip_filename=zip_filename,
        checkpoint_path=checkpoint_path,
    )
    assert isinstance(model, LamaPretrainedModel)
