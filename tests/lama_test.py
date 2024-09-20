from lama.configuration_lama import LamaConfig
from lama.modeling_lama import LamaModel


def test_lama():
    config = LamaConfig()
    model = LamaModel(config)
