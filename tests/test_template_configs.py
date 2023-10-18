import importlib.resources
from stort.config import Config


CONFIG_TYPES = ["minimal", "default", "full"]


def test_template_configs():
    for config_type in CONFIG_TYPES:
        json_data = importlib.resources.read_text(
            package="stort.data",
            resource=f"config_{config_type}.json",
        )
        Config.model_validate_json(json_data)
