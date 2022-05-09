import yaml  # type: ignore

from timething import align  # type: ignore

# yaml file containing all of the models
MODELS_YAML = "timething/models.yaml"


def load_config(model: str) -> align.Config:
    """
    Load config object for the given model key
    """

    with open(MODELS_YAML, "r") as f:
        cfg = yaml.safe_load(f)
        return align.Config(
            cfg[model]["model"], cfg[model]["pin"], cfg[model]["sampling_rate"]
        )
