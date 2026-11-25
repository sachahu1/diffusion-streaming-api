from omegaconf import DictConfig
from omegaconf import OmegaConf


# Checkpoints saved before the framework was reorganised embed `_target_`
# paths that no longer exist. Longest prefixes first.
_LEGACY_TARGETS = (
  (
    "diffusion_models.gaussian_diffusion.",
    "diffusion_models.diffusion.diffusers.",
  ),
  ("diffusion_models.models.", "diffusion_models.diffusion.models."),
  ("diffusion_models.utils.schemas.", "diffusion_models.core.schemas."),
)


def load_config(config_yaml: str | None) -> DictConfig:
  """Rebuild the training config embedded in a checkpoint.

  Args:
    config_yaml: The ``config_yaml`` of a checkpoint.

  Returns:
    The config, with legacy `_target_` module paths mapped to the current ones.

  Raises:
    ValueError: If the checkpoint has no embedded config.
  """
  if config_yaml is None:
    raise ValueError(
      "The checkpoint has no embedded config, so the model cannot be rebuilt."
    )
  for legacy, current in _LEGACY_TARGETS:
    config_yaml = config_yaml.replace(legacy, current)
  return OmegaConf.create(config_yaml)
