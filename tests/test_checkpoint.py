import pytest

from diffusion_streaming_api.utils.checkpoint import load_config


def test_load_config_maps_legacy_targets():
  config = load_config(
    "model:\n"
    "  _target_: diffusion_models.models.SmallAttentionUnet.SmallAttentionUnet\n"
    "diffuser:\n"
    "  _target_: diffusion_models.gaussian_diffusion.ddimm_diffuser.DdimDiffuser\n"
  )
  assert config.model._target_.startswith("diffusion_models.diffusion.models.")
  assert config.diffuser._target_.startswith(
    "diffusion_models.diffusion.diffusers."
  )


def test_load_config_keeps_current_targets():
  config = load_config(
    "model:\n  _target_: diffusion_models.diffusion.models.X\n"
  )
  assert config.model._target_ == "diffusion_models.diffusion.models.X"


def test_load_config_requires_embedded_config():
  with pytest.raises(ValueError):
    load_config(None)
