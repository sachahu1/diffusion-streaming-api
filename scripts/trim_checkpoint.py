"""Shrink a training checkpoint to what the API needs for inference.

Training checkpoints carry optimizer state, the GradScaler and (for
autoencoders) the discriminator, which inference never reads. The API only
loads the EMA weights, so the raw weights are dropped too when EMA weights
exist. The fields stay present (emptied) so the framework's checkpoint
classes can still load the file.

    python scripts/trim_checkpoint.py [--metadata-only] <input.pt> <output.pt>

With `--metadata-only` the weights are dropped as well, keeping only the config,
latent normalization and beta schedule. Use it for a latent checkpoint whose
UNet is shipped as ONNX (see `export_unet_onnx.py`, which needs the weights, so
run it first).

Works for both diffusion and autoencoder checkpoints.
"""

import argparse
import pathlib

import torch


# Fields that are only needed to resume training. They are emptied, not
# removed, because the checkpoint dataclasses require some of them.
_EMPTIED = {
  "optimizer_state_dict": {},
  "scaler": None,
  "discriminator_state_dict": None,
  "discriminator_optimizer_state_dict": None,
}


def trim(checkpoint: dict, metadata_only: bool = False) -> dict:
  trimmed = dict(checkpoint)
  for key, value in _EMPTIED.items():
    if key in trimmed:
      trimmed[key] = value
  trimmed["model_state_dict"] = {}
  if metadata_only:
    trimmed["ema_model_state_dict"] = None
  elif trimmed.get("ema_model_state_dict") is None:
    raise ValueError("No EMA weights to keep: the API loads the EMA weights.")
  return trimmed


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--metadata-only", action="store_true")
  parser.add_argument("source", type=pathlib.Path)
  parser.add_argument("destination", type=pathlib.Path)
  arguments = parser.parse_args()

  checkpoint = torch.load(
    arguments.source, weights_only=True, map_location="cpu"
  )
  torch.save(trim(checkpoint, arguments.metadata_only), arguments.destination)
  megabytes = lambda path: path.stat().st_size / 1e6  # noqa: E731
  print(  # noqa: T201
    f"{arguments.source}: {megabytes(arguments.source):.0f} MB -> "
    f"{arguments.destination}: {megabytes(arguments.destination):.0f} MB"
  )


if __name__ == "__main__":
  main()
