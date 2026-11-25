"""Export the UNet of a latent diffusion checkpoint to ONNX for the API.

The API runs the UNet with ONNX Runtime, which fuses the graph and is faster
than eager PyTorch on CPU. The batch size is fixed to one image, which is what
the streaming route generates.

    python scripts/export_unet_onnx.py <checkpoint.pt> <unet.onnx>

The checkpoint needs its EMA weights, so export before stripping it with
`trim_checkpoint.py --metadata-only`.
"""

import argparse
import pathlib

import torch
from diffusion_models.diffusion.schemas import Checkpoint
from hydra.utils import instantiate

from diffusion_streaming_api.utils.checkpoint import load_config
from diffusion_streaming_api.utils.onnx_model import INPUT_NAMES
from diffusion_streaming_api.utils.onnx_model import OUTPUT_NAME


def export(checkpoint_path: pathlib.Path, destination: pathlib.Path) -> None:
  checkpoint = Checkpoint.from_file(
    str(checkpoint_path), map_location="cpu", load_optimizer=False
  )
  config = load_config(checkpoint.config_yaml)
  model = instantiate(config.model)
  model.load_state_dict(checkpoint.get_model_state_dict(use_ema=True))
  model.eval()

  images = torch.randn(
    1, checkpoint.image_channels, config.image_size, config.image_size
  )
  timestep = torch.zeros(1, dtype=torch.int64)
  with torch.no_grad():
    torch.onnx.export(
      model,
      (images, timestep),
      str(destination),
      input_names=list(INPUT_NAMES),
      output_names=[OUTPUT_NAME],
      opset_version=17,
      dynamo=False,
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("checkpoint", type=pathlib.Path)
  parser.add_argument("destination", type=pathlib.Path)
  arguments = parser.parse_args()
  export(arguments.checkpoint, arguments.destination)
  size = arguments.destination.stat().st_size / 1e6
  print(f"Wrote {arguments.destination} ({size:.0f} MB)")  # noqa: T201


if __name__ == "__main__":
  main()
