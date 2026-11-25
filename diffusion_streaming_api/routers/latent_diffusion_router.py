import asyncio
import os
from collections.abc import Generator
from threading import Lock
from typing import Annotated

import torch
from diffusion_models.autoencoders.base import BaseAutoencoder
from diffusion_models.diffusion.diffusion_inference import DiffusionInference
from diffusion_models.diffusion.schemas import Checkpoint
from diffusion_models.latent.normalization import LatentNormalization
from fastapi import APIRouter
from fastapi import Depends
from hydra.utils import instantiate
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import StreamingResponse
from torchvision.transforms import v2

from diffusion_streaming_api.utils.checkpoint import load_config
from diffusion_streaming_api.utils.image import image_to_bytes
from diffusion_streaming_api.utils.onnx_model import OnnxDiffusionModel


latent_diffusion_router = APIRouter()

CHECKPOINT_FILE_PATH = os.environ.get(
  "LATENT_CHECKPOINT_FILE_PATH", "latent_checkpoint.pt"
)
AUTOENCODER_FILE_PATH = os.environ.get(
  "LATENT_AUTOENCODER_FILE_PATH", "latent_autoencoder.pt"
)
UNET_FILE_PATH = os.environ.get("LATENT_UNET_FILE_PATH", "latent_unet.onnx")

# Sampling settings (`diffusion-models infer steps=... eta=1.0`). They are fixed
# because the diffuser is shared by all requests. Time grows linearly with the
# steps.
NUMBER_OF_STEPS = 200
ETA = 1.0
# Decoding the latents is a significant part of a step's cost, so only about
# `NUMBER_OF_PREVIEWS` evenly spaced steps (and always the last) are decoded and
# streamed, however many steps are run.
NUMBER_OF_PREVIEWS = 10


_inference: DiffusionInference | None = None
_inference_lock = Lock()


def get_inference() -> DiffusionInference:
  """Load the model once per container and reuse it across requests."""
  global _inference
  with _inference_lock:
    if _inference is None:
      _inference = _load_inference()
  return _inference


def _load_inference() -> DiffusionInference:
  checkpoint = Checkpoint.from_file(
    CHECKPOINT_FILE_PATH, map_location="cpu", load_optimizer=False
  )
  if checkpoint.latent_scale_factor is None:
    raise ValueError(
      f"{CHECKPOINT_FILE_PATH} is not a latent diffusion checkpoint."
    )
  # The checkpoint carries the resolved config of the training run, so the
  # diffuser and beta scheduler are rebuilt exactly as they were trained.
  config = load_config(checkpoint.config_yaml)
  diffuser = instantiate(config.diffuser)
  diffuser.number_of_steps = NUMBER_OF_STEPS
  diffuser.eta = ETA
  # The UNet (EMA weights) was exported to ONNX, so the checkpoint's own weights
  # are not needed.
  model = OnnxDiffusionModel(UNET_FILE_PATH, diffuser)

  # The config points at the autoencoder of the training machine, so load it
  # from the file shipped alongside the checkpoint instead.
  autoencoder = BaseAutoencoder.load(AUTOENCODER_FILE_PATH, map_location="cpu")
  autoencoder.eval()
  normalization = LatentNormalization(
    scale=checkpoint.latent_scale_factor,
    shift=checkpoint.latent_shift,
    whitening=checkpoint.latent_whitening,
  )

  def decode(samples: torch.Tensor) -> torch.Tensor:
    return autoencoder.decode(
      normalization.denormalize(samples.float())
    ).clamp(-1, 1)

  output_channels = config.get("eval_image_channels", 3)
  reverse_transforms = v2.Compose(
    [
      v2.Normalize(mean=[-1.0] * output_channels, std=[2.0] * output_channels),
      v2.Resize((config.reverse_image_size, config.reverse_image_size)),
    ]
  )

  return DiffusionInference(
    model=model,
    reverse_transforms=reverse_transforms,
    # The model denoises latents, which `decode` turns into images.
    image_shape=(checkpoint.image_channels, config.image_size),
    device="cpu",
    decode=decode,
  )


@torch.no_grad()
def _denoise(inference: DiffusionInference) -> Generator:
  """Denoise one image, yielding about `NUMBER_OF_PREVIEWS` decoded previews.

  This is `DiffusionInference.get_generator` without the decode on every step.
  It also passes the diffuser's `eta` on: `get_generator` does not, so it
  always samples with eta=0.
  """
  diffuser = inference.model.diffuser
  images = torch.randn(
    (1, inference.image_channels, inference.image_size, inference.image_size),
    device=inference.device,
  )
  steps = diffuser.steps
  decode_every = max(1, len(steps) // NUMBER_OF_PREVIEWS)
  for position, step in enumerate(steps, start=1):
    timestep = diffuser.get_timestep(images.shape[0], step)
    images = diffuser._denoise_step(
      images, model=inference.model, timestep=timestep, eta=diffuser.eta
    )
    images = diffuser.clip(images)
    if position % decode_every == 0 or position == len(steps):
      yield inference._visualise_images(images)


async def get_generator() -> Generator:
  # Generators are single use, so build a fresh one per request.
  inference = await asyncio.to_thread(get_inference)
  return _denoise(inference)


@latent_diffusion_router.get("/newImage")
async def generate_new_image(
  generator: Annotated[Generator, Depends(get_generator)],
):
  """Stream the denoising of a single image as JPEG previews.

  About `NUMBER_OF_PREVIEWS` previews are decoded along the way, the last one
  being the final image. The steps run in a worker thread to keep the server
  responsive.
  """

  async def image_to_io():
    async for image in iterate_in_threadpool(generator):
      yield image_to_bytes(image)

  return StreamingResponse(
    content=image_to_io(), media_type="application/json"
  )
