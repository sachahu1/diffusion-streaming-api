import asyncio
from threading import Lock
from typing import Annotated
from typing import Generator

from diffusion_models.diffusion_inference import DiffusionInference
from diffusion_models.utils.schemas import Checkpoint
from fastapi import APIRouter
from fastapi import Depends
from hydra.utils import instantiate
from omegaconf import OmegaConf
from starlette.responses import StreamingResponse
from torchvision.transforms import v2

from diffusion_streaming_api.utils.image import image_to_bytes


diffusion_router = APIRouter()

CHECKPOINT_FILE_PATH = "new_checkpoint.pt"


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
  checkpoint = Checkpoint.from_file(CHECKPOINT_FILE_PATH, map_location="cpu")
  if checkpoint.config_yaml is None:
    raise ValueError(
      "The checkpoint has no embedded config, so the model cannot be rebuilt."
    )
  # The checkpoint carries the resolved config of the training run, so the
  # model, diffuser and beta scheduler are rebuilt exactly as they were trained.
  config = OmegaConf.create(checkpoint.config_yaml)

  model = instantiate(config.model)
  # The EMA weights give better samples than the raw ones.
  model.load_state_dict(checkpoint.get_model_state_dict(use_ema=True))
  model.eval()

  reverse_transforms = v2.Compose(
    [
      v2.Normalize(
        mean=[-1.0] * checkpoint.image_channels,
        std=[2.0] * checkpoint.image_channels,
      ),
      v2.Resize((config.reverse_image_size, config.reverse_image_size)),
    ]
  )

  return DiffusionInference(
    model=model,
    reverse_transforms=reverse_transforms,
    image_shape=(checkpoint.image_channels, config.image_size),
    device="cpu",
  )


async def get_generator() -> Generator:
  # Generators are single use, so build a fresh one per request.
  inference = await asyncio.to_thread(get_inference)
  return inference.get_generator(number_of_images=1)


@diffusion_router.get("/health")
async def health():
  """Warm-up endpoint: ensures the model is loaded, then returns immediately."""
  await asyncio.to_thread(get_inference)
  return {"status": "ok"}


@diffusion_router.get("/newImage")
async def generate_new_image(generator: Annotated[Generator, Depends(get_generator)]):
  """

  Returns:

  """
  async def image_to_io():
    for image in generator:
      image_bytes = image_to_bytes(image)
      yield image_bytes
      await asyncio.sleep(0)

  return StreamingResponse(
    content=image_to_io(), media_type="application/json"
  )
