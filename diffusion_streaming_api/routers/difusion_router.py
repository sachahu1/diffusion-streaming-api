import asyncio
from threading import Lock
from typing import Annotated
from typing import Generator

from diffusion_models.diffusion_inference import DiffusionInference
from diffusion_models.gaussian_diffusion.ddimm_diffuser import DdimDiffuser
from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.utils.schemas import Checkpoint
from fastapi import APIRouter
from fastapi import Depends
from starlette.responses import StreamingResponse
from torchvision.transforms import v2

from diffusion_streaming_api.utils.image import image_to_bytes


diffusion_router = APIRouter()


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
  checkpoint_file_path = "new_checkpoint.pt"

  checkpoint = Checkpoint.from_file(checkpoint_file_path, map_location="cpu")
  diffuser = DdimDiffuser.from_checkpoint(checkpoint)

  model = SimpleUnet(
    image_channels=checkpoint.image_channels, diffuser=diffuser
  )
  model.load_state_dict(checkpoint.model_state_dict)

  reverse_transforms = v2.Compose(
    [
      v2.Lambda(lambda x: (x + 1) / 2),
      v2.Resize((128, 128)),
    ]
  )

  return DiffusionInference(
    model=model, reverse_transforms=reverse_transforms, device="cpu"
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
