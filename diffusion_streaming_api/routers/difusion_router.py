import asyncio

from diffusion_models.diffusion_inference import DiffusionInference
from diffusion_models.gaussian_diffusion.gaussian_diffuser import (
  GaussianDiffuser,
)
from diffusion_models.models.SimpleUnet import SimpleUnet
from diffusion_models.utils.schemas import Checkpoint
from fastapi import APIRouter
from starlette.responses import StreamingResponse
from torchvision.transforms import v2

from diffusion_streaming_api.utils.image import image_to_bytes


diffusion_router = APIRouter()


def get_generator():
  checkpoint_file_path = "new_checkpoint.pt"

  checkpoint = Checkpoint.from_file(checkpoint_file_path, map_location="cpu")
  gaussian_diffuser = GaussianDiffuser.from_checkpoint(checkpoint)

  model = SimpleUnet(
    image_channels=checkpoint.image_channels, diffuser=gaussian_diffuser
  )
  model.load_state_dict(checkpoint.model_state_dict)

  reverse_transforms = v2.Compose(
    [
      v2.Lambda(lambda x: (x + 1) / 2),
      v2.Resize((128, 128)),
    ]
  )

  inference = DiffusionInference(
    model=model, reverse_transforms=reverse_transforms, device="cpu"
  )
  generator = inference.get_generator(number_of_images=1)
  return generator


@diffusion_router.get("/newImage")
async def generate_new_image():
  """

  Returns:

  """
  generator = get_generator()

  async def image_to_io():
    for image in generator:
      image_bytes = image_to_bytes(image)
      yield image_bytes
      await asyncio.sleep(0)

  return StreamingResponse(
    content=image_to_io(), media_type="application/json"
  )
