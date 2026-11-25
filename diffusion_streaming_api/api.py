import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from diffusion_streaming_api.routers.difusion_router import diffusion_router
from diffusion_streaming_api.routers.difusion_router import get_inference
from diffusion_streaming_api.routers.latent_diffusion_router import (
  get_inference as get_latent_inference,
)
from diffusion_streaming_api.routers.latent_diffusion_router import (
  latent_diffusion_router,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
  # Load the model at container start, off the event loop, so the server
  # can answer the Lambda adapter's readiness check while it loads.
  load_tasks = [
    asyncio.create_task(asyncio.to_thread(loader))
    for loader in (get_inference, get_latent_inference)
  ]
  yield
  for load_task in load_tasks:
    load_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://www.sachahu.com", "https://sachahu.com"],
  allow_credentials=False,
  allow_methods=["GET"],
  allow_headers=["*"],
)

app.include_router(diffusion_router, prefix="/diffusion", tags=["diffusion"])
app.include_router(
  latent_diffusion_router,
  prefix="/latent-diffusion",
  tags=["latent-diffusion"],
)


@app.get("/health", tags=["health"])
async def health():
  """Warm-up endpoint: waits until every model is loaded, then returns."""
  await asyncio.gather(
    asyncio.to_thread(get_inference),
    asyncio.to_thread(get_latent_inference),
  )
  return {"status": "ok"}
