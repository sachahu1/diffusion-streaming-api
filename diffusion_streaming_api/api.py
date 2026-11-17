import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from diffusion_streaming_api.routers.difusion_router import diffusion_router
from diffusion_streaming_api.routers.difusion_router import get_inference


@asynccontextmanager
async def lifespan(_: FastAPI):
  # Load the model at container start, off the event loop, so the server
  # can answer the Lambda adapter's readiness check while it loads.
  load_task = asyncio.create_task(asyncio.to_thread(get_inference))
  yield
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
