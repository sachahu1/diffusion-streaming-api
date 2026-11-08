from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from diffusion_streaming_api.routers.difusion_router import diffusion_router


app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://www.sachahu.com", "https://sachahu.com"],
  allow_credentials=False,
  allow_methods=["GET"],
  allow_headers=["*"],
)

app.include_router(diffusion_router, prefix="/diffusion", tags=["diffusion"])
