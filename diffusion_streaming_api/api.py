from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from diffusion_streaming_api.routers.difusion_router import diffusion_router


app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(diffusion_router, prefix="/diffusion", tags=["diffusion"])
