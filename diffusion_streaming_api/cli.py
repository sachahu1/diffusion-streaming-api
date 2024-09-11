import os

import uvicorn

from diffusion_streaming_api.api import app


def run():
  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
  run()
