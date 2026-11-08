import io

import PIL.Image


def image_to_bytes(image: PIL.Image.Image) -> bytes:
  buffer = io.BytesIO()
  image.save(buffer, "JPEG")
  buffer.seek(0)
  return buffer.read()
