import pytest
import torch

from diffusion_streaming_api.utils.onnx_model import INPUT_NAMES
from diffusion_streaming_api.utils.onnx_model import OUTPUT_NAME
from diffusion_streaming_api.utils.onnx_model import OnnxDiffusionModel


class _TinyUnet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

  def forward(self, images, timestep):
    return self.conv(images) * timestep.float().reshape(-1, 1, 1, 1)


@pytest.fixture
def exported(tmp_path):
  torch.manual_seed(0)
  model = _TinyUnet().eval()
  path = tmp_path / "tiny.onnx"
  torch.onnx.export(
    model,
    (torch.randn(1, 4, 8, 8), torch.zeros(1, dtype=torch.int64)),
    str(path),
    input_names=list(INPUT_NAMES),
    output_names=[OUTPUT_NAME],
    opset_version=17,
    dynamo=False,
  )
  return model, str(path)


def test_runs_the_exported_unet_like_pytorch(exported):
  model, path = exported
  onnx_model = OnnxDiffusionModel(path, diffuser=object())
  images = torch.randn(1, 4, 8, 8)
  timestep = torch.tensor([3])
  with torch.no_grad():
    expected = model(images, timestep)
  actual = onnx_model(images, timestep)
  assert isinstance(actual, torch.Tensor)
  torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_stands_in_for_a_diffusion_model(exported):
  _, path = exported
  diffuser = object()
  onnx_model = OnnxDiffusionModel(path, diffuser)
  assert onnx_model.to("cpu").eval() is onnx_model
  assert onnx_model.diffuser is diffuser


def test_refuses_other_devices(exported):
  _, path = exported
  with pytest.raises(ValueError):
    OnnxDiffusionModel(path, object()).to("cuda")
