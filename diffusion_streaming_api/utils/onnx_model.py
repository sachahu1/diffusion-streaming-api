import os

import onnxruntime
import torch


# The names the UNet is exported with (see scripts/export_unet_onnx.py).
INPUT_NAMES = ("images", "timestep")
OUTPUT_NAME = "noise"


class OnnxDiffusionModel:
  """A diffusion model whose UNet runs with ONNX Runtime on the CPU.

  It stands in for a `BaseDiffusionModel`: the diffuser calls it with the noisy
  images and the timesteps and gets the predicted noise back, and
  `DiffusionInference` only needs `to`, `eval` and `diffuser`. The batch size
  is fixed by the export.
  """

  def __init__(self, path: str, diffuser) -> None:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
      onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.inter_op_num_threads = 1
    # Left unset, ONNX Runtime sizes its pool from the physical cores it detects
    # and pins its threads to them, ignoring the CPUs this process may use (such
    # as the vCPUs of a Lambda function). Use exactly those.
    options.intra_op_num_threads = len(os.sched_getaffinity(0))
    self.session = onnxruntime.InferenceSession(
      path, options, providers=["CPUExecutionProvider"]
    )
    self.diffuser = diffuser

  def __call__(
    self,
    images: torch.Tensor,
    timestep: torch.Tensor,
    labels: torch.Tensor | None = None,
  ) -> torch.Tensor:
    (noise,) = self.session.run(
      [OUTPUT_NAME],
      {INPUT_NAMES[0]: images.numpy(), INPUT_NAMES[1]: timestep.numpy()},
    )
    return torch.from_numpy(noise)

  def to(self, device: str = "cpu") -> "OnnxDiffusionModel":
    if device != "cpu":
      raise ValueError("ONNX Runtime is only set up to run on the CPU here.")
    return self

  def eval(self) -> "OnnxDiffusionModel":
    return self
