"""Torch-native TensorRT runner -- a drop-in for Ditto's vendored TRTWrapper.

Why this exists
---------------
Ditto's own ``core/utils/tensorrt_utils.TRTWrapper`` needs the ``cuda-python``
package (``from cuda import cuda, cudart, nvrtc``) and is written against the
TensorRT-8 Linux ABI (it dlopen's ``libnvinfer.so.8``). Neither holds on this
box: ``cuda-python`` isn't installed and we run TensorRT 10 on Windows
(``nvinfer_10.dll``). Importing that module fails before any engine can load.

This runner reproduces the *exact* call interface the Ditto model wrappers use
(``core/models/decoder.py`` etc.):

    m.setup({"feature": np_array})   # named numpy inputs
    m.infer()                        # run
    out = m.buffer["output"][0]      # host numpy output, indexed by tensor name

...but drives TensorRT 10 directly through torch CUDA tensors (``data_ptr()`` +
the current torch CUDA stream). That needs only ``tensorrt`` + ``torch`` -- both
already present -- and keeps the engine on the same stream as the PyTorch stages
(warp stays PyTorch), so the hybrid pipeline interops cleanly with no extra deps.

ASCII-only on purpose (server .py files are cp1252-safe, per CLAUDE.md).
"""
from __future__ import annotations

import numpy as np
import tensorrt as trt
import torch

_LOGGER = trt.Logger(trt.Logger.ERROR)


def _torch_dtype(np_dt: np.dtype) -> torch.dtype:
    return {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
    }[np.dtype(np_dt).type]


class TRTRunner:
    """Run one serialized TensorRT engine via torch CUDA buffers.

    Mirrors the (setup/infer/buffer) interface of Ditto's TRTWrapper so the
    vendored model wrappers can use it unchanged. ``plugin_file_list`` is accepted
    for signature-compatibility but unused (the models we convert need no plugin;
    the one that does -- warp_network's GridSample3D -- stays on PyTorch).
    """

    def __init__(self, trt_file: str, plugin_file_list: list | None = None) -> None:
        self.model = trt_file
        with open(trt_file, "rb") as f, trt.Runtime(_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"failed to deserialize engine {trt_file}"
        self.context = self.engine.create_execution_context()

        self._names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self._is_input = {
            n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self._names
        }
        self._np_dtype = {n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self._names}
        # Persistent device tensors (kept alive across infer calls) + host mirrors.
        self._dev: dict[str, torch.Tensor] = {}
        self.buffer: dict[str, list] = {}

    def setup(self, input_data: dict) -> None:
        """Upload named numpy inputs to GPU and (re)bind all tensor addresses."""
        # Inputs: cast to the engine dtype, copy to a CUDA tensor, fix the shape.
        for name, arr in input_data.items():
            np_dt = self._np_dtype[name]
            arr = np.ascontiguousarray(arr, dtype=np_dt)
            t = torch.from_numpy(arr).to("cuda", non_blocking=True)
            self._dev[name] = t
            self.context.set_input_shape(name, tuple(arr.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        # Outputs: now that input shapes are set, output shapes resolve. Allocate
        # (or reuse) a CUDA tensor for each and bind it.
        for name in self._names:
            if self._is_input[name]:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            t = self._dev.get(name)
            if t is None or tuple(t.shape) != shape:
                t = torch.empty(shape, dtype=_torch_dtype(self._np_dtype[name]), device="cuda")
                self._dev[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

    def infer(self, stream: int = 0) -> None:
        """Execute on torch's current CUDA stream, then mirror outputs to host.

        The wrappers read ``buffer[name][0]`` as a host numpy array right after,
        so we synchronize and copy back (the decode/warp/putback path is numpy on
        the CPU side between GPU stages)."""
        s = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(s)
        torch.cuda.current_stream().synchronize()
        for name in self._names:
            if self._is_input[name]:
                continue
            self.buffer[name] = [self._dev[name].detach().cpu().numpy()]
