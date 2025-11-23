import queue

import numpy as np
import openvino as ov


_NP_TO_OV_TYPE = {
    np.dtype("float32"): ov.Type.f32,
    np.dtype("float16"): ov.Type.f16,
    np.dtype("uint8"): ov.Type.u8,
    np.dtype("int8"): ov.Type.i8,
}


def wrap_tensor(array, remote_context):
    """
    Wrap numpy arrays into tensors backed by the provided remote context.
    Falls back to the original data if wrapping is not possible.
    """
    if remote_context is None or not isinstance(array, np.ndarray):
        return array
    ov_type = _NP_TO_OV_TYPE.get(array.dtype)
    if ov_type is None:
        return array
    try:
        tensor = remote_context.create_host_tensor(ov_type, ov.Shape(list(array.shape)))
        np.copyto(tensor.data, array, casting="unsafe")
        return tensor
    except Exception:
        return array


class InferRequestPool:
    """
    Simple pool of reusable infer requests to avoid reallocation overhead.
    """

    def __init__(self, compiled_model, pool_size=2, tensor_wrapper=None):
        self.compiled_model = compiled_model
        self.pool = queue.Queue()
        self.tensor_wrapper = tensor_wrapper
        self.output_count = len(compiled_model.outputs)
        jobs = max(1, pool_size)
        for _ in range(jobs):
            self.pool.put(compiled_model.create_infer_request())

    def _prepare_inputs(self, inputs):
        if self.tensor_wrapper is None:
            return inputs
        if isinstance(inputs, dict):
            return {key: self.tensor_wrapper(value) for key, value in inputs.items()}
        if isinstance(inputs, (list, tuple)):
            wrapped = [self.tensor_wrapper(value) for value in inputs]
            return type(inputs)(wrapped)
        return self.tensor_wrapper(inputs)

    def infer(self, inputs):
        infer_request = self.pool.get()
        try:
            prepared = self._prepare_inputs(inputs)
            infer_request.infer(prepared)
            return [
                np.array(infer_request.get_output_tensor(i).data, copy=True)
                for i in range(self.output_count)
            ]
        finally:
            self.pool.put(infer_request)
