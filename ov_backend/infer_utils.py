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

    def __init__(self, compiled_model, pool_size=2):
        self.compiled_model = compiled_model
        self.pool = queue.Queue()
        self.output_count = len(compiled_model.outputs)
        jobs = max(1, pool_size)
        for _ in range(jobs):
            self.pool.put(compiled_model.create_infer_request())

    def infer(self, feed_dict):
        if not isinstance(feed_dict, dict):
            raise ValueError("feed_dict must be a mapping of input_name -> ndarray")
        infer_request = self.pool.get()
        try:
            for name, arr in feed_dict.items():
                tensor = infer_request.get_tensor(name)
                np.copyto(tensor.data, arr, casting="unsafe")
            infer_request.infer()
            return [
                np.array(infer_request.get_output_tensor(i).data, copy=True)
                for i in range(self.output_count)
            ]
        finally:
            self.pool.put(infer_request)
