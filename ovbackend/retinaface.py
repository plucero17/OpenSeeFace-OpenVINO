import json
import os
import queue

import cv2
import numpy as np
import openvino as ov

from .infer_utils import InferRequestPool, wrap_tensor
from .utils import resolve, temporary_full_cpu_affinity


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def decode(loc, priors, variances):
    data = (
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
    )
    boxes = np.concatenate(data, 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class RetinaFaceDetector:
    def __init__(
        self,
        model_path=None,
        json_path=None,
        device="GPU",
        min_conf=0.4,
        nms_threshold=0.4,
        top_k=1,
        res=(640, 640),
        async_requests=2,
        remote_context=None,
        core=None,
    ):
        if model_path is None:
            model_path = resolve(os.path.join("ov-models", "retinaface_640x640_opt.xml"))
        if json_path is None:
            json_path = resolve(os.path.join("models", "priorbox_640x640.json"))
        self.core = core or ov.Core()
        self.device = device
        self.core.set_property(self.device, {"PERFORMANCE_HINT": "LATENCY"})
        model = self.core.read_model(model_path)
        if remote_context is not None:
            self.session = self.core.compile_model(model, remote_context)
        else:
            with temporary_full_cpu_affinity(self.device):
                self.session = self.core.compile_model(model, self.device)
        self.input_name = self.session.input(0).get_any_name()
        self.output_count = len(self.session.outputs)
        self.res_w, self.res_h = res
        with open(json_path, "r") as prior_file:
            self.priorbox = np.array(json.loads(prior_file.read()))
        self.min_conf = min_conf
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.umat_enabled = hasattr(cv2, "UMat")
        self.remote_context = remote_context
        self.tensor_wrapper = (lambda data: wrap_tensor(data, self.remote_context)) if remote_context else None
        pool_size = max(1, async_requests)
        self.sync_pool = InferRequestPool(self.session, pool_size=pool_size, tensor_wrapper=self.tensor_wrapper)
        self.results = queue.Queue()
        self.async_queue = None
        if async_requests > 0:
            self.async_queue = ov.AsyncInferQueue(self.session, jobs=async_requests)
            self.async_queue.set_callback(self._async_callback)

    def _to_umat(self, image):
        if self.umat_enabled and image is not None and not isinstance(image, cv2.UMat):
            return cv2.UMat(image)
        return image

    def _from_umat(self, image):
        if self.umat_enabled and isinstance(image, cv2.UMat):
            return image.get()
        return image

    def _prepare_input(self, frame):
        h, w = frame.shape[0:2]
        work = self._to_umat(frame)
        im = cv2.resize(work, (self.res_w, self.res_h), interpolation=cv2.INTER_LINEAR)
        im = self._from_umat(im)
        im = np.float32(im)
        scale = np.array([w, h, w, h], dtype=np.float32)
        im -= (104, 117, 123)
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, 0)
        return im, scale

    def _postprocess(self, loc, conf, scale):
        boxes = decode(loc, self.priorbox, [0.1, 0.2])
        boxes = boxes * scale
        scores = conf[:, 1]

        inds = np.where(scores > self.min_conf)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        if dets.size == 0:
            return []
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        dets = dets[: self.top_k, 0:4]
        dets[:, 2:4] = dets[:, 2:4] - dets[:, 0:2]

        upsize = dets[:, 2:4] * np.array([[0.15, 0.2]], dtype=np.float32)
        dets[:, 0:2] -= upsize
        dets[:, 2:4] += upsize * 2
        return list(map(tuple, dets))

    def detect_retina(self, frame):
        tensor, scale = self._prepare_input(frame)
        outputs = self.sync_pool.infer({self.input_name: tensor})
        loc, conf = outputs[0][0], outputs[1][0]
        return self._postprocess(loc, conf, scale)

    def background_detect(self, frame):
        if self.async_queue is None:
            return False
        if not self.async_queue.is_ready():
            return False
        tensor, scale = self._prepare_input(frame)
        tensor = self.tensor_wrapper(tensor) if self.tensor_wrapper else tensor
        self.async_queue.start_async({self.input_name: tensor}, scale)
        return True

    def get_results(self):
        detections = []
        while True:
            try:
                det = self.results.get(False)
            except queue.Empty:
                break
            detections.extend(det)
        return detections

    def _async_callback(self, infer_request, scale):
        if scale is None:
            return
        loc = np.array(infer_request.get_output_tensor(0).data, copy=True)
        conf = np.array(infer_request.get_output_tensor(1).data, copy=True)
        dets = self._postprocess(loc[0], conf[0], scale)
        if dets:
            self.results.put(dets)


if __name__ == "__main__":
    retina = RetinaFaceDetector(top_k=40, min_conf=0.2)
