import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat

from .utils import temporary_full_cpu_affinity

MEAN_VALUES = [0.485, 0.456, 0.406]
STD_VALUES = [0.229, 0.224, 0.225]


def build_preprocessed_model(
    core,
    model_path,
    device,
    input_hw,
    remote_context=None,
    batch_size=1,
    precision=ov.Type.f32,
    compile_props=None,
):
    model = core.read_model(model_path)
    ppp = PrePostProcessor(model)
    tensor_info = ppp.input().tensor()
    tensor_info.set_element_type(ov.Type.u8)
    tensor_info.set_layout(ov.Layout("NHWC"))
    tensor_info.set_shape(ov.PartialShape([batch_size, input_hw[0], input_hw[1], 3]))
    tensor_info.set_color_format(ColorFormat.BGR)
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    preprocess = ppp.input().preprocess()
    preprocess.convert_element_type(precision)
    preprocess.convert_color(ColorFormat.RGB)
    preprocess.mean([m * 255.0 for m in MEAN_VALUES])
    preprocess.scale([s * 255.0 for s in STD_VALUES])
    model = ppp.build()
    compile_props = dict(compile_props) if compile_props is not None else {}
    compile_props.setdefault("PERFORMANCE_HINT", "LATENCY")
    with temporary_full_cpu_affinity(device):
        if remote_context is not None:
            return core.compile_model(model, remote_context, compile_props)
        return core.compile_model(model, device, compile_props)
