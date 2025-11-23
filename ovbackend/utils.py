import os
from contextlib import contextmanager


def _project_root():
    return os.path.dirname(os.path.dirname(__file__))


def resolve(path):
    return os.path.join(_project_root(), path)


def get_ov_model_base_path(model_dir):
    model_base_path = resolve(os.path.join("ov-models"))
    if model_dir is None:
        if not os.path.exists(model_base_path):
            model_base_path = resolve(os.path.join("..", "ov-models"))
    else:
        model_base_path = model_dir
    return model_base_path


def get_model_base_path(model_dir):
    model_base_path = resolve(os.path.join("models"))
    if model_dir is None:
        if not os.path.exists(model_base_path):
            model_base_path = resolve(os.path.join("..", "models"))
    else:
        model_base_path = model_dir
    return model_base_path


@contextmanager
def temporary_full_cpu_affinity(device):
    """
    Some OpenVINO CPU routines crash when the current affinity mask does not include CPU 0.
    Temporarily broaden the mask to all CPUs while compiling CPU models, then restore it.
    """
    if device != "CPU":
        yield
        return
    get_aff = getattr(os, "sched_getaffinity", None)
    set_aff = getattr(os, "sched_setaffinity", None)
    if get_aff is None or set_aff is None:
        yield
        return
    try:
        original = get_aff(0)
    except Exception:
        yield
        return
    if 0 in original:
        yield
        return
    expanded = False
    try:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = max(original) + 1
        set_aff(0, set(range(cpu_count)))
        expanded = True
    except Exception:
        pass
    try:
        yield
    finally:
        if expanded:
            try:
                set_aff(0, original)
            except Exception:
                pass
