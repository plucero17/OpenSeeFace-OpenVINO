import os
from contextlib import contextmanager
from pathlib import Path


def _project_root():
    return Path(__file__).resolve().parent.parent


def resolve(path):
    return str(_project_root() / path)


def get_ov_model_base_path(model_dir):
    if model_dir is not None:
        return model_dir

    candidate_paths = [
        resolve(os.path.join("ov_models")),
        resolve(os.path.join("..", "ov_models")),
        resolve(os.path.join("ov-models")),
        resolve(os.path.join("..", "ov-models")),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate ov_models or legacy ov-models directory.")


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
