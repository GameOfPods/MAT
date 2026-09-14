import ctranslate2
import pytest

from MAT.utils.device import ct2_compute_type


@pytest.mark.parametrize("supported, expected", [
    ({"float32", "int8_float32", "int8"}, "int8_float32"),  # GTX 1080 Ti
    ({"float32", "float16", "int8_float32", "int8_float16", "int8"}, "int8_float16"),  # RTX cards
    ({"float32"}, "float32"),
])
def test_auto_picks_fastest_supported(monkeypatch, supported, expected):
    seen = []

    def fake(device):
        seen.append(device)
        return supported

    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", fake)
    assert ct2_compute_type("cuda:0") == expected
    assert seen == ["cuda"]


def test_explicit_type_is_kept(monkeypatch):
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: pytest.fail("should not be called"))
    assert ct2_compute_type("cuda", requested="float32") == "float32"


def test_cpu_auto_really_works():
    assert ct2_compute_type("cpu") in {"int8_float32", "int8", "float32"}


def test_resolve_device(monkeypatch):
    import torch
    from MAT.utils.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


@pytest.mark.parametrize("capability, preferred, expected", [
    ((6, 1), "bfloat16", "float16"),  # GTX 1080 Ti
    ((7, 5), "bfloat16", "float16"),  # RTX 20xx
    ((8, 6), "bfloat16", "bfloat16"),  # RTX 30xx
    ((8, 6), "float16", "float16"),
    ((6, 1), "float32", "float32"),
])
def test_torch_dtype_on_gpu(monkeypatch, capability, preferred, expected):
    import torch
    from MAT.utils.device import torch_dtype

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert torch_dtype("cuda", preferred) == getattr(torch, expected)


def test_torch_dtype_on_cpu_and_bad_input():
    import torch
    from MAT.utils.device import torch_dtype

    assert torch_dtype("cpu") == torch.float32
    with pytest.raises(ValueError):
        torch_dtype("cuda", "int4")


def test_free_gpu_memory_works_without_gpu():
    from MAT.utils.device import free_gpu_memory
    free_gpu_memory()
