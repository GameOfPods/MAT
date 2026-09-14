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
