import sys
import textwrap

import pytest

from MAT import registry


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    created = []

    def make(name, body):
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
        created.append(name)
        return name

    yield make
    for name in created:
        sys.modules.pop(name, None)


def test_require_reports_missing_modules():
    with pytest.raises(registry.MissingDependencies) as error:
        registry.require("json", "mat_definitely_missing", "also_missing.submodule", extra="demo")
    assert error.value.missing == ("mat_definitely_missing", "also_missing.submodule")
    assert "uv sync --extra demo" in str(error.value)


def test_backend_with_missing_extra_is_skipped(fake_module):
    module = fake_module("mat_fake_missing", """
        from MAT.registry import require
        require("mat_definitely_missing", extra="demo")
        raise AssertionError("never reached")
    """)
    try:
        assert registry.load_optional(module, slot="transcriber", name="test-missing", extra="demo") is None
        skipped = registry.find("test-missing")
        assert isinstance(skipped, registry.SkippedBackend)
        assert skipped.extra == "demo"
        with pytest.raises(registry.BackendError, match="isn't installed"):
            registry.get("transcriber", "test-missing")
    finally:
        registry.unregister("transcriber", "test-missing")


def test_installed_backend_registers(fake_module):
    module = fake_module("mat_fake_ok", """
        from MAT.registry import register
        @register("transcriber", "test-ok", description="demo")
        class Demo:
            pass
    """)
    try:
        backend = registry.load_optional(module, slot="transcriber", name="test-ok", extra="demo")
        assert backend.cls.__name__ == "Demo"
        assert backend.cls.section == "test-ok"
        assert "test-ok" in [b.name for b in registry.backends("transcriber")]
    finally:
        registry.unregister("transcriber", "test-ok")


def test_module_that_forgets_to_register(fake_module):
    module = fake_module("mat_fake_forgot", "x = 1\n")
    with pytest.raises(RuntimeError, match="didn't register"):
        registry.load_optional(module, slot="transcriber", name="test-forgot", extra="demo")


def test_unknown_backend_lists_installed_ones():
    import MAT.tools  # noqa: F401
    with pytest.raises(registry.BackendError, match="Installed: .*whisper"):
        registry.get("transcriber", "nope")


def test_default_backends_are_installed_in_the_dev_environment():
    import MAT.tools  # noqa: F401
    names = {(b.slot, b.name) for b in registry.backends()}
    assert {("transcriber", "whisper"), ("diarizer", "sortformer"), ("identifier", "pyannote"),
            ("summarizer", "llm"), ("splitter", "spacy"), ("ner", "gliner")} <= names
