from types import SimpleNamespace

from MAT import registry
from MAT.tools.diarizators.nemo import DiarizerNEMO, DiarizerStreamingSortformer
from MAT.utils.config import Config


def test_streaming_backend_has_its_own_section_and_defaults(tmp_path):
    assert registry.find("sortformer-streaming", "diarizer").cls is DiarizerStreamingSortformer
    config = Config({"sortformer": {"segment-length": 120}}, work_directory=str(tmp_path))
    streaming = config.options(DiarizerStreamingSortformer)
    assert streaming.model == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert streaming.segment_length == 4 * 3600
    assert config.options(DiarizerNEMO).segment_length == 120


def test_streaming_settings_are_applied_to_the_model():
    model = SimpleNamespace(sortformer_modules=SimpleNamespace())
    options = DiarizerStreamingSortformer.Options(chunk_len=6, chunk_right_context=7, fifo_len=188,
                                                  spkcache_update_period=144, spkcache_len=188)
    DiarizerStreamingSortformer()._configure_model(model, options)
    assert vars(model.sortformer_modules) == {"chunk_len": 6, "chunk_right_context": 7, "fifo_len": 188,
                                              "spkcache_update_period": 144, "spkcache_len": 188}


def test_normal_sortformer_leaves_the_model_alone():
    model = SimpleNamespace(sortformer_modules=SimpleNamespace())
    DiarizerNEMO()._configure_model(model, DiarizerNEMO.Options())
    assert vars(model.sortformer_modules) == {}
