import math

from MAT.tools.diarizators.nemo import DiarizerNEMO
from MAT.utils.audio import Window


def test_merge_windows_shifts_times_and_drops_overlap_duplicates():
    windows = [Window(0.0, 65.0, -math.inf, 60.0), Window(55.0, 120.0, 60.0, math.inf)]
    per_window = [
        {"a": [(10.0, 20.0), (58.0, 61.0)]},
        # (3, 6) is the same segment as (58, 61) above, seen again in the overlap of the second piece
        {"a": [(3.0, 6.0)], "b": [(20.0, 30.0)]},
    ]
    result = DiarizerNEMO._merge_windows(windows, per_window)
    assert sorted(result.get_diarization("a")) == [(10.0, 20.0), (58.0, 61.0)]
    assert result.get_diarization("b") == [(75.0, 85.0)]


def test_single_window_keeps_everything():
    windows = [Window(0.0, 30.0, -math.inf, math.inf)]
    result = DiarizerNEMO._merge_windows(windows, [{"a": [(0.0, 1.0), (29.0, 30.0)]}])
    assert result.get_diarization("a") == [(0.0, 1.0), (29.0, 30.0)]
