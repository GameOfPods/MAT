import math

import numpy as np
import pytest

from MAT.utils.audio import frame_energy, plan_windows

RATE = 1000  # low sample rate keeps the test audio small


def _noise_with_gaps(seconds, gaps, seed=1):
    samples = np.random.default_rng(seed).normal(0, 1000, seconds * RATE)
    for gap in gaps:
        samples[int(gap * RATE):int((gap + 1) * RATE)] = 0
    return samples


def test_short_audio_is_one_window():
    windows = plan_windows(np.zeros(30 * RATE), RATE, max_length=60)
    assert len(windows) == 1
    assert (windows[0].start, windows[0].end) == (0.0, 30.0)
    assert windows[0].owns(0, 30)


def test_cuts_land_in_the_quiet_gaps():
    gaps = [100, 205, 318, 430, 540]
    windows = plan_windows(_noise_with_gaps(600, gaps), RATE, max_length=120, search=30)
    cuts = [w.keep_end for w in windows[:-1]]
    assert len(cuts) == len(gaps)
    for cut, gap in zip(cuts, gaps):
        assert gap <= cut <= gap + 1
    assert all(w.duration <= 120 for w in windows)
    assert windows[0].start == 0 and windows[-1].end == 600


def test_overlap_and_ownership():
    windows = plan_windows(_noise_with_gaps(600, [100, 205, 318, 430, 540]), RATE, max_length=120, overlap=10)
    for before, after in zip(windows, windows[1:]):
        assert before.keep_end == after.keep_start
        assert math.isclose(before.end - after.start, 10)
    assert all(w.duration <= 120 + 1e-9 for w in windows)
    # every point in time belongs to exactly one window, also right at the cuts
    for t in np.arange(0, 600, 0.37).tolist() + [w.keep_end for w in windows[:-1]]:
        assert sum(w.owns(t, t) for w in windows) == 1


def test_no_quiet_spot_still_makes_progress():
    windows = plan_windows(np.random.default_rng(2).normal(0, 1000, 500 * RATE), RATE, max_length=60)
    assert windows[-1].end == 500
    assert all(0 < w.duration <= 60 for w in windows)


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        plan_windows(np.zeros(RATE), RATE, max_length=10, overlap=5)


def test_frame_energy_handles_integer_audio_in_blocks():
    samples = np.full(25_000 * 10, 100, dtype=np.int16)
    energy = frame_energy(samples, sample_rate=100, frame_seconds=0.1)
    assert len(energy) == 25_000
    assert np.allclose(energy, 100)
