import zipfile
from pathlib import Path

import pytest
from pydub.generators import Sine

from MAT.bench.datasets import pack, Clip
from MAT.bench.datasets.ami import Ami, AmiOptions, parse_words, words_to_turns
from MAT.bench.datasets.bundestag import Bundestag, BundestagOptions, parse_kaldi, utterance_key, wav_name
from MAT.bench.datasets.fleurs import Fleurs, FleursOptions, parse_tsv, select_rows
from MAT.bench.datasets.voxconverse import VoxConverse, VoxConverseOptions, parse_rttm


def _wav(path: Path, seconds: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Sine(440).to_audio_segment(duration=int(seconds * 1000)).set_frame_rate(16000).export(str(path), format="wav")
    return path


def test_pack_joins_clips_with_gaps(tmp_path):
    clips = [Clip(f"c{i}", _wav(tmp_path / f"c{i}.wav"), f"text {i}") for i in range(3)]
    packed = pack(clips, tmp_path / "packed", "x", max_seconds=3.0, gap=1.0)
    assert len(packed) == 2
    first_audio, first_turns = packed[0]
    assert [(t.start, t.end, t.text) for t in first_turns] == [(0.0, 1.0, "text 0"), (2.0, 3.0, "text 1")]
    assert [(t.start, t.end) for t in packed[1][1]] == [(0.0, 1.0)]
    # a second call reuses the files
    mtime = first_audio.stat().st_mtime_ns
    assert pack(clips, tmp_path / "packed", "x", max_seconds=3.0, gap=1.0) == packed
    assert first_audio.stat().st_mtime_ns == mtime


FLEURS_TSV = (
    "1\tf1.wav\tErster Satz.\terster satz\te r s t e r\t16000\tMALE\n"
    "1\tf2.wav\tErster Satz.\terster satz\te r s t e r\t16000\tFEMALE\n"
    "2\tf3.wav\tZweiter Satz!\tzweiter satz\tz w e i\t16000\tMALE\n"
)


def test_fleurs_rows():
    rows = parse_tsv(FLEURS_TSV)
    assert [(r.id, r.file, r.text, r.samples) for r in rows][0] == ("1", "f1.wav", "Erster Satz.", 16000)
    assert [r.file for r in select_rows(rows, None)] == ["f1.wav", "f3.wav"]
    assert [r.file for r in select_rows(rows, 1)] == ["f1.wav"]


def test_fleurs_from_a_local_copy(tmp_path):
    folder = tmp_path / "fleurs" / "data" / "de_de"
    folder.mkdir(parents=True)
    (folder / "test.tsv").write_text(FLEURS_TSV)
    for name in ("f1.wav", "f3.wav"):
        _wav(folder / "audio" / "test" / name)
    options = FleursOptions(type="fleurs", languages=["de"], path=str(tmp_path / "fleurs"), pack_minutes=0.05)
    (item,) = Fleurs(options, cache=tmp_path / "cache").items()
    assert (item.id, item.language, item.has_words, item.has_speakers) == ("de_de-test-01", "de", True, False)
    assert [(t.start, t.text) for t in item.turns] == [(0.0, "Erster Satz."), (2.0, "Zweiter Satz!")]
    assert item.audio.parent == tmp_path / "cache" / "fleurs" / "packed"


def test_voxconverse_from_a_local_copy(tmp_path):
    rttm = "SPEAKER abc 1 0.50 2.00 <NA> <NA> spk00 <NA> <NA>\nSPEAKER abc 1 3.00 1.00 <NA> <NA> spk01 <NA> <NA>\n"
    assert parse_rttm(rttm) == {"spk00": [(0.5, 2.5)], "spk01": [(3.0, 4.0)]}
    (tmp_path / "vox" / "test").mkdir(parents=True)
    (tmp_path / "vox" / "test" / "abc.rttm").write_text(rttm)
    # macOS metadata files like the ones in the VoxConverse zips
    (tmp_path / "vox" / "test" / "._abc.rttm").write_bytes(b"\0")
    (tmp_path / "vox" / "audio").mkdir(parents=True)
    (tmp_path / "vox" / "audio" / "._abc.wav").write_bytes(b"\0")
    _wav(tmp_path / "vox" / "audio" / "abc.wav")
    (item,) = VoxConverse(VoxConverseOptions(type="voxconverse", path=str(tmp_path / "vox")), cache=tmp_path).items()
    assert item.id == "abc" and item.audio.name == "abc.wav"
    assert item.reference_segments() == {"spk00": [(0.5, 2.5)], "spk01": [(3.0, 4.0)]}
    assert not item.has_words and item.has_speakers


AMI_WORDS = b"""<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="M1.A.words" xmlns:nite="http://nite.sourceforge.net/">
   <w nite:id="a0" starttime="0.5" endtime="0.9">Okay</w>
   <w nite:id="a1" starttime="0.9" endtime="0.9" punc="true">.</w>
   <vocalsound nite:id="a2" starttime="1.0" endtime="1.2" type="laugh"/>
   <w nite:id="a3" starttime="1.0" endtime="1.4">let's</w>
   <w nite:id="a4" starttime="5.0" endtime="5.5">start</w>
   <w nite:id="a5">untimed</w>
</nite:root>"""


def test_ami_words_and_turns():
    words = parse_words(AMI_WORDS)
    assert words == [(0.5, 0.9, "Okay"), (1.0, 1.4, "let's"), (5.0, 5.5, "start")]
    turns = words_to_turns({"A": words, "B": [(2.0, 3.0, "hi")]}, gap=1.0)
    assert [(t.start, t.end, t.speakers, t.text) for t in turns] == [
        (0.5, 1.4, ("A",), "Okay let's"), (2.0, 3.0, ("B",), "hi"), (5.0, 5.5, ("A",), "start")]


def test_ami_from_a_local_copy(tmp_path):
    root = tmp_path / "ami"
    (root / "words").mkdir(parents=True)
    (root / "words" / "M1.A.words.xml").write_bytes(AMI_WORDS)
    (root / "words" / "M1.B.words.xml").write_bytes(AMI_WORDS.replace(b"M1.A", b"M1.B"))
    (root / "lists").mkdir()
    (root / "lists" / "test.meetings.txt").write_text("M1\nM2\n")
    (root / "only_words" / "rttms" / "test").mkdir(parents=True)
    (root / "only_words" / "rttms" / "test" / "M1.rttm").write_text(
        "SPEAKER M1 1 0.50 1.00 <NA> <NA> MEE001 <NA> <NA>\n")
    (root / "uems" / "test").mkdir(parents=True)
    (root / "uems" / "test" / "M1.uem").write_text("M1 1 0.000 6.000\n")
    _wav(root / "amicorpus" / "M1" / "audio" / "M1.Mix-Headset.wav")
    (item,) = Ami(AmiOptions(type="ami", path=str(root), limit=1), cache=tmp_path).items()
    assert (item.id, item.language, item.start, item.end) == ("M1", "en", 0.0, 6.0)
    assert {t.speakers for t in item.turns} == {("A",), ("B",)}
    assert item.reference_segments() == {"MEE001": [(0.5, 1.5)]}


def test_bundestag_helpers():
    assert parse_kaldi("u1 hallo welt\nu2\nu3\tmit  tab\n") == {"u1": "hallo welt", "u2": "", "u3": "mit  tab"}
    assert sorted(["7_0_522_3", "7_0_44", "7_1_5", "6_2_1"], key=utterance_key) == [
        "6_2_1", "7_0_44", "7_0_522_3", "7_1_5"]
    assert wav_name("u1", "/data/x/wavs/u1.wav") == "u1.wav"
    assert wav_name("u1", "sox in.flac -t wav - |") == "u1.wav"
    assert wav_name("u1", None) == "u1.wav"


def _bundestag_folder(root: Path) -> Path:
    folder = root / "asr_bundestag_clean"
    (folder / "test").mkdir(parents=True)
    (folder / "test" / "text").write_text("7_0_10\tzweiter satz\n7_0_2\terster satz\n")
    (folder / "test" / "wav.scp").write_text("7_0_10 /somewhere/wavs/7_0_10.wav\n7_0_2 /somewhere/wavs/7_0_2.wav\n")
    _wav(folder / "wavs" / "7_0_10.wav")
    _wav(folder / "wavs" / "7_0_2.wav")
    return folder


@pytest.mark.parametrize("zipped", [False, True], ids=["folder", "zip"])
def test_bundestag_from_a_local_copy(tmp_path, zipped):
    folder = _bundestag_folder(tmp_path / "bt")
    path = tmp_path / "bt"
    if zipped:
        path = tmp_path / "asr_bundestag_clean.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for file in folder.rglob("*"):
                archive.write(file, file.relative_to(tmp_path / "bt").as_posix())
    options = BundestagOptions(type="bundestag", path=str(path), pack_minutes=1)
    (item,) = Bundestag(options, cache=tmp_path / "cache").items()
    assert [t.text for t in item.turns] == ["erster satz", "zweiter satz"]
    assert item.language == "de" and not item.has_speakers
    if zipped:
        assert (tmp_path / "cache" / "bundestag" / "clean" / "wavs" / "7_0_2.wav").is_file()
