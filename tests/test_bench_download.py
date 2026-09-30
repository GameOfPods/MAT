import io
import tarfile
import zipfile

import pytest

from MAT.bench.download import RemoteFile, cache_root, download, extract_from_tar, extract_member


def test_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MAT_BENCH_CACHE", str(tmp_path / "env"))
    assert cache_root("/configured") == cache_root("/configured")
    assert str(cache_root("/configured")) == "/configured"
    assert cache_root() == tmp_path / "env"
    monkeypatch.delenv("MAT_BENCH_CACHE")
    assert cache_root().parts[-3:] == (".cache", "mat", "bench")


def test_remote_file_reads_single_zip_members():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("big/filler.bin", bytes(range(256)) * 4000)
        archive.writestr("wanted/file.txt", "hello")
    data = buffer.getvalue()
    fetched = []

    def fetch(start, end):
        fetched.append((start, end))
        return data[start:end + 1]

    remote = io.BufferedReader(RemoteFile("http://example", size=len(data), fetch=fetch, block=4096), 4096)
    archive = zipfile.ZipFile(remote)
    assert archive.read("wanted/file.txt") == b"hello"
    # the filler in front was never downloaded
    assert sum(end - start + 1 for start, end in set(fetched)) < len(data) / 2


def test_extract_member(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a/b.txt", "content")
    target = extract_member(zipfile.ZipFile(buffer), "a/b.txt", tmp_path / "out" / "b.txt")
    assert target.read_text() == "content"
    assert not (tmp_path / "out" / "b.txt.part").exists()


def test_extract_from_tar(tmp_path):
    archive = tmp_path / "audio.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("test/a.wav", "test/b.wav"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    extract_from_tar(archive, {"test/b.wav": tmp_path / "b.wav"})
    assert (tmp_path / "b.wav").read_bytes() == b"abc"
    with pytest.raises(FileNotFoundError, match="test/c.wav"):
        extract_from_tar(archive, {"test/c.wav": tmp_path / "c.wav"})


class _Response:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 416:
            raise OSError(self.status_code)

    def iter_content(self, size):
        yield self._body


class _Session:
    def __init__(self, data):
        self.data, self.ranges = data, []

    def get(self, url, headers=None, stream=False, timeout=None):
        header = (headers or {}).get("Range")
        self.ranges.append(header)
        if header:
            start = int(header.split("=")[1].rstrip("-"))
            return _Response(206, self.data[start:])
        return _Response(200, self.data)


def test_download_and_resume(tmp_path):
    session = _Session(b"0123456789")
    target = download("http://example/file", tmp_path / "file", session=session)
    assert target.read_bytes() == b"0123456789" and session.ranges == [None]

    (tmp_path / "other.part").write_bytes(b"0123")
    download("http://example/file", tmp_path / "other", session=session)
    assert (tmp_path / "other").read_bytes() == b"0123456789"
    assert session.ranges[-1] == "bytes=4-"

    # existing files aren't downloaded again
    download("http://example/file", tmp_path / "other", session=None)
