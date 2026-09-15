#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
"""
Downloads for datasets: the cache folder, resumable file downloads and reading single files out of big remote zips.

Files are written as NAME.part and renamed when complete, so an aborted download never looks finished. Don't
download the same dataset from two machines into a shared cache at the same time.
"""
import io
import logging
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional, Union

_LOGGER = logging.getLogger(__name__)

CACHE_ENV = "MAT_BENCH_CACHE"
CHUNK = 1 << 20


def cache_root(configured: Optional[Union[str, os.PathLike]] = None) -> Path:
    """The configured folder, else $MAT_BENCH_CACHE, else ~/.cache/mat/bench."""
    value = configured or os.environ.get(CACHE_ENV) or os.path.join("~", ".cache", "mat", "bench")
    return Path(os.path.expandvars(str(value))).expanduser()


def _session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "MAT-bench (https://github.com/GameOfPods/MAT)"
    return session


def download(url: str, target: Union[str, os.PathLike], session=None) -> Path:
    """Downloads url to target unless target already exists. Continues a .part file from an earlier try."""
    target = Path(target)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    session = session or _session()
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with session.get(url, headers=headers, stream=True, timeout=60) as response:
        if done and response.status_code == 416:
            # the part file already has everything
            part.replace(target)
            return target
        response.raise_for_status()
        if done and response.status_code != 206:
            done = 0
        length = response.headers.get("Content-Length")
        total = int(length) + done if length else None
        _LOGGER.info(f"Downloading {url}" + (f" ({total / 1e6:.0f} MB)" if total else ""))
        last_log = time.monotonic()
        with open(part, "ab" if done else "wb") as f:
            for chunk in response.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - last_log > 30:
                    last_log = time.monotonic()
                    _LOGGER.info(f"  {done / 1e6:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else ""))
    part.replace(target)
    return target


class RemoteFile(io.RawIOBase):
    """Read only, seekable file behind HTTP range requests. zipfile only reads the directory at the end of an archive
    and the members it's asked for, so single files come out of a 60 GB zip without downloading all of it."""

    def __init__(self, url: str, size: Optional[int] = None, fetch: Optional[Callable[[int, int], bytes]] = None,
                 block: int = 4 * CHUNK):
        super().__init__()
        if fetch is None:
            session = _session()

            def fetch(start: int, end: int) -> bytes:
                response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                response.raise_for_status()
                if response.status_code != 206:
                    raise OSError(f"{url} doesn't support range requests")
                return response.content

            if size is None:
                head = session.head(url, allow_redirects=True, timeout=60)
                head.raise_for_status()
                size = int(head.headers["Content-Length"])
        if size is None:
            raise ValueError("size is needed when fetch is given")
        self.url = url
        self.size = size
        self._fetch = fetch
        self._block = block
        self._blocks: Dict[int, bytes] = {}
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def _get(self, index: int) -> bytes:
        data = self._blocks.get(index)
        if data is None:
            start = index * self._block
            data = self._fetch(start, min(start + self._block, self.size) - 1)
            self._blocks[index] = data
            if len(self._blocks) > 16:
                del self._blocks[next(iter(self._blocks))]
        return data

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        wanted = max(0, min(len(view), self.size - self._pos))
        done = 0
        while done < wanted:
            index, offset = divmod(self._pos, self._block)
            chunk = self._get(index)[offset:offset + wanted - done]
            if not chunk:
                break
            view[done:done + len(chunk)] = chunk
            done += len(chunk)
            self._pos += len(chunk)
        return done


def open_zip(location: Union[str, os.PathLike]) -> zipfile.ZipFile:
    """A local zip file or one behind an http(s) URL."""
    location = str(location)
    if location.startswith(("http://", "https://")):
        _LOGGER.info(f"Reading the file list of {location}")
        return zipfile.ZipFile(io.BufferedReader(RemoteFile(location), CHUNK))
    return zipfile.ZipFile(location)


def extract_member(archive: zipfile.ZipFile, member: str, target: Union[str, os.PathLike]) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with archive.open(member) as source, open(part, "wb") as f:
        shutil.copyfileobj(source, f, CHUNK)
    part.replace(target)
    return target


def extract_from_tar(archive: Union[str, os.PathLike], wanted: Dict[str, Path]) -> None:
    """Extracts the members in wanted (member name -> target file) from a tar archive in one pass."""
    missing = {name: Path(target) for name, target in wanted.items() if not Path(target).exists()}
    if not missing:
        return
    _LOGGER.info(f"Extracting {len(missing)} files from {archive}")
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            target = missing.pop(member.name, None)
            if target is None or not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with tar.extractfile(member) as source, open(part, "wb") as f:
                shutil.copyfileobj(source, f, CHUNK)
            part.replace(target)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"{archive} doesn't contain {', '.join(sorted(missing)[:3])}")


__all__ = ["CACHE_ENV", "cache_root", "download", "RemoteFile", "open_zip", "extract_member", "extract_from_tar"]
