from typing import Optional, Type, Dict, Callable, Set, Iterator
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path


class ResultTypes(Enum):
    PODCAST = auto()
    BOOK = auto()


class ParsedResult(ABC):
    def __init__(self, parent: "MATResult"):
        self._parent = parent

    @property
    def parent(self) -> "MATResult":
        return self._parent


class MATResult(ABC):

    def __init__(self, path: Path):
        self._path = path
        self._files = set()
        self._get_content: Callable[[Set[str]], Dict[str, bytes]]
        self._path_seperator = None
        if self._path.is_file():
            from zipfile import ZipFile
            try:
                self._path_seperator = "/"
                with ZipFile(self._path, "r") as zip_file:
                    for f in (x for x in zip_file.infolist() if not x.is_dir()):
                        self._files.add(f.filename)

                def get_content_zip(files: Set[str]) -> Dict[str, bytes]:
                    ret = {}
                    if len(files) <= 0:
                        return ret
                    with ZipFile(self._path, "r") as _zip_file:
                        for _path in files:
                            ret[_path] = _zip_file.read(_path)
                    return ret

                self._get_content = get_content_zip
            except Exception as e:
                self.get_logger().exception(f"Error reading MAT-result file {path}: {e}", exc_info=e)
        elif self._path.is_dir():
            # os.sep is the directory separator. os.pathsep is the PATH separator (":" on linux) and was used here by mistake
            from os import sep, listdir
            from os.path import isfile
            try:
                self._path_seperator = sep

                def custom_walk(_path: str):
                    if isfile(str(self._path.absolute()) + sep + _path):
                        self._files.add(_path)
                        return
                    for _element in listdir(str(self._path.absolute()) + sep + _path):
                        custom_walk(_path + (sep if len(_path) > 0 else "") + _element)

                custom_walk("")

                def get_content_folder(files: Set[str]) -> Dict[str, bytes]:
                    ret = {}
                    if len(files) <= 0:
                        return ret
                    for _file in files:
                        with open(str(self._path.absolute()) + sep + _file, "rb") as _file_handle:
                            ret[_file] = _file_handle.read()
                    return ret
                self._get_content = get_content_folder
            except Exception as e:
                self.get_logger().exception(f"Error reading MAT-result directory {path}: {e}", exc_info=e)

    @property
    def pathsep(self):
        return self._path_seperator

    @property
    def filelist(self) -> Set[str]:
        return set(self._files)

    def get_content(self, files: Set[str]) -> Dict[str, Optional[bytes]]:
        avail_files = self.filelist
        avail_content = self._get_content({x for x in files if x in avail_files})
        return {k: avail_content.get(k, None) for k in files}

    @abstractmethod
    def get_results(self, t: ResultTypes) -> Iterator[ParsedResult]:
        raise NotImplementedError()

    @classmethod
    def get_logger(cls):
        import logging
        return logging.getLogger(cls.__name__)

    @classmethod
    @abstractmethod
    def supported_version(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def read(cls, path: Path) -> Optional["MATResult"]:
        import inspect

        readers = {}

        def collect_readers(base: type[MATResult] = MATResult):
            if inspect.isclass(base):
                if not inspect.isabstract(base):
                    if base.supported_version() in readers:
                        raise RuntimeError(f"Two or more readers for version {base.supported_version()} defined")
                    readers[str(base.supported_version())] = base
                for sub_base in base.__subclasses__():
                    collect_readers(sub_base)

        collect_readers()

        cls.get_logger().debug(
            f"Found {len(readers)} readers supporting versions {', '.join(str(x) for x in readers.keys())}"
        )

        version = None
        if path.is_file():
            from zipfile import ZipFile
            import json
            try:
                with ZipFile(path, "r") as zip_file:
                    if "meta.json" in [x.filename for x in zip_file.infolist()]:
                        content = zip_file.read("meta.json")
                        metadata = json.loads(content)
                        version = metadata["version"]
            except:
                pass

        elif path.is_dir():
            import json
            try:
                if "meta.json" in [x.name for x in path.iterdir()]:
                    with open(path.joinpath("meta.json"), "r") as meta_file:
                        metadata = json.loads(meta_file.read())
                        version = metadata["version"]
            finally:
                pass

        if version is None:
            cls.get_logger().error("Could not determine version from MAT-result")
            return None

        if version not in readers:
            cls.get_logger().error(f"Version {version} not supported currently to read")
            return None

        return readers[str(version)](path)


from MAT.reader.v1 import *
__all__ = ["MATResult", "ResultTypes"]
