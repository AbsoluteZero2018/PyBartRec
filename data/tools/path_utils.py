# _*_ coding: utf-8 _*_
# @Time    :   2023/01/05 11:17:17
# @FileName:   path_utils.py
# @Author  :   423A35C7
# @Software:   VSCode
import logging
from pathlib import Path, WindowsPath, PosixPath
from config import config
import os
from typing import Generator, ClassVar, TypeVar

_cls = WindowsPath if os.name == 'nt' else PosixPath
loggers = {}


# raw-data、clean-data、*tmp.py、json-data、*.jsonl、flask.jsonl、proj_*.json、feature_*.csv

class _PrefixPath(_cls):
    """
    抽象的类，不应直接实例化此类。
    join_path、rglob和with_suffix会直接返回自定义类型，所以不需要重写。
    """
    prefix: ClassVar[str] = ""

    # def __init__(self, prefix:str=prefix, *args, **kwargs) -> None:
    #     super().__init__(prefix, *args, *kwargs)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and issubclass(type(args[0]), _PrefixPath):
            return args[0].change_to_path(cls)
        return super().__new__(cls, cls.prefix, *args, **kwargs)

    def __getnewargs__(self):  # 不知道为什么这个方法不会被Python内部调用，只能手动调用了
        return self.get_relative_path().parts

    def __reduce__(self):  # 防止序列化后出现重复前缀
        return type(self), self.__getnewargs__()

    def change_to_path(self, path_type: type, *args, **kwargs):  # 切换为另一个_PrefixPath类型
        # if issubclass(path_type, type(self)):
        #     return path_type(self.get_relative_path(), prefix=self.prefix, *args, **kwargs)
        return path_type(self.get_relative_path(), prefix=path_type.prefix, *args, **kwargs)

    def make_parent_dirs(self, mode: int = 0o777, parents: bool = True,
                         exist_ok: bool = True) -> None:  # pathlib里mode的默认值就是这样
        return self.parent.mkdir(mode, parents, exist_ok)

    def get_relative_path(self):
        return Path(self.relative_to(self.prefix))

    def get_logger(self) -> logging.Logger:
        return self.change_to_path(LogFilePath).logger

    # def joinpath(self, *other):
    #     return type(self)(self.prefix,
    #         super().joinpath(*other).relative_to(self.prefix))

    # def rglob(self, pattern: str) -> Generator:
    #     for path in super().rglob(pattern):
    #         yield type(self)(self.prefix, path.relative_to(self.prefix))

    # def with_suffix(self, suffix: str):
    #     return type(self)(self.prefix,
    #         super().with_suffix(suffix).relative_to(self.prefix))


_PrefixPathType = TypeVar('PrefixPathType', bound=_PrefixPath)


class _SuffixPath(_PrefixPath):
    """抽象的类，不应直接实例化此类"""
    suffix: ClassVar[str] = ""

    def __new__(cls: type, *args, **kwargs):
        obj = super(_PrefixPath, cls).__new__(_cls, *args, **kwargs)  # 这里不需要加前缀
        obj = _SuffixPath.with_suffix(obj, cls.suffix)
        return super().__new__(cls, obj)  # 这里会加前缀

    def with_suffix(self, suffix):
        """Return a new path with the file suffix changed.  If the path
        has no suffix, add given suffix.  If the given suffix is an empty
        string, remove the suffix from the path.
        """
        f = self._flavour
        if f.sep in suffix or f.altsep and f.altsep in suffix:
            raise ValueError("Invalid suffix %r" % (suffix,))
        if suffix == '.':
            raise ValueError("Invalid suffix %r" % (suffix))
        name = self.name
        if not name:
            raise ValueError("%r has an empty name" % (self,))
        old_suffix = self.suffix
        if not old_suffix:
            name = name + suffix
        else:
            name = name[:-len(old_suffix)] + suffix
        return self._from_parsed_parts(self._drv, self._root,
                                       self._parts[:-1] + [name])


class RawFilePath(_PrefixPath):
    prefix = config["datasets"]["raw-data"]
    # prefix = "datasets/raw-data" # 测试相对路径用


class CleanFilePath(_PrefixPath):
    prefix = config["datasets"]["clean-data"]


class JsonFilePath(_PrefixPath):
    prefix = config["datasets"]["json-data"]


class EncodedFilePath(_PrefixPath):
    prefix = config["datasets"]["encoded-data"]


class _LogFilePath(_PrefixPath):
    prefix = config["datasets"]["log"]


class ModuleApisJsonFilePath(_SuffixPath, JsonFilePath):
    suffix = "_module_apis.json"


class DefApisJsonFilePath(_SuffixPath, JsonFilePath):
    suffix = "_def_apis.json"


class JsonListEncodedFilePath(_SuffixPath, EncodedFilePath):
    suffix = ".jsonl"


class LogFilePath(_SuffixPath, _LogFilePath):
    suffix = ".log"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        return cls.__attach_logger(obj)

    # def __getattr__(self, item):  # pathlib中的逻辑会捕获AttributeError，所以不能用这个了
    #     print(self.logger)
    #     return getattr(self.logger, item)

    @staticmethod
    def __create_logger(obj, name):
        logger = logging.getLogger(name)
        logger.setLevel(level=logging.DEBUG)

        console = logging.StreamHandler()
        console.setLevel(level=logging.INFO)  # 控制台输出INFO级别以上的信息
        logger.addHandler(console)

        file = logging.FileHandler(str(obj), encoding="utf-8", mode="w")
        file.setLevel(level=logging.DEBUG)  # 文件输出DEBUG级别以上信息（全部信息）
        formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
        file.setFormatter(formatter)
        logger.addHandler(file)

        logger.debug('-' * 100)
        logger.debug('Start print log')
        return logger

    @staticmethod
    def __attach_logger(obj):
        obj.make_parent_dirs()
        name = str(obj.get_relative_path())
        if name in loggers:
            obj.logger = loggers[name]
        else:
            obj.logger = loggers[name] = LogFilePath.__create_logger(obj, name)
        return obj


class LogFilePath(_SuffixPath, _LogFilePath):
    suffix = ".log"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        return cls.__get_logger(obj)

    # def __getattr__(self, item):  # pathlib中的逻辑会捕获AttributeError，所以不能用这个了
    #     print(self.logger)
    #     return getattr(self.logger, item)

    @staticmethod
    def __get_logger(obj):
        logger = logging.getLogger(str(obj.get_relative_path()))
        logger.setLevel(level=logging.DEBUG)

        console = logging.StreamHandler()
        console.setLevel(level=logging.INFO)  # 控制台输出INFO级别以上的信息
        logger.addHandler(console)

        file = logging.FileHandler(str(obj), encoding="utf-8", mode="w")
        file.setLevel(level=logging.DEBUG)  # 文件输出DEBUG级别以上信息（全部信息）
        formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
        file.setFormatter(formatter)
        logger.addHandler(file)

        logger.debug('-' * 100)
        logger.debug('Start print log')

        obj.logger = logger
        return obj


if __name__ == "__main__":
    import pickle

    raw_file_path = RawFilePath("flask")
    print(raw_file_path)
    # print(type(raw_file_path))
    # print(raw_file_path.joinpath("123"))
    # print(type(raw_file_path.joinpath("123")))
    # # for i in raw_file_path.rglob("*"):
    # #     print(i, type(i))
    # print(raw_file_path.with_suffix(".456"))
    # print(type(raw_file_path.with_suffix(".456")))
    # module_path = ModuleApisJsonFilePath("101")
    # print(raw_file_path.change_to_path(CleanFilePath))
    # print(raw_file_path.change_to_path(ModuleApisJsonFilePath))
    # print(raw_file_path.change_to_path(JsonListEncodedFilePath))
    # print(module_path.change_to_path(RawFilePath))
    # print(module_path.change_to_path(JsonFilePath))
    # print(raw_file_path.glob("*"))
    # clean = CleanFilePath()
    # print(clean)
    # print(clean.get_relative_path())
    # print(raw_file_path.get_relative_path())
    # print(RawFilePath(raw_file_path))
    # print(EncodedFilePath(raw_file_path))
    # raw_pickle = pickle.dumps(raw_file_path)
    # print(raw_pickle)
    # raw_load = pickle.loads(raw_pickle)
    # print(raw_load)
    log_path = raw_file_path.change_to_path(LogFilePath)
    print(log_path)
    log_path.logger.debug("3333")
    log_path = raw_file_path.change_to_path(LogFilePath)
    print(log_path)
    log_path.logger.debug("3333")
