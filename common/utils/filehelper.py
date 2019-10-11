from os import path, makedirs, remove
from shutil import rmtree
from typing import Union, Iterable


def create_and_clean_dir(dir_name) -> None:
    if path.exists(dir_name):
        rmtree(dir_name)
    makedirs(dir_name)


def remove_if_exists(path_name: Union[str, Iterable[str]]) -> None:
    if isinstance(path_name, str):
        if path.isfile(path_name):
            remove(path_name)
        elif path.isdir(path_name):
            rmtree(path_name)
    else:
        for entry in path_name:
            remove_if_exists(entry)


def create_dir_if_not_exists(path_name, is_file=False) -> None:
    if is_file:
        path_name = path.dirname(path_name)

    if not path.exists(path_name):
        makedirs(path_name)
