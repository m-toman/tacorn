""" Utils for handling files. """
import os
import shutil
import urllib.request


def ensure_dir(directory):
    """ Ensures a directory exists by creating it if not. """
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_files(src, dst):
    """ Copies all files in src into dst. """
    if os.path.exists(src) and os.path.exists(dst):
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file),
                        os.path.join(dst, file))


def copy_file(src, dst):
    """ Copies a file from src to dst. """
    shutil.copyfile(src, dst)


def move_file(src, dst):
    """ Moves a file or folder from src to dst. """
    shutil.move(src, dst)


def download_file(url, target):
    """ Downloads file from url into target file. """
    with urllib.request.urlopen(url) as response, open(target, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
