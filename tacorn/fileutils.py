#!/usr/bin/env python
import os
import shutil
import urllib.request


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def copy_files(src, dst):
    if os.path.exists(src) and os.path.exists(dst):
        for f in os.listdir(src):
            shutil.copy(os.path.join(src, f),
                        os.path.join(dst, f))


def copy_file(src, dst):
    shutil.copyfile(src, dst)


def download_file(url, target):
    with urllib.request.urlopen(url) as response, open(target, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
