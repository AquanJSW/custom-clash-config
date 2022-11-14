import os
import io
import requests
from zipfile import ZipFile
import gzip
import pathlib
import sys
import argparse
import logging

logging.getLogger('requests').setLevel(logging.DEBUG)

clash_linux_url = (
    'https://release.dreamacro.workers.dev/latest/clash-linux-amd64-latest.gz'
)
clash_windows_url = (
    'https://release.dreamacro.workers.dev/latest/clash-windows-amd64-latest.zip'
)
mmdb_url = 'https://git.io/GeoLite2-City.mmdb'

clash_linux_name = 'clash-linux-amd64'


def get_script_path():
    cwd = pathlib.Path().absolute()
    script_path_relative_to_cwd = sys.argv[0]
    script_path = os.path.join(cwd, script_path_relative_to_cwd)
    return script_path


def get_parent(path):
    return str(pathlib.Path(path).absolute().parent)


script_dir = get_parent(get_script_path())
root_dir = get_parent(script_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--clash',
    help='directory for saving clash binary.',
    default=f'{root_dir}/assets',
)
parser.add_argument(
    '--mmdb',
    help='path for saving mmdb database',
    default=f'{root_dir}/assets/GeoLite2-City.mmdb',
)
parser.add_argument('--proxy', help='specify a proxy for downloading.', default=None)
args = parser.parse_args()


def preprocess_args():
    os.makedirs(get_parent(args.clash), exist_ok=True)
    os.makedirs(get_parent(args.mmdb), exist_ok=True)


preprocess_args()


def receive_data(url, proxy=None) -> bytes:
    if proxy:
        r = requests.get(url, proxies={'https': proxy})
    else:
        r = requests.get(url)
    return r.content


def decompress_zip(data: bytes, dir: str):
    zf = ZipFile(io.BytesIO(data))
    zf.extractall(dir)


def decompress_gzip(data: bytes, path: str):
    data = gzip.decompress(data)
    with open(path, 'wb') as fd:
        fd.write(data)


def main():
    clash_url = clash_windows_url if os.name == 'nt' else clash_linux_url
    decompress = decompress_zip if os.name == 'nt' else decompress_gzip

    data_mmdb = receive_data(mmdb_url, args.proxy)
    data_clash = receive_data(clash_url, args.proxy)

    with open(args.mmdb, 'wb') as fd:
        fd.write(data_mmdb)

    decompress(
        data_clash,
        args.clash if os.name == 'nt' else os.path.join(args.clash, clash_linux_name),
    )

    os.chmod(os.path.join(args.clash, clash_linux_name), 0o775)


if __name__ == '__main__':
    main()
