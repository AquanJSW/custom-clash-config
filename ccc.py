#! /usr/bin/env python3
import argparse
import io
import ipaddress
import json
import logging
import multiprocessing
import os
import signal
import socket
import subprocess
import tempfile
import time
from typing import Iterable, Union

import geoip2.database
import geoip2.errors
import geoip2.models
import requests
import yaml
from requests.exceptions import ProxyError, SSLError, Timeout

default_format_v4 = '{region} - {number:02d}'
default_format_v6 = '[IPv6] {region} - {number:02d}'

parser = argparse.ArgumentParser()
parser.add_argument('bin', help='clash bin path')
parser.add_argument('config', help='clash config file/url')
parser.add_argument('templates', help='template clash config files path/url', nargs='+')
parser.add_argument('--workers', help='for multiprocessing', default=0, type=int)
parser.add_argument(
    '--proxy', help='existing proxy used to download files', default=None
)
parser.add_argument('-o', '--out', help='output config file pathes', nargs='+')
parser.add_argument(
    '--out-ss',
    help='output ss config pathes, same sequence as argument `template`, place \
        the templates to the end of its sequence so that this argument\' length \
            may differ from `template` if you want to disable outputing ss configs.',
    nargs='*',
    default=[],
)
parser.add_argument(
    '--mmdb',
    help='mmdb database path, not perform geolookup by default.',
    default=None,
)
parser.add_argument(
    '--enable-rename',
    help='Rename proxy names, see also argument `out-ss`',
    nargs='*',
    type=int,
    default=[],
)
parser.add_argument(
    '--format-v4',
    help=f'format string to rename proxy (IPv4)',
    default=default_format_v4,
)
parser.add_argument(
    '--format-v6',
    help=f'format string to rename proxy (IPv6)',
    default=default_format_v6,
)
parser.add_argument(
    '--log-level',
    help='log level',
    choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
    default='WARNING',
)
parser.add_argument('--log-file', help='log file', default=None)
args = parser.parse_args()


def prepare_args():
    length = len(args.templates)
    assert len(args.out) == length
    for arg, default in zip(
        [args.out_ss, args.enable_rename],
        [False, False],
    ):
        assert len(arg) <= length
        arg.extend([default] * (length - len(arg)))


prepare_args()

logging.basicConfig(
    filename=args.log_file,
    filemode='w',
    level=getattr(logging, args.log_level),
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%m-%d %H:%M:%S',
    encoding='utf-8',
)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


def make_separate_line(msg, marker='-', min_length=80, min_pad=3):
    assert len(marker) == 1

    total_marker = min_length - min_pad * 2 - len(msg)
    if total_marker < 0:
        min_length -= total_marker
    len_marker = [
        (min_length - len(msg)) // 2,
    ]
    len_marker.append(min_length - len_marker[0] - len(msg))
    return marker * len_marker[0] + msg + marker * len_marker[1]


def load_config_from_file(path):
    with open(path, 'r', encoding='utf-8') as fd:
        conf = yaml.safe_load(fd)
        return conf


def load_config_from_url(url, proxy=None):
    headers = {'charset': 'utf-8'}
    if proxy is not None:
        proxy = {'https': proxy}
    r = requests.get(url=url, headers=headers, proxies=proxy)
    conf = yaml.safe_load(r.text)
    return conf


def load_config(path_or_url, proxy=None):
    logging.debug('loading config from {}'.format(path_or_url))
    if os.path.isfile(path_or_url):
        return load_config_from_file(path_or_url)
    return load_config_from_url(path_or_url, proxy)


def dump_config(conf, path):
    with open(path, 'w', encoding='utf-8') as fd:
        yaml.safe_dump(conf, fd, allow_unicode=True)
    logging.info('config is saved to {}'.format(path))


def dump_config_json(conf, path):

    with open(path, 'w', encoding='utf-8') as fd:
        json.dump(conf, fd, indent=2, ensure_ascii=False)

    logging.info(f'ss config is saved to {path}')


class GeoLookup:
    def __init__(self, mmdb):
        logging.debug('loading geometry database from {}'.format(mmdb))
        self.reader = geoip2.database.Reader(mmdb)

    def __call__(self, ip):
        """
        # Exception
        - `AddressNotFoundError`
        # Return
        `City` object
        """
        try:
            r = self.reader.city(ip)
            return r
        except geoip2.errors.AddressNotFoundError:
            logging.warning('{} isn\'t in geometry database'.format(ip))


acceptable_exceptions = (SSLError, ProxyError, Timeout)


def get_external_ip(local_proxy=None):
    """
    # Exception
    - `ProxyError`
    """
    if local_proxy is None:
        r = requests.get('https://api64.ipify.org', timeout=5)
        return r.text
    r = requests.get(
        'https://api64.ipify.org', proxies={'https': local_proxy}, timeout=5
    )
    return r.text


class Clash:
    config_template = """
external-controller: 'localhost:{control_port}'
port: {proxy_port}
ipv6: true
mode: global
"""

    def __init__(self, clash_bin_path, control_port, proxy_port, proxies):
        self.proxies = proxies
        self.log_fd = tempfile.SpooledTemporaryFile()
        self.conf_path = Clash.__make_temp_subconf(control_port, proxy_port, proxies)
        logging.debug(f'spawning clash instance, log file: {self.conf_path}')
        self.clash_proc = subprocess.Popen(
            '{} -f {}'.format(clash_bin_path, self.conf_path).split(' '),
            stdout=self.log_fd,
            stderr=self.log_fd,
        )
        time.sleep(3)

        self.local_proxy_addr = 'http://localhost:{}'.format(proxy_port)
        self.control_addr = 'http://localhost:{}'.format(control_port)

    @staticmethod
    def __make_temp_subconf(control_port, proxy_port, proxies):
        """
        # Notice
        Temporary file need to be deleted by hand after using.

        # Return
        Name of the temporary file.
        """
        subconf = yaml.safe_load(
            io.StringIO(
                Clash.config_template.format(
                    control_port=control_port, proxy_port=proxy_port
                )
            )
        )
        subconf['proxies'] = proxies
        fd = tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False)
        if not fd.writable:
            print('Can not write temporary subconfig')
            raise PermissionError
        yaml.safe_dump(subconf, fd, allow_unicode=True)
        fd.close()
        return fd.name

    def publish_clash_log(self):
        # self.log_fd.close()
        self.log_fd.seek(0)
        log_str = ''.join([str(b, encoding='utf-8') for b in self.log_fd.readlines()])
        logging.critical(
            '\n'
            + make_separate_line('START Clash log')
            + '\n'
            + log_str
            + '\n'
            + make_separate_line('END Clash log')
        )

    def __del__(self):
        self.log_fd.close()
        self.clash_proc.kill()  # terminate() in linux ?


clash_bin_path = None


class ProxyExternalIPLookup:
    id_failed_to_get_ext_ip = None

    def __init__(self, clash_bin_path, control_port, proxy_port, proxies):
        self.proxies = proxies
        self.clash = Clash(clash_bin_path, control_port, proxy_port, proxies)

    def __switch_proxy(self, proxy):
        payload = {'name': proxy['name']}
        try:
            r = requests.put(
                url=self.clash.control_addr + '/proxies/GLOBAL',
                data=json.dumps(payload),
                timeout=5,
            )
            return r.ok
        except:
            return False

    def __call__(self):
        ext_ips = []
        has_valid_ip = False
        for proxy in self.proxies:
            if not self.__switch_proxy(proxy):
                logging.error('failed to switch proxy')
                self.clash.publish_clash_log()
                return None, has_valid_ip
            try:
                ext_ip = get_external_ip(self.clash.local_proxy_addr)
                ext_ips.append(ext_ip)
                has_valid_ip = True
            except acceptable_exceptions as e:
                name = proxy['name']
                logging.warning(f'local clash proxy[{name}] error: {e}')
                ext_ips.append(ProxyExternalIPLookup.id_failed_to_get_ext_ip)

            logging.info('{} {}'.format(proxy['name'], ext_ips[-1]))
        return ext_ips, has_valid_ip

    def __del__(self):
        del self.clash


class ProxyClass:
    def __init__(
        self,
        proxy_dict: dict = None,
        ext_ip: str = None,
        geo: geoip2.models.City = None,
    ):
        self.proxy_dict = proxy_dict
        self.ext_ip = ext_ip
        self.geo = geo

    def rename(self, name):
        self.proxy_dict['name'] = name

    @property
    def region(self):
        return self.geo.country.iso_code

    @property
    def name(self):
        return self.proxy_dict['name']

    def is_ipv6(self):
        return isinstance(ipaddress.ip_address(self.ext_ip), ipaddress.IPv6Address)

    def __hash__(self) -> int:
        return hash(self.ext_ip)

    def __eq__(self, po):
        return hash(self) == hash(po)

    def __lt__(self, po):
        return hash(self) < hash(po)

    def __repr__(self):
        return '{} {}'.format(self.proxy_dict['name'], self.ext_ip)


def clash_pool_worker(ports, proxies):
    return ProxyExternalIPLookup(clash_bin_path, *ports, proxies)()


def clash_pool_init_worker(clash_bin_path_):
    global clash_bin_path
    clash_bin_path = clash_bin_path_
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class NoValidIPError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ProxyExternalIPLookupAndFilter:
    @staticmethod
    def is_port_in_use(port, addr='localhost', timeout=0.01):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((addr, int(port)))
            return True
        except socket.error:
            return False
        finally:
            if s != None:
                s.close()

    @staticmethod
    def acquire_ports(count, start=7900):
        assert count > 0 and start > 0
        ports = []
        port = start
        while count != 0 and port < 65536:
            if not ProxyExternalIPLookupAndFilter.is_port_in_use(port):
                ports.append(port)
                count -= 1
            port += 1
        if count == 0:
            return ports
        logging.critical('no enough ports')
        exit(0)

    @staticmethod
    def divide_iterable(iterable, max_pieces):
        assert len(iterable) > 0 and max_pieces > 0
        ret = []
        [ret.append([]) for _ in range(max_pieces)]
        for i, it in enumerate(iterable):
            ret[i % max_pieces].append(it)
        return ret[: len(iterable)]

    @staticmethod
    def __has_valid_ip(proc_results):
        has = False
        for result in proc_results:
            has |= result[1]
        return has

    def __call__(self, proxy_dicts, clash_bin_path, workers=0) -> Iterable[ProxyClass]:
        """Update external IP and filter out external-IP-duplicated proxies.

        # Return
        proxy name ordered `ProxyObject` list
        """
        assert workers >= 0
        if workers == 0:
            workers = multiprocessing.cpu_count() + 1
        divided_proxies = self.divide_iterable(proxy_dicts, workers)
        ports = self.acquire_ports(2 * len(divided_proxies))
        divided_ports = self.divide_iterable(ports, len(divided_proxies))

        logging.debug('multiprocessing start')

        try:
            pool = multiprocessing.Pool(
                len(divided_ports), clash_pool_init_worker, (clash_bin_path,)
            )
            proc_results = pool.starmap(
                clash_pool_worker, zip(divided_ports, divided_proxies)
            )
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            exit(0)

        logging.debug('multiprocessing end')

        if not self.__has_valid_ip(proc_results):
            err_msg = 'no valid IPs, hns restarting is recommended.'
            logging.error(err_msg)
            raise NoValidIPError(err_msg)

        IPs = []
        for r in proc_results:
            IPs += r[0]

        proxies = []
        for p in divided_proxies:
            proxies += p

        # filter out external-ip duplicated
        proxy_objs = set([ProxyClass(proxy, ip) for proxy, ip in zip(proxies, IPs)])
        # filter out invalid proxy (external-ip == None)
        proxy_objs = list(filter(lambda p: p.ext_ip != None, proxy_objs))
        # sort
        proxy_objs = sorted(proxy_objs, key=lambda p: p.proxy_dict['name'])

        logging.info(
            'stat: {}/{}, {:.1f}% unique and valid proxies'.format(
                len(proxy_objs),
                len(proxy_dicts),
                100 * len(proxy_objs) / len(proxy_dicts),
            )
        )
        logging.debug(
            '\n'
            + make_separate_line('START unique and valid proxies')
            + '\n'
            + '\n'.join([repr(p) for p in proxy_objs])
            + '\n'
            + make_separate_line('END unique and valid proxies')
        )

        return proxy_objs


class ProxiesClass:
    def __init__(self, init_arg: Union[Iterable[dict], Iterable[ProxyClass]]):
        self.proxy_objs = self.parse_proxies(init_arg)
        self.map_region_proxyobjects = None
        self.map_region_renamedproxyobjects = None

    def parse_proxies(self, proxies):
        if isinstance(proxies[0], dict):
            return [ProxyClass(pd) for pd in proxies]
        elif isinstance(proxies[0], ProxyClass):
            return proxies
        else:
            logging.critical(
                'Unknown init arg type ({}) for ProxyObjects.'.format(
                    str(type(proxies))
                )
            )
            exit(1)

    @property
    def proxy_dicts(self):
        return [po.proxy_dict for po in self.proxy_objs]

    @property
    def proxy_names(self):
        return sorted([po.name for po in self.proxy_objs])

    def external_ip_lookup_and_filter(self, clash_bin_path, workers):
        start = time.time()
        self.proxy_objs = ProxyExternalIPLookupAndFilter()(
            self.proxy_dicts, clash_bin_path, workers
        )
        end = time.time()
        logging.info(
            'time cost of updating external-ip and fitering: {}'.format(
                time.strftime('%M min %S s', time.gmtime(end - start))
            )
        )

    def update_geometry(self, mmdb_path):
        gl = GeoLookup(mmdb_path)

        for p in self.proxy_objs:
            p.geo = gl(p.ext_ip)

        logging.debug(
            '\n'
            + make_separate_line('START geo check')
            + '\n'
            + '\n'.join(
                [
                    '[{}] {}'.format(p.geo.country.names['zh-CN'], p.proxy_dict['name'])
                    for p in self.proxy_objs
                ]
            )
            + '\n'
            + make_separate_line('END geo check')
        )

        map_region_proxyobjects = {}
        for po in self.proxy_objs:
            if po.region not in map_region_proxyobjects.keys():
                map_region_proxyobjects[po.region] = [po]
            else:
                map_region_proxyobjects[po.region].append(po)
        self.map_region_proxyobjects = map_region_proxyobjects

    def rename(
        self,
        format_v6=default_format_v6,
        format_v4=default_format_v4,
    ):
        self.map_region_renamedproxyobjects = {}
        for region, POs in self.map_region_proxyobjects.items():
            POs = sorted(POs)
            self.map_region_renamedproxyobjects[region] = []
            for i, po in enumerate(POs):
                format_ = format_v6 if po.is_ipv6() else format_v4
                po.rename(format_.format(region=region, number=i + 1))
                self.map_region_renamedproxyobjects[region].append(po)
        return self.map_region_renamedproxyobjects


class Config:
    def __init__(self, config) -> None:
        self.config = config

    @property
    def config_ss_android(self):
        r = []
        for proxy in self.config['proxies']:
            if proxy['type'] != "ss":
                continue
            r.append(
                {
                    'remarks': proxy['name'],
                    'server': proxy['server'],
                    'server_port': proxy['port'],
                    'method': proxy['cipher'],
                    'password': proxy['password'],
                }
            )
        return r

    def has_region_key(self):
        for pg in self.config['proxy-groups']:
            if 'region' in pg.keys():
                return True
        return False

    @staticmethod
    def __add_proxy_names(proxy_group, proxy_names):
        if 'proxies' in proxy_group.keys():
            proxy_group['proxies'] = proxy_names + proxy_group['proxies']
        else:
            proxy_group['proxies'] = proxy_names

    def update(self, proxiesobject: ProxiesClass, enable_rename):
        """Update proxies and proxy-groups."""
        if not self.has_region_key():
            self.config['proxies'] = proxiesobject.proxy_dicts
            for pg in self.config['proxy-groups']:
                self.__add_proxy_names(pg, proxiesobject.proxy_names)
            return
        elif not enable_rename:
            map_region_PO_s = proxiesobject.map_region_proxyobjects
        else:
            map_region_PO_s = proxiesobject.map_region_renamedproxyobjects

        # update proxies
        self.config['proxies'] = []
        for _, po_s in map_region_PO_s.items():
            self.config['proxies'] += ProxiesClass(po_s).proxy_dicts

        # update update-groups
        for pg in self.config['proxy-groups']:
            if 'region' in pg.keys():
                has_region_added = False
                for region in pg['region']:
                    try:
                        self.__add_proxy_names(
                            pg, ProxiesClass(map_region_PO_s[region]).proxy_names
                        )
                        has_region_added = True
                    except KeyError:
                        pass
                if not has_region_added:
                    logging.warning(
                        'No valid regions for proxy-group {}, adding all the proxies to this group.'.format(
                            pg['name']
                        )
                    )
                    self.__add_proxy_names(pg, proxiesobject.proxy_names)
                pg.pop('region')
            else:
                self.__add_proxy_names(pg, proxiesobject.proxy_names)

    def validate(self, clash_bin_path):
        log_fd = tempfile.SpooledTemporaryFile()

        conf_fd = tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False)
        yaml.safe_dump(self.config, conf_fd, allow_unicode=True)
        conf_path = conf_fd.name
        conf_fd.close()

        proc = subprocess.Popen(
            '{} -f {} -t'.format(clash_bin_path, conf_path).split(' '),
            stdout=log_fd,
            stderr=log_fd,
        )

        for _ in range(10):
            if proc.poll is not None:
                break
            time.sleep(1)
        else:
            logging.warning('Failed to validate output config.')
            proc.kill()
            log_fd.close()
            os.remove(conf_path)
            return True

        proc.kill()
        log_fd.seek(0)
        log_str = ''.join([str(l, encoding='utf-8') for l in log_fd.readlines()])

        ret = False
        if log_str.find('successful') > -1 or log_str.find('failed') == -1:
            logging.info('pass config validation.')
            os.remove(conf_path)
            ret = True
        else:
            logging.error('not pass config validation.')
            logging.error(
                '\n'
                + make_separate_line('START clash test log')
                + '\n'
                + log_str
                + '\n'
                + make_separate_line('END clash test log')
            )
        try:
            os.remove(conf_path)
        except:
            pass
        return ret


def is_configs_has_region_key(configs: Iterable[Config]):
    for config in configs:
        if config.has_region_key():
            return True
    else:
        return False


def check_arg_mmdb(template_configs):
    if not args.mmdb and is_configs_has_region_key(template_configs):
        logging.critical('Please provide mmdb database.')
        exit(1)


def get_proxiesobject():
    upstream_config = load_config(args.config, args.proxy)
    proxiesobject = ProxiesClass(upstream_config['proxies'])
    proxiesobject.external_ip_lookup_and_filter(args.bin, args.workers)
    if args.mmdb:
        proxiesobject.update_geometry(args.mmdb)
        proxiesobject.rename(format_v4=args.format_v4, format_v6=args.format_v6)
    return proxiesobject


def main():
    template_configs = [Config(load_config(t, args.proxy)) for t in args.templates]
    # check arg mmdb
    check_arg_mmdb(template_configs)

    proxiesobject = get_proxiesobject()

    [
        t.update(proxiesobject, enable_rename)
        for t, enable_rename in zip(template_configs, args.enable_rename)
    ]

    for template_config, out, out_ss in zip(template_configs, args.out, args.out_ss):
        if template_config.validate(args.bin):
            dump_config(template_config.config, out)
            if out_ss:
                dump_config_json(template_config.config_ss_android, out_ss)


if __name__ == '__main__':
    main()
