#! /usr/bin/env python3
import argparse
import logging
import os
import time
from typing import Iterable, Union

import requests
import yaml

__parser = argparse.ArgumentParser()
__parser.add_argument('bin', help='clash bin path')
__parser.add_argument('config', help='clash config file/url')
__parser.add_argument('template', help='template clash config file path/url')
__parser.add_argument('--workers', help='for multiprocessing', default=0, type=int)
__parser.add_argument('--proxy', help='used to get config from url.', default=None)
__parser.add_argument(
    '--out', help='output config file path', default='./config/output.yml'
)
__parser.add_argument(
    '--mmdb',
    help='mmdb database path.',
    default=None,
)
__parser.add_argument(
    '--disable-rename', help='Rename proxy names', action='store_true', default=False
)
__parser.add_argument(
    '--format-v4',
    help='format string to rename proxy (IPv4)',
    default='{region} - {number:02d}',
)
__parser.add_argument(
    '--format-v6',
    help='format string to rename proxy (IPv6)',
    default='[IPV6] {region} - {number:02d}',
)
__parser.add_argument(
    '--log-level',
    help='log level',
    choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
    default='WARNING',
)
__parser.add_argument('--log-file', help='log file', default=None)
args = __parser.parse_args()

logging.basicConfig(
    filename=args.log_file,
    level=getattr(logging, args.log_level),
    format='[%(asctime)s] [%(process)d.%(thread)d] [%(levelname)s] %(message)s',
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


import geoip2.database
import geoip2.errors
import geoip2.models


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


import requests
from requests.exceptions import ProxyError, SSLError

acceptable_exceptions = (SSLError, ProxyError)


def get_external_ip(local_proxy=None):
    """
    # Exception
    - `ProxyError`
    """
    if local_proxy is None:
        r = requests.get('https://api64.ipify.org')
        return r.text
    r = requests.get('https://api64.ipify.org', proxies={'https': local_proxy})
    return r.text


import io
import json
import multiprocessing
import os
import socket
import subprocess
import tempfile
import time

import requests
import yaml


class Clash:
    config_template = """
external-controller: 'localhost:{control_port}'
mixed-port: {proxy_port}
ipv6: true
mode: global
"""

    def __init__(self, clash_bin_path, control_port, proxy_port, proxies):
        self.proxies = proxies
        self.log_fd = tempfile.SpooledTemporaryFile()
        self.conf_path = Clash.__make_temp_subconf(control_port, proxy_port, proxies)
        logging.debug('spawning clash instance')
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
        self.log_fd.close()
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


class ExternalIPLookup:
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
                exit(0)
            try:
                ext_ip = get_external_ip(self.clash.local_proxy_addr)
                ext_ips.append(ext_ip)
                has_valid_ip = True
            except acceptable_exceptions:
                ext_ips.append(ExternalIPLookup.id_failed_to_get_ext_ip)

            logging.debug('{} {}'.format(proxy['name'], ext_ips[-1]))
        return ext_ips, has_valid_ip

    def __del__(self):
        del self.clash


import ipaddress


class ProxyObject:
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
        return self.geo.country.names['zh-CN']

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


def wrapper_proc(ports, proxies):
    return ExternalIPLookup(clash_bin_path, *ports, proxies)()


def initializer(clash_bin_path_):
    global clash_bin_path
    clash_bin_path = clash_bin_path_


class UpdateExternalIP:
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
            if not UpdateExternalIP.is_port_in_use(port):
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

    def __call__(self, proxy_dicts, clash_bin_path, workers=0) -> Iterable[ProxyObject]:
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

        pool = multiprocessing.Pool(len(divided_ports), initializer, (clash_bin_path,))
        proc_results = pool.starmap(wrapper_proc, zip(divided_ports, divided_proxies))
        pool.close()
        pool.join()

        if not self.__has_valid_ip(proc_results):
            raise Exception('No valid IP. The IP Lookup API may be broken.')

        IPs = []
        for r in proc_results:
            IPs += r[0]

        proxies = []
        for p in divided_proxies:
            proxies += p

        # filter out external-ip duplicated
        proxy_objs = set([ProxyObject(proxy, ip) for proxy, ip in zip(proxies, IPs)])
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


class ProxyObjects:
    def __init__(self, init_arg: Union[Iterable[dict], Iterable[ProxyObject]]):
        self.proxy_objs = self.__cvt_init_arg(init_arg)
        self.map_region_PO_s = None
        self.map_region_renamedPO_s = None

    def __cvt_init_arg(self, init_arg):
        if isinstance(init_arg[0], dict):
            return [ProxyObject(pd) for pd in init_arg]
        elif isinstance(init_arg[0], ProxyObject):
            return init_arg
        else:
            logging.critical(
                'Unknown init arg type ({}) for ProxyObjects.'.format(
                    str(type(init_arg))
                )
            )
            exit(1)

    @property
    def proxy_dicts(self):
        return [po.proxy_dict for po in self.proxy_objs]

    @property
    def proxy_names(self):
        return [po.name for po in self.proxy_objs]

    def update_external_ip(self, clash_bin_path, workers):
        start = time.time()
        self.proxy_objs = UpdateExternalIP()(self.proxy_dicts, clash_bin_path, workers)
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

        map_region_PO_s = {}
        for po in self.proxy_objs:
            if po.region not in map_region_PO_s.keys():
                map_region_PO_s[po.region] = [po]
            else:
                map_region_PO_s[po.region].append(po)
        self.map_region_PO_s = map_region_PO_s

    def rename(
        self,
        format_v6='[IPV6] {region} - {number:02d}',
        format_v4='{region} - {number:02d}',
    ):
        self.map_region_renamedPO_s = {}
        for region, POs in self.map_region_PO_s.items():
            POs = sorted(POs)
            self.map_region_renamedPO_s[region] = []
            for i, po in enumerate(POs):
                format_ = format_v6 if po.is_ipv6() else format_v4
                po.rename(format_.format(region=region, number=i + 1))
                self.map_region_renamedPO_s[region].append(po)


class Config:
    def __init__(self, config) -> None:
        self.config = config

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

    def update(self, proxy_objects: ProxyObjects, enable_rename):
        """Update proxies and proxy-groups."""
        if not self.has_region_key():
            self.config['proxies'] = proxy_objects.proxy_dicts
            for pg in self.config['proxy-groups']:
                self.__add_proxy_names(pg, proxy_objects.proxy_names)
            return
        elif not enable_rename:
            map_region_PO_s = proxy_objects.map_region_PO_s
        else:
            map_region_PO_s = proxy_objects.map_region_renamedPO_s

        # update proxies
        self.config['proxies'] = []
        for _, po_s in map_region_PO_s.items():
            self.config['proxies'] += ProxyObjects(po_s).proxy_dicts

        # update update-groups
        for pg in self.config['proxy-groups']:
            if 'region' in pg.keys():
                has_region_added = False
                for region in pg['region']:
                    try:
                        self.__add_proxy_names(
                            pg, ProxyObjects(map_region_PO_s[region]).proxy_names
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
                    self.__add_proxy_names(pg, proxy_objects.proxy_names)
                pg.pop('region')
            else:
                self.__add_proxy_names(pg, proxy_objects.proxy_names)

    def validate(self, clash_bin_path):
        log_fd = tempfile.SpooledTemporaryFile()

        conf_fd = tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False)
        yaml.safe_dump(self.config, conf_fd, allow_unicode=True)
        conf_path = conf_fd.close()

        proc = subprocess.Popen(
            '{} -f {} -t'.format(clash_bin_path, conf_path),
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
        os.remove(conf_path)
        return ret


def main():
    # check args
    config_template = load_config(args.template, args.proxy)
    config_obj_template = Config(config_template)
    if not args.mmdb and config_obj_template.has_region_key():
        logging.critical('Please provide mmdb database.')
        exit(1)

    config_origin = load_config(args.config, args.proxy)
    proxy_objs = ProxyObjects(config_origin['proxies'])
    proxy_objs.update_external_ip(args.bin, args.workers)
    if args.mmdb:
        proxy_objs.update_geometry(args.mmdb)
        if not args.disable_rename:
            proxy_objs.rename(format_v4=args.format_v4, format_v6=args.format_v6)

    config_obj_template.update(proxy_objs, not args.disable_rename)

    if config_obj_template.validate(args.bin):
        dump_config(config_obj_template.config, args.out)


if __name__ == '__main__':
    main()
