from .cloud import Cloud, CloudInteractive, MultiCloudInteractive

with open('/'.join(__file__.split('/')[:-2]) + '/VERSION.txt') as f:
    __version__ = f.readline().strip('\n')
