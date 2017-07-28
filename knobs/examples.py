from .cloud import Cloud, CloudInteractive, MultiCloudInteractive
from .physics import hydrogen_lines

import seaborn as sns
sns.set_context('poster', font_scale=1.5)
sns.set_style('ticks', rc={'xtick.direction': 'in', 'ytick.direction': 'in'})

__all__ = ["high_z_abs", "two_em", "two_abs", "em_abs",
           "example1", "example2", "example3", "example4"]


def high_z_abs(labels=True):
    """High redshift two-component absorption"""
    clouds = [
        Cloud(z=0.3, sigma=50, n=0.01, absorption=True),
        Cloud(z=1, sigma=50, n=0.01, absorption=True, continuum=0.01)
    ]
    return MultiCloudInteractive(100, 1000, clouds,
                                 zmax=3, nmax=1, labels=labels,
                                 widgets=('z', 'lyman', 'balmer', 'paschen'))


def two_em(labels=True):
    """Two emission lines around Halpha"""
    clouds = [
        Cloud(0, 300, 1),
        Cloud(0.001, 400, 1)
    ]
    return MultiCloudInteractive(650, 660, clouds,
                                 zmax=0.01, nmin=0.1, nmax=2, labels=labels,
                                 widgets=('z', 'n'))


def two_abs(labels=True):
    """Two absorption lines around Halpha"""
    clouds = [Cloud(0, 150, 0.2, absorption=True),
              Cloud(0.0005, 100, 0.2, absorption=True, continuum=0.01)]
    return MultiCloudInteractive(650, 660, clouds,
                                 zmax=0.001, nmin=0.1, nmax=1,
                                 smin=50, smax=200, labels=labels,
                                 widgets=('z', 'sigma'))


def em_abs(labels=True):
    """Hgamma emission at low z covering Lyalpha abs at high z"""
    hgamma = hydrogen_lines([2], n_upper=5)[0]
    lalpha = hydrogen_lines([1], n_upper=2)[0]
    z = (hgamma - lalpha) / lalpha + 0.0001

    clouds = [
        Cloud(z=0, sigma=50, n=0.1, absorption=False),
        Cloud(z=z, sigma=100, n=0.01, absorption=True)
    ]
    return MultiCloudInteractive(432, 436, clouds, labels=labels,
                                 zmax=3, nmax=0.5, smin=10, smax=100,
                                 widgets=('n', 'sigma', 'lyman', 'balmer'))


def example1():
    return high_z_abs(labels=False)

def example2():
    return two_em(labels=False)

def example3():
    return two_abs(labels=False)

def example4():
    return em_abs(labels=False)
