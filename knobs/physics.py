import numpy as np
from astropy import units as u
from astropy import constants
from astropy.analytic_functions import blackbody_lambda

c = constants.c.to(u.km / u.s).value
flux_unit = u.erg * u.s**-1 * u.cm**-2 * u.Hz**-1


def hydrogen_lines(series=None, n_upper=6):
    """rydberg formula -> hydrogen lines
    for Lyman (n = 1), Balmer (n = 2), Paschen (n = 3), and Brackett (n = 4)
    series up to n = n_upper
    """
    if series is None:
        series = range(1, 5)
    lines = []
    for nf in series:
        for ni in range(nf + 1, n_upper + 1):
            lines.append(1 / (1.097e7 * 1.0 / nf**2 - 1.0 / ni**2))
    lines = np.array(lines)
    lines.sort()
    lines = lines * u.m.to(u.nm) * u.nm
    return lines


def gauss(x, mu, sigma):
    var = sigma**2
    chi2 = (x - mu)**2 / (2 * var)
    return np.exp(-chi2) / np.sqrt(2 * np.pi * var)
