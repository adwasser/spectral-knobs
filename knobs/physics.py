import numpy as np
from scipy import ndimage
from astropy import units as u
from astropy import constants
# from astropy.analytic_functions import blackbody_lambda

c = constants.c.to(u.km / u.s).value
R = constants.Ryd.to(u.nm**-1).value
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
            lines.append(1 / (R * (1.0 / nf**2 - 1.0 / ni**2)))
    lines = np.array(lines)
    lines.sort()
    return lines


def gauss(x, mu, sigma):
    var = sigma**2
    chi2 = (x - mu)**2 / (np.sqrt(2) * var)
    return np.exp(-chi2) / np.sqrt(2 * np.pi * var)


def line_profile(wv, line, z, sigma, n, smooth=True):
    delta_lam = z * line
    sigma_lam = sigma * line / c
    a = n / np.sqrt(2 * np.pi) / sigma_lam
    line_flux = a * gauss(wv, line + delta_lam, sigma_lam)
    if smooth:
        return ndimage.gaussian_filter1d(line_flux, sigma=sigma_lam)
    return line_flux
