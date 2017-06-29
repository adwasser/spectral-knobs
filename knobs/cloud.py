import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.analytic_functions import blackbody_lambda
import matplotlib as mpl
from matplotlib import pyplot as plt
import ipywidgets as widgets

"""
rydberg formula -> hydrogen lines
for Lyman (n = 1), Balmer (n = 2), Paschen (n = 3), and Brackett (n = 4)
series up to n = 6
"""
R = 1.097e7 # inverse meters
hydrogen_lines = []
hydrogen_lines = np.array([1.0 / (R * (1.0 / nf**2 - 1.0 / ni**2))
                           for nf in range(1, 5)
                           for ni in range(nf + 1, 7)]) * u.m.to(u.nm) * u.nm
hydrogen_lines.sort()

flux_unit = u.erg * u.s**-1 * u.cm**-2 * u.Hz**-1

class Cloud(widgets.interactive):
    pass

class EmissionCloud(Cloud):
    """Line occupation based on temperature."""
    pass


class AbsorptionCloud(Cloud):
    """Lines subtracted from continuum source"""
    def __init__(redshift, temperature, density,
                 continuum=lambda wv: 3.6e-24 * flux_unit):
        """
        redshift : float, unitless
        temperature : quantity
        density : quantity
        continuum : func: wavelength -> specific intensity
            default is uniform continuum of around 10th magnitude
        """
    pass
