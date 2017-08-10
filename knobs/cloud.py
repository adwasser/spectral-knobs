from itertools import cycle
from string import ascii_lowercase
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import FloatSlider, Checkbox, fixed, interactive

from .physics import hydrogen_lines, line_profile, c


class Cloud:
    """Hydrogen cloud

    Parameters
    ----------
    z : redshift
    sigma : velocity dispersion in km/s
    n : column density (arbitrary units for now)
    P : period in years
    v : velocity amplitude in km/s
    t : time in years
    lyman, balmer, etc... : boolean flags for including the specified series
    absorption : bool, if true, subtract flux from continuum instead of
                 adding emission lines
    continuum : func: wv -> flux, only used if absorption lines
                or float for a flat continuum
    """
    def __init__(self, z, sigma, n, P=0, v=30, t=0,
                 lyman=True, balmer=True, paschen=False,
                 absorption=False, continuum=1.0, n_upper=8):
        self.z = z
        self.sigma = sigma
        self.n = n
        self.P = P
        self.v = v
        self.t = t
        self.lyman = lyman
        self.balmer = balmer
        self.paschen = paschen
        self.absorption = absorption
        if callable(continuum):
            self.continuum = continuum
        else:
            self.continuum = lambda wv: continuum * np.ones(wv.shape)
        self.n_upper = n_upper

    @property
    def series(self):
        """Construct a list of integers representing the series"""
        series = []
        if self.lyman:
            series.append(1)
        if self.balmer:
            series.append(2)
        if self.paschen:
            series.append(3)
        return series

    @series.setter
    def series(self, integers):
        integers = list(map(int, integers))
        for i in integers:
            if i not in range(1, 4):
                raise ValueError("Input needs to be from {1, 2, 3}")
        self.lyman = True if 1 in integers else False
        self.balmer = True if 2 in integers else False
        self.paschen = True if 3 in integers else False

    def line_flux(self, wv, weights=None):
        """Get line fluxes for the specified wavelength array."""
        lines = hydrogen_lines(self.series, self.n_upper)
        if weights is None:
            weights = np.ones(lines.shape)
        flux = np.zeros(wv.shape)
        for i, line in enumerate(lines):
            z = self.z + np.cos(self.P * self.t) / c
            flux += line_profile(wv, line, z, self.sigma, self.n)
        if self.absorption:
            return self.continuum(wv) - flux
        return flux


class CloudInteractive(interactive):
    """
    Interactive wrapper to a hydrogen cloud.

    Parameters
    ----------
    wvmin : minimum wavelength in nm
    wvmax : maximum wavelength in nm
    cloud : Cloud object (if None, then construct from default)
    cloud_kwargs : keyword arguments to pass to Cloud constructor
    show_labels : bool, if True include labels in the widgets
    widgets : iterable of strings, indicating which widgets to construct
    """
    def __init__(self, wvmin, wvmax, cloud=None, cloud_kwargs={}, show_labels=True,
                 widgets=('z', 'sigma', 'n', 'lyman', 'balmer', 'paschen'),
                 zmin=0.00, zmax=0.10,
                 smin=1, smax=500,
                 nmin=0, nmax=0.1,
                 Pmin=0, Pmax=10,
                 vmin=1, vmax=100,
                 tmin=0, tmax=20):
        if cloud is None:
            self.cloud = Cloud(z=0.00,
                               sigma=(smin + smax) / 2.0,
                               n=(nmin + nmax) / 2.0, **cloud_kwargs)
            cloud = self.cloud
        else:
            self.cloud = cloud
        dv = (smax + smin) / 8.0
        dlam = dv / c * (wvmax - wvmin) / 2.0
        wv = np.linspace(wvmin, wvmax, int((wvmax - wvmin) / dlam))
        self.wv = wv

        # set max height of graph
        old_sigma = cloud.sigma
        old_n = cloud.n
        cloud.sigma = (smin + smax) / 2.0
        cloud.n = nmax
        self.ymax = 1.1 * np.amax(cloud.line_flux(wv))
        cloud.sigma = old_sigma
        cloud.n = old_n

        # construct widget dictionary
        widget_dict = {}

        # float sliders
        keys = ['z', 'sigma', 'n', 'P', 'v', 't']
        labels = ['Redshift: ', 'Dispersion: ', 'Density: ',
                  'Period: ', 'Amplitude: ', 'Time: ']
        widget_kwargs = {"disabled": False,
                         "continuous_update": False,
                         "orientation": "horizontal",
                         "readout": True,
                         "readout_format": ".4f"}
        values = [cloud.z, cloud.sigma, cloud.n,
                  cloud.P, cloud.v, cloud.t]
        bounds = [(zmin, zmax), (smin, smax), (nmin, nmax),
                  (Pmin, Pmax), (vmin, vmax), (tmin, tmax)]
        letter = cycle(ascii_lowercase)
        for i, key in enumerate(keys):
            value = values[i]
            if key not in widgets:
                widget_dict[key] = fixed(value)
                continue
            if show_labels:
                label = labels[i]
            else:
                label = "({})".format(next(letter))
            lower, upper = bounds[i]
            widget_dict[key] = FloatSlider(value=value,
                                           min=lower,
                                           max=upper,
                                           step=(upper - lower) / 100,
                                           description=label,
                                           **widget_kwargs)

        # boolean checkboxes
        keys = ['lyman', 'balmer', 'paschen']
        labels = [s.capitalize() + ": " for s in keys]
        widget_kwargs = {"disabled": False}
        values = [cloud.lyman, cloud.balmer, cloud.paschen]
        for i, key in enumerate(keys):
            value = values[i]
            if key not in widgets:
                widget_dict[key] = fixed(value)
                continue
            if show_labels:
                label = labels[i]
            else:
                label = "({})".format(next(letter))
            widget_dict[key] = Checkbox(value=value,
                                        description=label,
                                        **widget_kwargs)

        self.widget_dict = widget_dict
        super().__init__(self.plot, **widget_dict)

    def plot(self, z, sigma, n, P, v, t, lyman, balmer, paschen):
        self.cloud.z = z
        self.cloud.sigma = sigma
        self.cloud.n = n
        self.cloud.P = P
        self.cloud.v = v
        self.cloud.t = t
        self.cloud.lyman = lyman
        self.cloud.balmer = balmer
        self.cloud.paschen = paschen
        flux = self.cloud.line_flux(self.wv)
        plt.plot(self.wv, flux)
        plt.xlabel("Wavelength  [nm]")
        plt.ylabel("Flux density [arbitrary units]")
        # plt.ylabel(r"Flux density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")
        plt.ylim(0, self.ymax)
        plt.show()


class MultiCloudInteractive(interactive):
    """
    Interactive with multiple clouds.

    Parameters
    ----------
    wvmin : minimum wavelength in nm
    wvmax : maximum wavelength in nm
    clouds : list of Cloud objects
    show_labels : bool, if True include labels in the widgets
    widgets : iterable of strings, indicating which widgets to construct
    """
    def __init__(self, wvmin, wvmax, clouds, show_labels=True,
                 widgets=('z', 'sigma', 'n', 'lyman', 'balmer', 'paschen'),
                 zmin=0.00, zmax=0.10,
                 smin=1, smax=500,
                 nmin=0, nmax=0.1,
                 Pmin=0, Pmax=10,
                 vmin=1, vmax=100,
                 tmin=0, tmax=20):
        self.clouds = clouds
        self.ncomponents = len(clouds)
        dv = (smax + smin) / 8.0
        dlam = dv / c * (wvmax - wvmin) / 2.0
        wv = np.linspace(wvmin, wvmax, int((wvmax - wvmin) / dlam))
        self.wv = wv

        # set max height of graph
        cloud = clouds[0]
        old_sigma = cloud.sigma
        old_n = cloud.n
        cloud.sigma = (smin + smax) / 2.0
        cloud.n = nmax
        self.ymax = 1.1 * np.amax(cloud.line_flux(wv))
        cloud.sigma = old_sigma
        cloud.n = old_n

        # construct widget dictionary        
        widget_dict = {}
        letter = cycle(ascii_lowercase)
        for i, cloud in enumerate(self.clouds):
            # float sliders
            keys = ['z', 'sigma', 'n', 'P', 'v', 't']
            labels = ['Redshift: ', 'Dispersion: ', 'Density: ',
                      'Period: ', 'Amplitude: ', 'Time: ']
            widget_kwargs = {"disabled": False,
                             "continuous_update": False,
                             "orientation": "horizontal",
                             "readout": True,
                             "readout_format": ".4f"}
            values = [cloud.z, cloud.sigma, cloud.n,
                      cloud.P, cloud.v, cloud.t]
            bounds = [(zmin, zmax), (smin, smax), (nmin, nmax),
                      (Pmin, Pmax), (vmin, vmax), (tmin, tmax)]
            values = [cloud.z, cloud.sigma, cloud.n]
            bounds = [(zmin, zmax), (smin, smax), (nmin, nmax)]
            for j, key in enumerate(keys):
                value = values[j]
                if key not in widgets:
                    widget_dict[key + str(i)] = fixed(value)
                    continue
                if show_labels:
                    label = labels[j]
                else:
                    label = "({})".format(next(letter))
                lower, upper = bounds[j]
                widget_dict[key + str(i)] = FloatSlider(value=value,
                                                        min=lower,
                                                        max=upper,
                                                        step=(upper - lower) / 100,
                                                        description=label,
                                                        **widget_kwargs)
            # boolean checkboxes
            keys = ['lyman', 'balmer', 'paschen']
            labels = [s.capitalize() + ": " for s in keys]
            widget_kwargs = {"disabled": False}
            values = [cloud.lyman, cloud.balmer, cloud.paschen]
            for j, key in enumerate(keys):
                value = values[j]
                if key not in widgets:
                    widget_dict[key + str(i)] = fixed(value)
                    continue
                if show_labels:
                    label = labels[j]
                else:
                    label = "({})".format(next(letter))
                widget_dict[key + str(i)] = Checkbox(value=value,
                                                     description=label,
                                                     **widget_kwargs)

        super().__init__(self.plot, **widget_dict)

    def plot(self, **kwargs):
        flux = np.zeros(self.wv.shape)
        for i, cloud in enumerate(self.clouds):
            cloud.z = kwargs['z' + str(i)]
            cloud.sigma = kwargs['sigma' + str(i)]
            cloud.n = kwargs['n' + str(i)]
            cloud.P = kwargs['P' + str(i)]
            cloud.v = kwargs['v' + str(i)]
            cloud.t = kwargs['t' + str(i)]
            cloud.lyman = kwargs['lyman' + str(i)]
            cloud.balmer = kwargs['balmer' + str(i)]
            cloud.paschen = kwargs['paschen' + str(i)]
            flux += cloud.line_flux(self.wv)
        plt.plot(self.wv, flux)
        plt.xlabel("Wavelength  [nm]")
        plt.ylabel("Flux density [arbitrary units]")
        # plt.ylabel(r"Flux density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")
        plt.ylim(0, self.ymax)
        plt.show()
