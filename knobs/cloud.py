import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
# import ipywidgets as widgets
from ipywidgets import FloatSlider, Checkbox, fixed, interactive

from .physics import hydrogen_lines, line_profile, c


class Cloud:
    """Hydrogen cloud

    Parameters
    ----------
    z : redshift
    sigma : velocity dispersion in km/s
    n : column density (arbitrary units for now)
    lyman, balmer, etc... : boolean flags for including the specified series
    """
    def __init__(self, z, sigma, n,
                 lyman=True, balmer=True, paschen=False):
        self.z = z
        self.sigma = sigma
        self.n = n
        self.lyman = lyman
        self.balmer = balmer
        self.paschen = paschen

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

    def line_flux(self, wv, weights=None, n_upper=6):
        """
        Get line fluxes for the specified wavelength array.
        """
        lines = hydrogen_lines(self.series(), n_upper)
        if weights is None:
            weights = np.ones(lines.shape)
        flux = np.zeros(wv.shape)
        for i, line in enumerate(lines):
            flux += line_profile(wv, line, self.z, self.sigma, self.n)
        return flux


class EmissionCloud(Cloud):
    """Line occupation based on temperature."""
    pass


class AbsorptionCloud(Cloud):
    """Hydrogen cloud in absorption

    Parameters
    ----------
    z : redshift
    sigma : velocity dispersion in km/s
    n : column density (arbitrary units for now)
    continuum : func: wv -> flux
    lyman, balmer, etc... : boolean flags for including the specified series
    """
    def __init__(self, z, sigma, n,
                 continuum=lambda wv: np.ones(wv.shape),
                 lyman=True, balmer=True, paschen=False):
        self.continuum = continuum
        super().__init__(z, sigma, n, lyman, balmer, paschen)

    def line_flux(self, wv, weights=None):
        flux = super().line_flux(wv, weights)
        return np.maximum(self.continuum(wv) - flux, 0)


class CloudInteractive(interactive):
    """
    Interactive wrapper to a hydrogen cloud.

    Parameters
    ----------
    cloud : Cloud object
    wvmin : minimum wavelength in nm
    wvmax : maximum wavelength in nm
    series_flags : if True, include toggles for the different series
    """
    def __init__(self, cloud, wvmin, wvmax, series_flags=True,
                 zmin=0.00, zmax=0.10,
                 smin=1, smax=500,
                 nmin=0, nmax=0.1):
        self.cloud = cloud
        dv = (smax + smin) / 8.0
        dlam = dv / c * (wvmax - wvmin) / 2.0
        wv = np.linspace(wvmin, wvmax, int((wvmax - wvmin) / dlam))
        # wv = []
        # idx = wvmin
        # while idx < wvmax:
        #     wv.append(idx)
        #     idx += dv / c * idx
        # wv = np.array(wv)
        self.wv = wv

        # set max height of graph
        old_sigma = cloud.sigma
        old_n = cloud.n
        cloud.sigma = (smin + smax) / 2.0
        cloud.n = nmax
        self.ymax = 1.1 * np.amax(cloud.line_flux(wv))
        # if isinstance(cloud, AbsorptionCloud):
        #     self.ymax *= 1.1
        cloud.sigma = old_sigma
        cloud.n = old_n

        # construct widget dictionary
        widget_dict = {}
        widget_dict['z'] = FloatSlider(value=cloud.z,
                                       min=zmin,
                                       max=zmax,
                                       step=(zmax - zmin) / 100,
                                       description="Redshift: ",
                                       disabled=False,
                                       continuous_update=False,
                                       orientation="horizontal",
                                       readout=True,
                                       readout_format=".3f")
        widget_dict['sigma'] = FloatSlider(value=cloud.sigma,
                                           min=smin,
                                           max=smax,
                                           step=(smax - smin) / 100,
                                           description="Dispersion: ",
                                           disabled=False,
                                           continuous_update=False,
                                           orientation="horizontal",
                                           readout=True,
                                           readout_format=".3f")
        widget_dict['n'] = FloatSlider(value=cloud.n,
                                       min=nmin,
                                       max=nmax,
                                       step=(nmax - nmin) / 100,
                                       description="Density: ",
                                       disabled=False,
                                       continuous_update=False,
                                       orientation="horizontal",
                                       readout=True,
                                       readout_format=".3f")
        if series_flags:
            widget_dict['lyman'] = Checkbox(value=cloud.lyman,
                                            description="Lyman series",
                                            disabled=False)
            widget_dict['balmer'] = Checkbox(value=cloud.balmer,
                                             description="Balmer series",
                                             disabled=False)
            widget_dict['paschen'] = Checkbox(value=cloud.paschen,
                                              description="Paschen series",
                                              disabled=False)
        else:
            widget_dict['lyman'] = fixed(self.lyman)
            widget_dict['balmer'] = fixed(self.balmer)
            widget_dict['paschen'] = fixed(self.paschen)
        self.widget_dict = widget_dict
        super().__init__(self.plot, **widget_dict)

    def plot(self, z, sigma, n, lyman, balmer, paschen):
        self.cloud.z = z
        self.cloud.sigma = sigma
        self.cloud.n = n
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



# TODO: finish this
class MultiCloudInteractive(interactive):
    def __init__(self, clouds, wvmin, wvmax,
                 zmin=0.00, zmax=0.10,
                 tmin=100, tmax=10000,
                 dmin=0, dmax=0.1):
        """
        clouds : list of Cloud objects
        wvmin : minimum wavelength in nm
        wvmax : maximum wavelength in nm
        """
        self.clouds = clouds
        self.ncomponents = len(clouds)
        dv = np.mean([cloud.sigma_v.value for cloud in self.clouds]) / 4.
        dv = dv * u.km / u.s
        wv = []
        idx = wvmin
        while idx < wvmax:
            wv.append(idx)
            idx += (dv / c).to(u.dimensionless_unscaled).value * idx
        wv = np.array(wv) * u.nm
        self.wv = wv
        widget_dict = {}
        for i, cloud in enumerate(self.clouds):
            key = 'z' + str(i)
            widget_dict[key] = FloatSlider(value=cloud.z,
                                           min=zmin,
                                           max=zmax,
                                           step=(zmax - zmin) / 100,
                                           description="Redshift: ",
                                           disabled=False,
                                           continuous_update=False,
                                           orientation="horizontal",
                                           readout=True,
                                           readout_format=".3f")
            key = 't' + str(i)
            widget_dict[key] = FloatSlider(value=cloud.temp,
                                           min=tmin,
                                           max=tmax,
                                           step=(tmax - tmin) / 100,
                                           description="Temperature: ",
                                           disabled=False,
                                           continuous_update=False,
                                           orientation="horizontal",
                                           readout=True,
                                           readout_format="d")
            key = 'd' + str(i)
            widget_dict[key] = FloatSlider(value=cloud.dens,
                                           min=dmin,
                                           max=dmax,
                                           step=(dmax - dmin) / 100,
                                           description="Density: ",
                                           disabled=False,
                                           continuous_update=False,
                                           orientation="horizontal",
                                           readout=True,
                                           readout_format=".3f")
        super().__init__(self.plot, **widget_dict)

    def plot(self, **kwargs):
        flux = np.zeros(self.wv.shape)
        for i, cloud in enumerate(self.clouds):
            z = kwargs['z' + str(i)]
            t = kwargs['t' + str(i)]
            d = kwargs['d' + str(i)]
            cloud.z = z
            cloud.temp = t
            cloud.dens = d
            flux += cloud.line_flux(self.wv)
        plt.plot(self.wv, flux)
        plt.xlabel("Wavelength  [nm]")
        plt.ylabel("Flux density [arbitrary units]")
        # plt.ylabel(r"Flux density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")
        plt.ylim(0, 1)
        plt.show()
        
