import numpy as np
from scipy import ndimage
from astropy import units as u
from matplotlib import pyplot as plt
import ipywidgets as widgets

from .physics import hydrogen_lines, flux_unit, gauss, c


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
                 lyman=True, balmer=True, paschen=True, brackett=True):
        self.z = z
        self.sigma = sigma
        self.n = n
        self.lyman = lyman
        self.balmer = balmer
        self.paschen = paschen
        self.brackett = brackett

    def series(self):
        """Construct a list of integers representing the series"""
        series = []
        if self.lyman:
            series.append(1)
        if self.balmer:
            series.append(2)
        if self.paschen:
            series.append(3)
        if self.brackett:
            series.append(4)
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
            delta_lam = self.z * line
            sigma_lam = self.sigma * line / c
            line_flux = weights[i] * gauss(wv, line + delta_lam, sigma_lam)
            flux += ndimage.gaussian_filter1d(line_flux, sigma=sigma_lam)
        return self.n * flux * flux_unit


class EmissionCloud(Cloud):
    """Line occupation based on temperature."""
    pass


class AbsorptionCloud(Cloud):
    """Lines subtracted from continuum source"""
    def __init__(self, z, temp, dens,
                 continuum=lambda wv: 3.6e-24 * flux_unit * np.ones(wv.shape)):
        """
        redshift : float, unitless
        temperature : quantity
        density : quantity
        continuum : func: wavelength -> specific intensity
            default is uniform continuum of around 10th magnitude
        """
        self.continuum = continuum
        super().__init__(z, temp, dens)

    def line_flux(self, wv, weights=None):
        flux = super().line_flux(wv, weights)
        return np.maximum(self.continuum(wv) - flux, 0)


class CloudInteractive(widgets.interactive):
    def __init__(self, cloud, wvmin, wvmax, series_flags=True,
                 zmin=0.00, zmax=0.10,
                 smin=1, smax=500,
                 nmin=0, nmax=0.1):
        """
        cloud : Cloud object
        wvmin : minimum wavelength in nm
        wvmax : maximum wavelength in nm
        """
        self.cloud = cloud
        dv = cloud.sigma / 4.
        wv = []
        idx = wvmin
        while idx < wvmax:
            wv.append(idx)
            idx += dv / c * idx
        wv = np.array(wv)
        self.wv = wv
        z_widget = widgets.FloatSlider(value=cloud.z,
                                       min=zmin,
                                       max=zmax,
                                       step=(zmax - zmin) / 100,
                                       description="Redshift: ",
                                       disabled=False,
                                       continuous_update=False,
                                       orientation="horizontal",
                                       readout=True,
                                       readout_format=".3f")
        s_widget = widgets.FloatSlider(value=cloud.sigma,
                                       min=smin,
                                       max=smax,
                                       step=(smax - smin) / 100,
                                       description="Dispersion: ",
                                       disabled=False,
                                       continuous_update=False,
                                       orientation="horizontal",
                                       readout=True,
                                       readout_format=":.3f")
        n_widget = widgets.FloatSlider(value=cloud.dens,
                                       min=nmin,
                                       max=nmax,
                                       step=(nmax - nmin) / 100,
                                       description="Density: ",
                                       disabled=False,
                                       continuous_update=False,
                                       orientation="horizontal",
                                       readout=True,
                                       readout_format=".3f")
        lyman_widget = widgets.Checkbox(value=cloud.lyman,
                                        description="Lyman series",
                                        disabled=False,
                                        continuous_update=False)
        balmer_widget = widgets.Checkbox(value=cloud.balmer,
                                         description="Balmer series",
                                         disabled=False,
                                         continuous_update=False)
        paschen_widget = widgets.Checkbox(value=cloud.paschen,
                                          description="Paschen series",
                                          disabled=False,
                                          continuous_update=False)
        brackett_widget = widgets.Checkbox(value=cloud.brackett,
                                           description="Brackett series",
                                           disabled=False,
                                           continuous_update=False)
        if series_flags:
            lyman = lyman_widget
            balmer = balmer_widget
            paschen = paschen_widget
            brackett = brackett_widget
        else:
            lyman = self.lyman
            balmer = self.balmer
            paschen = self.paschen
            brackett = self.brackett

        super.__init__(self.plot,
                       z=z_widget,
                       sigma=s_widget,
                       n=n_widget,
                       lyman=lyman,
                       balmer=balmer,
                       paschen=paschen,
                       brackett=brackett)

    def plot(self, z, sigma, n, lyman, balmer, paschen, brackett):
        self.cloud.z = z
        self.cloud.sigma = sigma
        self.cloud.n = n
        self.cloud.lyman = lyman
        self.cloud.balmer = balmer
        self.cloud.paschen = paschen
        self.cloud.brackett = brackett
        flux = self.cloud.line_flux(self.wv)
        plt.plot(self.wv, flux)
        plt.xlabel("Wavelength  [nm]")
        plt.ylabel("Flux density [arbitrary units]")
        # plt.ylabel(r"Flux density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]")
        plt.ylim(0, 1)
        plt.show()


class MultiCloudInteractive(widgets.interactive):
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
            idx += (dv / c.c).to(u.dimensionless_unscaled).value * idx
        wv = np.array(wv) * u.nm
        self.wv = wv
        widget_dict = {}
        for i, cloud in enumerate(self.clouds):
            key = 'z' + str(i)
            widget_dict[key] = widgets.FloatSlider(value=cloud.z,
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
            widget_dict[key] = widgets.FloatSlider(value=cloud.temp,
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
            widget_dict[key] = widgets.FloatSlider(value=cloud.dens,
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
        
