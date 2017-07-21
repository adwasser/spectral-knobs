import numpy as np
from scipy import ndimage
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

flux_unit = u.erg * u.s**-1 * u.cm**-2 * u.Hz**-1
hydrogen_lines = np.array([1.0 / (1.097e7 * (1.0 / nf**2 - 1.0 / ni**2))
                           for nf in range(1, 5)
                           for ni in range(nf + 1, 7)])
hydrogen_lines.sort()
hydrogen_lines = hydrogen_lines * u.m.to(u.nm) * u.nm

def gauss(x, mu, sigma):
    var = sigma**2
    chi2 = (x - mu)**2 / (2 * var)
    return np.exp(-chi2) / np.sqrt(2 * np.pi * var)


class Cloud:
    """Hydrogen cloud"""
    def __init__(self, z, temp, dens):
        """
        z : redshift
        temp : temperature in K
        dens : density
        """
        self.z = z
        self.temp = temp
        self.dens = dens

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, temp):
        fudge_factor = 10
        self._temp = temp
        self.sigma_v = np.sqrt(3 * c.k_B * fudge_factor * self.temp * u.K /
                               c.m_p)
        self.sigma_v = self.sigma_v.to(u.km / u.s)

    @u.quantity_input(wv=u.nm)
    def line_flux(self, wv, weights=None):
        """
        Get line fluxes for the specified wavelength array.
        """
        if weights is None:
            weights = np.ones(hydrogen_lines.shape)
        flux = np.zeros(wv.shape)
        for i, line in enumerate(hydrogen_lines):
            delta_lam = self.z * line
            sigma_lam = self.sigma_v * line / c.c
            line_flux = weights[i] * gauss(wv.to(u.nm).value,
                                           (line + delta_lam).to(u.nm).value,
                                           sigma_lam.to(u.nm).value)
            flux += ndimage.gaussian_filter1d(line_flux,
                                              sigma=sigma_lam.to(u.nm).value)
        return self.dens * flux * flux_unit

    
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

    @u.quantity_input(wv=u.nm)
    def line_flux(self, wv, weights=None):
        flux = super().line_flux(wv, weights)
        return np.maximum(self.continuum(wv) - flux, 0)
        

class CloudInteractive(widgets.interactive):
    
    def __init__(self, cloud, wvmin, wvmax,
                 zmin=0.00, zmax=0.10,
                 tmin=100, tmax=10000,
                 dmin=0, dmax=0.1):
        """
        cloud : Cloud object
        wvmin : minimum wavelength in nm
        wvmax : maximum wavelength in nm
        """
        self.cloud = cloud
        dv = cloud.sigma_v / 4.
        wv = []
        idx = wvmin
        while idx < wvmax:
            wv.append(idx)
            idx += (dv / c.c).to(u.dimensionless_unscaled).value * idx
        wv = np.array(wv) * u.nm
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
        temp_widget = widgets.FloatSlider(value=cloud.temp,
                                          min=tmin,
                                          max=tmax,
                                          step=(tmax - tmin) / 100,
                                          description="Temperature: ",
                                          disabled=False,
                                          continuous_update=False,
                                          orientation="horizontal",
                                          readout=True,
                                          readout_format="d")
        dens_widget = widgets.FloatSlider(value=cloud.dens,
                                          min=dmin,
                                          max=dmax,
                                          step=(dmax - dmin) / 100,
                                          description="Density: ",
                                          disabled=False,
                                          continuous_update=False,
                                          orientation="horizontal",
                                          readout=True,
                                          readout_format=".3f")
        super().__init__(self.plot,
                         z=z_widget,
                         temp=temp_widget,
                         dens=dens_widget)
        
    def plot(self, z, temp, dens):
        self.cloud.z = z
        self.cloud.temp = temp
        self.cloud.dens = dens
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
        
