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
    lyman, balmer, etc... : boolean flags for including the specified series
    absorption : bool, if true, subtract flux from continuum instead of
                 adding emission lines
    continuum : func: wv -> flux, only used if absorption lines
                or float for a flat continuum
    """
    def __init__(self, z, sigma, n,
                 lyman=True, balmer=True, paschen=False,
                 absorption=False, continuum=1.0, n_upper=8):
        self.z = z
        self.sigma = sigma
        self.n = n
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
            flux += line_profile(wv, line, self.z, self.sigma, self.n)
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
    labels : bool, if True include labels in the widgets
    widgets : iterable of strings, indicating which widgets to construct
    """
    def __init__(self, wvmin, wvmax, cloud=None, cloud_kwargs={}, labels=True,
                 widgets=('z', 'sigma', 'n', 'lyman', 'balmer', 'paschen'),
                 zmin=0.00, zmax=0.10,
                 smin=1, smax=500,
                 nmin=0, nmax=0.1):
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
        widget_kwargs = {"disabled": False,
                         "continuous_update": False,
                         "orientation": "horizontal",
                         "readout": True,
                         "readout_format": ".4f"}
        key = 'z'
        if key in widgets:
            desc = "Redshift: " if labels else "-"
            widget_dict[key] = FloatSlider(value=cloud.z,
                                           min=zmin,
                                           max=zmax,
                                           step=(zmax - zmin) / 100,
                                           description=desc,
                                           **widget_kwargs)
        else:
            widget_dict[key] = fixed(cloud.z)

        key = 'sigma'
        if key in widgets:
            desc = "Dispersion: " if labels else "-"
            widget_dict[key] = FloatSlider(value=cloud.sigma,
                                           min=smin,
                                           max=smax,
                                           step=(smax - smin) / 100,
                                           description=desc,
                                           **widget_kwargs)
        else:
            widget_dict[key] = fixed(cloud.sigma)

        key = 'n'
        if key in widgets:
            desc = "Density: " if labels else "-"
            widget_dict[key] = FloatSlider(value=cloud.n,
                                           min=nmin,
                                           max=nmax,
                                           step=(nmax - nmin) / 100,
                                           description=desc,
                                           **widget_kwargs)
        else:
            widget_dict[key] = fixed(cloud.n)

        key = "lyman"
        if key in widgets:
            desc = "Lyman series" if labels else "-"
            widget_dict[key] = Checkbox(value=cloud.lyman,
                                        description=desc,
                                        disabled=False)
        else:
            widget_dict[key] = fixed(cloud.lyman)

        key = "balmer"
        if key in widgets:
            desc = "Balmer series" if labels else "-"
            widget_dict[key] = Checkbox(value=cloud.balmer,
                                        description=desc,
                                        disabled=False)
        else:
            widget_dict[key] = fixed(cloud.balmer)

        key = "paschen"
        if key in widgets:
            desc = "Paschen series" if labels else "-"
            widget_dict[key] = Checkbox(value=cloud.paschen,
                                        description=desc,
                                        disabled=False)
        else:
            widget_dict[key] = fixed(cloud.paschen)

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


class MultiCloudInteractive(interactive):
    """
    Interactive with multiple clouds.

    Parameters
    ----------
    wvmin : minimum wavelength in nm
    wvmax : maximum wavelength in nm
    clouds : list of Cloud objects
    labels : bool, if True include labels in the widgets
    widgets : iterable of strings, indicating which widgets to construct
    """
    def __init__(self, wvmin, wvmax, clouds, labels=True,
                 widgets=('z', 'sigma', 'n', 'lyman', 'balmer', 'paschen'),
                 zmin=0.00, zmax=0.10,
                 smin=1, smax=500,
                 nmin=0, nmax=0.1):
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

        widget_dict = {}
        widget_kwargs = {"disabled": False,
                         "continuous_update": False,
                         "orientation": "horizontal",
                         "readout": True,
                         "readout_format": ".4f"}
        for i, cloud in enumerate(self.clouds):
            key = 'z'
            if key in widgets:
                desc = "Redshift: " if labels else "-"
                widget_dict[key + str(i)] = FloatSlider(value=cloud.z,
                                                        min=zmin,
                                                        max=zmax,
                                                        step=(zmax - zmin) / 100,
                                                        description=desc,
                                                        **widget_kwargs)
            else:
                widget_dict[key + str(i)] = fixed(cloud.z)

            key = 'sigma'
            if key in widgets:
                desc = "Dispersion: " if labels else "-"
                widget_dict[key + str(i)] = FloatSlider(value=cloud.sigma,
                                                        min=smin,
                                                        max=smax,
                                                        step=(smax - smin) / 100,
                                                        description=desc,
                                                        **widget_kwargs)
            else:
                widget_dict[key + str(i)] = fixed(cloud.sigma)

            key = 'n'
            if key in widgets:
                desc = "Density: " if labels else "-"
                widget_dict[key + str(i)] = FloatSlider(value=cloud.n,
                                                        min=nmin,
                                                        max=nmax,
                                                        step=(nmax - nmin) / 100,
                                                        description=desc,
                                                        **widget_kwargs)
            else:
                widget_dict[key + str(i)] = fixed(cloud.n)

            key = 'lyman'
            if key in widgets:
                desc = "Lyman series" if labels else "-"
                widget_dict[key + str(i)] = Checkbox(value=cloud.lyman,
                                                     description=desc,
                                                     disabled=False)
            else:
                widget_dict[key + str(i)] = fixed(cloud.lyman)

            key = 'balmer'
            if key in widgets:
                desc = "Balmer series" if labels else "-"
                widget_dict[key + str(i)] = Checkbox(value=cloud.balmer,
                                                     description=desc,
                                                     disabled=False)
            else:
                widget_dict[key + str(i)] = fixed(cloud.balmer)

            key = 'paschen'
            if key in widgets:
                desc = "Paschen series" if labels else "-"
                widget_dict[key + str(i)] = Checkbox(value=cloud.paschen,
                                                     description=desc,
                                                     disabled=False)
            else:
                widget_dict[key + str(i)] = fixed(cloud.paschen)

        super().__init__(self.plot, **widget_dict)

    def plot(self, **kwargs):
        flux = np.zeros(self.wv.shape)
        for i, cloud in enumerate(self.clouds):
            cloud.z = kwargs['z' + str(i)]
            cloud.sigma = kwargs['sigma' + str(i)]
            cloud.n = kwargs['n' + str(i)]
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
