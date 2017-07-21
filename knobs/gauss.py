import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import ipywidgets as widgets

def gauss(x, mu, sigma):
    var = sigma**2
    chi2 = (x - mu)**2 / (2 * var)
    return np.exp(-chi2) / np.sqrt(2 * np.pi * var)


class GaussInteractive(widgets.interactive):
    def __init__(self, xmin=-10.0, xmax=10.0, size=200):
        self.x = np.linspace(xmin, xmax, size)
        self.xmin = xmin
        self.xmax = xmax
        mu_widget = widgets.FloatSlider(value=(xmin + xmax) / 2.0,
                                        min=xmin,
                                        max=xmax,
                                        step=(xmax - xmin) / 200.0,
                                        description='Location: ',
                                        disabled=False,
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='.1f')
        sigma_widget = widgets.FloatSlider(value=xmax / 10.0,
                                           min=xmax / 100.0,
                                           max=xmax / 2.0,
                                           step=(xmax - xmin) / 200.0,
                                           description='Spread: ',
                                           disabled=False,
                                           continuous_update=False,
                                           orientation='horizontal',
                                           readout=True,
                                           readout_format='.1f')
        super().__init__(self.plot, mu=mu_widget, sigma=sigma_widget)

    def plot(self, mu, sigma):
        plt.plot(self.x, gauss(self.x, mu, sigma))
        plt.ylim(0, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
