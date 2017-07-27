from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "VERSION.txt"), encoding="utf-8") as f:
    __version__ = f.readline().strip('\n')

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="spectral-knobs",
      version=__version__,
      long_description=long_description,
      description="Interactive spectral lines",
      url="https://github.com/adwasser/spectral-knobs",
      author="Asher Wasserman",
      author_email="adwasser@ucsc.edu",
      license="GPLv3",
      packages=["knobs"],
      install_requires=[
          "numpy",
          "scipy",
          "astropy",
          "matplotlib",
          "seaborn",
          "ipython",
          "ipywidgets"],
      classifiers=[
          "Development Status :: 1 -Planning",
          "Intended Audience :: Education",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Programming Language :: Python :: 3.6",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Visualization"],
      python_requires=">=3.2")
