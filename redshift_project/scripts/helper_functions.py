import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.io import fits
import pandas as pd
import os
import sys
from glob import glob
    

pretty_plot_params = {'font.size': 18, 
                     'text.usetex': True, 
                     'font.family': 'STIXGeneral',
                     'xtick.top': True,
                     'ytick.right': True,
                     'xtick.major.size': 10.,
                     'xtick.minor.size': 5.,
                     'xtick.major.width': 1.5,
                     'xtick.minor.width': 1.,
                     'ytick.major.size': 10.,
                     'ytick.minor.size': 5.,
                     'ytick.major.width': 1.5,
                     'ytick.minor.width': 1.,
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.minor.visible': True,
                     'ytick.minor.visible': True}

def read_spectrum(filename):
    """
    Reads a FITS file from SDSS and returns the spectrum data as an Astropy Table.
    
    Parameters:
    filename (str): The path to the FITS file.
    
    Returns:
    astropy.table.Table: The spectrum data.
    """
    
    t = Table.read(filename)

    wavelength = t['wavelength']
    flux = t['flux']

    return wavelength, flux

# Pre-made function to plot a spectrum
def plot_spectrum(filename, wave_range=None, fig=None, ax=None):
    """
    Plots the spectrum from the given filename.

    Parameters:
        filename (str): Path to the spectrum file.
        wave_range (tuple, optional): (wave_min, wave_max) for x-axis limits.
        fig (matplotlib.figure.Figure, optional): Existing figure to use.
        ax (matplotlib.axes.Axes, optional): Existing axes to use.

    Returns:
        fig, ax: The figure and axis containing the plot.
    """
    wave, flux = read_spectrum(filename)

    # Apply wavelength range if specified
    if wave_range is not None:
        wave_min, wave_max = wave_range

        # Create a mask to isolate the wavelength region
        mask = (wave >= wave_min) & (wave <= wave_max)

        # Apply the mask to the wavelength and flux arrays
        wave = wave[mask]
        flux = flux[mask]

    # Create figure and axis if not provided
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the spectrum
    ax.step(wave, flux, where='mid', color='black', lw=1)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Flux')

    if wave_range is not None:
        ax.set_xlim(wave_min, wave_max)

    return fig, ax


# Define a Gaussian function for fitting
# A Gaussian is also the bell curve
def gaussian(x, amp, cen, wid, offset):
    """
    x : an array of wavelengths
    amp : amplitude of the Gaussian
    cen : center of the Gaussian (mean)
    wid : width of the Gaussian (standard deviation)
    offset : vertical offset of the Gaussian

    Returns the Gaussian array.
    """
    return amp * np.exp(-(x-cen)**2 / (2*wid**2)) + offset


# Pre-made function to fit Gaussian profile to an emission line
def fit_gaussian(filename, wave_range=[4980, 5020]):
    wave, flux = read_spectrum(filename)

    wave_min, wave_max = wave_range
    # Create a mask to isolate the wavelength region
    mask = (wave >= wave_min) & (wave <= wave_max)
    # Apply the mask to the wavelength and flux arrays
    wave, flux = wave[mask], flux[mask]

    # Fit a Gaussian profile to the emission line
    # p0 is the initial guess for the parameters: [amplitude, center, width, offset]
    amplitude_guess = 10 * np.median(flux)
    try:
        center_guess = wave[np.argmax(flux)]
    except ValueError:
        center_guess = np.mean(wave)
    width_guess = 10  # This is a guess, you might need to adjust it
    offset_guess = 0  # Assuming no offset for simplicity
    p0 = [amplitude_guess, center_guess, width_guess, offset_guess]  # Initial guess for the parameters
    fit, cov = curve_fit(gaussian, wave, flux, p0=p0)
    # fit contains the best-fit parameters (amp, cen, wid, offset)
    # cov is the "covariance" of the fit
    # We can turn cov into the "error" of the fit like this:
    fit_err = np.sqrt(np.diag(cov))

    return fit, fit_err
