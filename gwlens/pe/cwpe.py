import bilby
import numpy as np
from matplotlib.pyplot as plt
import h5py
from scipy.constants import G, c
from scipy.interpolate import RectBivariateSpline
from astropy import units as u
from bilby.core.utils import random

# Constants required for dimensionless frequency
Msun = 1.988e30
constant = (8 * np.pi * G / c**3) * Msun

class CWPE:
    """
    A lensed waveform generator class of continuous waves

    """
    def __init__(self,time):
        """
        Time array, always need to be the first argument for bilby 
        """
        self.time = time

    def impact_parameter(self,t0,y0,TE):

        """
        The time varying impact parameter

        """
        x = (self.time - t0) / TE

        return np.sqrt(y0**2 + x**2)


    def model(time,h0,Sn,f0,MLz,t0,y0,TE,Tobs):

        """



        """

        return ampfac(time,f0,MLz,t0,y0,TE)*h0*np.sqrt(Tobs)/(Sn)
