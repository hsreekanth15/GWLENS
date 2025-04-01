"""
author: Sreeakanth Harikumar
"""

import numpy as np
import scipy.special as sc
from mpmath import hyp1f1
import cmath
import mpmath
from astropy.constants import c,G,M_sun


#Define constants   
c = c.value
G = G.value
M_sun = M_sun.value

def DimensionlessFrequency(lensmass,f):
    
    const = 8*np.pi*G/c**3

    w = const*lensmass*M_sun*f

    return w



def Point_multi(w, x):
    """
    A point mass lens analytical amplification factor from Takahashi(2003). The function is suitable for 
    microlensing (fixed w and time varyinig impact paramater)

    Parameter :
    w : Dimensionless frequency
    y : Impact parameter

    returns : the magnitude of the amplification factor
    """
    Freal = []
    Fimag = []
    y = np.abs(x)
    for i in range(len(w)):
        
        for j in range(len(y)):
            #y[i] = np.abs(y[i])
            xm = (y[j] + np.sqrt(y[j]**2 + 4)) / 2
            phim = (xm - y[j])**2 / 2 - np.log(xm)
            expon = np.exp(np.pi * w[i] / 4 + 1j * w[i] / 2 * (np.log(w[i] / 2) - 2 * phim))
            mp_w = mpmath.mpc(w[i])
            mp_y = mpmath.mpc(y[j])
            f = expon * mpmath.gamma(1 - 1j / 2 * mp_w) * mpmath.hyp1f1(1j / 2 * mp_w, 1, 1j / 2 * mp_w * mp_y**2)
            Freal.append(f.real)
            Fimag.append(f.imag)   
    return Freal, Fimag



def Point(w,y):
    xm = (y + np.sqrt(y**2 + 4)) / 2
    phim = (xm - y)**2 / 2 - np.log(xm)
    expon = np.exp(np.pi * w / 4 + 1j * w / 2 * (np.log(w / 2) - 2 * phim))
    mp_w = mpmath.mpc(w)
    mp_y = mpmath.mpc(y)
    f = expon * mpmath.gamma(1 - 1j / 2 * mp_w) * mpmath.hyp1f1(1j / 2 * mp_w, 1, 1j / 2 * mp_w * mp_y**2)
    #Fan.append(complex(f.real, f.imag)) 
    F = complex(f.real, f.imag)
    return F



