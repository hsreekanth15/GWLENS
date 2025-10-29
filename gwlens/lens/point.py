"""
author: Sreeakanth Harikumar
"""

import numpy as np
import scipy.special as sc
from mpmath import hyp1f1,exp, mp,sqrt,mpc,pi,log,gamma, besselj
import cmath
import mpmath
from astropy.constants import c,G,M_sun

mp.dps= 60

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
            f = expon * mpmath.gamma(1 - 1j / 2 *w) * hyp1f1(1j / 2 * w, 1, 1j / 2 * w * y**2,100)
            Freal.append(f.real)
            Fimag.append(f.imag)   
    return Freal, Fimag



def Point(w,y):

    w = mpc(w)
    y = mpc(y)
    xm = (y + sqrt(y**2 + 4)) / 2
    phim = (xm - y)**2 / 2 - log(xm)
    expon = exp(pi * w / 4 + 1j * w / 2 * (log(w / 2) - 2 * phim))
    #mp_w = mpmath.mpc(w)
    #mp_y = mpmath.mpc(y)
    f = expon *gamma(1 - 1j / 2 * w) * hyp1f1(1j / 2 * w, 1, 1j / 2 * w * y**2, maxterms=1e+9)
    #Fan.append(complex(f.real, f.imag)) 
    #F = complex(f.real, f.imag)
    return f




def psi(x):
    
    return log(x)
    

def func(x, w, y):
    # x: real integration variable (mpf)
    # w, y, amp, core, p: mpc or mpf
    sqrt2x = sqrt(2 * x)
    arg_bessel = w * y * sqrt2x
    bessel_val = besselj(0, arg_bessel)
    psi_val = psi(sqrt2x)
    return bessel_val * exp(-1j * w * psi_val)

def func2(x, w, y):
    return func(x, w, y) * exp(1j * w * x)

def dfunc(x, w, y):
    sqrt2x = sqrt(2 * x)
    psi_val = psi(sqrt2x)
    J1 = besselj(1, w * y * sqrt2x)
    prefactor = -w * y / sqrt2x
    return prefactor * J1 * exp(-1j * w * psi_val) - (1j * w / (2 * x)) * func(x, w, y)

def ddfunc(x, w, y):
    sqrt2x = sqrt(2 * x)
    psi_val = psi(sqrt2x)
    #denom = (sqrt2x**2 + core**2)**(1 - p/2)
    #dpsi = amp * p * sqrt2x / denom
    
    # derivative squared term:
    #d2psi = amp * p * (1 - p / 2) * (2 * x)**(-0.5) * (1 - 2 * x / (2 * x + core**2)) / (2 * x + core**2)**(1 - p/2)
    
    term1 = (w * y) / (2 * x * sqrt2x) * (2 + 1j * w) * besselj(1, w * y * sqrt2x) * exp(-1j * w * psi_val)
    term2 = -1 / (2 * x) * (w**2 * y**2 - 1j * w / x) * func(x, w, y)
    term3 = -1j * w / (2 * x) * dfunc(x, w, y)
    return term1 + term2 + term3

def TakPoint(w, y):

    #w = mpc(w)
    #y = mpc(y)
    #rs = mpc(rs)
    a = 0.00001

    if w < 1 :
        b = 100 / w
    elif 1 < w < 500:
        b = 1000/w 
    else: b = 10000/w   
    #zzp = mpc(-1)
    
    
    # mpmath.quad with complex integrand
    zz = quad(lambda x: func2(x, w, y), [a, b], error=True, maxdegree=10)
    zz_val = zz[0]  # integral value
    
    # Add tail correction terms (at b)
    tail = (-func(b, w, y) / (1j * w) * exp(1j * w * b)
            - dfunc(b, w, y) / (w**2) * exp(1j * w * b)
            + ddfunc(b, w, y) / (1j * w**3) * exp(1j * w * b))
    
    zz_val += tail
        
    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val
