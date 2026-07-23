"""
author: Sreeakanth Harikumar
"""

import numpy as np
from astropy.constants import c,G,M_sun,kpc


#Define constants
c = c.value
G = G.value
M_sun = M_sun.value


def EinsteinCrossingTime(Dl,Ds,v,M):
    """
    Computes the Einstein Crossing time used in Microlensing analysis

    Parameters:
    
    Dl :  Distance to lens in kpc
    Ds : Distance to lens in kpc
    v :  velocity of the lens (not relative velocity) in 1 km/s
    M :  Mass of the lens in Solar Mass
    
    returns: einstein crossing time in days
    """
    prefactor = np.sqrt(1*kpc.value)*np.sqrt(M_sun)*  1/(1000)
    oneday = 60*60*24
    x = Dl/Ds
    te = (1/oneday ) * prefactor*np.sqrt(G/c**2) * np.sqrt(4*x * (1 -x )) * np.sqrt(Ds) * np.sqrt(M) * (v)**(-1)
    return te

def RelativeVelocity(Dl,Ds,vS,vL,vO):

    """
    Computes the relative velocity of the lensing system (Source, Lens and Observer)
    Based on eq.10 in https://doi.org/10.3847/1538-4357/ab1087 

    Parameters:

    Dl: Distance to the lens in kpc
    Ds: Distance to source in kpc
    vs: Velocity of the source
    vL: Velocity of the lens
    vO:  Velocity of the observer

    Returns the relative velocity in same units as the velocities of the source, lens and observer
    """

    x = Ds/Dl
    return vS - x*vL + (x-1)*vO
    
def MicrolensingImpactparameter(t,y0,t0,tE):
    """
    Computes the Microlensing impact parameter based on Einstein crossing time

    Parameters:

    tE :  The Einstein Crossing Time in days
    t: time in days
    y0:  Closest impact parameter
    t0: Day on which the maximum aplification is observed 
 
    Returns: An array of dimensionless impact parameter values. Spans in the range [-0.5,0.5]. The extreme
    values correspond to the left and right edges of the Einstein Ring.

    """
    x = (t-t0)/tE
    y = np.sqrt(y0**2 + x**2)
    return y    
 