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
    x = Ds/Dl
    return vS - x*vL + (x-1)*vO
    
def MicrolensingImpactparameter(t,y0,t0,tE):
    x = (t-t0)/tE
    y = np.sqrt(y0**2 + x**2)
    return y    
 