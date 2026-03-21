#author : Sreekanth Harikumar
#A class for Primodial DM mini halo 
# See https://arxiv.org/pdf/2303.17601 and references there in for more details


import numpy as np
from .base import Lens
from astropy import units as u
from general import amplification
from mpmath import log,exp


class PBH(Lens):

    def __init__(self,m_PBH,f_PBH):
        
        """
        Parameters
        ------------

        m_PBH :  Primodial Black Hole mini halo mass (in solar units)
        f_PBH :  dark matter fraction

        """

        self.m_PBH = m_PBH
        self.f_PBH = f_PBH


    def proper_size(self):

        """
        Returns propersize of the halo in pc
        """

        return 14* self.m_PBH**(1/3) * min(1, 1e-3 * self.f_PBH**(-4/3))*(u.parsec)

    def minihalo_mass(self):

        return (1-self.f_PBH)*self.m_PBH * min(50,0.28/self.f_PBH)

    def PBH_EinsteinRadius(self,Dl,Ds,Dls):

        Deff = Dl*Ds/Dls

        xi0 = sqrt(4*G*self.m_PBH/c**2  * Deff)

        return xi0

    def point_psi(x):

        return log(x)

    def psi_halo(self,x):

        const = 12*(1-self.f_PBH)*(self.m_PBH)**(1/4) * (self.Rh)**(3/4)

        if x < self.Rh:
           pot =  (x/self.Rh)**(3/4)

        elif x > self.Rh:
           pot = 1 + 3/4 * log(x/)

        return const*pot

    def psi(self,x):


        return self.point_psi(x) + self.psi_halo(self,x)
    
    







