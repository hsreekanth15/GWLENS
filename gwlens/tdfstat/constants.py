import numpy as np
import math
import lal
from scipy.constants import c,G

C_OMEGA_R = 7.2921151467064e-5  ## rotational angular velocity of the Earth in rad/s
C_SIDDAY = 2 * np.pi / C_OMEGA_R  ## sidereal day in seconds
C_SI = 299792458  ## speed of light in m/s
EPSILON = 0.40909280422232891          # obliquity of the ecliptic [rad]
SIDEREAL_DAY = 86164.09890369719       # seconds
C_SI = lal.C_SI                        # 299792458.0 m/s
_DEG = 180.0 / math.pi                 # the C 'deg' factor (radians per ... see usage)

M_sun = 1.98e+30
dimensionless_frequency_constant = (8*np.pi*G)/c**3 * M_sun