import numpy as np

C_OMEGA_R = 7.2921151467064e-5  ## rotational angular velocity of the Earth in rad/s
C_SIDDAY = 2 * np.pi / C_OMEGA_R  ## sidereal day in seconds
C_SI = 299792458  ## speed of light in m/s