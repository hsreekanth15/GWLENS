"""
WIP

"""



import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def convergence_shear(lens_model,x,y,kwargs_lens,diff=None):

    f_xx, f_xy, f_yx, f_yy = lens_model.hessian(x,y,kwargs_lens, diff=diff)
    kappa = 0.5* (f_xx + f_yy)
    gamma1 =  0.5*(f_xx - f_yy)
    gamma2 = 0.5* (f_xy + f_yx)

    return kappa, gamma1,gamma2

def shear_frame(kappa, gamma1, gamma2):
    """
    In Shear frame after rotation    
    Shear amplitude, angle and the eigenvalues of the magnification matrix.
    
    """
    gamma = np.hypot(gamma1, gamma2)
    phi = 0.5 * np.arctan2(gamma2, gamma1)          # rotation angle [rad]
    lam_t, lam_r = 1 - kappa - gamma, 1 - kappa + gamma
    mu = 1.0 / (lam_t * lam_r)
    return gamma, phi, lam_t, lam_r, mu
 
def image_type(lam_t, lam_r):
    if lam_t > 0 and lam_r > 0:
        return "minimum (type I)"
    if lam_t * lam_r < 0:
        return "saddle  (type II)"
    return "maximum (type III)"
 

lens_model_list = ["SIE", "SHEAR"]
kwargs_lens = [
    {"theta_E": 1.0, "e1": 0.1, "e2": -0.05, "center_x": 0.0, "center_y": 0.0},
    {"gamma1": 0.03, "gamma2": 0.01, "ra_0": 0.0, "dec_0": 0.0},
    ]

zl, zs = 0.5, 2.0
beta_x, beta_y = 0.05, 0.03   

lens_model = LensModel(lens_model_list, z_lens=zl, z_source=zs, cosmo=cosmo)
solver = LensEquationSolver(lens_model)
x1_img, x2_img = solver.image_position_from_source(beta_x, beta_y, kwargs_lens, min_distance=0.01, search_window=5,precision_limit=1e-10, num_iter_max=100)
 
kappa,gamma1,gamma2 = convergence_shear(lens_model, x1_img, x2_img,kwargs_lens)

dt = lens_model.arrival_time(x1_img, x2_img, kwargs_lens)  
dt -= dt.min()