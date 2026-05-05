import lenstronomy
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lensinggw.constants.constants as const
Msun_Kg         = const.M_sun                             # Msun (Kg)
Mpc             = const.Mpc                               # 1 Megaparsec in meters
arcsec2rad      = np.pi/(648000)                          # conversion factor from arcsec to radian
G               = const.G                                 # gravitational constant (m3*Kg-1*s-2)
c               = const.c  




def TimeDelay(Img_ra, Img_dec, source_pos_x, source_pos_y, zL, zS, lens_model_list, kwargs_lens_list, scaled=False, scale_factor=None, cosmo=None): 

   
    # set a default cosmology if not specified
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM              
        cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # transform the redshift to angular diameter distance
    DL         = cosmo.angular_diameter_distance(zL)
    DS         = cosmo.angular_diameter_distance(zS)
    DLS        = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D          = DLS/(DL*DS)
    D          = np.float64(D/(Mpc))
    prefactor  = 1 #*(1+zL)/(D*c)
    shift2     = (Img_ra-source_pos_x)**2+(Img_dec-source_pos_y)**2
    potential  = 0.0

    # compute the time delay, Eq. 6 of Diego et al https://www.aanda.org/articles/aa/pdf/2019/07/aa35490-19.pdf
    potential = lens_model.potential(Img_ra, Img_dec, kwargs_lens_list)    
    td        = prefactor*(0.5*shift2 - potential)
    
    # if scaled, account for it
    if (scaled):
        if scale_factor is None:
            sys.stderr.write('\n\nMust specify a scale factor to use scaled units\n')
            sys.stderr.write('Aborting\n')
            exit(-1)
        else:
            td *= scale_factor**2
            
    # normalize the time delays so that one of the images has delay '0'
    td = td #-np.min(td)
    
    
        
    return td

####TFunc for histogram counting########
def TFunc(Img_ra, Img_dec, source_pos_x, source_pos_y, zL, zS, lens_model_list, kwargs_lens_list, scaled=False, scale_factor=None, cosmo=None): 

   
    # set a default cosmology if not specified
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM              
        cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # transform the redshift to angular diameter distance
    DL         = cosmo.angular_diameter_distance(zL)
    DS         = cosmo.angular_diameter_distance(zS)
    DLS        = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D          = DLS/(DL*DS)
    D          = np.float64(D/(Mpc))
    prefactor  = 1 #*(1+zL)/(D*c)
    shift2     = (Img_ra-source_pos_x)**2+(Img_dec-source_pos_y)**2
    potential  = 0.0

    # compute the time delay, Eq. 6 of Diego et al https://www.aanda.org/articles/aa/pdf/2019/07/aa35490-19.pdf
    potential = lens_model.potential(Img_ra, Img_dec, kwargs_lens_list)    
    td        = prefactor*(0.5*shift2 - potential)
    
    # if scaled, account for it
    if (scaled):
        if scale_factor is None:
            sys.stderr.write('\n\nMust specify a scale factor to use scaled units\n')
            sys.stderr.write('Aborting\n')
            exit(-1)
        else:
            td *= scale_factor**2
            
    # normalize the time delays so that one of the images has delay '0'
    td = td #-np.min(td)
    
    
        
    return td

def dTFunc(Img_ra, Img_dec, source_pos_x, source_pos_y, zL, zS, lens_model_list, kwargs_lens_list, scaled=False, scale_factor=None, cosmo=None, diff=None): 

    """    
    Computes the gradient of the time delay (in seconds) for a given lens model 
    
    :param Img_ra: images right ascensions (arbitrary units)
    :type Img_ra: array
    :param Img_dec: images declinations (arbitrary units)
    :type Img_dec: array
    :param source_pos_x: source right ascension (arbitrary units)
    :type source_pos_x: float
    :param source_pos_y: source declination (arbitrary units)
    :type source_pos_y: float
    :param zL: lens redshift
    :type zL: float
    :param zS: source redshift
    :type zS: float
    :param lens_model_list: names of the lens profiles to be considered for the lens model
    :type lens_model_list: list of strings
    :param kwargs_lens_list: keyword arguments of the lens parameters matching each lens profile in *lens_model_list*
    :type kwargs_lens_list: list of dictionaries
    :param scaled: specifies if the input is given in arbitrary units, *optional*. If not specified, the input is assumed to be in radians
    :type scaled: bool
    :param scale_factor: scale factor, *optional*. Used to account for the proper conversion factor in the time delays when coordinates are given in arbitrary units, as per :math:`x_{a.u.} = x_{radians}/scale\_factor`. Only considered when *scaled* is *True*
    :type scale_factor: float
    :param cosmo: cosmology used to compute angular diameter distances, *optional*. If not specified, a :math:`\\mathrm{FlatLambdaCDM}` instance with :math:`H_0=69.7, \Omega_0=0.306, T_{cmb0}=2.725` is considered
    :type cosmo: instance of the astropy cosmology class
    :param diff: If set, computes the deflection as a finite numerical differential of the lensing
     potential. This differential is only applicable in the single lensing plane where the form of the lensing
     potential is analytically known. *Optional*, default is *None*
    :type diff: None or float
    
    :returns: :math:`\\frac{\\partial t_d}{\\partial x_0},\\frac{\\partial t_d}{\\partial x_1}`
    :rtype: array, array       
    """
    
    # set a default cosmology if not specified
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM              
        cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
        
    # transform the redshift to angular diameter distance
    DL         = cosmo.angular_diameter_distance(zL)
    DS         = cosmo.angular_diameter_distance(zS)
    DLS        = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D          = DLS/(DL*DS)
    D          = np.float64(D/(Mpc))
    prefactor  = 1 #(1+zL)/(D*c)
    shiftX     = Img_ra-source_pos_x
    shiftY     = Img_dec-source_pos_y
    psiX       = 0.0
    psiY       = 0.0
    
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # overall lensing potential derivatives
    psiX, psiY = lens_model.alpha(Img_ra, Img_dec, kwargs_lens_list, diff=diff)
    
    # time delays derivatives, Eq. 6 of Diego et al https://www.aanda.org/articles/aa/pdf/2019/07/aa35490-19.pdf
    tdX = prefactor*(shiftX - psiX)    
    tdY = prefactor*(shiftY - psiY)

    # if scaled, account for it
    if (scaled):
        if scale_factor is None:
            sys.stderr.write('\n\nMust specify a scale factor to use scaled units\n')
            sys.stderr.write('Aborting\n')
            exit(-1)
        else:
            tdX *= scale_factor
            tdY *= scale_factor

    return tdX, tdY


####Morse phase
def getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff=None):

    """
    from lensinggw python package
    to determine if images are minima, maxima or saddle points of the time delay
    
    :param Img_ra: images right ascensions (arbitrary units)
    :type Img_ra: array
    :param Img_dec: images declinations (arbitrary units)
    :type Img_dec: array
    :param lens_model_list: names of the lens profiles to be considered for the lens model
    :type lens_model_list: list of strings
    :param kwargs_lens_list: keyword arguments of the lens parameters matching each lens profile in *lens_model_list*
    :type kwargs_lens_list: list of dictionaries
    :param diff: If set, computes the deflection as a finite numerical differential of the lensing
     potential. This differential is only applicable in the single lensing plane where the form of the lensing
     potential is analytically known. *Optional*, default is *None*
    :type diff: None or float
        
    :returns: list of Morse indices matching the images; :math:`0,1,1/2` if images are minima, maxima or saddle points respectively
    :rtype: list
    
    Raises
    ------
    StandardError
        If the test is inconclusive
    """       
    
    # store the results of the test here
    res = []
    
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # second derivative test
    for i in range(len(Img_ra)):
        f_xx, f_xy, f_yx, f_yy = lens_model.hessian(Img_ra[i], Img_dec[i], kwargs_lens_list, diff=diff)
        td_xx = 1-f_xx     # derivatives of the time delay, Eq. (12) in https://iopscience.iop.org/article/10.1086/377430/fulltext/ 
        td_yy = 1-f_yy     
        td_xy = -f_xy
        td_yx = -f_yx
        D = td_xx*td_yy-td_xy*td_yx
        if (D<0):          # it's a saddle point
            res.append(0.5)
        elif (D>0):        
            if (td_xx>0):   # it's a minimum
                res.append(0)
            elif (td_xx<0): # it's a maximum
                res.append(1)
            else:
                sys.stderr.write("\n\nInconclusive test for image # {0}\n".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
        else:              # inconclusive test
            if (td_xx>0) or (td_yy>0):
                sys.stderr.write("\n\nInconclusive test for image # {0}. But it is not a maximum\n".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
            elif (td_xx<0) or (td_yy<0):
                sys.stderr.write("\n\nInconclusive test for image # {0}. But it is not a minimum\n".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
            else:
                sys.stderr.write("\n\nInconclusive test for image # {0}. No possibility can be ruled out.".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
                
    return res



####Morse phase_flag
def getMinMaxSaddle_flag(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff=None):

    """
    from lensinggw python package
    to determine if images are minima, maxima or saddle points of the time delay
    
    :param Img_ra: images right ascensions (arbitrary units)
    :type Img_ra: array
    :param Img_dec: images declinations (arbitrary units)
    :type Img_dec: array
    :param lens_model_list: names of the lens profiles to be considered for the lens model
    :type lens_model_list: list of strings
    :param kwargs_lens_list: keyword arguments of the lens parameters matching each lens profile in *lens_model_list*
    :type kwargs_lens_list: list of dictionaries
    :param diff: If set, computes the deflection as a finite numerical differential of the lensing
     potential. This differential is only applicable in the single lensing plane where the form of the lensing
     potential is analytically known. *Optional*, default is *None*
    :type diff: None or float
        
    :returns: list of Morse indices matching the images; :math:`0,1,1/2` if images are minima, maxima or saddle points respectively
    :rtype: list
    
    Raises
    ------
    StandardError
        If the test is inconclusive
    """       
    
    # store the results of the test here
    res = []
    
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # second derivative test
    for i in range(len(Img_ra)):
        f_xx, f_xy, f_yx, f_yy = lens_model.hessian(Img_ra[i], Img_dec[i], kwargs_lens_list, diff=diff)
        td_xx = 1-f_xx     # derivatives of the time delay, Eq. (12) in https://iopscience.iop.org/article/10.1086/377430/fulltext/ 
        td_yy = 1-f_yy     
        td_xy = -f_xy
        td_yx = -f_yx
        D = td_xx*td_yy-td_xy*td_yx
        if (D<0):          # it's a saddle point
            res.append('saddle')
        elif (D>0):        
            if (td_xx>0):   # it's a minimum
                res.append('min')
            elif (td_xx<0): # it's a maximum
                res.append('max')
            else:
                sys.stderr.write("\n\nInconclusive test for image # {0}\n".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
        else:              # inconclusive test
            if (td_xx>0) or (td_yy>0):
                sys.stderr.write("\n\nInconclusive test for image # {0}. But it is not a maximum\n".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
            elif (td_xx<0) or (td_yy<0):
                sys.stderr.write("\n\nInconclusive test for image # {0}. But it is not a minimum\n".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
            else:
                sys.stderr.write("\n\nInconclusive test for image # {0}. No possibility can be ruled out.".format(i))
                sys.stderr.write("Aborting\n")
                exit(-1)
                
    return res
    
def amplification_from_data(frequencies, mu, td, n):
    """
    Computes the geometrical optics amplification :math:`F(f)` over a frequency band, given a set of magnifications, time delays and Morse indices
    
    :param frequencies: frequency band for the computation of the amplification factor
    :type frequencies: array
    :param mu: images magnifications
    :type mu: indexable object
    :param td: images time delays
    :type td: indexable object
    :param n: Morse indices
    :type n: indexable object
    
    :returns: :math:`F(f)`
    :rtype: array
    """
    
    Fmag = np.zeros(len(frequencies), dtype=np.complex128)
    for i in range(len(mu)):
        Fmag += np.sqrt(np.abs(mu[i]))* np.exp(1j*np.pi*(2.*frequencies*td[i] - n[i]))
    return Fmag    