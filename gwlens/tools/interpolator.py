import configparser
import argparse
import os
import numpy as np
from mpmath import mp
from gwlens.lens import point,sis,nfw,powerlaw,dpbh
from gwlens.tools import halo
from gwlens.tools import logutils
import matplotlib.pyplot as plt


def make_grid(lower, upper, samples, gridspace):
    gridspace = gridspace.lower()

    if gridspace == "linear":
        return np.linspace(lower, upper, samples)

    elif gridspace == "log":
        if lower <= 0 or upper <= 0:
            raise ValueError("Log-spaced grids require lower and upper limits > 0.")
        return np.logspace(np.log10(lower), np.log10(upper), samples)

    else:
        raise ValueError(
            f"Unknown gridspace '{gridspace}'. Use 'linear' or 'log'."
        )



def interpolate():


    parser = argparse.ArgumentParser(description="Generate interpolation data")
    parser.add_argument("config", help="Path to the parameters .ini file")
    args = parser.parse_args()

        # Read config
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)


    #Reading Interpolation Data

    #Valid lens models
    valid_lens_models = ['point', 'sis' ,'nfw', 'powerlaw','dpbh']

    n_parallel = int(config['Interpolation']['n-parallel'])

    #Store result
    output_folder = config['Output']['outdir']

    # Subfolder for interpolation outputs
    interpolation_folder = os.path.join(output_folder, 'interpolation')
    os.makedirs(interpolation_folder, exist_ok=True)  # Create if it doesn't exist

    interpolator_output_filename = config['Interpolation']['filename']

    if not interpolator_output_filename.lower().endswith('.hdf5'):
        interpolator_output_filename += '.hdf5'

    output_path = os.path.join(interpolation_folder ,interpolator_output_filename)
    #Lens_Model
    lensmodel = config['Interpolation']['lensmodel']
    
    logger = logutils.setup_logger(output_folder, logger_name="interpolation_logger")

    # Validate lens model
    if lensmodel not in valid_lens_models:
        raise KeyError(f"Invalid lens model specified: '{lensmodel}'. Valid options are {valid_lens_models}")
    
    Ds  = float(config['Source']['Ds'])
    Dls = float(config['Lens']['Dls'])
    Dl  = float(config['Lens']['Dl'])

    
    

    

    logger.info(f"Using Lens model: {lensmodel}")
    logger.info(f"Distances: Dl={Dl} kpc, Dls={Dls} kpc, Ds={Ds} kpc")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Number of parallel processes: {n_parallel}")

    w_lower = float(config['Interpolation']['dimensionless_frequency_lower_limit'])
    w_upper = float(config['Interpolation']['dimensionless_frequency_upper_limit'])
    w_samples = int(config['Interpolation']['dimensionless_frequency_samples'])
    w_gridspace = config['Interpolation']['dimensionless_frequency_gridspace']


    y_lower = float(config['Interpolation']['impact_parameter_lower_limit'])
    y_upper = float(config['Interpolation']['impact_parameter_upper_limit'])
    y_samples = int(config['Interpolation']['impact_parameter_samples'])
    y_gridspace = config['Interpolation']['impact_parameter_gridspace']

    if lensmodel == "nfw":

        rs_lower = float(config['Interpolation']['scale_radius_lower_limit'])
        rs_upper = float(config['Interpolation']['scale_radius_upper_limit'])
        rs_samples = int(config['Interpolation']['scale_radius_samples'])
        rs_gridspace =  config['Interpolation']['rs_gridspace']
        rs = make_grid(rs_lower,rs_upper,rs_samples,rs_gridspace)

        logger.info(f"NFW parameters: scale radius range=({rs_lower}, {rs_upper}), samples={rs_samples}") 
        logger.info(f"Using {rs_gridspace} gridspacing for rs")
    
    elif lensmodel == "powerlaw":

        amp = float(config['Interpolation']['powerlaw_amplitude'])
        core = float(config['Interpolation']['powerlaw_coreradius'])
        p = float(config['Interpolation']['powerlaw_exponent'])

        logger.info(f"Power-law parameters: amplitude={amp}, core radius={core}, exponent={p}")

    elif lensmodel == "dpbh":

        f_pbh_lower = float(config['Interpolation']['f_pbh_lower_limit'])
        f_pbh_upper = float(config['Interpolation']['f_pbh_upper_limit'])
        f_pbh_samples = int(config['Interpolation']['f_pbh_samples'])
        f_pbh_gridspace = config['Interpolation']['f_pbh_gridspace']
   
        f_pbh = make_grid(f_pbh_lower,f_pbh_upper,f_pbh_samples,f_pbh_gridspace)

        m_pbh_lower = float(config['Interpolation']['m_pbh_lower_limit'])
        m_pbh_upper = float(config['Interpolation']['m_pbh_upper_limit'])
        m_pbh_samples = int(config['Interpolation']['m_pbh_samples'])
        m_pbh_gridspace =  config['Interpolation']['m_pbh_gridspace']
        m_pbh = make_grid(m_pbh_lower,m_pbh_upper,m_pbh_samples,m_pbh_gridspace)

        logger.info(f"PBH parameters: f_PBH range=({f_pbh_lower}, {f_pbh_upper}), m_PBH range=({m_pbh_lower}, {m_pbh_upper})")

    
    mp.dps = float(config['Interpolation']['precision'])
    logger.info(f"Interpolation precision set to {mp.dps} decimal places")

    w = make_grid(w_lower,w_upper,w_samples,w_gridspace)
    y = make_grid(y_lower,y_upper,y_samples,y_gridspace)


    if lensmodel == "point":
        print(f"Lens : {lensmodel}")
        point.compute_point_lens_grid(w, y, output_path, num_processes=n_parallel)

    elif lensmodel == "sis":
        print(f"Lens: {lensmodel}")
        sis.compute_sis_lens_grid(w,y,output_path, num_processes=n_parallel)    


    elif lensmodel == "nfw":
        print(f"Lens : {lensmodel}")
        nfw.compute_nfw_lens_test(w,y,rs,output_path,num_jobs=n_parallel)

    elif lensmodel == "powerlaw":
        print(f"Lens : {lensmodel}")
        powerlaw.compute_powerlaw_lens_grid(w,y,amp,core,p,output_path,num_processes = n_parallel)

    elif lensmodel == "dpbh":
        print(f"Lens: {lensmodel} ")
        dpbh.compute_pbh_lens_grid(w,y,f_pbh,m_pbh,Dl,Ds,Dls,output_path,num_processes = n_parallel)




