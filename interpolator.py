import configparser
import os
import numpy as np
from lens import DimensionlessFrequency,Point
import nfw
import utilities 
import matplotlib.pyplot as plt
import shutil


#Read the config file
config = configparser.ConfigParser()
config.read('parameters.ini')

#Reading Interpolation Data

#Valid lens models
valid_lens_models = ['point', 'nfw']


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

# Validate lens model
if lensmodel not in valid_lens_models:
    raise KeyError(f"Invalid lens model specified: '{lensmodel}'. Valid options are {valid_lens_models}")

#Dimensional_frequency
w_lower = float(config['Interpolation']['dimensionless_frequency_lower_limit'])
w_upper = float(config['Interpolation']['dimensionless_frequency_upper_limit'])
w_samples = int(config['Interpolation']['dimensionless_frequency_samples'])


y_lower = float(config['Interpolation']['impact_parameter_lower_limit'])
y_upper = float(config['Interpolation']['impact_parameter_upper_limit'])
y_samples = int(config['Interpolation']['impact_parameter_samples'])

if lensmodel == "nfw":

    rs_lower = float(config['Interpolation']['scale_radius_lower_limit'])
    rs_upper = float(config['Interpolation']['scale_radius_upper_limit'])
    rs_samples = int(config['Interpolation']['scale_radius_samples'])
    rs = np.linspace(rs_lower,rs_upper,rs_samples) 



w = np.linspace(w_lower,w_upper,w_samples)
y = np.linspace(y_lower,y_upper,y_samples)


if lensmodel == "point":
    print(f"Lens : {lensmodel}")
    ampf = Point
    utilities.compute_and_store_results_2D(output_path,w,y,ampf)


elif lensmodel == "nfw":
    print(f"Lens : {lensmodel}")
    ampf = nfw.NFW
    utilities.compute_and_store_results_3D(interpolator_output_filename,w,y,rs,ampf)




