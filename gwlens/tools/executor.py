import configparser
import argparse
import os
import numpy as np
from gwlens.lens import point
from gwlens import microlensing
from . import utilities
import matplotlib.pyplot as plt
import shutil




def execute():

    parser = argparse.ArgumentParser(description="Generate microlensing data")
    parser.add_argument("config", help="Path to the parameters .ini file")
    args = parser.parse_args()

        # Read config
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)



    #EinsteinCrossingTime calculation
    Dl = float(config['Lens']['Dl'])
    Ds = float(config['Source']['Ds'])
    v = float(config['Microlens']['v'])
    M = float(config['Lens']['lensmass'])
    tE = microlensing.EinsteinCrossingTime(Dl,Ds,v,M)
    print("Einstein Crossing Time",tE)

    #ImpactParameter calculation

    y0 = float(config['Microlens']['closest_approach'])
    t0 = float(config['Microlens']['Time_of_closest_approach'])
    Tobs = float(config['Microlens']['Tobs'])
    samples = float(config['Microlens']['Number_of_samples'])
    time = Tobs/2
    t = np.linspace(-time, time, 1000)
    ImpactParameter = microlensing.MicrolensingImpactparameter(t,y0,t0,tE)

    #Amplification Factor

    f = float(config['Source']['frequency'])

    w = point.DimensionlessFrequency(M,f)
    print("w",w)
    y = ImpactParameter

    result_F_real = np.array(point.Point(w, y)[0])
    result_F_imag = np.array(point.Point(w, y)[1])

    #Save results
    output_folder = config['Output']['outdir']

    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Copy the configuration file to the output folder
    ini_copy_path = os.path.join(output_folder, 'parameters.ini')
    shutil.copy('parameters.ini', ini_copy_path)



    output_filename_part = config['Lens']['lensmodel']
    specified_word = "amplification_factor"  # The word specified in the code
    final_filename_real = f"{output_filename_part}_lens_{specified_word}_real.txt"
    final_filename_imag = f"{output_filename_part}_lens_{specified_word}_imag.txt"

    output_path = os.path.join(output_folder, final_filename_real)
    np.savetxt(output_path, result_F_real)
    output_path = os.path.join(output_folder, final_filename_imag)
    np.savetxt(output_path, result_F_imag)
    output_path = os.path.join(output_folder,"source_position.txt")
    np.savetxt(output_path,y)
    output_path = os.path.join(output_folder,"dimensionless_frequency.txt")
    w = utilities.ensure_1d_array(w)
    np.savetxt(output_path,w)

    #plots

    generate_plots = config['Plots']['generate_plots'].strip().lower() == 'true'

    if generate_plots:
        F = result_F_real + 1j*result_F_imag
        F_absolute = np.abs(F)
        plt.plot(t,F_absolute, color= "red", ls="dotted", label=f"w = {np.around(w[0],2)}")
        plt.title("Microlensing Amplfication")
        plt.grid()
        plt.xlabel("Time (Days)")
        plt.ylabel("$F(t)$")
        plt.legend()


        output_path = os.path.join(output_folder, "point.png")
        plt.savefig(output_path,dpi=300)