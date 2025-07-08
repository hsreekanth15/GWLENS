# GWLENS-  Gravitational Wave Lensing Exploration of Non-visible Structures

Author: Sreekanth Harikumar

The package can be installed by downloading the entire code and using pip

`cd GWLENS`

`pip install .`

GWLENS is completely written in  python and uses arbitrary precision library 'mpmath' to solve 
the Fersnal Kirchhoff diifraction integral accurately. GWLENS can also used with an .ini file (parameters.ini).
To generate such an .ini file  simply run

`gwlens_generate_default_ini`

This .ini file can be used for all puporses (such as the solving the amplification factor for short transient signals and 
long duration signals from pulsars). 

To solve the amplifaction factor for a particular lens model, edit appropriate sections in the parameters.ini file generated and 
provide the range of values for the impact parameter $y$ and the dimensionless frequencies $w$. Amplification factor can now be produced by running the command

`gwlens_interpolate parameters.ini`

This will produce and output with the name you provided and the interpolation file in hdf5 format. The can also be found in the same directory for troubleshooting.
GWLENS also support parallel processing, this can be enabled by simply providing the number of cpus in the parmaters.ini file ( for ex. `n-paralle= 8` ). This will speed up the
amplification factor evaluation.

GWLENS also comes with lensed waveforms which can be used with the popular paramter estimation tool `bilby`. The waveforms are located in the pe/waveform folder..


