# this script uses windspharm to compute streamfunction data out of ERA5 u/v fields to use with the library
# conda environment: windspharm
#
# cdo command for pre-processing: cdo remapbil,n32 ifile ofile
# for i in $(ls); do cdo remapbil,n32 ${i} regrid/${i}-t42.nc; done
#
# then get a interative job on the hpc for this script (it needs more than 4GB RAM): 
# srun --qos=priority --ntasks-per-node=14 --pty bash 
#
# merge final result with cdo 
# cdo mergetime *nc outfile.nc

import iris
import numpy as np 
from windspharm.iris import VectorWind

DATA_DIR = '/p/projects/ou/labs/ai/reanalysis/era5/T42/'
SAVE_DIR = '/p/projects/ou/labs/ai/reanalysis/era5/T42/streamfunction/hourly/'

U_DIR = 'u_component_of_wind/hourly/'
V_DIR = 'v_component_of_wind/hourly/'

U_FILENAME_BASE = 'ERA5-u-200500800hPa-'
V_FILENAME_BASE = 'ERA5-v-200500800hPa-'
PSI_FILENAME_BASE = 'ERA5-sf-200500800hPa-'
#years = np.concatenate((range(79,100,range(00,21)))
years = np.concatenate((range(99,100),range(00,21)))

for iyear in years: 
    # Read u and v wind components from file.
    u = iris.load_cube(DATA_DIR+U_DIR+U_FILENAME_BASE+f'{iyear:02d}'+'.nc-t42.nc')
    v = iris.load_cube(DATA_DIR+V_DIR+V_FILENAME_BASE+f'{iyear:02d}'+'.nc-t42.nc')

    # Create an instance of the VectorWind class to do the computations.
    w = VectorWind(u, v)

    # Call methods to compute streamfunction and relative vorticity.
    psi = w.streamfunction()

    iris.save(psi, SAVE_DIR+PSI_FILENAME_BASE+str(iyear)+'.nc')