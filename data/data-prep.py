# this script uses windspharm to compute streamfunction data out of ERA5 u/v fields to use with the library
# conda environment: windspharm

import iris
import numpy as np 
from windspharm.iris import VectorWind

DATA_DIR = '/p/projects/ou/labs/ai/reanalysis/era5/resolution_025/'
SAVE_DIR = '/p/projects/ou/labs/ai/reanalysis/era5/resolution_025/streamfunction/hourly/'

U_DIR = 'u_component_of_wind/hourly-200-500-800hPa/'
V_DIR = 'v_component_of_wind/hourly-200-500-800hPa/'

U_FILENAME_BASE = 'ERA5-u-200500800hPa-'
V_FILENAME_BASE = 'ERA5-v-200500800hPa-'
PSI_FILENAME_BASE = 'ERA5-sf-200500800hPa-'
years = np.concatenate((range(77,99),range(00,20)))

for iyear in years: 
    # Read u and v wind components from file.
    u = iris.load_cube(DATA_DIR+U_DIR+U_FILENAME_BASE+str(iyear)+'.nc')
    v = iris.load_cube(DATA_DIR+U_DIR+V_FILENAME_BASE+str(iyear)+'.nc')

    # Create an instance of the VectorWind class to do the computations.
    w = VectorWind(u, v)

    # Call methods to compute streamfunction and relative vorticity.
    psi = w.streamfunction()

    iris.save(psi, SAVE_DIR+PSI_FILENAME_BASE+str(iyear)+'.nc')