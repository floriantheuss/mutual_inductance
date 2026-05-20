import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mutual_inductance.mutual_inductance_calibration import MICalibration
from mutual_inductance.calculate_mutual_inductance import CalcMutualInductance
from mutual_inductance.mutual_inductance_data import MIData

path      = str( Path(__file__).absolute() )

# create data filename
name      = '2025-10-31_Al_foil_cooling'
temp      = path.split('/')
temp[-3]  = 'data'
temp[-1]  = f'data/{name}.dat'
data_name = '/'.join(temp)

# create save name
temp      = path.split('/')
# temp[-1]  = f'{name}_calib.dat'
temp[-1]  = f'{name}_calib_no_phase_correction.dat'
save_name = '/'.join(temp)

# load data
dat = np.loadtxt(data_name, skiprows=1)
X = dat[:,12]/10 # the factor of 10 is because on the lockin we had "expand" on, which multiplies everything with a factor of 10
Y = dat[:,13]/10 # the factor of 10 is because on the lockin we had "expand" on, which multiplies everything with a factor of 10
T = dat[:,5]
f = 6e4
V = 3 # drive current voltage
R = 9.93e3 # drop down resistance
I = V/R


# process data	
MI = MICalibration(X,Y,T,f,I)
# MI.perform_calibration(save_name=save_name, phase_correction=True)
MI.perform_calibration(save_name=save_name, phase_correction=False)