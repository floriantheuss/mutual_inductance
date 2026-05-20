import os
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
name      = '2025-09-03_LSNO_M1_2C_warming'
temp      = path.split('/')
temp[-3]  = 'data'
temp[-1]  = f'data/{name}.dat'
data_name = '/'.join(temp)

dat = np.loadtxt(data_name, skiprows=1)
X = dat[:,12]/10
Y = dat[:,13]/10
T = dat[:,5]
f = 6e4
V = 3 # drive current voltage
R = 9.93e3 # drop down resistance
I = V/R


######################################################
# calib
leakage = 9.66893011248539e-10
angle   = 1.603212699619901

lookup_table_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../0_Al_calibration_measurements/analysis')
lookup_table_file1 = '20251031_Al_foil_2.5x5_calibration'
lookup_table_file2 = 'lookup_table_6.8nm_2025-10-31_Al_foil_cooling_calib.dat'
lookup_table_path = f'{lookup_table_folder}/{lookup_table_file1}/{lookup_table_file2}'

save_name = f'{name}_processed_20251031_calib.dat'
path      = str( Path(__file__).absolute() )
temp      = path.split('/')
temp[-1]  = save_name
save_name = '/'.join(temp)

MI = MIData(X,Y,T,f,I,leakage=leakage, correction_angle=angle, residual_capacitance=0)
MI.process_data(Tmin=11, save_name=save_name, n=100)
MI.convert_MI_to_PD (lookup_table_path, save_path=save_name)