# Generates the theoretical MI(λ) lookup table used to convert normalised MI → penetration depth.
# Two steps:
#   1. Fit the coil separation h to the measured normal-state MI from the calibration run,
#      so that the model geometry exactly matches the physical setup.
#   2. Sweep λ over the expected range and compute MI/MI_∞ for each value, saving the result
#      as a lookup table that process_sample_data.py reads via interpolation.
# Run process_calibration_data.py first — its output file is read here.

import os
import sys
# The line below makes this script runnable from any working directory without installing the package.
# In your own analysis code, add the repo root to your PYTHONPATH instead
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from mutual_inductance.calculate_mutual_inductance import CalcMutualInductance
import time

here = os.path.dirname(os.path.abspath(__file__))

############################################################
# coil geometry parameters
############################################################
coilDiam = 8 * 2.54e-5  # inner diameter of the innermost loop (m)
d_wire   = 19e-6        # wire diameter (m)
rmin     = (coilDiam + d_wire) / 2  # radius to the centre of the innermost turn

# drive coil: total turns = n_dl * n_dt
nTurnsD = 50
rat     = 0.28           # aspect ratio n_turns/n_layers
n_dl    = int(np.round(np.sqrt(nTurnsD / rat)))   # number of layers
n_dt    = int(np.round(np.sqrt(nTurnsD * rat)))   # turns per layer
r_d     = rmin

# pickup coil: total turns = n_pl * n_pt
nTurnsP = 400
rat     = 0.622          # aspect ratio (width / height)
n_pl    = int(np.round(np.sqrt(nTurnsP / rat)))
n_pt    = int(np.round(np.sqrt(nTurnsP * rat)))
r_p     = rmin


############################################################
# step 1: fit coil separation to the normal-state MI
############################################################
# Load the corrected MI_x curve from the calibration run and take its average
# in the normal state (T > Tc) as the target value for the fit.
calib_name = 'calibration_data_example_calib_no_phase_correction.dat'
calib_file = os.path.join(here, calib_name)
dat  = np.loadtxt(calib_file, delimiter=',')
T    = dat[:, 0]
MIx  = dat[:, -2]   # MIx_corrected is the second-to-last column

Tmin = 2.5   # temperature range for the normal-state average (K)
Tmax = 5
mask    = (T > Tmin) & (T < Tmax)
MIx_ave = np.mean(MIx[mask])

plt.figure()
plt.plot(T, MIx)
print()
print('calibration curve loaded; normal-state MI averaged between', Tmin, 'K and', Tmax, 'K')
print('target MI_x value:', MIx_ave)
print()
plt.show()

# d_film=0 removes the film from the model so we fit purely to the coil geometry
h      = 1.07e-3   # initial guess for coil separation (m)
d_film = 0
MI = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
MI.fit_coil_distance(MIx_data=MIx_ave)
h = MI.h   # fitted coil separation — used for the lookup table below
print('fitted coil separation (m):', h)


############################################################
# step 2: compute MI(λ) lookup table for the sample film
############################################################
# Adjust PDList bounds if the measured MI_x_norm falls outside [min, max] of this range.
PDList = np.linspace(5e-9, 5e-3, 500)

t1     = time.time()
d_film = 5.1e-9   # sample film thickness (m) — update for each sample
save_name = os.path.join(here, f'lookup_table_{np.round(d_film*1e9, 1)}nm_{calib_name}')

MI = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
m, mn = MI.calc_MI_PD_array(PDList, save_name)
print()
print(f'lookup table for {np.round(d_film*1e9, 1)} nm film done in {time.time()-t1:.1f} s')

plt.figure()
plt.xlabel('Penetration depth (m)')
plt.ylabel('MI (H)')
plt.scatter(PDList, m)

plt.figure()
plt.xlabel('Penetration depth (m)')
plt.ylabel('MI / MI$_\\infty$')
plt.scatter(PDList, mn)
plt.show()
