# Processes a calibration measurement taken on a thick superconducting film (λ ≪ d_film).
# At low T the film acts as a perfect magnetic screen, which lets us extract two corrections
# needed for every sample measurement:
#   1. phase mixing angle  — cable/capacitive coupling rotates the X/Y lock-in channels
#   2. field leakage       — flux that bypasses the sample and reaches the pickup coil
# Both values are saved together with the corrected MI curves.

import os
import sys
# The line below makes this script runnable from any working directory without installing the package.
# In your own analysis code, add the repo root to your PYTHONPATH instead
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mutual_inductance.mutual_inductance_calibration import MICalibration

here = os.path.dirname(os.path.abspath(__file__))

name      = 'calibration_data_example'
data_name = os.path.join(here, 'example_data', f'{name}.dat')
save_name = os.path.join(here, f'{name}_calib.dat')

# columns: T (K), X (V), Y (V)  — X/Y are the in-phase and quadrature lock-in outputs
dat = np.loadtxt(data_name, skiprows=1)
T = dat[:,0]
X = dat[:,1]
Y = dat[:,2]

f = 6e4       # lock-in reference frequency (Hz)
V = 3         # drive voltage (V)
R = 9.93e3    # series resistance (Ω) — sets the drive current
I = V/R

MI = MICalibration(X, Y, T, f, I)
# phase_correction=True: let the calibration routine determine (and correct for) the phase angle from the low-T SC state;
MI.perform_calibration(save_name=save_name, phase_correction=True)
