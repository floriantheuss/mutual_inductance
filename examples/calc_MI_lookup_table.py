import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mutual_inductance.mutual_inductance_calibration import MICalibration
from mutual_inductance.calculate_mutual_inductance import CalcMutualInductance
from mutual_inductance.mutual_inductance_data import MIData
import time

############################################################
# coil parameters
############################################################
coilDiam = 8 * 2.54e-5 # inner diameter of inner most loop
d_wire    = 19e-6 # diameter of wire
rmin     = (coilDiam+d_wire)/2

# drive coil
nTurnsD = 50 # total number of turns in coil (i.e. n_row*n_col)
rat     = 0.28 # the ratio n_col/n_row
n_dl    = int(np.round(np.sqrt(nTurnsD/rat)))
n_dt    = int(np.round(np.sqrt(nTurnsD*rat)))
r_d     = rmin

# pickup coil
nTurnsP = 400
rat     = 0.622  # This is width / height
n_pl    = int(np.round(np.sqrt(nTurnsP/rat)))
n_pt    = int(np.round(np.sqrt(nTurnsP*rat)))
r_p     = rmin


########################################################################################################################
########################################################################################################################
# first fit the distance between the coils so that the resulting mutual inductance is what is measured in the calibration in the normal state
calib_name = f'2025-10-31_Al_foil_cooling_calib.dat'
path       = str( Path(__file__).absolute() )
temp       = path.split('/')
temp[-1]   = calib_name
calib_file = '/'.join(temp)
dat = np.loadtxt(calib_file, delimiter=',')
T = dat[:,0]
MIx = dat[:,-2]
Tmin = 2.5
Tmax = 5
mask = (T>Tmin)&(T<Tmax)
MIx_ave = np.mean(MIx[mask])
plt.figure()
plt.plot(T, MIx)
print()
print('this is the calibration curve you have chosen;')
print('normal state value is evalulated as the average of the data between ', Tmin, ' K and ', Tmax, ' K;')
print('the average mutual inductance value is: ', MIx_ave)
print()
plt.show()



h = 1.07e-3
d_film = 0 # set to zero, mimicking nothing in between the coils
MI = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
MI.fit_coil_distance(MIx_data=MIx_ave)
h = MI.h
print(h)


########################################################################################################################
########################################################################################################################
# try to fit some extreme values of the measured mutual inductance to see where to put the bounds for the lookup table

# d_film = 5e-9  # Sample film thickness
# h = 0.0010339878004774965 # coil distance from fit above
# MI = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
# MI.fit_penetration_depth (MIx_data=1e-12, lam_guess=1e-6, ray_fit=False)


########################################################################################################################
########################################################################################################################
# create lookup table of mutual inductance as a fct of penetration depth with numbers from above
PDList = np.linspace(5e-9, 5e-3, 500)

print('h is: ', h)

t1 = time.time()
d_film = 5.1e-9  # Sample film thickness
save_name  = f'lookup_table_{np.round(d_film*1e9,1)}nm_{calib_name}'
temp[-1]  = save_name
save_name = '/'.join(temp)
MI = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
m,mn = MI.calc_MI_PD_array(PDList, save_name)
print()
print('calc for ', np.round(d_film*1e9,1), ' done in ', time.time()-t1, ' seconds')


plt.figure()
plt.scatter(PDList, m)


plt.figure()
plt.scatter(PDList, mn)
plt.show()