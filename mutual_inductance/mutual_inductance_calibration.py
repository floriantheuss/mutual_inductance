import matplotlib.pyplot as plt
from scipy import special, integrate, constants
import numpy as np
import itertools
from time import time
from copy import deepcopy
import ray
from psutil import cpu_count


# Processes a calibration measurement (thick superconductor, lam << d_film) to extract
# the phase mixing angle and magnetic-field leakage used to correct subsequent sample data.
class MICalibration:
	def __init__ (self, X, Y, T, f, I):
		# sort according to temperature
		# mask = np.argsort(T)
		# real and imaginary part of the calibration data
		# self.X = X[mask]
		# self.Y = Y[mask]
		# temperature
		# self.T = T[mask]

		self.X, self.Y, self.T = X, Y, T
		if np.mean(self.T[:10]) > np.mean(self.T[-10:]):
			self.X, self.Y, self.T = self.X[::-1], self.Y[::-1], self.T[::-1]

		# frequency of experiment
		self.f = f
		# current through drive coil
		self.I = I
		# angle between X and Y
		self.angle = 0
		self.Xcorr = None
		self.Ycorr = None

		# leakage deep in the superconducting phase with thick superconductor
		self.leakage = 0

		# mutual inductance before and after correction for leaking magnetic field
		self.MIx, self.MIy = None, None
		self.MIxcorr, self.MIycorr = None, None

	def phase_correct_data (self, X, Y, T, Tmin=0, Tmax=None):
		"""
		for a perfect superconductor Y should be zero at T=0K;
		we assume that any remnant Y part is due to capacitative coupling in lines etc.;
		we correct for it here
		"""
		# use average of points between Tmin and Tmax to get angle between X and Y
		if Tmax is None: # just in case you don't quite know what Tmax should be, take the first 100 data points
			Tmax = T[100]
		print('----------------------------------')
		print('temperature range selected for phase correction is:')
		print(f'Tmin = {Tmin} K')
		print(f'Tmax = {Tmax} K')
		print('----------------------------------')

		mask = (T>Tmin)&(T<Tmax)
		Xave = np.mean(X[mask])
		Yave = np.mean(Y[mask])
		angle = np.angle(Xave+1j*Yave)

		# Rotate so the SC-state signal lands in the Y quadrature (Xcorr≈0 deep in SC phase).
		# +π/2 is needed because the lock-in convention maps Y→real(MI), not X.
		complex_corr = np.exp(1j * (-angle+np.pi/2)) * (X+1j*Y)
		Xcorr = np.real(complex_corr)
		Ycorr = np.imag(complex_corr)
		return angle, Xcorr, Ycorr
	
	def convert_XY_to_MI (self, X, Y, f, I):
		MIx = Y/2/np.pi/f/I # real part of the mutual inductance
		MIy = X/2/np.pi/f/I # imaginary part of the mutual inductance
		return MIx, MIy
	
	def find_leakage (self, MIx, MIy, T, Tmin=0, Tmax=None):
		# for a thick film superconductor at T=0, both the real and imaginary part of the mutual inductance should be zero
		# (i.e. penetration depth smaller than film thickenss)
		# we already took care of the imaginary part by correcting for the phase
		# what is left to be non-zero in MIx, must be field "leaking" around the sample into the pickup coil
		# we correct for it by just subtracting the low temperature value from the data
		if Tmax is None: # just in case you don't quite know what Tmax should be, take the first 100 data points
			Tmax = T[100]
		print('----------------------------------')
		print('temperature range selected for leakage is:')
		print(f'Tmin = {Tmin} K')
		print(f'Tmax = {Tmax} K')
		print('----------------------------------')
		
		mask    = (T>Tmin)&(T<Tmax)
		leakage = np.mean(MIx[mask])
		MIxcorr = MIx-leakage
		MIycorr = MIy
		return leakage, MIxcorr, MIycorr
	
	def perform_calibration (self, save_name=None, Tphase=[0,None], Tleakage=[0,None], phase_correction=True):
		if phase_correction:
			self.angle, self.Xcorr, self.Ycorr = self.phase_correct_data(self.X, self.Y, self.T, Tmin=Tphase[0], Tmax=Tphase[1])
		else:
			self.angle, self.Xcorr, self.Ycorr = np.nan, self.X, self.Y
		self.MIx, self.MIy                       = self.convert_XY_to_MI(self.Xcorr, self.Ycorr, self.f, self.I)
		self.leakage, self.MIxcorr, self.MIycorr = self.find_leakage(self.MIx, self.MIy, self.T, Tmin=Tleakage[0], Tmax=Tleakage[1])

		print(f'the corrected phase angle is: {self.angle/np.pi*180} degrees')
		print(f'the corrected leakage in Mx is: {self.leakage}')
		print()

		if save_name is not None:
			self.save_calibration_results(save_name)

		plt.figure()
		plt.xlabel('Tempperature ( K )')
		plt.ylabel('Voltage')
		plt.plot(self.T, self.X, c='red', ls='-', label='$X$')
		plt.plot(self.T, self.Xcorr, c='red', ls='--', label='$X^{phase~corrected}$')
		plt.plot(self.T, self.Y, c='blue', ls='-', label='$Y$')
		plt.plot(self.T, self.Ycorr, c='blue', ls='--', label='$Y^{phase~corrected}$')
		plt.legend()
		
		plt.figure()
		plt.xlabel('Tempperature ( K )')
		plt.ylabel('Mutual Inductance')
		plt.plot(self.T, self.MIx, c='red', ls='-', label='MI$_x$')
		plt.plot(self.T, self.MIxcorr, c='red', ls='--', label='MI$_x^{leakage~corrected}$')
		plt.plot(self.T, self.MIy, c='blue', ls='-', label='MI$_y$')
		plt.plot(self.T, self.MIycorr, c='blue', ls='--', label='MI$_y^{leakage~corrected}$')
		plt.legend()
		plt.show()
		return Tphase, Tleakage
	
	def save_calibration_results (self, filename):
		header = f'phase correction angle: {self.angle}  (i.e. {self.angle*180/np.pi} degrees)\n'
		header+= f'corrected leakage in Mx: {self.leakage}\n'
		header+= 'T (K), X (V), Y (V), Xcorr (V), Y corr (V), MIx (H), MIy (H), MIxcorr (H), MIycorr (H)'
		save_data = np.array([self.T, self.X, self.Y, self.Xcorr, self.Ycorr, self.MIx, self.MIy, self.MIxcorr, self.MIycorr]).T
		np.savetxt(filename, save_data, delimiter=',', header=header)