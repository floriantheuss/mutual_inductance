import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

class MIData:
	def __init__ (self, X, Y, T, f, I, leakage=0, residual_capacitance=None, correction_angle=np.pi/2):
		# sort according to temperature
		# mask = np.argsort(T)
		# mutual inductance data
		# self.X = X[mask] # real part
		# self.Y = Y[mask] # imaginary part
		# self.T = T[mask] # temperature
		self.X, self.Y, self.T = X, Y, T
		self.T, self.X, self.Y = self.remove_nan (self.T, self.X, self.Y)
		if np.mean(self.T[:10]) > np.mean(self.T[-10:]):
			self.X, self.Y, self.T = self.X[::-1], self.Y[::-1], self.T[::-1]
		
		self.f = f                          # frequency
		self.I = I                          # current in drive coil
		self.leakage = leakage              # leakage of magnetic field around the sample - determined from calibration sample
		self.res_cap = residual_capacitance # the capacitive signal should be zero at zero T=0; it isn't due to setup we will subtract this amount here from the data to make it zero
										    # if None: will find it by taking the minimum value that exists and shift by that amount
		self.angle   = correction_angle     # mix X and Y by this angle - also determined from calibration sample

		# X and Y but corrected for a phase of self.angle between the two
		self.Xcorr = None
		self.Ycorr = None
		# mutual inductance before and after correction for leaking magnetic field
		self.MIx, self.MIy = None, None
		self.MIxcorr, self.MIycorr = None, None
		# normalized mutual inductance data
		self.MIxnorm, self.MIynorm = None, None
		# penetration depth
		self.penet_dep = None

	def remove_nan (self, T, X, Y):
		Xnan_mask = np.invert(np.isnan(self.X))
		X, Y, T = X[Xnan_mask], Y[Xnan_mask], T[Xnan_mask]
		Ynan_mask = np.invert(np.isnan(Y))
		X, Y, T = X[Ynan_mask], Y[Ynan_mask], T[Ynan_mask]
		Tnan_mask = np.invert(np.isnan(T))
		X, Y, T = X[Tnan_mask], Y[Tnan_mask], T[Tnan_mask]
		return T, X, Y

	
	def phase_correct_data (self, X, Y, angle, Tmin=0, Tmax=None, T=None):
		if angle is None:
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

		complex_corr = np.exp(1j * (-angle+np.pi/2)) * (X+1j*Y)
		Xcorr = np.real(complex_corr)
		Ycorr = np.imag(complex_corr)
		self.angle = angle
		return Xcorr, Ycorr
	
	def convert_XY_to_MI (self, X, Y, f, I):
		MIx = Y/2/np.pi/f/I # real part of the mutual inductance
		MIy = X/2/np.pi/f/I # imaginary part of the mutual inductance
		return MIx, MIy
	
	def moving_average(self, x, y, n):
		ret = np.cumsum(y, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return x[n-1:], ret[n - 1:] / n
	
	def median_filter (self, data, threshold=1, window=10000):
		median     = pd.DataFrame(data).rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill').to_numpy().flatten()
		difference = np.abs((data - median)/median)
		good_bool  = difference <= threshold
		bad_bool   = np.invert(good_bool)	
		return bad_bool, good_bool
	
	def correct_leakage (self, MIx, MIy, T, leakage, residual_capacitance=None, n=1000):
		MIxcorr = MIx - leakage
		
		if residual_capacitance is None:
			tave, miyave = self.moving_average(T, MIy, n)
			plt.figure()
			plt.plot(T, MIy)
			plt.plot(tave, miyave)
			plt.show()
			MIycorr = MIy - min(miyave)
			self.res_cap = min(miyave)
		else:
			MIycorr = MIy - residual_capacitance
		return MIxcorr, MIycorr
	
	def normalize_MI (self, MIx, MIy, T, Tmin=None, Tmax=None):
		if Tmax is None:
			Tmax = T[-1]
		if Tmin is None:
			Tmin = T[-100]
		print('----------------------------------')
		print('temperature range selected to normalize Mx is:')
		print(f'Tmin = {Tmin} K')
		print(f'Tmax = {Tmax} K')
		print('----------------------------------')

		mask = (T>Tmin)&(T<Tmax)
		temp = MIx[mask]
		temp = temp[np.invert(np.isnan(temp))]
		MIave = np.mean(temp)
		MIxnorm = MIx/MIave
		MIynorm = MIy/MIave		
		return MIxnorm, MIynorm
	
	def process_data (self, Tmin=None, Tmax=None, save_name=None, n=1000):
		self.Xcorr, self.Ycorr     = self.phase_correct_data(self.X, self.Y, self.angle, T=self.T)
		self.MIx, self.MIy         = self.convert_XY_to_MI(self.Xcorr, self.Ycorr, self.f, self.I)
		self.MIxcorr, self.MIycorr = self.correct_leakage(self.MIx, self.MIy, self.T, self.leakage, self.res_cap, n=n)
		self.MIxnorm, self.MIynorm = self.normalize_MI(self.MIxcorr, self.MIycorr, self.T, Tmin=Tmin, Tmax=Tmax)

		if save_name is not None:
			self.save_results(save_name)

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

		return self.MIxnorm, self.MIynorm, self.T
	
	def save_results (self, filename, additional_header=''):
		if self.penet_dep is None:
			header = additional_header
			header+= f'phase correction angle: {self.angle}  (i.e. {self.angle*180/np.pi} degrees)\n'
			header+= f'corrected leakage in Mx: {self.leakage}\n'
			header+= f'corrected residual capacitance in My: {self.res_cap}\n'
			header+= 'T (K), X (V), Y (V), X_phase_corr (V), Y_phase_corr (V), MIx_phase_corr (H), MIy_phase_corr (H), MIx_leak_corr (H), MIy_leak_corr (H), MIx_norm, MIy_norm = sigma_1'
			save_data = np.array([self.T, self.X, self.Y, self.Xcorr, self.Ycorr, self.MIx, self.MIy, self.MIxcorr, self.MIycorr, self.MIxnorm, self.MIynorm]).T
			np.savetxt(filename, save_data, delimiter=',', header=header)
		else:
			header = additional_header
			header+= f'phase correction angle: {self.angle}  (i.e. {self.angle*180/np.pi} degrees)\n'
			header+= f'corrected leakage in Mx: {self.leakage}\n'
			header+= f'corrected residual capacitance in My: {self.res_cap}\n'
			header+= 'T (K), X (V), Y (V), X_phase_corr (V), Y_phase_corr (V), MIx_phase_corr (H), MIy_phase_corr (H), MIx_leak_corr (H), MIy_leak_corr (H), MIx_norm, MIy_norm = sigma_1, penetration depth (m)'
			save_data = np.array([self.T, self.X, self.Y, self.Xcorr, self.Ycorr, self.MIx, self.MIy, self.MIxcorr, self.MIycorr, self.MIxnorm, self.MIynorm,self.penet_dep]).T
			np.savetxt(filename, save_data, delimiter=',', header=header)

	def convert_MI_to_PD (self, lookup_table_path, save_path=None):
		lookup_table    = np.loadtxt(lookup_table_path, delimiter=',')
		PD_lookup       = lookup_table[:,0]
		MIx_norm_lookup = lookup_table[:,2]
		
		lookup_func     = interp1d(MIx_norm_lookup, PD_lookup)
		mask            = (self.MIxnorm>np.min(MIx_norm_lookup))&(self.MIxnorm<np.max(MIx_norm_lookup))
		penet_dep       = np.nan*self.T
		penet_dep[mask] = lookup_func(self.MIxnorm[mask])
		self.penet_dep  = penet_dep
		
		print('-----------------------------------------------------------')
		print(self.MIxnorm[np.invert(mask)])
		print(f'threw away {len(self.MIxnorm[np.invert(mask)])} points compared to {len(self.MIxnorm)} points total')
		print()

		if save_path is not None:
			self.save_results(save_path, additional_header=f'penetration depth lookup table: {lookup_table_path}\n')
		
		plt.figure()
		plt.plot(self.T, self.penet_dep)
		plt.xlabel('Temperature ( K )')
		plt.ylabel('Penetration depth ( m )')

		plt.figure()
		plt.plot(self.T, (self.penet_dep[0]/self.penet_dep)**2)
		plt.xlabel('Temperature ( K )')
		plt.ylabel('$\lambda_0^2 / \lambda(T)^2$')
		plt.show()






if __name__ == '__main__':
	print(1)