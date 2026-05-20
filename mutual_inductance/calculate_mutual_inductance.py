import matplotlib.pyplot as plt
from scipy import special, integrate, constants
import numpy as np
import itertools
from time import time
from copy import deepcopy
import ray
from psutil import cpu_count
from lmfit import minimize, Parameters


# Computes mutual inductance between a drive and pickup coil separated by a given distance with a
# superconducting thin film in between. Each coil is modelled as a stack
# of single circular loops whose geometry is determined by the winding parameters.
class CalcMutualInductance:
	def __init__ (self, n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire):
		# initialize geometry parameters
		# drive (pickup) coil has n_dl (n_pl) layers and n_dt (n_pt) turns per layer
		self.n_dl = n_dl
		self.n_dt = n_dt
		self.n_pl = n_pl
		self.n_pt = n_pt
		# thickness d_film of measured superconducting film
		self.d_film = d_film
		# distance between closest loops in drive and pickup coils
		self.h = h
		# radius of drive r_d and pickup r_p coils 
		self.r_d = r_d
		self.r_p = r_p
		# diameter of the wire used for the coils
		self.d_wire = d_wire

		# counter to see which iteration of the fit we are currently at
		self.fit_counter = 0


	def assemble_individual_coil (self, n_l, n_t, r, d_wire):
		"""
		n_l: number of layers of loops in the coil
		n_t: number of turns in each layer
		r: radius of coil, i.e. imagine the coil is wound around some cylinder, this is the radius of the cylinder
		d_wire: diameter of used wire
		"""
		# actual radius of inner-most loop of the coil
		r_min   = r + d_wire/2
		# array of all radii of all individual turns
		r_array = r_min + d_wire*np.arange(n_t)

		# height of the first layer - this assumes that the coil is lying at zero
		# (we introduce the actual distance between two coils later, here we jsut assemble one coil "lying on the ground"
		h_min   = d_wire/2
		# array of all heights of all layers
		h_array = h_min + d_wire*np.arange(n_l)

		return h_array, r_array
	
	def assemble_full_coil_geometry (self, n_dl, n_dt, r_d, n_pl, n_pt, r_p, d_wire, h, d_film):
		"""
		Build the full drive+pickup coil geometry and return the unique loop separations.

		h is the minimum gap between the closest loops of the two coils.
		Returns rd_array, rp_array (loop radii) and coil_sep, the unique list of
		center-to-center separations between individual drive and pickup loops. Deduplication
		exploits the fact that many loop pairs share the same separation (e.g. layer i of the
		drive coil is as far from layer j of the pickup as layer i+1 of the drive is from layer j+1),
		so M_single only needs to be evaluated once per unique distance.
		"""
		hd_array, rd_array = self.assemble_individual_coil(n_dl, n_dt, r_d, d_wire)
		hp_array, rp_array = self.assemble_individual_coil(n_pl, n_pt, r_p, d_wire)

		# coil_sep is a 1D array which gives all unique distances between individual loops of pickup vs drive coils
		# - for example consider one loop in the drive coil: its distance is the same to all turns within one layer in the pickup coil
		# - but also the second layer in drive coil has the same distance to the first layer in pickup coil as the first layer in drive coil to second layer in pickup coil
		grid1, grid2 = np.meshgrid(hd_array, hp_array, indexing='ij')
		coil_sep     = grid1 + grid2 + h - d_film
		coil_sep     = coil_sep.flatten()
		coil_sep     = np.unique(coil_sep)
		return rd_array, rp_array, coil_sep


	def M_single (self, r_d, r_p, coilSep, d_film, lam):
		# mutual inductance integral for two circular loops separated by a
		# superconducting film of thickness d_film and London penetration depth lam.
		def M_integrand_single(x, r_d, r_p, coilSep, d_film, lam):
			Xi          = np.sqrt(x**2 + 1/lam**2)
			denominator = np.cosh(d_film*Xi) + (x**2+Xi**2)/(2*x*Xi) * np.sinh(d_film*Xi)
			enumerator  = np.exp(-x*coilSep) * special.jv(1, x*r_d) * special.jv(1, x*r_p)
			# enumerator  = np.exp(-x*coilSep) * special.j1(x*r_d) * special.j1(x*r_p)
			return enumerator/denominator

		integral, _ = integrate.quad(M_integrand_single, 0, np.inf, args=(r_d,r_p,coilSep,d_film,lam))
		factor   = constants.pi * constants.mu_0 * r_d * r_p
		return integral * factor
	

	def calc_MI_mat (self, rd_array, rp_array, coil_sep, n_dl, n_pl, d_film, lam):
		MI_mat_partial = np.zeros((len(rd_array), len(rp_array),len(coil_sep)))
		MI_mat_full    = np.zeros((len(rd_array), len(rp_array), n_dl, n_pl))
		for ii, r_d in enumerate(rd_array):
			for jj, r_p in enumerate(rp_array):
				for kk, sep in  enumerate(coil_sep):
					MI_mat_partial[ii,jj,kk] = self.M_single(rd_array[ii], rp_array[jj], coil_sep[kk], d_film, lam)
		indices = np.arange(n_dl)[:, None] + np.arange(n_pl)
		MI_mat_full = MI_mat_partial[:, :, indices]
		return MI_mat_full
		
	# ray is used for parallel computation
	def start_ray (self, nb_workers=None):
		if nb_workers is None:
			nb_workers = cpu_count(logical=False)
		print('number of workers: ', nb_workers)
		ray.init(num_cpus=nb_workers, include_dashboard=False, log_to_driver=False)
	
	def stop_ray (self):
		ray.shutdown()
	
	@staticmethod
	@ray.remote  # Ray remote version of M_single for parallel evaluation across workers
	def M_single_ray (r_d, r_p, coilSep, d_film, lam):

		def M_integrand_single(x, r_d, r_p, coilSep, d_film, lam):
			Xi          = np.sqrt(x**2 + 1/lam**2)
			denominator = np.cosh(d_film*Xi) + (x**2+Xi**2)/(2*x*Xi) * np.sinh(d_film*Xi)
			enumerator  = np.exp(-x*coilSep) * special.jv(1, x*r_d) * special.jv(1, x*r_p)
			# enumerator  = np.exp(-x*coilSep) * special.j1(x*r_d) * special.j1(x*r_p)
			return enumerator/denominator
		
		integral, _ = integrate.quad(M_integrand_single, 0, np.inf, args=(r_d,r_p,coilSep,d_film,lam))
		factor   = constants.pi * constants.mu_0 * r_d * r_p
		return integral * factor
	

	def calc_MI_mat_ray (self, rd_array, rp_array, coil_sep, n_dl, n_pl, d_film, lam):
		MI_mat_full    = np.zeros((len(rd_array), len(rp_array), n_dl, n_pl))
		futures = []
		for ii, r_d in enumerate(rd_array):
			for jj, r_p in enumerate(rp_array):
				for kk, sep in  enumerate(coil_sep):
					futures.append(self.M_single_ray.remote(rd_array[ii], rp_array[jj], coil_sep[kk], d_film, lam))
		results = ray.get(futures)
		MI_mat_partial = np.array(results).reshape(len(rd_array), len(rp_array), len(coil_sep))

		# accounts for the fact that many different loops will have the same distance between each other
		# for ii in np.arange(n_dl):
			# for jj in np.arange(n_pl):
				# MI_mat_full[:,:,ii,jj] = MI_mat_partial[:,:,ii+jj]
		# replaces the two loops with broadcasting
		indices = np.arange(n_dl)[:, None] + np.arange(n_pl)
		MI_mat_full = MI_mat_partial[:, :, indices]
		return MI_mat_full
	
	def calc_MI_PD_array (self, lambda_array, save_name=None):
		# lambda_array: London penetration depths (m) for which to evaluate MI;
		# MI_norm = MI / MI_inf normalises to the normal-state (lam→∞) value
		self.start_ray()
		
		rd_array, rp_array, coil_sep = self.assemble_full_coil_geometry(self.n_dl, self.n_dt, self.r_d, self.n_pl, self.n_pt, self.r_p, self.d_wire, self.h, self.d_film)
		
		MI_mat_full = np.zeros((len(rd_array), len(rp_array), self.n_dl, self.n_pl, len(lambda_array)))
		MI_array    = np.zeros(len(lambda_array))
		
		counter = len(lambda_array)
		for ii, lam in enumerate(lambda_array):
			print(f'{counter} calculations to go ...')
			counter-=1

			mat = self.calc_MI_mat_ray (rd_array, rp_array, coil_sep, self.n_dl, self.n_pl, self.d_film, lam)
			MI_mat_full[:,:,:,:,ii] = mat
			MI_array[ii] = np.sum(mat)

		# mutual inductance for infinite penetration depth
		mat_inf = self.calc_MI_mat_ray (rd_array, rp_array, coil_sep, self.n_dl, self.n_pl, self.d_film, np.inf)
		MI_norm = MI_array/np.sum(mat_inf)
		
		self.stop_ray()
		if save_name is not None:
			self.save_MI_array(save_name, lambda_array, MI_array, MI_norm)
		return MI_array, MI_norm
	
	def save_MI_array (self, save_name, lam_array, MI_array, MI_norm):
		header = f'drive coil dimensions: n_layers={self.n_dl}; n_turns_per_layer={self.n_dt}\n'
		header+= f'pickup coil dimensions: n_layers={self.n_pl}; n_turns_per_layer={self.n_pt}\n'
		header+= f'smallest distance between coils (m): {self.h}\n'
		header+= f'smallest coil radius (m): drive coil = {self.r_d}; pickup coil = {self.r_p}\n'
		header+= f'wire diameter (m): {self.d_wire}\n'
		header+= f'film thickness (m): {self.d_film}\n'
		header+= 'penetration depth (m), Mx (H), Mx/Mx_inf'
		
		data = np.array([lam_array, MI_array, MI_norm]).T
		np.savetxt(save_name, data, header=header, delimiter=',')

	
	def residual_function (self, params, ray_fit=False):
		"""
		lmfit residual: returns MIx_calc - MIx_data (scalar, so leastsq treats it as one residual).

		params must contain:
		  h         — coil separation (m), vary=True when fitting coil distance
		  lam       — London penetration depth (m), vary=True when fitting penetration depth
		  MIx_data  — measured mutual inductance (H), always vary=False (passed as a frozen Parameter
		              rather than a regular argument because lmfit only forwards params to the residual)
		"""
		h        = params['h'].value
		MIx_data = params['MIx_data'].value
		lam      = params['lam'].value

		rd_array, rp_array, coil_sep = self.assemble_full_coil_geometry(self.n_dl, self.n_dt, self.r_d, self.n_pl, self.n_pt, self.r_p, self.d_wire, h, self.d_film)
		if ray_fit:
			mat = self.calc_MI_mat_ray(rd_array, rp_array, coil_sep, self.n_dl, self.n_pl, self.d_film, lam)
		else:
			mat = self.calc_MI_mat(rd_array, rp_array, coil_sep, self.n_dl, self.n_pl, self.d_film, lam)
		MIx_calc = np.sum(mat)

		self.fit_counter+=1
		self.h = h
		print(f'fit iteration {self.fit_counter} done')
		print(f'MIx calc: {MIx_calc} --- data: {MIx_data}')
		print(f'coil separation: {self.h}')
		print(f'penetration depth: {lam}')
		print('-------------------------------------------------')
		return MIx_calc - MIx_data
	
	def fit_coil_distance (self, MIx_data, ray_fit=False):
		# uses residual function above to fit the distance between coils to mutual inductance data
		if ray_fit:
			self.start_ray()

		self.fit_counter = 0
		params = Parameters()
		params.add('h', value=self.h, vary=True, min=self.h/2, max=self.h*1.5)#, min=h_init/5, max=h_init*5)
		params.add('MIx_data', value=MIx_data, vary=False)
		params.add('lam', value=np.inf, vary=False)

		output = minimize(self.residual_function, params, method='leastsq', args=[ray_fit])
		# output = minimize(self.residual_function, params, method='differential_evolution')
		print('fit done!')

		if ray_fit:
			self.stop_ray()

	def fit_penetration_depth (self, MIx_data, lam_guess=1e-5, ray_fit=False):
		# uses residual function above to fit the penetration depth to mutual inductance data
		if ray_fit:
			self.start_ray()

		self.fit_counter = 0
		params = Parameters()
		params.add('h', value=self.h, vary=False)#, min=h_init/5, max=h_init*5)
		params.add('MIx_data', value=MIx_data, vary=False)
		params.add('lam', value=lam_guess, vary=True)

		output = minimize(self.residual_function, params, method='leastsq', args=[ray_fit])
		print('fit done!')

		if ray_fit:
			self.stop_ray()


if __name__ == '__main__':
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

	h      = 1.07e-3 # distance between coils
	d_film = 40e-9  # Sample film thickness
	lam    = np.inf # penetration depth
	MI     = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
	MIcalc, MIcalc_norm = MI.calc_MI_PD_array([lam])
	print(MIcalc)
	print(MIcalc_norm)
	print()

	h      = 1.07e-3 # distance between coils
	d_film = 20e-9  # Sample film thickness
	lam    = np.inf # penetration depth
	MI     = CalcMutualInductance(n_dl, n_dt, n_pl, n_pt, d_film, r_d, r_p, h, d_wire)
	MIcalc, MIcalc_norm = MI.calc_MI_PD_array([lam])
	print(MIcalc)
	print(MIcalc_norm)