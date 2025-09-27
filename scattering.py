# from halo import sim_halo
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from astropy import constants as const
# from scipy import stats
# import h5py
# import os
# import utilities as ut

# # constants
# gamma = 5/3
# xi = 10
# k_b = const.k_B.cgs

# # setting up halo and halo gas
# sim_name = 'm12f_res7100'
# halo = sim_halo(sim_name)
# halo.read_gas_files(cooling = True)
# halo_gas = halo.in_halo(region = 'cgm', save = True)

# # nfw potential
# max_potential = np.max(halo.gas_data[2]['potential'])
# max_nfw_potential = np.max(halo.gas_data[0]['nfw_potential'])

# halo_gas = pd.concat(halo_gas)

# # finding shattered cells
# metal = halo_gas['metallicity']
# y = halo_gas['helium']
# x = 1 - metal - y
# f_e = halo_gas['electron_abundance']
# rho = halo_gas['density']
# d = halo_gas['bin_size']

# mu = 1/((1-y) + y/4 + (1-y)*f_e)
# mbar = mu*ut.constant.proton_mass           # mean molecular mass
# n = rho/mbar

# # u = chunk['internal_energy']
# temp = halo_gas['temperature']

# # t_cross = chunk['bin_size']*3.09e16/c_s                 # crossing time
# t_cool = halo_gas['cooling_times']                               # cooling time in cgs, positive when heating, negative when cooing
# rate_cool = np.abs(halo_gas['cooling_rates'])
# t_ff = (2*halo_gas['dist_tot']/3)*np.sqrt(-1/(halo_gas['nfw_potential']) )          # freefall time in cgs, positive

# # select particles that meet shattering criteria
# shattered_idx = ((-t_cool/t_ff) < xi) & ((-t_cool/t_ff) > 0)
# not_shattered_idx = np.logical_not(shattered_idx)
# shattered = halo_gas[shattered_idx]

# # percentage of gas shattered
# print(str(len(shattered)/len(halo_gas)*100) + "%" + " shattered")

# # hot phase and cold phase density
# T_c = 10**4.3   # cold temperature
# n_h = (k_b*temp)/((gamma-1)*rate_cool*xi*t_ff*(mu**2))
# n_c = (n_h*temp/T_c)[shattered_idx]

# t_cool_cold, rate_cool_cold = halo.get_cooling(phase = 'Cold')
# t_cool_cold = np.concat(t_cool_cold)

# c_s = np.sqrt(gamma*k_b*T_c/mbar[shattered_idx])           # speed of sound
# l = np.abs(t_cool_cold)*c_s                      # shattered cloudlet size[cm]
# f_v = (n[shattered_idx]/n_h[shattered_idx] - 1)/(temp[shattered_idx]/T_c - 1)          # volume filling fraction
# N_e = l*n_c*f_e[shattered_idx]*x[shattered_idx]*mu[shattered_idx]                        # electron column density
# f_a = f_v*(d[shattered_idx]/l)                             # number of cloudlet incercepted

