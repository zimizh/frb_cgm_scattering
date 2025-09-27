import numpy as np

from pygrackle import chemistry_data
from pygrackle.fluid_container import FluidContainer

from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc, mass_sun_cgs, cm_per_kpc

# Cooling rates at logT = 4.3 instead of virial temp, using grackle

# Test -- recreating Liang & Remming 2020 fig 2

num_rho = 300
num_T = 200
num_points = num_rho * num_T

solar_abund = np.array(
[
    7.0649785e-01, 2.8055534e-01, 2.0665436e-03, 8.3562563e-04,
    5.4926244e-03, 1.4144605e-03, 5.9070642e-04, 6.8258739e-04,
    4.0898522e-04, 6.4355001e-05, 1.1032152e-03
])


elements = [
     'H', 'He', 'C', 'N', 'O',
     'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
element_names = [
     "Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen",
     "Neon", "Magnesium", "Silicon", "Sulphur", "Calcium", "Iron"]


Zarr = solar_abund * 0.3
#Zarr = 

#rho = np.logspace(-34, -30, num_rho)     # cgs units 

rho = np.logspace(-4, 0, num_rho) * mass_hydrogen_cgs
temperature = np.logspace(3.5, 7.5, num_T)
#temperature = [10**4.3, 10**4.7, 10**5, 10**6]
redshift = 0.0

chem = chemistry_data()
chem.use_grackle = 1
chem.with_radiative_cooling = 0     # Default example has this as 0
chem.metal_cooling = 1 
chem.UVbackground = 1               # Add heating from UV background (only used if primordial_chemistry != 0)
chem.primordial_chemistry = 0       # Tabulated cooling (no chemistry). Assumes ionization equilibrium.
chem.H2_self_shielding = 0 
chem.self_shielding_method = 0      # Paper says self-shielding is sketchy when UV background is low or density is high.
                                    # Self-shielding is not applied to metal cooling. You'd need a separate CLOUDY table with it applied
chem.dust_chemistry = 0 

chem.grackle_data_file = bytearray(
    "/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
    #"/home/alanman/Desktop/repositories/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
    #"/home/alanman/Desktop/repositories/grackle/grackle_data_files/input/CloudyData_noUVB.h5",
    #"/home/alanman/Desktop/repositories/grackle/grackle_data_files/input/CloudyData_UVB=FG2011_shielded.h5",
    'utf-8')
chem.use_specific_heating_rate = 1
chem.use_volumetric_heating_rate = 1

# Set units (used to scale internal values to be near 1)
chem.comoving_coordinates = 0 # proper units
chem.a_units = 1.0
chem.a_value = 1.0 / (1.0 + redshift) / \
    chem.a_units
chem.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g/cm3
chem.length_units = cm_per_mpc         # 1 Mpc in cm
chem.time_units = sec_per_Myr          # 1 Gyr in s
chem.set_velocity_units()

chem.initialize()
fc = FluidContainer(chem, num_points)

rho_g, T_g = map(np.ndarray.flatten, np.meshgrid(rho, temperature, indexing='ij'))
fc['density'][:] = rho_g / chem.density_units
#fc['energy'][:] = u_int / chem.velocity_units**2       # Specific energy has velocity^2 units
fc['metal'][:] = rho_g / chem.density_units * np.sum(Zarr[2:], axis=-1)

fc.calculate_mean_molecular_weight()
fc["energy"] = T_g / \
    fc.chemistry_data.temperature_units / \
    fc["mu"] / (chem.Gamma - 1.0)


fc.calculate_cooling_time()

density_proper = fc["density"] / \
        (chem.a_units *
         chem.a_value)**(3*chem.comoving_coordinates)
cooling_rate = fc.chemistry_data.cooling_units * fc["energy"] / \
        fc["cooling_time"] / density_proper

## cooling_time_g = cooling_rate


## import pylab as pl
## import IPython; IPython.embed()
## 
## import sys; sys.exit()

cooling_time_g = fc['cooling_time'] * chem.time_units
#ct = cooling_time_g.reshape((num_rho, num_T))

cs_g = np.sqrt(chem.Gamma * (chem.Gamma - 1) * fc['energy'] * chem.energy_units)

rc_pc = -(cs_g * cooling_time_g / cm_per_kpc * 1000).reshape((num_rho, num_T))

from matplotlib import pyplot as plt

mu = fc["mu"]
#print(mu)
num_dens = rho/(mass_hydrogen_cgs)

plt.pcolormesh(num_dens, temperature, rc_pc.T, norm='log')
# pl.colorbar(label='log rc [pc]')
plt.xlabel("log n [cm^-3]")
plt.ylabel("Temperature [K]")
levels = 10.0**np.arange(-4, 5)  * 1000
cs = plt.contour(num_dens, temperature, rc_pc.T, levels=levels, norm='log', colors='k')
plt.clabel(cs, fmt=lambda x: f"{np.log10(x):.1f}", inline_spacing=1)

## for ii in range(num_T):
##     vals = rc_pc[:, ii]
##     neg = vals < 0
##     p = pl.plot(rho[neg], np.abs(vals[neg]), ls='--')
##     pl.plot(rho[~neg], vals[~neg], label=f"{np.log10(temperature[ii])}", color=p[-1].get_color())
## pl.legend()
plt.xscale('log'); plt.yscale('log')
plt.show()


# Plot the rc vs number density logT = 4.3
Tind = np.argmin(np.abs(np.log10(temperature) - 4.3))
rc_slc = rc_pc[:, Tind]
plt.plot(num_dens, rc_slc)
plt.xscale("log"); plt.yscale('log')
plt.show()

# import IPython; IPython.embed()
