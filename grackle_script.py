########################################################################
# Adapted from cooling_rate.py example script
#    Compute the cooling rate for different metallicities, densities, temperatures.
########################################################################

import numpy as np
import os
import h5py
import sys
from astropy import coordinates as ac, units as un, constants as con
from scipy.interpolate import LinearNDInterpolator
from utilities.constant import gram_per_sun, cm_per_kpc, centi_per_kilo, proton_mass, boltzmann

# from pygrackle import chemistry_data
# from pygrackle.fluid_container import FluidContainer
# from pygrackle.utilities.convenience import check_convergence, setup_fluid_container

# from pygrackle.utilities.physical_constants import \
#     mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc, mass_sun_cgs, cm_per_kpc


# def cooling_func(rho, Zarr, redshift, u_int=None, temperature=None, chem_units=False, return_time=False, uvbackground = 1):
#     """
#     Parameters
#     ----------
#     rho: array[float]
#         Mass density in g cm^-3
#         (num_points,)
#     Zarr: array[float]
#         Metal mass fractions
#         Second axis is element type:
#             Illustris: [H, He, C, N, O, Ne, Mg, Si, Fe, extra]
#             GIZMO:     [H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe]
#         (num_points, num_species)
#     redshift: float
#         Redshift
#     u_int: array[float] or None
#         Specific internal energy in cm s^-1
#         Either u_int or temperature must be provided
#         (num_points,)
#     temperature: array[bool] or None
#         Temperature in K
#         Either u_int or temperature must be provided
#         (num_points,)
#     chem_units: bool
#         If True, return in the units in the chemistry_data instance
#     return_time: bool
#         Return cooling time as well

#     Returns
#     -------
#     array[float]
#         Cooling rates in erg s^-1 cm^3
#         (num_points,)
#     array[float]
#         Cooling timescale in seconds
#         (num_points,)
#     pygrackle.grackle_wrapper.chemistry_data
#         Chemistry data instance (essential information on units
#         and various grackle settings)
        
#     """
#     num_points = rho.size
#     # TODO assert all other arrays same size.
#     has_u, has_T = u_int is not None, temperature is not None
#     if (has_u and has_T) or (not has_u and not has_T):
#         raise ValueError("Must specify either u_int or temperature, not both.")

#     chem = chemistry_data()
#     chem.use_grackle = 1
#     chem.with_radiative_cooling = 1     # Default example has this as 0
#     chem.metal_cooling = 1 
#     chem.UVbackground = uvbackground      # Add heating from UV background
#     chem.primordial_chemistry = 0       # Tabulated cooling (no chemistry). Assumes ionization equilibrium.
#     chem.H2_self_shielding = 0 
#     chem.self_shielding_method = 0      # Paper says self-shielding is sketchy when UV background is low or density is high.
#                                         # Self-shielding is not applied to metal cooling. You'd need a separate CLOUDY table with it applied
#     chem.dust_chemistry = 0 

#     chem.grackle_data_file = bytearray(
#         #"/home/alanman/Desktop/repositories/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
#         "/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
#         'utf-8')
#     chem.use_specific_heating_rate = 1
#     chem.use_volumetric_heating_rate = 1

#     # Set units (used to scale internal values to be near 1)
#     chem.comoving_coordinates = 0 # proper units
#     chem.a_units = 1.0
#     chem.a_value = 1.0 / (1.0 + redshift) / \
#         chem.a_units
#     chem.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g/cm3
#     chem.length_units = cm_per_mpc         # 1 Mpc in cm
#     chem.time_units = sec_per_Myr          # 1 Gyr in s
#     chem.set_velocity_units()

#     chem.initialize()
#     fc = FluidContainer(chem, num_points)


#     fc['density'][:] = rho / chem.density_units
#     fc['metal_density'][:] = rho / chem.density_units * Zarr[:, 0]
    
#     # Start supporting non-equilibrium?
#     tiny_number = 1e-20
#     d_to_h_ratio = 3.4e-5
#     # Should I be setting the fractions based on the free-electron abundance?
#     if chem.primordial_chemistry > 0:
#         fc["HII"][:] = Zarr[:, 0] * fc["density"]
#         fc["HI"][:] = tiny_number * fc["density"]
#         fc["HeI"][:] = Zarr[:, 1] * fc["density"]
#         fc["HeII"][:] = tiny_number * fc["density"]
#         fc["HeIII"][:] = tiny_number * fc["density"]
#         fc["de"][:] = fc["HII"] + fc["HeII"] / 4.0 + fc["HeIII"] / 2.0
#     if chem.primordial_chemistry > 1:
#         fc["HM"][:] = tiny_number * fc["density"]
#         fc["H2I"][:] = tiny_number * fc["density"]
#         fc["H2II"][:] = tiny_number * fc["density"]
#     if chem.primordial_chemistry > 2:
#         fc["DI"][:] = 2.0 * d_to_h_ratio * fc["density"]
#         fc["DII"][:] = tiny_number * fc["density"]
#         fc["HDI"][:] = tiny_number * fc["density"]

#     fc["specific_heating_rate"][:] = 0.
#     fc["volumetric_heating_rate"][:] = 0.
#     fc["x_velocity"][:] = 0.0
#     fc["y_velocity"][:] = 0.0
#     fc["z_velocity"][:] = 0.0

#     # fc.calculate_mean_molecular_weight()

#     if has_u:
#         fc['internal_energy'][:] = u_int / chem.velocity_units**2       # Specific energy has velocity^2 units
#     else: 
#         fc['internal_energy'][:] = temperature / chem.temperature_units / \
#                           fc["mean_molecular_weight"] / (chem.Gamma - 1.0)
        
#     fc.calculate_mean_molecular_weight()

#     fc.calculate_cooling_time()

#     cooling_rate = fc['internal_energy'] / fc['cooling_time'] / fc['density']
#     cooling_time = fc['cooling_time']# * chem.time_units

#     if not chem_units:
#         cooling_time *= chem.time_units
#         cooling_rate *= chem.cooling_units

#     return cooling_rate, cooling_time, chem

def cooling_from_table(n_H, n, Zarr, redshift, temperature=None, chem_units=False, return_time=False, uvbackground = 1):
    table = h5py.File("/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5", 'r')

    # Axes = (log_nH, redshift, temperature)

    cool = table['CoolingRates/Primordial/Cooling'][()]
    heat = table['CoolingRates/Primordial/Heating'][()]
    metal_cool = table['CoolingRates/Metals/Cooling'][()]
    metal_heat = table['CoolingRates/Metals/Heating'][()]

    logT = np.log10(table['CoolingRates/Primordial/Cooling'].attrs['Temperature'])
    log_nH = table['CoolingRates/Primordial/Cooling'].attrs['Parameter1']

    log_nH_g, logT_g = map(np.ndarray.flatten, np.meshgrid(log_nH, logT, indexing='ij'))

    # From the grackle source code, it seems that one should combine the metal and primordial tables by scaling the metal
    # rates by the metallicity relative to solar.
    cool_tot_03Z = (cool + metal_cool * 0.3)
    heat_tot_03Z = (heat + metal_heat * 0.3)
    net_cool = cool_tot_03Z - heat_tot_03Z      # positive = net cooling


    # Now get cooling timescales
    # Axes = (log_nH, temperature)
    interp = LinearNDInterpolator(list(zip(log_nH_g, logT_g)), net_cool[:,0].flatten())
    cooling_rate = interp(np.log10(n_H), np.log10(temperature))* un.erg /un.s * un.cm**3
    cooling_time = (3/2) * con.k_B * temperature * un.K / (n / un.cm**3 * cooling_rate)
    

    return np.array(cooling_rate.cgs), np.array(cooling_time.cgs)

def get_snapshot_nparts(snapshot_path, partnum):
    """
    Returns the number of particles in each file of snapshot
    """
    n_per_file = []
    for file_of_snapshot in range(4):
        fn = os.path.join(
            snapshot_path, f"output/snapdir_600/snapshot_600.{file_of_snapshot}.hdf5"
        )
        if not os.path.exists(fn):
            fn = fn.replace("snapdir_600/", "")  # One path doesn't have this level
        fh = h5py.File(fn, 'r')
        gas = fh[f"PartType{partnum}"]
        n_ = fh['Header'].attrs['NumPart_ThisFile'][partnum]
        n_per_file.append(n_)
    return n_per_file


def calc_cooling_gizmo(
        snapshot_path, file_of_snapshot=0, subsel=None, phase = 'cold'
    ):
    """
    Write a supplementary file with cooling rates and timescales from Gizmo data
    """

    fn = os.path.join(
       snapshot_path, f"output/snapdir_600/snapshot_600.{file_of_snapshot}.hdf5"
    )
    snapdir = os.path.dirname(fn)
    if not os.path.exists(snapdir):
        fn = fn.replace("snapdir_600/", "")  # One path doesn't have this level
        snapdir = snapdir.replace("snapdir_600", "")
    fh = h5py.File(fn, 'r')
    
    gas = fh['PartType0']
    n_gas = fh['Header'].attrs['NumPart_ThisFile'][0]

    h = fh['Header'].attrs["HubbleParam"]
    z = fh['Header'].attrs['Redshift']
    a = 1 / (1 + z)
    density_unit = 1e10 * h**2 / a**3 * gram_per_sun /  cm_per_kpc**3       # Multiply to convert to cgs
    if subsel is None:
        subsel = slice(None)

    Zarr = gas['Metallicity'][subsel, :]
    metallicity = Zarr[:,0]
    frac_He = Zarr[:,1]
    frac_H = 1 - metallicity - frac_He

    outdir = snapdir

    os.makedirs(outdir, exist_ok=True)
    fn_out = os.path.join(
        outdir, f"cooling_600_{phase}.{file_of_snapshot}.hdf5"
    )

    if phase == 'cold':
        # get mask for halo and shattered cells
        halo_mask = np.load(f"/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/halo_mask_{file_of_snapshot}.npy")
        shattered_mask = np.load(f"/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/shattered_mask_{file_of_snapshot}.npy")

        # Converting all to CGS
        n = np.load(f"/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/n_c_{file_of_snapshot}.npy")   # cold phase density in cgs
        n_H = n*frac_H
        Zarr = Zarr[halo_mask][shattered_mask]
        Tcool_peak = 10**4.3
        cool_rate_low, cool_time_low = cooling_from_table(n_H, Zarr, z, temperature=Tcool_peak, chem_units=True, return_time=True)

        fout = h5py.File(fn_out, 'a')
        print(f"\t writing to {fn_out}")

        n_gas = len(n)
        if not "Cold" in fout.keys():
            fout.create_group("Cold")
            fout['Cold'].attrs['T_cold'] = Tcool_peak
            fout.create_dataset("Cold/rate", (n_gas,), dtype="<f4", fillvalue=np.nan)
            fout.create_dataset("Cold/time", (n_gas,), dtype="<f4", fillvalue=np.nan)
            # fout["Cold/rate"].attrs["to_cgs"] = chem_low.cooling_units
            # fout["Cold/time"].attrs["to_cgs"] = chem_low.time_units
    
        fout['Cold/rate'][subsel] = cool_rate_low[()]
        fout['Cold/time'][subsel] = cool_time_low[()]
    
    elif phase == 'ambient':
        f_e = gas['ElectronAbundance'][subsel]
        mu = 1/((1-frac_He) + frac_He/4 + (1-frac_He)*f_e)
        mbar = mu*proton_mass           # mean molecular mass
        rho = gas['Density'][subsel] * density_unit     #cgs
        n = rho/mbar
        n_H = n*frac_H

        u_int = gas['InternalEnergy'][subsel] * centi_per_kilo**2     # km^2/s^2 -> cm^2/s^2
        temp = u_int *  (5 / 3 - 1)* mbar / boltzmann

        cool_rate, cool_time = cooling_from_table(n_H, n, Zarr, z, temperature=temp, chem_units=True, return_time=True)
        print(temp)
    
        fout = h5py.File(fn_out, 'a')
        print(f"\t writing to {fn_out}")
        
        if not "Ambient" in fout.keys():
            fout.create_group("Ambient")
            fout.create_dataset("Ambient/rate", (n_gas,), dtype="<f4", fillvalue=np.nan)
            fout.create_dataset("Ambient/time", (n_gas,), dtype="<f4", fillvalue=np.nan)
            # fout["Ambient/rate"].attrs["to_cgs"] = chem.cooling_units
            # fout["Ambient/time"].attrs["to_cgs"] = chem.time_units

        fout['Ambient/rate'][subsel] = cool_rate[()]
        fout['Ambient/time'][subsel] = cool_time[()]

    fout.close()
    fh.close()

# CHANGE THIS
snapdir = '/ceph/submit/data/user/z/zimi/analysis/FIRE/m12f_res7100'#sys.argv[1]
nchunks = 20 #int(sys.argv[2]) if len(sys.argv) >= 3 else 5000
ph = 'ambient' # sys.argv[3]

nparts_per = get_snapshot_nparts(snapdir, 0)

for fi, npts in enumerate(nparts_per):
    chunksize = npts // nchunks
    complete = np.zeros(npts).astype(bool)
    for ci in range(nchunks+1):
        slc = slice(ci*chunksize, min((ci+1)*chunksize, npts), 1)
        calc_cooling_gizmo(
            snapdir, 
            file_of_snapshot=fi,
            subsel=slc, phase = ph,
        )
        complete[slc] = True
        print(f"Chunk {ci} of {nchunks}")
    print(fi, all(complete))

