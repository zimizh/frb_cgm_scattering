########################################################################
# Adapted from cooling_rate.py example script
#    Compute the cooling rate for different metallicities, densities, temperatures.
########################################################################

from matplotlib import pyplot
import numpy as np
# import pandas as pd
import os
import h5py
import sys

from pygrackle import chemistry_data
from pygrackle.fluid_container import FluidContainer
from pygrackle.utilities.convenience import check_convergence, setup_fluid_container

from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc, mass_sun_cgs, cm_per_kpc


def cooling_func(rho, Zarr, redshift, u_int=None, temperature=None, chem_units=False, return_time=False):
    """
    Parameters
    ----------
    rho: array[float]
        Mass density in g cm^-3
        (num_points,)
    Zarr: array[float]
        Metal mass fractions
        Second axis is element type:
            Illustris: [H, He, C, N, O, Ne, Mg, Si, Fe, extra]
            GIZMO:     [H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe]
        (num_points, num_species)
    redshift: float
        Redshift
    u_int: array[float] or None
        Specific internal energy in cm s^-1
        Either u_int or temperature must be provided
        (num_points,)
    temperature: array[bool] or None
        Temperature in K
        Either u_int or temperature must be provided
        (num_points,)
    chem_units: bool
        If True, return in the units in the chemistry_data instance
    return_time: bool
        Return cooling time as well

    Returns
    -------
    array[float]
        Cooling rates in erg s^-1 cm^3
        (num_points,)
    array[float]
        Cooling timescale in seconds
        (num_points,)
    pygrackle.grackle_wrapper.chemistry_data
        Chemistry data instance (essential information on units
        and various grackle settings)
        
    """
    num_points = rho.size
    # TODO assert all other arrays same size.
    has_u, has_T = u_int is not None, temperature is not None
    if (has_u and has_T) or (not has_u and not has_T):
        raise ValueError("Must specify either u_int or temperature, not both.")

    chem = chemistry_data()
    chem.use_grackle = 1
    chem.with_radiative_cooling = 1     # Default example has this as 0
    chem.metal_cooling = 1 
    chem.UVbackground = 1               # Add heating from UV background
    chem.primordial_chemistry = 0       # Tabulated cooling (no chemistry). Assumes ionization equilibrium.
    chem.H2_self_shielding = 0 
    chem.self_shielding_method = 0      # Paper says self-shielding is sketchy when UV background is low or density is high.
                                        # Self-shielding is not applied to metal cooling. You'd need a separate CLOUDY table with it applied
    chem.dust_chemistry = 0 

    chem.grackle_data_file = bytearray(
        #"/home/alanman/Desktop/repositories/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
        "/pool001/zimi/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
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


    fc['density'][:] = rho / chem.density_units
    fc['metal'][:] = rho / chem.density_units * np.sum(Zarr[:, 2:], axis=-1)
    
    # Start supporting non-equilibrium?
    tiny_number = 1e-20
    d_to_h_ratio = 3.4e-5
    # Should I be setting the fractions based on the free-electron abundance?
    if chem.primordial_chemistry > 0:
        fc["HII"][:] = Zarr[:, 0] * fc["density"]
        fc["HI"][:] = tiny_number * fc["density"]
        fc["HeI"][:] = Zarr[:, 1] * fc["density"]
        fc["HeII"][:] = tiny_number * fc["density"]
        fc["HeIII"][:] = tiny_number * fc["density"]
        fc["de"][:] = fc["HII"] + fc["HeII"] / 4.0 + fc["HeIII"] / 2.0
    if chem.primordial_chemistry > 1:
        fc["HM"][:] = tiny_number * fc["density"]
        fc["H2I"][:] = tiny_number * fc["density"]
        fc["H2II"][:] = tiny_number * fc["density"]
    if chem.primordial_chemistry > 2:
        fc["DI"][:] = 2.0 * d_to_h_ratio * fc["density"]
        fc["DII"][:] = tiny_number * fc["density"]
        fc["HDI"][:] = tiny_number * fc["density"]

    fc["specific_heating_rate"][:] = 0.
    fc["volumetric_heating_rate"][:] = 0.
    fc["x-velocity"][:] = 0.0
    fc["y-velocity"][:] = 0.0
    fc["z-velocity"][:] = 0.0

    fc.calculate_mean_molecular_weight()

    if has_u:
        fc['energy'][:] = u_int / chem.velocity_units**2       # Specific energy has velocity^2 units
    else: 
        fc['energy'][:] = temperature / chem.temperature_units / \
                          fc["mu"] / (chem.Gamma - 1.0)

    fc.calculate_cooling_time()

    cooling_rate = fc['energy'] / fc['cooling_time'] / fc['density']
    cooling_time = fc['cooling_time']# * chem.time_units

    if not chem_units:
        cooling_time *= chem.time_units
        cooling_rate *= chem.cooling_units

    return cooling_rate, cooling_time, chem

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
    
    # get mask for halo and shattered cells
    halo_mask = np.load(f"/pool001/zimi/grackle/grackle_data_files/input/halo_mask_{file_of_snapshot}.npy")
    shattered_mask = np.load(f"/pool001/zimi/grackle/grackle_data_files/input/shattered_mask_{file_of_snapshot}.npy")
    gas = fh['PartType0']
    n_gas = fh['Header'].attrs['NumPart_ThisFile'][0]

    h = fh['Header'].attrs["HubbleParam"]
    z = fh['Header'].attrs['Redshift']
    a = 1 / (1 + z)
    density_unit = 1e10 * h**2 / a**3 * mass_sun_cgs /  cm_per_kpc**3       # Multiply to convert to cgs
    if subsel is None:
        subsel = slice(None)

    # Converting all to CGS
    rho = np.load(f"/pool001/zimi/grackle/grackle_data_files/input/n_c_{file_of_snapshot}.npy")
    u_int = gas['InternalEnergy'] #[subsel]
    u_int = u_int[halo_mask]
    u_int = u_int[shattered_mask] * 1e10     # km^2/s^2 -> cm^2/s^2
    Zarr = gas['Metallicity']# [subsel, :]
    Zarr = Zarr[halo_mask][shattered_mask]
    Tcool_peak = 10**4.3

    outdir = snapdir

    os.makedirs(outdir, exist_ok=True)
    fn_out = os.path.join(
        outdir, f"cooling_600_{phase}.{file_of_snapshot}.hdf5"
    )
#    fout = h5py.File(fn_out, 'a')
#    if "Cooling" in fout.keys():
#        if any(np.isnan(fout['Cooling/rate'][()])):
#            fout.close()
#            os.remove(fn_out)
#    if not os.path.exists(fn_out):
    cool_rate, cool_time, chem = cooling_func(rho, Zarr, z, u_int=u_int, chem_units=True, return_time=True)
    cool_rate_low, cool_time_low, chem_low = cooling_func(rho, Zarr, z, temperature=Tcool_peak, chem_units=True, return_time=True)
    fout = h5py.File(fn_out, 'a')
    print(f"\t writing to {fn_out}")
    if not "Ambient" in fout.keys():
        fout.create_group("Ambient")
        fout.create_dataset("Ambient/rate", (n_gas,), dtype="<f4", fillvalue=np.nan)
        fout.create_dataset("Ambient/time", (n_gas,), dtype="<f4", fillvalue=np.nan)
        fout["Ambient/rate"].attrs["to_cgs"] = chem.cooling_units
        fout["Ambient/time"].attrs["to_cgs"] = chem.time_units

    if not "Cold" in fout.keys():
        fout.create_group("Cold")
        fout['Cold'].attrs['T_cold'] = Tcool_peak
        fout.create_dataset("Cold/rate", (n_gas,), dtype="<f4", fillvalue=np.nan)
        fout.create_dataset("Cold/time", (n_gas,), dtype="<f4", fillvalue=np.nan)
        fout["Cold/rate"].attrs["to_cgs"] = chem_low.cooling_units
        fout["Cold/time"].attrs["to_cgs"] = chem_low.time_units

    #fout['Ambient/rate'][subsel] = cool_rate[()]
    #fout['Ambient/time'][subsel] = cool_time[()]
    fout['Cold/rate'][subsel] = cool_rate_low[()]
    fout['Cold/time'][subsel] = cool_time_low[()]

    fout.close()
    fh.close()

# CHANGE THIS
ph = 'cold'

snapdir = sys.argv[1]
nchunks = int(sys.argv[2]) if len(sys.argv) >= 3 else 5000

nparts_per = get_snapshot_nparts(snapdir, 0)

for fi, npts in enumerate(nparts_per):
    chunksize = npts // nchunks
    complete = np.zeros(npts).astype(bool)
    for ci in range(nchunks+1):
        slc = slice(ci*chunksize, min((ci+1)*chunksize, npts), 1)
        calc_cooling_gizmo(
            snapdir, 
            file_of_snapshot=fi,
            subsel=None, phase = ph,
        )
        complete[slc] = True
        print(f"Chunk {ci} of {nchunks}")
    print(fi, all(complete))

