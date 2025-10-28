import functools

import numpy as np
import os
import h5py
from pygrackle import chemistry_data
from pygrackle.fluid_container import FluidContainer
from pygrackle.utilities.physical_constants import (
    mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc, cm_per_kpc, mass_sun_cgs, cm_per_km, boltzmann_constant_cgs
)

from scipy.optimize import root_scalar



# note: I'm going to submit a PR to add a slightly modified version of this function
#       directly to grackle
def infer_eint_for_temperature_tabulated(
    chem,
    target_temperature,
    density,
    metal_density,
    *,
    cgs_density=False,
    xtol=None,
    rtol=None,
    maxiter=None
):
    """Infers internal energy assocaited with the target temperature

    Parameters
    ----------
    chem
        Chemistry object
    target_temperature
        Holds the target temperature (units of Kelvin)
    density
        Holds the total mass density (units are determined by the
        ``cgs_density`` kwarg)
    metal_density
        Holds the metal mass density (units are determined by the
        ``cgs_density`` kwarg)

    Note
    ----
    This function uses numerical root finding.

    While we could theoretically lookup a temperature by directly reading
    the cloudy table, that result wouldn't be exactly right. As the Grackle
    paper explains, Grackle uses a "dampener" when it computes the
    temperature a cloudy table. Consequently, it
    is "more correct" to use numerical root finding.
    """
    densityU = chem.density_units
    eintU = chem.get_velocity_units()**2
    gm1 = chem.Gamma - 1.0

    density = np.array(density)
    metal_density = np.array(metal_density)
    target_temperature = np.array(target_temperature)

    # todo: get a little more flexible about broadcasting
    # todo: all metal_density to be omitted
    assert (density.ndim == 1) and (density.size > 0)
    assert density.shape == metal_density.shape == target_temperature.shape

    if chem.primordial_chemistry != 0:
        raise ValueError("Grackle wasn't configured in tabulated mode")

    # ensure that density & metal_density reference values in code units
    if cgs_density:
        density, metal_density = density / densityU, metal_density / densityU
    else:
        density, metal_density = density, metal_density

    # allocate the output array
    eint = np.empty(shape=density.shape, dtype="f8")

    # we are going to repeatedly call scipy.optimize.root_scalar. Let's set some stuff
    # up that we will reuse every time...
    fc = FluidContainer(chem, n_vals=1)

    def f(specific_internal_energy, target_T):
        fc["internal_energy"][0] = specific_internal_energy
        fc.calculate_temperature()
        return fc["temperature"] - target_T

    common_kwargs = {
        "f": f, "method": "bisect", "xtol": xtol, "rtol": rtol, "maxiter": maxiter
    }

    # move onto the actual loop:
    for i in range(density.size):
        # lets get bounds on the specific internal energy:
        # -> we could give more precise bounds on the mmw, and the way
        #    that metal density influences the value by reading the code
        #    or reading the discussion in GH-issue#67 (the paper actually
        #    discusses this, but actually had a mistake)
        mmw_bounds = np.array([0.6, 2.0])
        eint_bounds_cgs = (
            (boltzmann_constant_cgs * target_temperature[i]) /
            (gm1 * mmw_bounds * mass_hydrogen_cgs)
        )
        # convert to code units
        eint_bounds = eint_bounds_cgs / eintU
 
        # define the function used in root finding
        fc["density"][0] = density[i]
        fc["metal_density"][0] = metal_density[i]
 
        root_result = root_scalar(
            args=(target_temperature[i],), bracket=eint_bounds, **common_kwargs
        )
        if root_result.converged:
            eint[i] = root_result.root
        else:
            eint[i] = np.nan
    return eint

# --------------------------------------------------

def calc_rho_metals_eint(chem, ndens_H_cgs, T_kelvin,  metal_mass_frac = 0.03):
    """
    Calculates the mass density, metal density, and (specific) internal energy
    using the cloudy table

    Note
    ----
    It's **REALLY** important that chem is already initialized
    """

    ndens_H_cgs = np.array(ndens_H_cgs)
    T_kelvin = np.array(T_kelvin)
    num_points = ndens_H_cgs.size
    assert num_points > 0
    if ndens_H_cgs.ndim != 1 or ndens_H_cgs.shape != T_kelvin.shape:
        raise ValueError("This function currently requires 1D arrays")
    elif chem.primordial_chemistry != 0:
        raise ValueError("This function currently just supports using cloudy tables")

    # internally grackle uses:
    #    rho_H = chem.HydrogenFractionByMass * (density - metal_density)
    #    (this choice is influenced by the way that grackle was defined)
    # so, let's now compute the metal-free mass density:
    rho_H_cgs = ndens_H_cgs * mass_hydrogen_cgs
    density_metalfree_cgs = rho_H_cgs / chem.HydrogenFractionByMass

    density_cgs = rho_H_cgs / (1.0 - metal_mass_frac)
    metal_density_cgs = density_cgs * metal_mass_frac

    density = density_cgs / chem.density_units
    metal_density = metal_density_cgs / chem.density_units

    return {
        "density": density,
        "metal_density": metal_density,
        "internal_energy": infer_eint_for_temperature_tabulated(
            chem=chem,
            target_temperature=T_kelvin,
            density=density,
            metal_density=metal_density,
            cgs_density=False,
            rtol=1e-6
        )
    }


def setup_chem(redshift=0.0):
    chem = chemistry_data()
    chem.use_grackle = 1
    chem.with_radiative_cooling = 1     # Default example has this as 0
    chem.metal_cooling = 1 
    chem.UVbackground = 1               # Add heating from UV background (only used if primordial_chemistry != 0)
    chem.primordial_chemistry = 0       # Tabulated cooling (no chemistry). Assumes ionization equilibrium.
    chem.H2_self_shielding = 0 
    chem.self_shielding_method = 0      # Paper says self-shielding is sketchy when UV background is low or density is high.
                                        # Self-shielding is not applied to metal cooling. You'd need a separate CLOUDY table with it applied
    chem.dust_chemistry = 0 

    # chem has been able to properly handle regular strings for a long time now...
    chem.grackle_data_file = (
        "/ceph/submit/data/user/z/zimi/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5"
    )

    # you need to set the specific and volumetric heating rates to 0 unless you
    # explicitly provide specific and volumetric heating rates
    chem.use_specific_heating_rate = 0
    chem.use_volumetric_heating_rate = 0

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
    return chem

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
    density_unit = 1e10 * h**2 / a**3 * mass_sun_cgs /  cm_per_kpc**3       # Multiply to convert to cgs
    if subsel is None:
        subsel = slice(None)

    Zarr = gas['Metallicity'][subsel, :]
    metallicity = Zarr[:,0]
    frac_He = Zarr[:,1]
    frac_H = 1 - metallicity - frac_He

    chem = setup_chem(redshift=z)

    outdir = snapdir
    os.makedirs(outdir, exist_ok=True)
    fn_out = os.path.join(
        outdir, f"cooling_600_{phase}.{file_of_snapshot}.hdf5"
    )
    
    if phase == 'cold':
        # get mask for halo and shattered cells
        halo_mask = np.load(snapshot_path + f"/grackle_input/halo_mask_{file_of_snapshot}.npy")
        shattered_mask = np.load(snapshot_path + f"/grackle_input/shattered_mask_{file_of_snapshot}.npy")

        # Converting all to CGS
        n = np.load(snapshot_path + f"/grackle_input/n_c_{file_of_snapshot}.npy")   # cold phase density in cg
        n_H = n*(frac_H[halo_mask][shattered_mask])
        Zarr = Zarr[halo_mask][shattered_mask]
        Tcool_peak = 10**4.3
        
        # Because it is most convenient for hydro codes, the core Grackle functionality was
        # designed to map specific internal energies and mass densities to other quantities.
        #
        # The following function returns a dict holding the mass density, metal density,
        # and internal energy (all in code units) that are consistent with the specified nH
        # and T
        rslt = calc_rho_metals_eint(chem, n_H, [Tcool_peak]*(n_H.size), metal_mass_frac = metallicity[halo_mask][shattered_mask])

        # now, use Grackle to compute the cooling time
        fc = FluidContainer(chem, rslt["density"].size)
        for key, arr in rslt.items():
            fc[key][...] = arr
        fc.calculate_cooling_time()
        cooling_time_s = fc["cooling_time"] * chem.time_units
        cooling_rate = fc['internal_energy'] / fc['cooling_time'] / fc['density'] * chem.cooling_units

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
    
        fout['Cold/rate'][subsel] = cooling_rate[()]
        fout['Cold/time'][subsel] = cooling_time_s[()]
    
    elif phase == 'ambient':
        f_e = gas['ElectronbAundance'][subsel]
        mu = 1/((1-frac_He) + frac_He/4 + (1-frac_He)*f_e)
        mbar = mu*mass_hydrogen_cgs           # mean molecular mass
        rho = gas['Density'][subsel] * density_unit     #cgs
        n = rho/mbar
        n_H = n*frac_H

        u_int = gas['InternalEnergy'][subsel] * cm_per_km**2     # km^2/s^2 -> cm^2/s^2
        # temp = u_int *  (5 / 3 - 1)* mbar / boltzmann_constant_cgs

        # rslt = calc_rho_metals_eint(chem, n_H, temp, metallicity = metallicity)
        
        # use Grackle to compute the cooling time
        fc = FluidContainer(chem, n_H.size)
        fc['density'][...] = rho/chem.density_units
        fc['metal_density'][...] = rho * metallicity/chem.density_units
        fc['internal_energy'][...] = u_int / chem.velocity_units**2
        fc.calculate_cooling_time()
        fc.calculate_cooling_rate()
        cooling_time_s = fc["cooling_time"] * chem.time_units
        # cooling_rate = fc['internal_energy'] / fc['cooling_time'] / fc['density'] * chem.cooling_units
        cooling_rate = fc["cooling_rate"]
    
        fout = h5py.File(fn_out, 'a')
        print(f"\t writing to {fn_out}")
        
        if not "Ambient" in fout.keys():
            fout.create_group("Ambient")
            fout.create_dataset("Ambient/rate", (n_gas,), dtype="<f4", fillvalue=np.nan)
            fout.create_dataset("Ambient/time", (n_gas,), dtype="<f4", fillvalue=np.nan)
            # fout["Ambient/rate"].attrs["to_cgs"] = chem.cooling_units
            # fout["Ambient/time"].attrs["to_cgs"] = chem.time_units

        fout['Ambient/rate'][subsel] = cooling_rate[()]
        fout['Ambient/time'][subsel] = cooling_time_s[()]

        # ndensH_cgs = np.geomspace(1e-6, 10.0, num=351)
        # T_kelvin = np.logspace(3.5, 7.5, num=201)
        # nH_flat_grid, T_flat_grid = map(
        #     np.ndarray.flatten, np.meshgrid(ndensH_cgs, T_kelvin, indexing='ij')
        # )

    fout.close()
    fh.close()

    
if __name__ == "__main__":
    # CHANGE THIS
    snapdir = '/ceph/submit/data/user/z/zimi/analysis/FIRE/m12f_res7100'#sys.argv[1]
    nchunks = 1 #int(sys.argv[2]) if len(sys.argv) >= 3 else 5000
    ph = 'cold' # sys.argv[3]

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



