import os
import numpy as np
import pandas as pd
import h5py
import scipy
import gizmo_analysis as gizmo
import halo_analysis as halo
import utilities as ut
from astropy import units as u
from pathlib import Path
from nfw_methods import phi_nfw
from utils import Rodrigues

class sim_halo:
    """
    An object for the primary halo of a simulation.
    """
    
    def __init__(self, name='m12f_res7100', sim_dir=None, 
                 cooling_dir = None,
                 binsize = 1,
                 emap_dir= None):
        """
        Parameters
        ----------
        name: str
            The name of the simulation, e.g. 'TNG205n2500'. Used for setting
            the default simulation and map directory paths.
        binsize: int, optional
            The size of the bin for the electron density map, in ckpc/h. Must
            divide the simulation boxsize. Default: 1
        sim_dir: str, optional
            Directory to simulation snapshot files. Defaults to the path on
            Illustris JupyterHub.
        emap_dir: str, optional
            Directory to electron density map. Defaults to './data/n_e_maps'
        """
        
        self.sim_name = name
        home_dir = '/ceph/submit/data/user/z/zimi/analysis/FIRE/'

        if sim_dir is None:
            self.sim_dir = home_dir + name
        else:
            self.sim_dir = sim_dir

        if cooling_dir is None:
            self.cooling_dir = home_dir + f'{name}/cooling_output'
        else:
            self.cooling_dir = cooling_dir

        # read halo information using gizmo
        # set halo and halo radius
        self.host = halo.io.IO.read_catalogs('redshift', 0, self.sim_dir, species='gas')
        self.host_index = self.host['host.index'][0]
        self.host_radius = self.host['radius'][self.host_index] * u.kpc
        self.host_mass = self.host['mass.200c'][self.host_index] * u.M_sun
        self.scale_radius = self.host['scale.radius'][self.host_index] * u.kpc
        self.host_pos = self.host['position'][self.host_index]
        
        self.binsize = binsize*u.kpc
        self.n_bins = int(self.host_radius / self.binsize)

        self.header = gizmo.io.Read.read_header(self.sim_dir)
        self.h = self.header['hubble']
        self.scalefactor = self.header['scalefactor']
        

    def read_gas(self, cooling = False):
        # read gas particles at z=0, store as python dictionary

        num_gas = []
        dir = Path(self.sim_dir + '/output/')
       
        for file in dir.glob(f"snapshot_600.*.hdf5"):
            part = h5py.File(file, 'r')
            gas = part["PartType0"]

            num_gas.append(len(gas['Coordinates']))

        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, self.sim_dir, assign_hosts='halo', assign_hosts_rotation=True)
        gas = part['gas']

        positions = gas['position']
        x = positions[:,0]
        y = positions[:,1]
        z = positions[:,2]

        dist = gas.prop('host.distance.principal')
        dist_x = dist[:,0]
        dist_y = dist[:,1]
        dist_z = dist[:,2]
        dist_tot = gas.prop('host.distance.total')

        dens = gas['density']
        electron_abundance = gas['electron.fraction']
        ne = electron_abundance*gas.prop('number.density') # electron number density, units:cm^-3
        binsize = gas['size']
        mol_mass = gas.prop('molecular.mass')
        # int_energy = gas.prop('internal.energy')
        temp = gas['temperature']
        metallicity = gas['massfraction'][:,0] # metals mass fraction
        helium = gas['massfraction'][:,1] # metals mass fraction
        potential = gas['potential']

        # make pandas dataframe
        data = {'x': x,
                'y': y,
                'z': z,
                'dist_x': dist_x,
                'dist_y':dist_y,
                'dist_z': dist_z,
                'dist_tot': dist_tot,
                'density': dens,
                'electron_abundance': electron_abundance,
                'n_e': ne,
                'smoothing_len': binsize,
                # 'internal_energy': int_energy,
                'temperature': temp,
                'metallicity': metallicity,
                'helium': helium,
                'potential': potential}
        
        if cooling:
            cooling_times, cooling_rates = self.get_cooling()
            data['cooling_rates'], data['cooling_times'] = cooling_rates, cooling_times

        self.gas_data = pd.DataFrame(data)
        self.num_particles = num_gas

    def read_gas_files(self, cooling = False):
        self.gas_data = []
        dir = Path(self.sim_dir + '/output/')

        if cooling:
            cooling_times, cooling_rates = self.get_cooling()

        # apply unit conversions
        mass_conversion = 1e10 / self.h  # multiply by this for [M_sun]
        length_conversion =  self.scalefactor / self.h  # multiply for [kpc physical]
        time_conversion = 1 / self.h  # multiply by this for [Gyr]

        def assign_hosts_rotation(
            part, species_name='gas', distance_max=10, mass_percent=90, age_percent=25
        ):
            '''
            Compute and assign rotation tensor and ratios of principal axes
            (defined via the moment of inertia tensor) for each host galaxy.
            By default, use stars for baryonic simulations, or if no stars in catalog, use gas.

            Parameters
            ----------
            part : dictionary class
                catalog of particles at snapshot
            species_name : string
                name of particle species to use to determine rotation
            distance_max : float
                maximum distance to select particles [kpc physical]
            mass_percent : float
                keep particles within the distance that encloses mass percent [0, 100] of all particles
                within distance_max
            age_percent : float
            keep youngest age_percent of (star) particles within distance cut
            '''

            principal_axes = ut.particle.get_principal_axes(
                part,
                species_name,
                distance_max,
                mass_percent,
                age_percent,
                center_positions=part.host['position'],
                center_velocities=part.host['velocity'],
                return_single_array=False,
                verbose=True,
            )

            if principal_axes is not None and len(principal_axes) > 0:
                for prop_name in principal_axes:
                    part.host[prop_name] = principal_axes[prop_name]
                    for spec_name in part:
                        part[spec_name].host[prop_name] = principal_axes[prop_name]

       
        for i, file in enumerate(sorted(dir.glob(f"snapshot_600.*.hdf5"))):
            with h5py.File(file, 'r') as part:
                part.info = self.header
                part.snapshot = {
                    'index': 600,
                    'redshift': 0,
                    'scalefactor': 1.0,
                    # 'time': self.header['time'],
                    'time.lookback': 0,
                    'time.hubble': None,
                }
                
                # HARD CODED CHANGE FOR EACH HALO
                # m12f
                part.host = {
                    'position': np.array([38711.78, 47665.06, 46817.31]),
                    'velocity': np.array([-156.13,  162.88,  110.12]),
                    'rotation': np.array([[ 0.08447786, -0.05280387,  0.99502525],
                                        [-0.83146679,  0.54657421,  0.09959724],
                                        [-0.54911426, -0.83574421,  0.00226876]]),
                    'axis.ratios': np.array([[0.155538  , 0.17101623, 0.90949266]]),
                }
                
                #m12b
                # part.host = {
                #     'position': np.array([39257.39045089, 41609.99354669, 39190.05643334]),
                #     'velocity': [],
                #     'acceleration': [],
                #     'rotation': np.array([[ 0.57996381, -0.81105026, -0.07641635],
                #                         [ 0.72333475,  0.55583914, -0.40967022],
                #                         [ 0.37473834,  0.1823193 ,  0.90902742]]),
                #     'axis.ratios': np.array([0.13753059, 0.15669618, 0.87768946])
                # }

                #m12z
                # part.host = {
                #     'position': np.array([72937.33, 73589.12, 73689.64]),
                #     'velocity': np.array([ 3.4749103, 74.35693  , 25.895018 ]),
                #     'acceleration': [],
                #     'rotation': np.array([[ 0.95412356,  0.2123023 ,  0.21113016],
                #             [-0.09696471,  0.8862399 , -0.4529644 ],
                #             [-0.28327733,  0.41171184,  0.8661682 ]]),
                #     'axis.ratios': np.array([0.36434507, 0.4700129 , 0.775181  ]),
                # }
                
                # part['PartType0'].host = self.host
                gas = part["PartType0"] 

                # unrotated x,y,z distances
                temp_dist = ut.coordinate.get_distances(
                            gas['Coordinates']/self.h,
                            self.host_pos,
                            part.info['box.length'],
                            part.snapshot['scalefactor'],
                        )  # [kpc physical]

                # want dist x, y, z relative to galactic center, and oriented s.t. galaxy is face on
                # get rotation matrix
                # assign_hosts_rotation(part, species_name='PartType0')

                
                # align distances with host principal axes
                rot_dist = ut.coordinate.get_coordinates_rotated(
                    temp_dist, part.host['rotation']
                )*u.kpc     # [kpc physical]

                #calculate total distance
                shape_pos = 1
                dist_tot = np.sqrt(np.sum(rot_dist**2, shape_pos))

                dens = gas['Density']* mass_conversion / length_conversion**3 * u.M_sun/(u.kpc**3) # units: to [M_sun]/[kpc^3]
                electron_abundance = gas['ElectronAbundance']
                
                #calculate number density
                h_number_dens = dens[:] * ut.constant.proton_per_sun * ut.constant.sun_massfraction['hydrogen'] / u.M_sun # hydrogen number density [H atoms/cm^3]
                ne = electron_abundance * h_number_dens # electron number density, units:kpc^-3
                
                binsize = gas['SmoothingLength']* length_conversion * (np.pi / 3) ** (1 / 3) / 2 * u.kpc

                metallicity = gas['Metallicity'][:,0] # metals mass fraction
                helium_mass_fracs = gas['Metallicity'][:,1] # helium mass fraction

                sfr = gas['StarFormationRate']

                # potential from data
                if 'potential' in gas.keys(): 
                    potential = gas['Potential'][:]/self.scalefactor * u.km**2 /(u.s**2)   # convert from km^2/s^2 to cm^2/s^2
                else: 
                    potential  = [-1]*len(dens) * u.km**2 /(u.s**2)             # if no potential information recorded, then return an array of -1
                # potential from nfw0
                nfw_potential = phi_nfw(dist_tot, self.scale_radius, self.host_mass, self.host_radius)
                
                ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
                mus = (1 + 4 * ys_helium) / (1 + ys_helium + electron_abundance)
                molecular_weights = mus * ut.constant.proton_mass
                int_energy = gas['InternalEnergy']
                int_energy *= (
                    ut.constant.centi_per_kilo**2
                    * (5 / 3 - 1)
                    * molecular_weights
                    / ut.constant.boltzmann
                )
                
                data = {'dist_x': rot_dist[:,0].cgs,
                        'dist_y': rot_dist[:,1].cgs,
                        'dist_z': rot_dist[:,2].cgs,
                        'dist_tot': dist_tot.cgs,       #[cm]
                        'density': dens.cgs,            # [g]/[cm^3]
                        'electron_abundance': electron_abundance,       # num electron per proton, unitless
                        'n_e': ne.cgs,                                      #[num electron]/[cm^3]
                        'smoothing_len': binsize.cgs,                #[cm]
                        'temperature': int_energy,          #[K]
                        'metallicity': metallicity,         #linear mass fraction
                        'helium': helium_mass_fracs,
                        'star_formation_rate': sfr,  #star formation rate[Msun]/[yr]
                        'potential': potential.cgs ,            #[cm^2]/[s^2]
                        'nfw_potential': nfw_potential.cgs}
            
                if cooling:
                    data['cooling_rates'], data['cooling_times'] = cooling_rates[i], cooling_times[i]
            
                data_chunk = pd.DataFrame(data)
                self.gas_data.append(data_chunk)
            

    def set_gas(self, gas_data):
        self.gas_data = gas_data
    def get_gas(self):
        return self.gas_data

    def set_shattered(self, shattered_data):
        self.shattered_gas = shattered_data
    def get_shattered(self, shattered_data):
        return self.shattered_gas
    
    def set_rotated(self, rotated_data):
        self.rotated_gas = rotated_data

    
    def get_cooling(self, phase = 'Ambient'):
        """
        returns cooling rate and times
        phase: Ambient or Cold
        """
        cooling_times = []
        cooling_rates = []
        
        dir = Path(self.cooling_dir)
        for file in sorted(dir.glob(f"cooling_600_{phase.lower()}.*.hdf5")):
            with h5py.File(file, 'r') as fh:
            
                cooling_time_seconds = fh[f'{phase}/time'][()] #* fh[f'{phase}/time'].attrs['to_cgs']     # Cooling times in seconds
                cooling_rate_cgs = fh[f'{phase}/rate'][()] #* fh[f'{phase}/rate'].attrs['to_cgs']    # Cooling rate in erg sec^-1 cm^3

                cooling_times.append(cooling_time_seconds)
                cooling_rates.append(cooling_rate_cgs)

        return cooling_times, cooling_rates


    def in_halo(self, data = None, region = 'all', save = False):
        """
        Given dataset of particles, return subset of particles in the halo

        Parameters
        --------
        data: Dataframe
            properties of particles. Must contain
        region:
            'all': disk and CGM, <= 1 virial radius
            'cgm': CGM only, 0.15 virial radius <= dist_tot <= 1 virial radius
            'disk': galactic disk only, dist_tot <= 0.15 virial radius
        """
        if data is None:
            data = self.gas_data

        result = []
        rad = self.host_radius.cgs

        for i, chunk in enumerate(data):
            if 'dist_tot' not in chunk.keys():
                raise KeyError('data must contain column "dist_tot"')
            
            if region == 'all':
                mask = (chunk['dist_tot'] <= rad) 
            elif region == 'cgm':
                mask = (chunk['dist_tot'] <= rad) & (chunk['dist_tot'] >= rad*0.15)
            elif region == 'disk':
                mask = (chunk['dist_tot'] <= rad*0.15)

            result.append(chunk[mask])

            if save:
                np.save(f'/ceph/submit/data/user/z/zimi/analysis/FIRE/{self.sim_name}/grackle_input/halo_mask_{i}', mask)

        return result

    def cooling_cold_prep():
        return 0
    
    def find_n_c(self):
        return 0

    def get_box_crossings(self, ray_location, gas):
        """
        Given an (x,y) location, finds the intersecting gas particles in z direction

        Parameters
        --------
        halo_gas: Dataframe
            properties of gas contained in halo
        rotation: (theta, phi)
        """
        # if shattered_only:
        #    gas = self.shattered_gas
        # else:
        #     gas = self.gas_data
        
        ray_location = np.asarray(ray_location)

        if not(ray_location.shape == (2,)):
            raise ValueError('location must be (2,) array')
        
        binsizes = gas['smoothing_len']
        # max_size = np.max(binsizes)
        
        # calculate distance from particle center to ray location
        line = gas[['dist_x', 'dist_y']] - ray_location

        # particles that are insize a max_size*max_size box outside ray location
        # inside_ind = (line['dist_x'] < max_size and line['dist_y'] < max_size)

        dist_sq = line['dist_x']**2+line['dist_y']**2
        gas['dist_ray_sq'] = dist_sq

        mask = (dist_sq < binsizes**2)
        gas_intersect = gas[mask]

        return gas_intersect

    def ray_trace_dm(self, ray_location, rotation = (0,0)):

        gas_intersect = self.get_box_crossings(ray_location, self.rotated_gas)

        # get chord length 
        # chord_len should have units of kpc
        chord_len = 2*np.sqrt(gas_intersect['smoothing_len']**2-gas_intersect['dist_ray_sq'])
        dm = np.sum(chord_len*gas_intersect['n_e_final'])
    
        return dm
    
    def ray_trace_scattering(self, ray_location):
        z_l = 0
        d_l = 3.085678e+27/2 # 500*u.mpc
        d_s = 3.085678e+27 # 1000*u.mpc
        d_ls = 3.085678e+27/2  #500*u.mpc

        gas_intersect = self.get_box_crossings(ray_location, self.rotated_gas)

        # get chord length 
        # chord_len should have units of cm
        chord_len = 2*np.sqrt(gas_intersect['smoothing_len']**2-gas_intersect['dist_ray_sq'])
        ratio = chord_len/gas_intersect['smoothing_len']

        # num_intercepted = gas_intersect['num_intercepted'].sum()
        # alpha_sq_tot = (ratio*gas_intersect['alpha_sq']).sum()
        
        N_img = 1 + (2*((1/(1+z_l))*(d_l*d_ls/d_s))**2) * np.sum(ratio*gas_intersect['num_intercepted']*(gas_intersect['delta_m']/gas_intersect['final_size'])**2)
        
        # tau = (1+z_l)/(2*const.c.cgs) * (d_l*d_ls/d_s) * (alpha_sq) # seconds

        return N_img #num_intercepted, N_img, alpha_sq_tot

