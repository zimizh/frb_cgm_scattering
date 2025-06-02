import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py
import gizmo_analysis as gizmo
import halo_analysis as halo
import utilities as ut
from astropy import units as u
import json
import os


def gas_in_halo(sim_name, region = 'all', assign_hosts = 'track', assign_hosts_rotation = True):
    sim_dir = f'/pool001/zimi/analysis/FIRE/{sim_name}/'

    # read gas particles at z=0, store as python dictionary
    part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, sim_dir, assign_hosts=assign_hosts, assign_hosts_rotation=assign_hosts_rotation)
    hal = halo.io.IO.read_catalogs('redshift', 0, sim_dir, species='gas')


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
    temp = gas['temperature']
    metallicity = gas['massfraction'][:,0] # metals mass fraction
    helium = gas['massfraction'][:,1] # metals mass fraction

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
            'bin_size': binsize,
            'temp': temp,
            'metallicity': metallicity,
            'helium': helium}

    gas_df = pd.DataFrame(data)

    host_index = hal['host.index'][0]
    host_radius = hal['radius'][host_index]

    if region == 'all':
        mask = (gas_df['dist_tot'] <= host_radius)
    elif region == 'cgm':
        mask = (gas_df['dist_tot'] <= host_radius) & (gas_df['dist_tot'] >= host_radius*0.15)
    
    halo_gas = gas_df[mask]

    return halo_gas, host_radius


def get_box_crossings_z(halo_gas, ray_location):
    """
    Given an (x,y) location, finds the intersecting gas particles along the z-axis
    (perpendicular to the face of galactic plane)

    Parameters
    --------
    halo_gas: Dataframe
        properties of gas contained in halo
    """

    ray_location = np.asarray(ray_location)

    if not(ray_location.shape == (2,)):
        raise ValueError('location must be (2,) array')
    
    binsizes = halo_gas['bin_size']
    
    # keep only the particle bins the ray goes through
    line = halo_gas[['dist_x', 'dist_y']] - ray_location

    dist_sq = line['dist_x']**2+line['dist_y']**2
    halo_gas['dist_ray_sq'] = dist_sq

    mask = (dist_sq < binsizes**2)
    gas_intersect = halo_gas[mask]

    return gas_intersect

def ray_trace(halo_gas, ray_location):

    gas_intersect = get_box_crossings_z(halo_gas, ray_location)

    # get chord length 
    # chord_len should have units of kpc
    chord_len = 2*np.sqrt(gas_intersect['bin_size']**2-gas_intersect['dist_ray_sq'])

    # get DM
    dm = np.sum(chord_len*gas_intersect['n_e'])* u.kpc/u.cm**-2

    return dm.to(u.pc/u.cm**-2)
            
        
if __name__ == "__main__":
    sim_name = 'm12f_res7100'
    halo_gas, host_radius = gas_in_halo(sim_name, 'cgm')

    dms = []
    x= []
    y= []


    for i in np.linspace(200, 356, 156, endpoint = False):
    # dms.append([])
        for j in np.linspace(-355, 355, 711):
            dm = ray_trace(halo_gas, [i,j]).value
            dms.append(dm)
            x.append(i)
            y.append(j)

    np.save("outputs/m12f_res7100/dms_355kpc_cgm_6",dms)


    # fig, ax = plt.subplots()

    # h = ax.hist2d(x, y, weights = dms, bins = [150,150], norm=matplotlib.colors.LogNorm())
    # ax.set_xlabel('$\Delta x$ [ckpc/h]')
    # ax.set_ylabel('$\Delta y$ [ckpc/h]')
    # ax.axis('equal')
    # fig.colorbar(h[3], label = 'DM [pc/$cm^3$]')

    # plt.show()
    # fig.savefig(os.path.join("outputs", sim_name, 'dm_355kpc_cgm'), dpi=200)





    
    


