import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py
import gizmo_analysis as gizmo
from halo import sim_halo
import utilities as ut
from astropy import units as u
from astropy import constants as const
import json
import os

gamma = 5/3
xi = 10
k_b = const.k_B.cgs

if __name__ == "__main__":
    sim_name = 'm12f_res7100'
    halo = sim_halo(sim_name)

    halo.read_gas_files(cooling = True)
    halo_gas = halo.in_halo(region = 'cgm', save = True)

    halo_gas = pd.concat(halo_gas)

    max_potential = np.max(halo.gas_data[2]['potential'])
    max_nfw_potential = np.max(halo.gas_data[0]['nfw_potential'])


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





    
    


